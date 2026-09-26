# The diagnosis, taken to four released SOTA checkpoints

**Date:** 2026-09-26. Inference only. FLowHigh, NU-Wave 2, AP-BWE, AudioSR (speech), their released
weights, their own inference code, one scorer.

## 0. Predictions, written before any SOTA output was scored

SNR splits by band. The low band is copied, so almost all error sits above the cut. For a high band
with true energy E, output energy pE and coherent fraction c = Re<Y,X>/<Y,Y>, the high-band error is
E(1 + p − 2c). The empty band (p = 0, c = 0) scores the ceiling C. So a model that reports SNR S, with
c ≈ 0 and a perfect low band, has

    p = 10^((C − S)/10) − 1.

Any low-band error only lowers S further, so this p is an upper bound. A reported SNR is a meter on
high-band energy.

Applied to NU-Wave 2's own Table 1, where the "Input" row is their ceiling on their own protocol:

| input | ceiling (their Input row) | NU-Wave 2 | implied p | implied energy |
|---|---|---|---|---|
| 8 kHz | 19.0 | 18.8 | ≤ 0.047 | ≤ −13.3 dB |
| 12 kHz | 22.1 | 21.6 | ≤ 0.122 | ≤ −9.1 dB |
| 16 kHz | 24.7 | 24.0 | ≤ 0.175 | ≤ −7.6 dB |
| 24 kHz | 29.4 | 28.4 | ≤ 0.259 | ≤ −5.9 dB |

**P1.** NU-Wave 2 at 12 kHz is too quiet above 6 kHz. Its broadband (energy-weighted) high-band ratio
is below −6 dB, unless its coherent fraction is above 0.1. Refuted if the measured broadband ratio is
above −6 dB with c < 0.05.

**P2.** Every sampler is under-dispersed. The SNR gap (mean of M draws against one draw) falls short
of 10 log10(2 / (1 + 1/M)). Refuted for a model if its gap is within 20 % of the target.

**P3.** No released model's measured SNR sits above the empty-band ceiling on the same utterances.
Refuted by any model with measured SNR above C.

**Found while wiring FLowHigh (before scoring):** as shipped, FLowHigh is not a sampler. `inference.py`
calls `sample(..., std_2=1.)` without `std_1`; `sample()` then resets both to `(1, sigma)`, with
`sigma = 1e-4`. The start point is `cond + 1e-4 N(0, I)`. Two seeds differ by 7.6e-6 relative RMS.

**P4.** Even with the trained prior restored (`std_1 = std_2 = 1`, `flowhigh_std1`), FLowHigh stays nearly
deterministic. Reason: its training target at t = 0 is `u = (x1 − x0) − ε` and the input is `x0 + ε`, so
the optimal field gives `x0 + ε + v*(x0 + ε, 0) = E[x1 | cond]` for every ε. One Euler step from t = 0
returns the conditional mean. It is regression to the mean by construction. Prediction: HB spread of
`flowhigh_std1` below 0.1 of calibrated. Refuted if above 0.3.

## The residual: level regresses per utterance (2026-09-27)

p236–p238 through the same harness (`sota/work/ourtest`, both input filters; the filter moves the
deficit by 0.04 dB for the sampler and 0.07 dB for det): the sampler's one draw is −4.3 dB per band there
against −0.7 on the paper split; det −10.2 against −6.2. Per utterance, the deficit tracks the truth's
high-band share: r = −0.61 for one draw and −0.58 for det over 62 utterances (`ourtest`, 30, plus `core`, 32;
an ad hoc computation, not a committed script). Bright utterances come out 6–10 dB under, dark ones 2–3 dB
over. **Reproducible baseline** (added after P5 was written; `demo/tools/make_results_sota.py` computes it):
over `wide` + `ourtest`, 270 utterances, es_dec_erb_l0.1 has r = −0.41 and slope −0.29 dB per dB; det
r = −0.42, slope −0.28. P5's "from −0.6" refers to the 62-utterance figure; judge P5 on the 270-utterance
pool, where the target (r above −0.2) is unchanged. The loud-frame deficit is −5.7 dB
against −1.5. The model emits the corpus's typical level; the truth moves around it. Regression to the mean,
one level up. High-band spread on these speakers is 0.23 of calibrated (0.46 on the paper split).

**Predictions P5–P7, written before training** (arms in `fast/arms_env.py`, trainer `fast/train_env.py`,
notebook `sampler/lisa_rtm_train_env.ipynb`; same batch stream as OV50, control `es_dec_erb_l0.1`):

- **P5.** `es_ctx_env_l0.1` (22 ms context + per-frame HB log-energy energy score): the deficit's
  correlation with the utterance's HB share goes from −0.6 to above −0.2; the loud-frame deficit on
  p236/p238 closes from −8/−6 dB to better than −3 dB. Refuted if the correlation stays below −0.4 or the
  loud-frame deficit stays below −5 dB.
- **P6.** Context moves it more than the objective: `es_ctx_erb_l0.1` improves the correlation by more
  than `es_env_l0.1` does. Refuted if `es_env_l0.1` alone reaches P5's target.
- **P7.** Neither arm changes the coherent fraction: stays below 0.03. Context fixes the level, not the
  phase.
