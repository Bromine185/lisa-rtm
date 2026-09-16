# Research note — the score's geometry, noise where the fine structure lives, and the right readout for each judge

**Date:** 2026-09-16 (run `OV3_es`, Colab A100-80GB; predictions written before the run, results filled in after)
**Question:** the 8 Sep sampler wins the proper score and loses the perceptual judges. Is that a fact about samplers, or about *this* sampler's distance, its noise pathway, and the statistic we read out of it?
**Rule of the day (user, 16 Sep):** a winner must improve CRPS and calibration *and* LSD + audio-mode ViSQOL together. Stochastic sampling is the object of study.

---

## 0. Three moves, one model class

The energy score is strictly proper for the law of $\phi(y)$ whenever $d(a,b) = \|\phi(a) - \phi(b)\|$. Nothing in that theorem says $\phi$ has to be the log-magnitude STFT that LSD happens to use; and nothing says the noise has to enter at the encoder. The 8 Sep sampler made both choices by default. Tonight varies each one at a time, on identical batches and one init.

1. **The geometry of $d$** (what the score is proper for, and what the judges measure). `es_marg` uses $\ell_1$ on log-magnitude bins: proper for per-bin marginals only. Two alternatives: the Euclidean norm over the whole log-magnitude spectrogram (`es_ged`, Gritsenko et al. 2020's choice, strictly proper for the joint law), and $\ell_1$ on log ERB-band energies (`es_erb`: 32 gammatone-like bands to 24 kHz, the neurogram ViSQOL's NSIM is computed on). Plus the weight of the spectral term inside $d$ raised from $10^{-2}$ to $10^{-1}$ (`es_marg_l0.1`).
2. **Where the noise enters.** LISAS feeds 8 Gaussian channels at the 12 kHz input; every 48 kHz output sample inside one input interval is then a deterministic function of the same three latents. The band above 6 kHz has period shorter than four output samples, so its randomness has to be smuggled through the latents. `LISASD` adds 4 Gaussian channels per *output* sample at the decoder input (+~600 weights): noise at the rate of the fine structure it has to generate.
3. **The readout.** LSD is squared error in log-power, so its minimiser is the conditional mean of the *log*-magnitude. The waveform ensemble mean estimates the mean waveform (its log-magnitude is far below), and a single draw estimates nothing. The sampler can produce the LSD-optimal object directly: per-bin mean of $\log|Y|$ across $M$ draws, phase of draw 0, resynthesised (`logmean16`). One model, three readouts: draw for CRPS and listening, waveform mean for SNR, log-magnitude mean for LSD.

## 1. Arms (identical batches, one init per class, batch 32 × 1 s, $\lambda$ inside $d$)

The literature synthesis (`notes/lit/2026-09-16-synthesis-experiment-menu.md`) sharpened the plan before launch: the deficit is a bias in an *aggregate* (band energy) that a per-bin proper score never constrains, and the waveform term dominates $d$ in a band that is 98 % incoherent with the input. So the spectral weight is raised ten-fold in every new arm, one arm restricts the waveform term to the low band (the sampler's `det_split`), and the ERB term is kept as an *added* aggregate-proper term on log band energies rather than a replacement.

| arm | class | $d$ | $\lambda$ | tests |
|---|---|---|---|---|
| `det` | LISAS, $\tau = 0$ | paper loss | $10^{-2}$ | reference |
| `es_marg` | LISAS | $d_w + \lambda d_m$ | $10^{-2}$ | reference sampler (8 Sep recipe) |
| `es_marg_l0.1` | LISAS | $d_w + \lambda d_m$ | $10^{-1}$ | spectral geometry dominant |
| `es_split_l0.1` | LISAS | $d_w(\Pi_{\mathrm{lo}}\cdot) + \lambda d_m$ | $10^{-1}$ | no waveform term above 6 kHz |
| `es_erb_l0.1` | LISAS | $d_w + \lambda (d_m + d_{\mathrm{ERB}})/2$ | $10^{-1}$ | plus the aggregate-proper term on log ERB-band energies |
| `es_dec_l0.1` | LISASD | $d_w + \lambda d_m$ | $10^{-1}$ | noise at the output rate |
| `es_dec_erb_l0.1` | LISASD | $d_w + \lambda (d_m + d_{\mathrm{ERB}})/2$ | $10^{-1}$ | both |

$d_{\mathrm{ERB}}(a,b) = \frac13\sum_s \mathrm{mean}_{b,t}\,\lvert \log\sqrt{E_{b,t}(a)} - \log\sqrt{E_{b,t}(b)}\rvert$ with $E_{b,t} = \sum_k W_{bk}\lvert Y_{t,k}\rvert^2$ over 32 triangular bands equally spaced on the ERB-rate scale from 50 Hz to 24 kHz; inside the two-draw estimator this is a sum of CRPS terms on each band's log energy, which is exactly the aggregate a bin-marginal score leaves free. Dropped from the first draft: `det_l0.1` (the 4 Sep $\lambda = 10^{-1}$ checkpoint already gives that ceiling) and `es_ged` (lowest expected payoff).

Validation loss: each arm's own objective on 8 fixed held-out batches (speakers p236–p237, utterances disjoint from the evaluation set), every 500 steps, fixed noise seeds, no anchor jitter. Training loss: EMA 0.98 of the per-step loss.

## 2. Predictions, written before the run

- **P1 (weight).** `es_marg_l0.1` beats `es_marg` on LSD and audio ViSQOL of a single draw, keeps a flat PIT and equal or lower CRPS. Refuted if CRPS rises by more than 5 % or the PIT end bins exceed 0.15.
- **P2 (low-band waveform term).** `es_split_l0.1` has a smaller single-draw energy deficit than `es_marg_l0.1` at equal or better CRPS, because no term asks for zero above 6 kHz. Refuted if its deficit is not smaller.
- **P3 (judge's geometry).** `es_erb_l0.1` gains at least 0.10 audio-mode ViSQOL over `es_marg_l0.1` at equal CRPS, and its single-draw high-band energy sits within 2 dB of the truth on the Hub set; LSD moves less than ViSQOL does.
- **P4 (decoder noise).** `es_dec_l0.1` closes the in-distribution energy gap: single-draw Hub deficit from −7.9 dB to within ±2 dB, and the SNR gap between mean-of-16 and one draw within 0.5 dB of the calibrated 2.75 dB. Refuted if the deficit stays below −5 dB.
- **P5 (combination).** `es_dec_erb_l0.1` is the only arm that improves on `es_marg` in both families at once.
- **P6 (deterministic ceiling).** No deterministic arm can win overall: `det`'s CRPS stays at the MAE of a point forecast, about 1.0, whatever its LSD.
- **P7 (readout).** `logmean16` of any calibrated sampler beats that sampler's single draw and the deterministic `det` on LSD, because it estimates the LSD minimiser directly. Refuted if `logmean16` LSD is not below both.

## 3. Results

*(filled in after the run)*

## 4. What it means

*(filled in after the run)*

## 5. Prior art this builds on

Energy score and its properness: Gneiting & Raftery 2007. Two-draw feed-forward training: DISCO Nets (Bouchacourt et al. 2016), Gritsenko et al. 2020 (spectral energy distance, the $\ell_2$ log-magnitude geometry), Engression (Shen & Meinshausen 2025). Noise at every layer: EnScale (Schillinger et al. 2025). The factor of two between a draw and the mean: Blau & Michaeli 2018. The readout argument is the perception–distortion reading of the ensemble: the optimal estimator under a distortion is the conditional mean *in that distortion's coordinates*, and the sampler gives it for any coordinates you name.
