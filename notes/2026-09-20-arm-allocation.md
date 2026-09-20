# Arm allocation for the 50-epoch run — four of seven

**Date:** 2026-09-20. **Decision:** the user's, after reviewing the `OV3_fast` scoreboard.
**Evidence:** [`2026-09-16-score-geometry-decoder-noise-readouts.md`](2026-09-16-score-geometry-decoder-noise-readouts.md)
§3.1/§3.3/§3.4, [`analysis/dynamics_OV3_fast.md`](analysis/dynamics_OV3_fast.md),
[`analysis/gated_OV3_fast.md`](analysis/gated_OV3_fast.md). All figures are step 16 000 = 7.62 epochs.

## The set

```python
ARMS = {
    "det":             ("det",         1e-2, "LISAS"),    # control
    "es_marg":         ("es_marg",     1e-2, "LISAS"),    # the 8 Sep reference sampler
    "es_erb_l0.1":     ("es_marg_erb", 1e-1, "LISAS"),    # the winner
    "es_dec_erb_l0.1": ("es_marg_erb", 1e-1, "LISASD"),   # decoder noise, best gating
}
```

21.504 Mrows/step (det 3.072 + 3 × 6.144). Cost law `t = 9.0 + 22.9367 · Mrows` → **502 ms/step**,
**14.6 h**, **$26.21** at $1.79/GPU/hr for 104 950 steps on one GPU. Against 925 ms / 27.0 h / $48.27
for all seven. (The law's intercept is statistically degenerate — see the optimisation plan §0.2 — so
treat these as ±10 % until M4b measures it.)

## Cut, and why

| arm | reason |
|---|---|
| **`es_split_l0.1`** | Not a sampler. Deficit with noise off −9.28, with noise on −9.41: a **0.13 dB swing** where the ERB arms swing 12.7 dB — the noise channel is not connected to anything. `def1 − def0` is −0.06 and moved *backwards* over training (−0.19 from 4k→16k). Worst CRPS of any sampler (0.913), worst calibration (PIT 0.609), worst deficit (−15.18). Most stalled: end-slope −0.05 with 5 upticks of 8 in the last quarter. Pre-registered prediction P2 refuted on every clause. |
| **`es_dec_l0.1`** | P4 refuted — decoder noise alone leaves the deficit at −10.79, below the −5 dB refutation line. Dominated by `es_marg` on CRPS (0.702 vs 0.697) and by `es_erb_l0.1` on everything. Its decoder-noise contrast is served more cleanly by `es_erb_l0.1` vs `es_dec_erb_l0.1`, which differ only in that channel. |
| **`es_marg_l0.1`** | The λ control. Cut deliberately — see the confound below. |

## The confound this creates, stated plainly

`es_erb_l0.1` runs at λ = 1e-1 and `es_marg` at λ = 1e-2, so **within this run the ERB geometry is
confounded with a 10× spectral weight.** The arm that separates them, `es_marg_l0.1`, is cut.

The defence is the `OV3_fast` measurement itself: at 7.62 epochs `es_marg_l0.1` was *worse* than
`es_marg` on CRPS (0.805 vs 0.697), LSD (1.004 vs 0.946) and audio ViSQOL (2.701 vs 2.760), with PIT
end bins 0.538 against 0.412 — P1 refuted on every clause. So raising λ alone does not help, and the
ERB term is the plausible cause of the winner's advantage.

**What this run cannot do is re-establish that at 50 epochs.** Any claim that the ERB geometry (rather
than the spectral weight) produces the win must cite the shorter run. If a reviewer asks whether
λ = 1e-1 alone would have caught up by epoch 50, the honest answer is that this run does not test it.
Re-adding `es_marg_l0.1` costs +$7.40 and would close it.

## Consequences for the rewrite

1. **`LISASD` stays.** `es_dec_erb_l0.1` keeps the decoder-noise class alive: `n_dec = 4`, the
   per-output-sample `eps_dec` tensor (~98 MB/step), and a first decoder layer of 101 inputs rather
   than 97. The class cannot be dropped to simplify the trainer.
2. **Both readouts are still needed.** `es_erb_l0.1` and `es_dec_erb_l0.1` are scored at
   `logmean16`, the other two at one draw — the evaluation must carry both paths.
3. **Three of four arms are ES**, so the step is 3 × 2 draws + 1 single = 7 forward sequences' worth
   of decoder rows, not 13 as at seven arms.

## Headroom

$60 ceiling − ~$31 all-in (training + calibration + staging + evaluation) leaves **~$29**.
A second seed of `es_erb_l0.1` costs **+$7.40** (502 → 643 ms/step, 14.6 → 18.8 h) and would give the
headline claim the confidence interval it currently lacks — the repo's own notes flag one seed and no
CIs as the weakest point in the result. Not taken as of this note; the four arms above stand.
