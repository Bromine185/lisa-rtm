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

---

## Fan-out: four separate 1-GPU rentals (decision, 2026-09-20)

**There is no 4-GPU SKU.** Live `rental-options` offers `gpuCount` 1 at $1.79/hr and 8 at $14.32/hr
only, priced exactly linearly. So "four GPUs" is four **independent** 1-GPU rentals, never the 8-GPU
node with four idle cards.

| plan | wall-clock | cost |
|---|---|---|
| 1 GPU, all four arms in one loop | 14.64 h | $26.21 |
| **4 x 1-GPU, each torn down when its arm ends** | **4.37 h** | **$27.62** |
| 8-GPU node, four cards idle | 4.37 h | $62.59 — rejected |

Independent billing is what makes this nearly free: `det` is 79.5 ms/step against an ES arm's
149.9 ms, so its rental terminates at 2.32 h ($4.15) while the three ES arms run to 4.37 h ($7.82
each). +$1.41 over the single-GPU plan for a 3.3x shorter wall-clock.

Sensitivity to the degenerate intercept: $27.62 / $30.96 / $34.09 at 9 / 25 / 40 ms.
**Unresolved:** at the pessimistic Blackwell bracket (time x2.29) this reaches ~$63 and breaches the
$60 ceiling. The $0.36 single-GPU calibration decides it before four instances are launched.

### Mandatory for this topology

1. **`batch_tag` / `val_tag` identical across all four rentals**, output-path tag varying. Passing
   `f"{tag}_{arm}"` as `tag` reaches `stream(f"{tag}/batches")` (`e2b_fast.py:591-592`), which is
   blake2b-hashed on the label, and silently gives each arm a different 105k-step batch stream and a
   different validation set. On four separate VMs there is no shared filesystem to catch it.
2. **Cross-machine G3.** Each instance hashes the first ~1000 drawn `(idx, starts)` arrays and
   publishes the digest *before* training starts; all four must match or the run aborts. Costs
   seconds, and it is the only thing standing between this topology and a silently void experiment.
3. **Per-arm `torch.Generator`** with a fixed per-arm key, for training and validation. Declared in
   the run log as CHANGES-TRAJECTORY (the arms currently draw from one shared generator in stack
   order, and Philox offsets follow `multiProcessorCount`, so the realised noise was never portable
   from the A100 anyway).
4. **Self-termination on completion and on failure**, per rental. A forgotten instance bills at
   $1.79/hr indefinitely.

### Operational notes

- Each VM stages the 37.3 h corpus independently (~15-20 min, inside its own rental clock): four
  independent failure points, and no shared volume since storage volumes appear to be bare-metal-only.
- Billing granularity is unverified. If it rounds to the hour, `det` costs $5.37 rather than $4.15.
- The `LISASD` arm (`es_dec_erb_l0.1`) carries the extra `eps_dec` traffic and a 101-input first
  decoder layer, so it — not the plain ES arms — most likely sets the 4.37 h pace. The cost law was
  fitted on the seven-arm mix and does not resolve the LISASD premium.
