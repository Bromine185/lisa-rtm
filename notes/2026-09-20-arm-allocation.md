# Arm allocation for the 50-epoch run — eight arms

**Date:** 2026-09-20. **Decision:** the user's, after reviewing the `OV3_fast` scoreboard.
**Supersedes** the four-arm version earlier in this file's history; the budget ceiling was lifted and
the set rebuilt as a factorial.
**Evidence:** [`2026-09-16-score-geometry-decoder-noise-readouts.md`](2026-09-16-score-geometry-decoder-noise-readouts.md)
§3.1/§3.3/§3.4, [`analysis/dynamics_OV3_fast.md`](analysis/dynamics_OV3_fast.md),
[`analysis/gated_OV3_fast.md`](analysis/gated_OV3_fast.md),
[`analysis/scale_OV3_fast.md`](analysis/scale_OV3_fast.md). All `OV3_fast` figures are step 16 000 =
7.62 epochs.

## The set

```python
ARMS = {
    "det_paper":       ("det",         0.0,  "LISAS"),   # pure L1 — LISA's released config
    "det":             ("det",         1e-2, "LISAS"),   # our deterministic baseline
    "es_marg":         ("es_marg",     1e-2, "LISAS"),   # reference sampler, no ERB
    "es_erb_l0.001":   ("es_marg_erb", 1e-3, "LISAS"),   # ┐
    "es_erb_l0.01":    ("es_marg_erb", 1e-2, "LISAS"),   # ├ ERB λ-ladder
    "es_erb_l0.1":     ("es_marg_erb", 1e-1, "LISAS"),   # ┘ (the OV3_fast winner)
    "es_dec_l0.01":    ("es_marg",     1e-2, "LISASD"),  # decoder noise at matched λ
    "es_dec_erb_l0.1": ("es_marg_erb", 1e-1, "LISASD"),  # decoder noise + ERB at high λ
}
```

All eight at **R = 4** (12 kHz → 48 kHz), batch 64, 104 950 steps. A 2× arm was considered and
dropped: its CRPS, deficit and HB-LSD are computed over 12–24 kHz rather than 6–24 kHz, so it cannot
share a scoreboard column with the others, and without a 2× deterministic baseline the result would
be descriptive rather than comparative.

## What each contrast isolates

| contrast | isolates |
|---|---|
| `det_paper` vs `det` | the multi-resolution spectral loss itself (λ = 0 against λ = 1e-2) |
| `det` vs `es_marg` | the energy score — deterministic against sampler, matched λ |
| **`es_marg` vs `es_erb_l0.01`** | **the ERB geometry, at matched λ = 1e-2** |
| `es_erb_l0.001 / 0.01 / 0.1` | the spectral weight, within the ERB geometry |
| `es_marg` vs `es_dec_l0.01` | decoder noise, at matched λ = 1e-2 |
| `es_erb_l0.1` vs `es_dec_erb_l0.1` | decoder noise, at high λ |

**The λ confound raised against the earlier four-arm set is closed.** `es_marg` and `es_erb_l0.01`
now sit at the same λ = 1e-2 and differ only in the ERB term, so the winner's advantage is
attributable within this run rather than by citing the shorter `OV3_fast`.

## `det_paper` — what "the original config" actually is

From `configs/audio/lisa.yaml` at ml-postech/LISA `master` (HEAD `4f3c1cd`), reached via
`scripts/setup.sh` → `config_file_path`:

```yaml
loss: l1
optimizer: {name: adam, args: {lr: 1.0e-3}}
epoch_max: 50
multi_step_lr: {milestones: [10,20,25,30,35,40], gamma: 0.5}
train_dataset: {..., input_sr: 8000, gt_sr: 48000, gt_aug_max: 3, sample_q: 8000}, batch_size: 64
val_dataset:   {..., input_sr: 12000, gt_sr: 48000}, batch_size: 8
model: imnet siren+ {dim_hidden: 128, num_layers: 5, dim_latent: 96}, encoder conv-enc {latent_dim: 32, kernel_size: 7}
```

Three findings.

1. **`loss: l1` — the released config has no spectral term at all.** So "the original config" is
   **λ = 0**, not our λ = 1e-2. This also explains the audit's observation that their Table 2 ablation
   of the multi-resolution spectral loss moves LSD by +0.00/+0.00/+0.01: the shipped config does not
   use it.
2. **`epoch_max: 50` with milestones `[10,20,25,30,35,40]`** is exactly 0.2/0.4/0.5/0.6/0.7/0.8 of 50,
   and `batch_size: 64` matches. This repo already inherited the paper's schedule and batch faithfully;
   "50 epochs" is genuinely their number.
3. **`dim_hidden: 128`, where this repo uses `dec_hidden = 144`.** A divergence nobody had flagged.
   `dim_latent: 96` = 3 × 32 neighbours, so the 97-wide decoder input matches; only the hidden width
   differs.

**Not reproduced literally:** their training wrapper is `input_sr: 8000` with `gt_aug_max: 3`, i.e.
1×–3× from 8 kHz, while validation is 12 kHz. The released config never trains the 4× setting of
Table 1 (as `2026-09-17-lisa-reported-numbers-audit.md` §3.2 notes). A literal reproduction would
train a different task and be incomparable to the other seven arms. So `det_paper` means **their loss
and hyperparameters, our task**.

**DECIDED:** decoder width stays at this repo's **144** for `det_paper`, not their 128, so that λ is
the only thing varying across the eight arms and no second architecture enters the run. The cost is
that `det_paper` is their *loss and schedule*, not their *architecture*; say so when reporting it.

**DECIDED — not taken:** with λ = 0 the det branch still computes the three STFT scales and
multiplies by zero. A two-line skip would make `det_paper` cheaper than `det`; judged not worth the
code risk, so both det arms cost the same.

## Cut from the original seven, and why

| arm | reason |
|---|---|
| `es_split_l0.1` | Not a sampler. Deficit −9.28 with noise off, −9.41 with it on: a **0.13 dB swing** where the ERB arms swing 12.7 dB. `def1 − def0` is −0.06 and moved *backwards* over training. Worst CRPS of any sampler (0.913), worst PIT (0.609), worst deficit (−15.18), most stalled (end-slope −0.05, 5 upticks of 8). P2 refuted on every clause. |
| `es_marg_l0.1` | Superseded. Its job — the λ control — is now done inside the ERB ladder. |
| `es_dec_l0.1` | Replaced by `es_dec_l0.01`, which puts decoder noise at the *matched* λ = 1e-2 against `es_marg` rather than at an unmatched 1e-1. |

## Cost

43.008 Mrows/step (2 det × 3.072 + 6 ES × 6.144). Cost law `t = 9.0 + 22.9367 · Mrows` → **995 ms/step**.

| topology | wall-clock | cost |
|---|---|---|
| 8 parallel 1-GPU rentals, each torn down when its arm ends | 4.67 h | **$59.5** |
| 8-GPU node | 4.67 h | $66.9 |
| 1 GPU, all eight arms | 29.0 h | $51.9 |

det arms finish at 2.32 h and their rentals terminate; ES arms run to 4.37 h. Add ~0.3 h staging per
instance. All figures are A100-equivalent and carry the unmeasured Blackwell factor.

**The hard limit is not the $60 preference, it is the account.** `maxOverdraftCents = 0`: the balance
stops at **$101** mid-step. Across the Blackwell bracket the parallel plan is $53.6 (fast) / $67
(bandwidth-scaled) / **$136 (pessimistic — exceeds the balance and the run dies)**. The $0.36
calibration is therefore a feasibility test, not a cost optimisation, and checkpoint/resume is
mandatory so that a balance stop is recoverable after a top-up rather than a total loss.

## Consequences for the rewrite

1. **`LISASD` stays** — two arms use it: `n_dec = 4`, the per-output-sample `eps_dec` tensor
   (~98 MB/step), and a 101-input first decoder layer.
2. **Both readouts are needed.** The ERB arms score at `logmean16`, the others at one draw.
3. **`batch_tag` / `val_tag` must be held identical across all eight rentals** while output-path tags
   vary. Passing `f"{tag}_{arm}"` as `tag` reaches `stream(f"{tag}/batches")`, which is blake2b-hashed
   on the label, and silently gives each arm a different batch stream. Eight separate VMs, no shared
   filesystem, no existing test catches it. Cross-machine G3 (each instance publishes a hash of its
   first ~1000 drawn index arrays; all eight must match or the run aborts) is the only guard.
4. **Per-arm `torch.Generator`**, fixed per-arm key, training and validation, declared in the run log
   as CHANGES-TRAJECTORY.
5. **Self-termination on completion and on failure**, per rental.

## Scale freedom, for the record

`analysis/scale_OV3_fast.md`: the decoder generalises off its trained coordinate lattice without any
scale augmentation — pooled median curvature ratio 1.18, nothing bumping at c = ±0.25 or ±0.75 on any
arm, and the ×8 query reproducing the ×4 query to 232 dB at shared samples. The operating rule is
**query at ×4 or above and decimate; below ×4 aliases**. The *input* rate remains fixed at 12 kHz and
is untested — these arms are any-output-rate, not any-to-any in LISA's sense, because nothing here
trains with input-rate augmentation.
