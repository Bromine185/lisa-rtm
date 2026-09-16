# Research note — evaluation audit, the training objective, and where the hypotheses stand

**Date:** 2026-09-01
**Project:** `lisa-rtm` — regression to the mean in LISA, and transport maps against it
**Runs referenced:** `checkpoints/FULL/lisa.pt` (λ = 1.0, the original run) and
`checkpoints/FULL_lam0.001/lisa.pt` (λ = 1e-3, retrained today on an A100-80GB). Both in Drive
`MyDrive/lisa_rtm/`. Full per-rung numbers for the second run are in Drive `RESEARCH_FULL.md`.
**Status of the repo:** `build_notebook.py`, `lisa_rtm.ipynb`, `TODO.md`, `audit_cell.py`, and this
note are uncommitted local changes. The corrected notebook also lives in Drive
`lisa_rtm/lisa_rtm.ipynb` and is what Colab now opens.

---

## 0. Summary

1. The alarming held-out plot (−35…−45 dB across *all* frequencies) did not reproduce on a clean
   kernel. It was stale kernel state, not a VCTK source mismatch. Held-out evaluation is sound.
2. The audit found a real defect the plot had hidden: the original model had the right magnitude
   spectrum and a **random-phase waveform** (SNR −5.8 dB, worse than emitting silence). Cause: the
   notebook weighted the multi-scale STFT term at **λ = 1.0**; the paper's official code uses
   **λ = 1e-3**, and the paper's own ablation shows the term is nearly inert. At λ = 1.0 the
   phase-blind term was 92 % of the loss and the waveform L1 *rose* during training.
3. Retraining at λ = 1e-3 fixed the phase (SNR 18.13 dB, equal to naive upsampling) and exposed the
   next layer: under a near-pure L1 objective, on 0.54 h of speech, the model's high band collapses
   to ≈ −20 dB. It is a sinc interpolator that emits nothing above 6 kHz.
4. LSD 1.6 is not a mystery: it is the −19.6 dB deficit restated in log10-power RMS units. LSD
   *worsened* when the model got better, which is hypothesis H4 observed directly.
5. Hypothesis status: the regression-to-the-mean **mechanism** is confirmed in the extreme; the
   metric critique (H4) is strongly supported; the transport claim (H2/H5) is **untested**, because
   neither run produced the object it targets — a *coherent but under-energetic* high band. The
   paper's own SNR (24.16 dB) implies that at paper scale the deficit is modest (−2…−6 dB, inferred),
   which is the risk the whole project stands on.

---

## 1. What was believed, and what was true

| believed at the start | found |
|---|---|
| Held-out cache came through a different pipeline (HF repack) and is not comparable to train | Same code path, clean kernel: baseband energy ratio −0.45 dB (held-out) vs −0.58 dB (train); high-band deficit −8.96 vs −11.82 dB. Held-out is fine — slightly better than train. |
| The −40 dB curve is either a caching bug, a representation mismatch, or catastrophic generalization failure | None of these. A hand-added scratch cell had rebound `test_utts = train_utts[:12]` (its output read `temporary test: (12,)`), and `EVAL_HAT` was evidently computed against a different `EVAL`. The curve never reproduced. |
| The model "learned to emit some high-frequency content" (−11 dB deficit) | It emitted the right *energy* with random *phase*. Band-energy ratios cannot see phase; SNR can. |
| `grad_clip = 1e-3` looked suspiciously aggressive | It is the paper's stated value ("gradient clipping with 0.001 max norm"). Not a bug. |
| "Retraining is cheap, so that isn't the problem" | Correct, and retraining was exactly the fix — but for the objective, not the data. |

---

## 2. Timeline of the diagnosis

### 2.1 Same-code-path audit (λ = 1.0 checkpoint, 12 train vs 12 held-out utterances)

```
                       TRAIN      HELD-OUT
target RMS             0.1120     0.0499
prediction RMS         0.1084     0.0480
SNR                   -5.84 dB   -5.77 dB
baseband energy ratio -0.58 dB   -0.45 dB
high-band deficit    -11.82 dB   -8.96 dB
weights vs lisa.pt     max |dw| = 0.0          input x2 -> output x1.988
```

Held-out passes every structural check. The anomaly is the SNR, identical on both splits.

### 2.2 Ruling out an offset

Full cross-correlation: lag 0 or +1 sample (≤ 0.02 ms) on every utterance; aligned SNR equals
as-is SNR. Not a shift.

### 2.3 Phase swap — the decisive test

```
                                        TRAIN      HELD-OUT
prediction as-is                       -5.81 dB    -5.73 dB
prediction magnitude + TARGET phase   +17.29 dB   +15.69 dB
target magnitude + PREDICTION phase    -5.94 dB    -5.89 dB
relative magnitude error              -17.22 dB   -15.77 dB
naive resample_poly upsample          +16.62 dB   +15.06 dB
```

Magnitudes are within ~16 dB of the target and, given the target's phase, beat naive upsampling.
The prediction's own phase is worthless. Correct spectrogram, random waveform.

### 2.4 Why: the loss

```
final loss terms   wave 0.0836   spec 1.0287   -> spectral term is 92 % of the loss
wave L1  first/last   0.0557 -> 0.0836    (rose: worse than the near-silent random init)
spec     first/last   2.7504 -> 1.0287
```

`lambda_spec = 1.0` in both presets. The multi-scale STFT term is magnitude-only; at 12× the L1
term it dominated the gradient, and the optimizer traded phase for magnitude. It optimized what it
was given.

### 2.5 What the paper actually does (arXiv:2111.00195v2; official repo `ml-postech/LISA`)

*Stated in the paper:* loss `L1_wave + λ · L_multi-scale-STFT`, λ never printed; Adam, lr 1e-3,
50 epochs, lr halved on a schedule, grad clip 1e-3; VCTK 48 kHz, first 99 speakers train; encoder
{7,3,3,1}/{16,32,64,32}, 5-layer ReLU MLP; SNR (Eq. 4) and LSD (Eq. 5, log|STFT|², all bins).
Table 1, ×4 (12k→48k): LISA SNR **24.16** / LSD **0.81**; WSRGlow 19.41 / 1.01; TFilm 19.51 / 2.02;
AudioUNet 18.55 / 2.11. Ablation "−spec": ΔSNR ≤ 0.08 dB, ΔLSD ≤ 0.01.

*From the released code (not the paper):* `spec_coeff = 0.001`; shipped config `loss: l1` (no
spectral term at all); batch 64; 1 s chunks with 8000 sampled output coordinates; `MultiStepLR`
milestones epochs 10/20/25/30/35/40, γ = 0.5; train speakers id < 350 (torchaudio VCTK = v0.80).
LSD: `n_fft 2048`, hop 1024, log10, ε = 1e-8 — and the code appears to compute `log10(power²)`,
i.e. 4·log10|STFT|, twice the quantity in its own Eq. (5). *(Inference from reading the code; not
verified empirically.)*

**Consequence:** LISA is, in practice, an L1-waveform model. The repro's λ was 1000× the reference.

### 2.6 Corrections made to `build_notebook.py` (notebook regenerated; smoke-validated end to end)

| change | where |
|---|---|
| `lambda_spec` 1.0 → **1e-3**, both presets | Config |
| `MultiStepLR`, halved at 20/40/50/60/70/80 % of steps (official milestones as fractions) | Config, §8 |
| Waveform SNR and naive-baseline SNR logged at every checkpoint; third panel on the training plot | §8 |
| Resume guard: a checkpoint whose optimisation config differs raises instead of continuing | §8 |
| Run dir keyed by λ (`FULL_lam0.001`); the λ = 1.0 checkpoint is preserved as a reference | Config |
| `save_ckpt`: write to local tmp, copy to Drive with retries (Colab's Drive FUSE dropped a directory under ~20 s writes and killed a run at step 3500); `ckpt_every` 500 → 1000 | §8 |
| §9 pre-flight: reload both splits from the `.npz` caches, warn on speaker overlap, reload weights | §9 |
| Gate 2a (SNR > naive) placed ahead of gate 2b (headroom); paper-basis LSD printed with caveat | §9, §15 |

Deliberately unchanged, to keep one variable: batch 16, 0.256 s segments, 10 speakers, 20k steps.

### 2.7 Retrain, λ = 1e-3 (A100-80GB, 20k steps, ~43 ms/step)

Windowed (1000-step) mean of the waveform L1 from the every-25-steps history:

```
window        wave mean   wave std     lr
    0-1000      0.0143     0.0095    1.0e-3
 1000-2000      0.0081     0.0024
 3000-4000      0.0056     0.0019
 5000-6000      0.0041     0.0013    5.0e-4
 8000-9000      0.0035     0.0013    2.5e-4
10000-11000     0.0035     0.0012    1.25e-4
12000-13000     0.0037     0.0013    6.25e-5
15000-16000     0.0034     0.0010    3.1e-5
```

Converged at 0.0034 ± 0.001 from step 8000 (the single-batch prints that looked like an increase
are ~3–5σ batches; the per-batch std is a third of the mean). Waveform L1 is 24× lower than the
λ = 1.0 run.

Dev probe: SNR 14.88 dB vs naive 14.89, **flat from step 11000**, while the high-band deficit
deepened −15 → **−24.2 dB**. The model became a sinc interpolator.

§9, 12 held-out utterances:

```
SNR        18.13 dB   naive 18.16 dB   paper 24.16 dB     -> GATE 2a FAIL by 0.03 dB (a tie)
LSD        1.609      paper-basis (2048/1024) 1.588        -> not like-for-like with 0.81
HB deficit -19.55 dB                                       -> GATE 2b pass (headroom exists)
```

### 2.8 The ladder on this model (pipeline evidence only — gate 2a failed)

| condition | SNR dB | HB-LSD | HB deficit dB |
|---|---|---|---|
| T0 identity | 18.13 | 1.846 | −19.68 |
| **T1 quantile (λ=1)** | 16.57 | **1.226** | −2.83 |
| T2 block (best λ=0.9) | 17.66 | 1.315 | −7.34 |
| T3 conditional (best λ=0.8) | 17.39 | 1.397 | −5.25 |
| S stochastic (1 draw) | 16.37 | 1.356 | −1.29 |
| control: frame shuffle | 17.28 | 1.585 | −6.11 |
| control: shaped noise | 17.14 | 1.746 | −5.57 |
| **control: mis-specified T1** | 16.52 | **1.370** | −5.76 |

CRPS: deterministic 1.352 → stochastic 1.134. Transport latency 29 ms per second of audio.

The mis-specified map beats T3 and the shaped-noise control on HB-LSD. With T0's high band on the
ε floor, *any* injected energy scores as a large HB-LSD gain. The controls were built for exactly
this and they fired on the first clean run.

---

## 3. Why LSD is 1.6 and not 0.8

**Arithmetic.** LSD = mean over frames of the RMS over bins of Δlog10(power). A −19.55 dB deficit
is 1.955 log10 units. In the evaluation basis (n_fft 1024, k_cut 128) 385 of 513 bins — 75 % — lie
above 6 kHz. Near-perfect baseband plus a uniform ~2-unit error above 6 kHz gives
√(0.75 · 1.955²) ≈ **1.69**. Observed: 1.609 full-band, 1.846 high-band-only. LSD 1.6 *is* the
deficit, in other units.

**LSD moved the wrong way when the model got better.** λ = 1.0: LSD 1.36, waveform garbage.
λ = 1e-3: LSD 1.61, waveform correct. LSD preferred the broken model because it only sees
magnitudes and the broken model matched them. Same story in the ladder: a wrong map improved
HB-LSD 1.85 → 1.37. This is H4 (§4) observed twice without trying.

**The paper's 0.81 means their model emits most of the high band, coherently.** 0.81 log10 units
≈ 8 dB RMS per-bin error against our ~18. Their SNR of 24.16 dB — several dB above any naive
baseline — is only reachable by reconstructing high-band *waveform*, which requires conditional
structure (harmonic continuation, per-phone fricative spectra) that a model can only learn from
data volume. Ours saw 0.54 h from 10 speakers (42 epochs of a 400-utterance subset) with the lr at
1e-5 by step 12k; theirs saw ~99 speakers for 50 epochs at batch 64. Under a pure L1 objective the
loss-optimal output for anything unpredictable is zero, and that is what the model produced.
Separately, 0.81 is on a different STFT basis and possibly a different scaling (§2.5), so no
single number here should be quoted as "the gap". SNR — Eq. (4), same formula as `snr_db` — is
the like-for-like metric: 18.13 vs 24.16 dB.

---

## 4. Hypothesis status

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| H1 | Deterministic loss on a one-to-many problem → the high band regresses to the conditional mean; output is muffled | **Confirmed as a mechanism, in the extreme.** Deficit deepened −15 → −24 dB *while* L1 was minimised. Regression to the conditional median, on camera. | §8 log, λ = 1e-3 run |
| H1′ | …and the deficit is large enough to matter at paper scale | **Refuted 2026-09-17 — the premise was wrong.** The paper's numbers cannot be used as evidence: 24.16 dB is 3.16 dB above the task's ceiling and its evaluation reports eight seconds of audio (`notes/2026-09-17-lisa-reported-numbers-audit.md`). The deficit measured here is real. Original verdict, retained for provenance: **Unknown; the paper's numbers argue against a large one.** SNR 24.16 dB with a ~2–3 % high-band energy fraction implies most high-band energy is recovered coherently; the residual deficit is plausibly **−2…−6 dB** (back-of-envelope, not measured). The notebook's own stop rule is 1–2 dB. The −20 dB measured here is data starvation, not LISA. | inference from Table 1 |
| H2 | Transport maps on low-order high-band statistics correct the deficit | **Machinery works; scientific claim untested.** T1 moved HB-LSD 1.85 → 1.23 and beat frame-shuffle (1.59), so it exploits per-frame structure even in a −20 dB residual. But the source is degenerate and a wrong map also "won" on LSD. Needs a coherent, under-energetic source. | §10–§11 |
| H3 | Only the stochastic rung can move a proper scoring rule | **Supported directionally.** CRPS 1.35 → 1.13 under the sampler only, on held-out speakers; provisional because the base model fails gate 2a. | §12 |
| H4 | LSD rewards energy matching / regression to the log-domain mean and must not be primary evidence | **Strongly supported, twice.** Broken model beats good model on LSD; wrong map beats shaped noise on HB-LSD. The controls caught it on the first clean run. | both runs, §11 |
| H5 | Transport beats classical noise-filling BWE | **Not adjudicated.** T1 1.23 vs shaped noise 1.75 on HB-LSD — but H4 says HB-LSD cannot be the judge here. | §11 |

**What the two runs together say.** λ = 1.0 gives structured high-band magnitudes with random
phase; λ = 1e-3 gives correct phase with no high band. Neither is "muffled but coherent", which is
the object the transport thesis presumes. That object has not yet been observed in this repro.

---

## 5. Decisions and next experiments

1. **Paired λ sweep, {1e-3, 1e-2, 1e-1}**, three models and three optimisers in one training loop on
   identical batches. Target regime: SNR > naive **and** a structured high band with a deficit in
   roughly the −5…−12 dB range. This is the cheapest route to a non-degenerate source. ~15 min each
   on the A100 sequentially; nearly free in parallel, since the GPU is >90 % idle.
2. **A paper-scale run** (99 speakers, batch 64, 1 s chunks, 50 epochs, official schedule) to measure
   H1′ directly. This is the experiment the project's premise depends on; if the deficit at paper
   scale is ≤ 2 dB, the honest result is the notebook's own stop rule.
3. **Throughput before either**: precompute `decimate` per utterance; disable
   `torch.use_deterministic_algorithms` during training (deterministic scatter-add in the gather
   backward is the likely hot spot); then batch 64 / 1 s. A step is ~43 ms for ~3 ms of GPU work.
4. **Do not quote** the λ = 1e-3 ladder numbers as results, and **do not quote** LSD against 0.81.
5. Before any LSD claim, run the official repo's `utils.py` LSD on identical audio to pin its
   scaling empirically.

---

## 6. Errata — what was wrong along the way, on both sides

- Notebook: λ = 1.0 (1000× reference); band-energy ratio used as the sole gate (phase-blind);
  `test_utts` rebound to training speakers by a scratch cell (made the earlier T0–T3 / CRPS numbers
  in-sample); §9 trusted cell-execution order; `fetch_vctk` URL returns 403 (irrelevant while the
  caches exist).
- Assistant: first hypothesis (evaluating random-init weights) was wrong — refuted by
  `max |dw| = 0.0`; suspected `grad_clip = 1e-3`, which is the paper's value; pasted the audit as a
  base64 one-liner, which froze the Colab tab and was unreviewable; resumed §8 without re-running
  §0 after adding imports (`NameError: tempfile`); initial checkpoint cadence (every ~19 s) tripped
  the Drive mount. All corrected; none affected the numbers above.

---

## 7. Provenance

- Data: Drive `lisa_rtm/fixtures/train_FULL.npz` (400 utts, p225–p234, 0.538 h),
  `test_FULL.npz` (120 utts, p236/p237/p238, 701 s). Speaker-disjoint. Not the paper's split
  (paper trains on id < 350, which includes p236–p238).
- Checkpoints: `checkpoints/FULL/lisa.pt` (λ = 1.0, step 20000; keep as the failure reference);
  `checkpoints/FULL_lam0.001/lisa.pt` (λ = 1e-3, step 20000).
- Logs: Drive `RESEARCH_FULL.md` (λ = 1e-3 run, written by §15); local `audit_cell.py` (audit,
  phase test, lag test, LSD-basis comparison as run today); `TODO.md`.
- Paper: arXiv:2111.00195v2, "Learning Continuous Representation of Audio for Arbitrary Scale
  Super Resolution", Kim, Lee, Hong, Ok (ICASSP 2022). Code: github.com/ml-postech/LISA.
