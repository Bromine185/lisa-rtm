# lisa-rtm — todo

Status as of 2026-09-01, after the evaluation audit on Colab (checkpoint step 20000).

## Settled — do not re-litigate

- [x] **Held-out evaluation is sound.** Same code path, clean kernel: baseband energy ratio
      -0.45 dB, high-band deficit -8.96 dB on p236/p237/p238. The earlier "-40 dB across all
      frequencies" plot did not reproduce; it was stale kernel state, not a VCTK source mismatch.
- [x] **Weights are the trained ones.** `max |dw|` vs `lisa.pt` = 0.0; input x2 -> output x1.988.
- [x] **No time offset.** Cross-correlation lag 0..+1 sample; aligned SNR == as-is SNR.
- [x] **Not a silence/eval artifact.** Silence trimming keeps 100% of samples and moves LSD by 0.002.
- [x] **Not a capacity problem.** 86,881 params vs the paper's ~89k.

## The real defect: the objective is phase-blind

Evidence: spec term is **92% of the loss** (wave 0.0836, spec 1.0287 at lambda_spec=1.0).
The waveform term went **backwards** during training: 0.0557 at step 0 -> 0.0836 at step 20000,
i.e. worse than the near-silent random init. Waveform SNR is -5.8 dB (worse than emitting
silence, which scores 0 dB), while pred-magnitude + target-phase resynthesis scores +17.3/+15.7 dB.
The model traded phase for magnitude and the loss let it.

**Confirmed against the paper + official repo (ml-postech/LISA):** the paper gives the loss as
`L_wave + lambda * L_spec` but never prints lambda; the released code uses **spec_coeff = 1e-3**,
and the shipped `configs/audio/lisa.yaml` sets `loss: l1` (waveform only, no spectral term at all).
The paper's own ablation ("-spec", Table 2) moves LSD by <=0.01 and SNR by <=0.08 dB. LISA is
effectively an L1-waveform model. **Our `lambda_spec = 1.0` is 1000x the reference value.**

- [x] **`lambda_spec = 1e-3`** in both presets (`build_notebook.py`, 2026-09-01). Run dir is now
      `checkpoints/FULL_lam0.001`, so the old lam=1.0 checkpoint at `checkpoints/FULL/` is kept as
      the reference for the failure mode and can never be resumed by accident.
- [x] **Retrained** on A100, 2026-09-01 (`checkpoints/FULL_lam0.001`). Result: SNR **18.13 dB** vs
      naive 18.16 dB (gate 2a fails by 0.03 dB -- a tie: phase is now correct, the model is a sinc
      interpolator that adds nothing above 6 kHz). HB deficit **-19.55 dB**. Wave L1 plateaued at
      0.0034 from step 8000; not diverging, converged. Full record in Drive `RESEARCH_FULL.md`.
- [ ] **Paired lambda sweep** {1e-3, 1e-2, 1e-1}: three models, three optimisers, ONE data pipeline
      and identical batches, in one training loop. Looking for the regime with SNR > naive AND a
      structured, under-energetic high band (deficit in the -5..-12 dB range with correct phase).
      That is the source the transport ladder actually needs.
- [ ] Ladder numbers from the lam=1e-3 run are pipeline diagnostics only: the mis-specified-map
      control improved HB-LSD (1.846 -> 1.370) because T0 sits on the eps floor above 6 kHz, so
      HB-LSD rewards any energy injection. Do not quote them.
- [x] Waveform SNR (and the naive baseline) logged at every checkpoint; third panel in the §8 plot.
- [x] Resume guard: a checkpoint whose training config differs on any optimisation field raises
      instead of silently continuing.
- [x] `grad_clip=1e-3` is **correct** — the paper explicitly specifies "gradient clipping with
      0.001 max norm". Not a bug; my earlier suspicion was wrong.

## The remaining gap: underfit, plus a budget mismatch

LSD 1.306 train vs 1.359 held out — a 0.05 gap. The model generalises fine and is simply
bias-limited.

| | this repro | paper / official code |
|---|---|---|
| loss | L1 + **1.0** * multi-scale STFT | L1 + **1e-3** * MS-STFT (config ships plain L1) |
| speakers | 10 (400 utts, 0.538 h) | 99 train (VCTK, id < 350) |
| batch | 16 | 64 |
| chunk | 0.256 s, all 12288 coords | 1 s, 8000 sampled coords |
| schedule | constant lr 1e-3, 20k steps | lr 1e-3, 50 epochs, halved at milestones |
| grad clip | 1e-3 | 1e-3 (matches) |
| SNR @ 12k->48k | **-5.8 dB** | **24.16 dB** |

- [x] LR schedule: `MultiStepLR`, halved at 20/40/50/60/70/80 % of steps (official code's
      epoch milestones 10,20,25,30,35,40 of 50, expressed as fractions).
- [ ] Raise batch size / chunk length toward 64 / 1 s if VRAM allows.
- [ ] Scale the training budget: more steps first (it is bias-limited, so more data is the second
      lever, not the first), then more speakers.

**Do not quote "LSD 1.36 vs 0.81" as the gap.** It is not apples-to-apples: the paper's LSD uses
n_fft=2048 / hop=1024 (from code; the paper states neither), and per the released code it computes
`log10(|power spectrogram|^2)` = 4*log10|STFT|, i.e. 2x the quantity printed in its own Eq. (5) and
2x what `lsd_db()` computes here. SNR (their Eq. 4) IS directly comparable and is the honest
headline: **-5.8 dB vs 24.16 dB**.

- [x] §9 now also prints LSD in the paper's STFT basis (n_fft 2048 / hop 1024) — labelled as
      orientation only, because the released code's log scaling appears to be 2x its own Eq. (5).
- [ ] If an LSD claim is ever needed: run the official repo's `utils.py` LSD on identical audio
      to pin the scaling factor empirically rather than by reading code.
- [ ] Note our speaker split (hold out p236-p238) is NOT the paper's split (train = id < 350, so
      p236-p238 are inside their training set). Fine for our purposes, but not a like-for-like number.

## Notebook robustness

- [x] **§9 is independent of cell execution order.** It reloads both splits from the `.npz` caches
      when they exist, warns loudly on any speaker overlap, and reloads `lisa.pt` itself. (A scratch
      cell had set `test_utts = train_utts[:12]`; that is what made the earlier T0-T3 / CRPS runs
      in-sample.)
- [x] `build_split` in the repo already checks the cache before `fetch_vctk`; the buggy helper was
      a pasted Colab scratch cell, not repo code. Nothing to fix here.
- [ ] `fetch_vctk` still points at a DataShare URL that returns 403. Irrelevant while the caches
      exist; replace the URL (or document the HF route) before anyone rebuilds the fixtures.
- [ ] `audit_cell.py` (phase test, lag test, LSD basis comparison) is still a loose file. Fold the
      phase test into §9 once the retrain passes gate 2a, or delete it.

## Throughput (the A100 is >90% idle at ~43 ms/step for a 2-3 ms job)

- [ ] Precompute `decimate` per utterance once; slice aligned segments instead of resampling per step.
- [ ] `torch.use_deterministic_algorithms(False)` during training (keep for eval): deterministic
      scatter_add in the gather backward is the likely hot spot.
- [ ] Then batch 64 / 1 s chunks (paper) -- overhead-bound, so ~free.

## Not done on purpose — one variable at a time

- [ ] batch 16 -> 64 and 0.256 s -> 1 s chunks (paper's code). ~16x compute per step; and the 1 s
      chunk only makes sense with the paper's 8000-coordinate subsampling, which the STFT term
      cannot use (it needs contiguous output). Change these after the lambda fix is verified in
      isolation, not alongside it.
- [ ] More speakers. Model is bias-limited today, so this is the last lever, not the first.

## Blocked until the retrain lands

- [ ] T0-T3 transport ladder on held-out speakers. (Magnitude-domain, so it would still "work" on
      the current checkpoint, but any claim built on it inherits a model whose waveform output is
      worse than silence.)
- [ ] Stochastic rung / CRPS held-out numbers.
- [ ] §14 listening examples.
