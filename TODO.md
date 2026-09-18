# lisa-rtm — todo

**2026-09-17: the paper's 24.16 dB is settled, and SNR is retired as a ranking metric.**
`notes/2026-09-17-lisa-reported-numbers-audit.md` and `notes/2026-09-17-snr-ceiling-gating-and-scale.md`.
24.16 dB sits 3.16 dB above the task's ceiling measured under LISA's own protocol; their evaluation
reports one batch of eight seconds, where a sinc interpolator's sd is 3.63 dB. Naive upsampling *is* the
optimal point predictor to within 0.1 dB, and every decibel of high band a model restores costs it SNR,
so the whole achievable range is about 1 dB. New defect found in our own sampler: its high band does not
gate with the speech (−11.7 dB on loud frames, +1.0 dB in the gaps), which the mean deficit hides.

**2026-09-18: the ceiling is written up as a paper, and the literature check moved the close.**
[`paper/bwe-information-ceiling.md`](paper/bwe-information-ceiling.md), reproduced end to end from a
clean clone by `audit/snr_scale.py`, `audit/protocol_fork.py` and `audit/lisa_paper_protocol.py` — no
GPU, no checkpoint, ~15 s. New: `audit/protocol_fork.py` measures what dropping the anti-alias filter
does (2.15 % of the high band reaches an anti-aliased observation against 100.1 % of an aliased one;
headroom above each protocol's own naive baseline +0.12 dB against ≥ +2.47 dB). The §9 close as
originally planned was **false** and is corrected in both 17 Sep notes: AudioUNet 18.55 / TFiLM 19.51 /
WSRGlow 19.41 are LISA's own reruns, not independent reports, and NU-Wave 2 reports 21.2–22.1 dB at the
same setting. The true and stronger statement is that WSRGlow spans 18.38 / 19.41 / 21.2 dB across three
protocols — wider than the whole within-protocol achievable span — and that NU-Wave 2's own table puts
the unprocessed input above every model.

- [ ] **Decide whether the paper goes out as a 2-page workshop submission or a poster.** The structure
      is already two pages; a LaTeX conversion is mechanical.
- [ ] **Run §9 of the 17 Sep audit note inside `ml-postech/LISA`** — delete the `if ii == 3:` guard
      (`eval_lisa.py:206`), swap the model for a sinc upsample, report the full validation set. It is the
      one thing that would remove the last reimplementation caveat from §7 of the paper.
- [ ] **A per-frame adaptive unfolder for `audit/protocol_fork.py`.** The fixed linear filter recovers
      1.07 % held out and the per-bin oracle 50.8 %; the gap is entirely signal adaptivity, and where a
      realizable estimator lands in between prices the aliased protocol properly.

## Next, from the 17 Sep audit

- [ ] **Add the gated deficit to `overnight3/e4_eval.py`** — `band_energy_ratio` already takes a
      `frame_mask`; split frames into loud / mid / quiet by energy and report all three. This is the
      defect a listener hears and no current metric reports it.
- [ ] **Energy-weighted ERB arm.** Weight `d_erb` by frame energy, or add a term on the high-band
      envelope against the low-band envelope: score the conditional, not the marginal. Prediction
      pre-registered in `notes/2026-09-17-snr-ceiling-gating-and-scale.md` §2.2.
- [ ] **Print the probe's naive SNR in the dashboard**, next to `SNR0`. The trainer already computes it
      and prints it on the `step` lines; without it `SNR0 14.3` reads as a failure when it is 0.05 dB
      off that utterance's ceiling.
- [ ] **Repeat `audit/scale_freedom.py` across all seven arms** after the run. A fixed-scale local INR
      generalising off its training coordinate lattice is publishable on its own and nobody has checked.
- [ ] **A/B our `resample_poly` decimation against torchaudio's looser sinc** (`audit/` has both). Worth
      ≤ 1.6 dB of oracle head-room and it is the one genuine asymmetry between our protocol and LISA's.

**2026-09-08 overnight run supersedes the ladder programme.** See
`notes/2026-09-08-network-as-transport-map.md`. Headline: the same 88k-parameter LISA trained under the
energy score (two draws per step, noise channels at the input) is a conditional sampler at zero
inference cost. CRPS −43 % vs the deterministic model and −23 % vs the best post-hoc rung; ensemble mean
beats the point-trained model on SNR; the band above 6 kHz is incoherent with the truth for every model
(coherent fraction ≤ 2 %), so the deterministic SNR ceiling is 0.3 dB above naive upsampling.

## Next, in priority order

**Evening 8 Sep — judged by LSD + ViSQOL (note §7):** speech-mode ViSQOL/PESQ cannot see the band (naive
wins on DataShare); audio-mode ViSQOL rewards deterministic high-band energy monotonically; winners are
λ = 1e-1 (Hub) and the wide-context det (DataShare), with the T1 rung adding +0.11/+0.15; the sampler's
speech-ViSQOL loss was baseband leakage and disappears with passthrough. Best model has 46 % of the
audio-mode range between empty and true high band.

- [ ] **Deterministic λ sweep {0.1, 0.3, 1.0} with baseband passthrough** (`overnight2/c8_launch_lambda.py`,
      A100 needed for the 37 h corpus; ~3 h for three arms). Judge on LSD + audio-mode ViSQOL with
      passthrough; expect the trend 1e-3 → 1e-1 to continue until the phase-blind term breaks the band's
      own structure.
- [ ] **Make baseband passthrough the default output** for every model (it improves every condition and
      is the honest system for BWE since the low band is given).
- [ ] **Wide context + high λ + T1 rung** as one model: the three things that each helped on LSD + ViSQOL.
- [ ] Retire speech-mode ViSQOL and PESQ as judges for 12 → 48 kHz; report audio-mode ViSQOL (and NSIM)
      with the floor/ceiling rows (1.57 / 4.73) alongside.

- [ ] **Resolve the two-test-set discrepancy.** The same checkpoint measures a high-band deficit 5–8 dB
      deeper on the Hub `test_utts` than on the DataShare `test_FULL.npz` for the same speakers (both
      mic1). The sampler is energy-calibrated on one and under-dispersed on the other. Until the
      provenance is settled, every number must name its set. Start by comparing one utterance's
      spectrum from both routes.
- [ ] **Close the in-distribution energy gap of the sampler** (single draw −7.9 dB on Hub at 9 epochs,
      PIT nearly flat). Candidates, one at a time: train to 50 epochs; weight the log-magnitude term
      higher than 1e-2; more noise channels or noise into the decoder; the β = 0.5 energy score.
- [ ] **Sliced score at higher weight / longer budget.** `es_slice` converged slower than `es_marg`
      and was under-dispersed (PIT top bin 41 %). It is the only arm proper for the joint law of a
      frame; it should not be judged at 9 epochs and weight 1e-2.
- [ ] **Listen.** Drive `lisa_rtm/ov2/audio/`: truth, naive, det, det_split, two draws of each
      sampler and each sampler at τ = 0, three utterances. Perceptual adjudication remains the binding
      constraint; no metric here settles it.
- [ ] **GAN baseline.** The community's default answer to over-smoothing; one adversarial arm on the
      same batches would settle whether the proper-score sampler matches it without a discriminator.
- [ ] Fold `LISAS`, the energy-score losses, `HostCorpus` and the coherence metrics into
      `build_notebook.py`; they are overnight-only (`overnight2/c*.py`).
- [x] **Settled 17 Sep.** With coherent prediction excluded (§1.4), the paper's 24.16 dB was settled
      against ml-postech/LISA, not by more training. See `notes/2026-09-17-lisa-reported-numbers-audit.md`.
      Their down-sampling is honest (Kaldi windowed sinc, anti-aliased); the number is 3.16 dB above the
      protocol's own ceiling (21.00 dB, measured their way on their own held-out speakers); and
      `eval_lisa.py` computes it on one batch of eight 1-second chunks, where a sinc interpolator's
      batch-to-batch sd is 3.63 dB. Reproduce: `venv/bin/python audit/lisa_paper_protocol.py`.

## Settled by the 2026-09-08 run

- [x] H2 (deterministic transport) is dead on an honest source and now strictly dominated: every
      deterministic rung and the post-hoc S rung lose to the learned sampler on CRPS.
- [x] H3 (only a sampler moves a proper score): confirmed, and the sampler can be the network itself.
- [x] The deterministic model's high-band energy is set by λ, not by predictability: κ = 0.06–0.12.
      Its deficit curve is not a predictability spectrum; the sampler's ensemble-mean coherence is.
- [x] No deterministic LISA of this class can beat naive upsampling by more than 0.3 dB on this data
      (coherent fraction ≤ 2 % above 6 kHz). The paper's 24.16 dB is not reachable from an 11-sample
      window unless the SNR convention differs.
- [x] Score geometry: waveform-only energy score gives energy without spectral structure; the
      log-magnitude marginal score is the best bargain at this budget.
- [x] Real-time: 1.5 ms per second of audio on an A100, 16.5 ms on an Apple M4 CPU, 0.5 ms
      algorithmic look-ahead.
- [x] Receptive field is not the lever: a 22 ms dilated-context encoder leaves the coherent fraction
      above 7 kHz at 0.000–0.005 under both objectives (`OV2_wide`).
- [x] Graph toy: shallow GNN + MSE loses 15.5 dB in the high graph-frequency bands; the energy score
      restores it in one draw; a deep no-skip GNN cannot under any objective. The ensemble-mean test
      separates objective- from architecture-induced over-smoothing with one model.

---

## Earlier (2026-09-04) — retained for provenance

**2026-09-04 overnight run supersedes much of what follows.** See
`notes/2026-09-04-overnight-lambda-frontier.md` and the results page. Headline: a 37.3 h,
paper-recipe, four-arm paired run found the muffled-but-coherent regime at `lambda_spec = 1e-2`
(deficit -4.5 dB, coherent phase, LSD 0.962 vs the paper's 0.81). The Fourier-feature control
cleared ReLU spectral bias as a cause. No arm beats naive upsampling; the paper's 24.16 dB SNR is
still unreproduced.

## Next, in priority order

- [ ] **Run the ladder on `relu_l1e-2`** (Drive `checkpoints/XL4_b64_1s/`). First legitimate source:
      coherent phase, systematically low high-band magnitudes. Judge on CRPS and the controls.
      Expect the mis-specified-map control to stop flattering now that T0's high band sits at
      -4.5 dB rather than on the epsilon floor -- that is the real test of whether HB-LSD ever
      measured anything.
- [ ] **Finish the budget**: 105k steps (50 epochs) at lambda=1e-3, one arm, ~10 h. Does an
      L1-only model eventually learn the high band? That is the paper's implicit claim and the
      remaining explanation for the SNR gap.
- [x] **Settled 17 Sep** — see the entry above and `notes/2026-09-17-lisa-reported-numbers-audit.md`.
      The observation that all three of the paper's baselines land within ~1 dB of naive on our data
      turned out to be the tell: the ceiling is real and they are all sitting on it.
- [ ] Fold `LISAFF`, `GPUCorpus` and the paired trainer into `build_notebook.py`; they are
      currently overnight-only (`overnight/cell*.py`).
- [ ] `fetch_vctk` still points at a 403 URL. The Hub route in `overnight/cell1_corpus.py` works;
      make it the notebook's default.

## Settled by the overnight run

- [x] H1' (magnitude): deficit is monotone in lambda -- -17.4 / -4.5 / -2.3 dB at 1e-3 / 1e-2 / 1e-1.
      At paper-like LSD the headroom is a few dB, not the 20 dB the starved runs implied.
- [x] Architecture confound: Fourier-feature decoder moves the deficit 0.6 dB on a 17 dB hole.
      Not spectral bias. Struck from the notebook's "Still open" list.
- [x] The paper's spectral-loss ablation is scale-dependent: one decade of lambda moves LSD 0.58
      here vs their <=0.01. Never quote it without the scale caveat.
- [x] Throughput: GPU-resident corpus + precomputed decimate + TF32 + determinism off during
      training -> 86 ms/step/arm.


Status as of 2026-09-01, after the evaluation audit on Colab (checkpoint step 20000).

## Earlier (2026-09-01 audit) — retained for provenance



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
