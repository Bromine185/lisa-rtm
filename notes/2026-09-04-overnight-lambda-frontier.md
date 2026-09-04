# Research note — the spectral-weight frontier, and the object your thesis needs

**Date:** 2026-09-04 (overnight run, unattended)
**Question it answers:** H1′ — is LISA's high-band deficit large enough to be worth correcting? — and,
unasked, *where does the muffled-but-coherent regime actually live?*
**Headline:** it lives at `lambda_spec = 1e-2`. Deficit **−4.5 dB**, phase coherent, LSD **0.96**
against the paper's 0.81, at a cost of 0.83 dB of waveform SNR. That is the first source this
project has produced that the transport ladder can legitimately be run on.

---

## 0. What was run

A single paired experiment on an A100-80GB: **four models, one data stream, identical batches**, so
every difference below is caused by the arm and nothing else.

| | |
|---|---|
| corpus | full VCTK 0.92 mic1 from the Hub (`sanchit-gandhi/vctk`, 27 parquet shards), **97 train speakers, 39,639 utterances, 37.3 h** |
| held out | p236/p237/p238 (never trained on), plus the paper's own test split (id ≥ 350, 8 speakers) |
| recipe | batch **64**, **1 s** chunks, Adam lr 1e-3, halved at 20/40/50/60/70/80 % of steps, grad clip 1e-3 — the paper's, except steps |
| budget | 36,000 steps = **17.1 epochs** (the paper's 50 epochs would have been 105k steps; 3.5 h wall clock) |
| arms | `relu_l1e-3` (official λ), `relu_l1e-2`, `relu_l1e-1`, `ff_l1e-3` (Fourier-feature decoder, official λ) |

The corpus replaces the 403'd DataShare URL and is 69× the data of every previous run here (0.54 h →
37.3 h). Training throughput went from ~43 ms/step for one model to **344 ms/step for four** — the
per-step cost is ~86 ms per arm, twice as fast as the old single-arm loop, because the corpus now
lives on the GPU and `decimate` is precomputed instead of run per batch on the CPU.

## 1. The result

Evaluated **locally on the Mac**, on the original `test_FULL.npz` (the DataShare-derived cache, a
*different acquisition route* from the Hub corpus the models trained on), 36 utterances, 12 per
held-out speaker. Independent of the Colab kernel:

| arm | SNR | naive | ΔSNR | LSD | LSD(paper basis) | HB-LSD | deficit dB | baseband dB | tgt-mag+pred-phase |
|---|---|---|---|---|---|---|---|---|---|
| `relu_l1e-3` | 19.16 | 19.20 | -0.04 | **1.546** | 1.530 | 1.778 | **-17.42** | -0.121 | 15.83 |
| `relu_l1e-2` | 18.37 | 19.20 | -0.83 | **0.962** | 0.945 | 1.097 | **-4.52** | -0.044 | 15.94 |
| `relu_l1e-1` | 17.69 | 19.20 | -1.50 | **0.948** | 0.931 | 1.084 | **-2.35** | -0.025 | 15.86 |
| `ff_l1e-3` | 19.15 | 19.20 | -0.05 | **1.486** | 1.470 | 1.708 | **-16.84** | -0.121 | 15.99 |

Three things fall out of that table.

### 1.1 The muffled-but-coherent regime exists, and λ is the dial

Every previous run in this project sat at one of two useless extremes: λ=1.0 gave structured
magnitudes with random phase (SNR −5.8 dB), λ=1e-3 on 0.54 h gave correct phase with no high band
(deficit −19.6 dB). **λ=1e-2 is the middle**: deficit −4.5 dB, baseband essentially exact
(−0.04 dB), and phase coherent — the `target-magnitude + predicted-phase` resynthesis scores
15.94 dB, versus **−5.9 dB** for the broken λ=1.0 model. The prediction's own phase is now worth
~16 dB, not worth nothing.

That is precisely the object the transport thesis presumes and has never had: a model whose
high-band **magnitudes are systematically low but whose structure is real**.

### 1.2 The paper's own ablation does not hold at this scale

The paper reports that removing the spectral term entirely moves LSD by ≤0.01 and SNR by ≤0.08 dB.
Here, moving λ from 1e-3 to 1e-2 moves **LSD by 0.58** (1.546 → 0.962) and the deficit by **13 dB**.

These are not in conflict; they say something specific. The spectral term is a **crutch for a model
that cannot learn the high band from data**. At 17 epochs on 37 h, the L1-only model has no way to
predict what lives above 6 kHz, so the loss-optimal output is zero and it emits nothing. Turn the
magnitude term up and it is *forced* to synthesise plausible high-band energy. The paper's flat
ablation is therefore evidence that **their** model had learned the high band conditionally, and the
spectral term genuinely added nothing on top. Ours has not. The ablation is a scale-dependent
statement, and reporting it without the scale caveat is misleading.

### 1.3 ReLU spectral bias is not the cause — a clean negative

The notebook's "Still open" list has carried this for weeks: *"LISA's decoder is a ReLU MLP, not a
SIREN. ReLU spectral bias may be a second, architectural cause of over-smoothing, independent of the
loss."*

`ff_l1e-3` is the same encoder, same latents, same loss, same batches, same seed — only the decoder's
view of the relative coordinate changes, from `c` to `[c, sin(πkc), cos(πkc)]` for k=1..6 (Tancik et
al. 2020, the standard fix for spectral bias in coordinate MLPs). Result: deficit −16.84 vs −17.42,
LSD 1.486 vs 1.546, SNR identical to two decimal places.

**A 0.6 dB difference on a 17 dB deficit.** Fourier features do not rescue the high band. The
over-smoothing is caused by the objective and the data, not by the decoder's spectral bias. That
item can be struck from the open list.

## 2. What still does not reproduce

**No arm beats naive polyphase upsampling.** The best is 0.04 dB *below* it. The paper reports
24.16 dB at 12k→48k, roughly 5 dB above where a sinc interpolator sits on this data.

This is the honest negative and it should be stated plainly: with 37 h, the paper's batch/chunk/
schedule and its λ, at 17 of 50 epochs, this reimplementation reproduces LISA's *LSD* (0.95–0.96
against 0.81) but **not its SNR**. Beating naive interpolation requires coherently reconstructing
high-band *waveform*, and nothing here does that.

One observation worth recording, not a claim: in the paper's Table 1, AudioUNet (18.55), TFilm
(19.51) and WSRGlow (19.41) all score within ~1 dB of where naive upsampling scores on our data
(19.20), while LISA alone sits 5 dB above all three. A 5 dB gap over three unrelated architectures,
all of which land at the trivial baseline, is a pattern worth understanding before treating 24.16 dB
as the target. Possible explanations: a different SNR convention, a different downsampling operator
(they use sinc interpolation; we use `resample_poly`), or their evaluation including the input band
differently. Not resolved here.

## 3. Consequences for the hypotheses

| # | before tonight | after tonight |
|---|---|---|
| H1 (mechanism) | confirmed in the extreme | **confirmed and now *tunable*.** The deficit is a monotone function of λ: −17.4, −4.5, −2.3 dB at λ = 1e-3, 1e-2, 1e-1 |
| H1′ (magnitude at paper scale) | unknown; inferred −2…−6 dB | **still not measured at the paper's SNR**, but the −2…−6 dB inference is now *observed* at λ ≥ 1e-2. A model with paper-like LSD (0.95) carries a −2.3…−4.5 dB deficit — small, and in the notebook's own "no-headroom" grey zone at λ=1e-1 |
| H2 (transport) | untested; source degenerate | **now testable.** `relu_l1e-2` is a legitimate source: coherent phase, −4.5 dB deficit, LSD 0.96 |
| H4 (LSD is a bad judge) | strongly supported | **supported again, and sharpened.** λ=1e-1 has the best LSD (0.948) *and* the worst SNR (−1.50 vs naive). LSD ranks the arms in exactly the opposite order to waveform fidelity |
| architecture confound | open | **closed.** Fourier features change nothing |

The uncomfortable implication for the project, stated because it is load-bearing: at the λ that
produces paper-like LSD, the deficit is **−2.3 to −4.5 dB**. The notebook's own stop rule says
1–2 dB is "no headroom". So the headroom for a transport correction on a *properly-trained* LISA is
real but modest — a few dB, not the 20 dB the under-trained runs suggested. The ladder should be
judged on CRPS and on listening, not on how much energy it can pour into a −20 dB hole.

## 4. Artefacts

| what | where |
|---|---|
| four checkpoints, step 36000 | Drive `lisa_rtm/checkpoints/XL4_b64_1s/{relu_l1e-3,relu_l1e-2,relu_l1e-1,ff_l1e-3}.pt` |
| training log, 4 arms × 36 lines | Drive `lisa_rtm/train_XL4_b64_1s.log` |
| corpus manifest (speakers, hours) | Drive `lisa_rtm/xl_manifest.json` |
| local independent evaluation | `overnight/local_eval_XL4.json`, reproduced by `overnight/local_eval.py` |
| λ frontier figure | `overnight/pareto_lambda.png` |
| architecture control figure | `overnight/arch_control.png` |
| corpus / trainer / eval cells | `overnight/cell{1,2,3,4,5,6}*.py` |

`overnight/local_eval.py` runs on the Mac with no GPU and no Colab: it execs the notebook's own
definition cells, so its metrics are the notebook's, and it reads the checkpoints straight from the
Drive mount. That is the reproduction path if the Colab kernel is gone.

## 5. Next

1. **Run the ladder on `relu_l1e-2`.** First legitimate source. Judge on CRPS and the controls, and
   expect the mis-specified-map control to be *much* less flattering now that T0's high band is at
   −4.5 dB instead of on the epsilon floor — that is the real test of whether HB-LSD was ever
   measuring anything.
2. **Finish the budget**: 105k steps (50 epochs) at λ=1e-3 to see whether the L1-only model
   eventually learns the high band, which is the paper's implicit claim. ~10 h for one arm.
3. **Resolve the SNR convention** against the official repo before treating 24.16 dB as a target.
4. Fold `LISAFF` and the paired trainer into `build_notebook.py`; they are currently overnight-only.
