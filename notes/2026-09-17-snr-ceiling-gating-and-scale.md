# Research note — what SNR can measure here, why the samples hiss, and a decoder that is scale-free by accident

**Date:** 2026-09-17, written while `OV3_fast` was still training. §2 and §3 are one arm
(`es_erb_l0.1`), one checkpoint (step 13500), one utterance; §1 is 64 held-out utterances.
**Companion:** [`2026-09-17-lisa-reported-numbers-audit.md`](2026-09-17-lisa-reported-numbers-audit.md)
covers where the paper's 24.16 dB comes from. This note covers what the same evening taught us about our
own models.
**Reproduce:** `audit/snr_scale.py`, `audit/band_gating.py`, `audit/scale_freedom.py`.

---

## 1. SNR cannot rank models on this task

A point predictor reproduces the low band exactly — sinc interpolation is the optimal reconstruction of
a band-limited signal — so its entire error is high-band energy it fails to reproduce **in phase**. SNR is
therefore a function of one quantity: the coherently recovered fraction of the band above 6 kHz. Per
utterance, averaged in dB, 64 held-out utterances:

| coherently recovered | SNR | vs naive |
|---|---|---|
| 0 % | 19.11 dB | +0.10 |
| 2 % — this model class | 19.19 | +0.19 |
| 10 % | 19.56 | +0.56 |
| 20 % | 20.08 | +1.07 |
| 50 % | 22.12 | +3.11 |

Naive sinc upsampling measures **19.01 dB** against a zero-recovery optimum of **19.11 dB**. Naive *is*
the zero-recovery solution, to within 0.1 dB. It is not a weak baseline to be beaten with a better
architecture; it is the MMSE answer when the missing band carries no recoverable phase.

Recovering a tenth of the high band buys 0.56 dB. Everything a real model could plausibly achieve is
compressed into about one decibel — and it sits on top of a per-utterance spread of **4.32 dB** (range
10.52 to 29.52 across the same 64 utterances). The metric has less dynamic range than its own noise.

**Worse, SNR punishes the task.** The 4 September λ sweep is the cleanest evidence in the repo:

| λ | SNR | LSD | HB deficit | |
|---|---|---|---|---|
| 1e-3 | 19.16 | 1.546 | −17.42 dB | empty high band |
| 1e-2 | 18.37 | 0.962 | −4.52 dB | |
| 1e-1 | 17.69 | 0.948 | −2.35 dB | band restored |

Monotone. The SNR-optimal model outputs *nothing* above 6 kHz and lands at 19.16, which is naive to
within 0.04 dB. So `det` at 18.37 is not losing to naive — it is paying 0.83 dB to move the deficit from
−17.4 dB to −4.5 dB, and that is the right side of the trade. Any energy a model adds above 6 kHz is
incoherent with the truth, and incoherent energy *adds* to the error.

This is Blau–Michaeli, and the field confirms it where it is honest. **Corrected 18 Sep** — the three
citations below were checked against the primary sources while writing
[`paper/bwe-information-ceiling.md`](../paper/bwe-information-ceiling.md), and two of the three were
overstated as first written:

- NU-Wave ([2104.02321](https://arxiv.org/abs/2104.02321)) Table 2, MultiSpeaker ×2 (24k→48k): their
  U-Net scores **9.86 dB against linear interpolation's 11.1**, i.e. 1.24 dB worse, not 0.9, while LSD
  goes 1.47 against 1.93. It is one cell of four — in the other three their U-Net *beats* linear on SNR —
  and it is ×2, not ×4. The "0.9" was the paper's *"improves SNR value by 0.18-0.9 dB from the best
  performing baseline"*, a different sentence.
- WSRGlow ([2106.08507](https://arxiv.org/abs/2106.08507)) §3.2.2, verbatim: *"WSRGlow w/o Enc_STFT
  outperforms the baseline models in terms of SNR, but performs poorly in LSD. This indicates that
  WSRGlow w/o Enc_STFT cannot synthesize the high-frequency part of the signal well."* That variant
  scores 17.45 at ×4 — above AudioUNet and MUGAN, below full WSRGlow's 18.38. It is not "the variant
  with the best SNR"; it is the variant that beats the *baselines* on SNR.
- AP-BWE ([2401.06387](https://arxiv.org/abs/2401.06387)) §V-A4, verbatim: *"Under the condition of
  extending from 8 kHz to 16 kHz, the performance of the sinc filter interpolation was already very
  close to the Ground Truth."* Quote confirmed, but Table IV's metrics are WER / CER / STOI, not SNR —
  it is an intelligibility claim. AP-BWE reports no SNR anywhere.

**What to do:** report SNR always beside naive on the same audio, and say in words that it is maximised
by doing nothing. Never rank arms on it. The axes with room are CRPS, the band-energy deficit, the
coherent fraction, and audio-mode ViSQOL.

### 1.1 The probe SNR is not the eval SNR

Recorded because it cost an hour. The dashboard's `SNR t0 / t1` come from `probe_metrics(m, probe, CFG,
naive)` on a **single** utterance, `test_utts[0]`, which is `p236_002` — VCTK 0.92 has no `p236_001`. Its
own naive baseline is **14.89 dB**, computed independently and confirmed against the trainer's stored
`snr_naive` history, because it is fricative-heavy: 3.23 % of its energy is above 6 kHz against 1.88 %
for these speakers. p236's first six utterances span 14.89 to 25.48 dB.

So the seven arms sitting at 14.3–14.8 are all within 0.6 dB of that utterance's ceiling, ordered
monotonically by how much high band they restore:

```
det 14.84 (-0.05, def1  -19.24) · es_marg     14.82 (-0.07, -16.18) · es_split 14.70 (-0.19, -16.93)
es_dec 14.62 (-0.27, -13.31) · es_dec_erb 14.50 (-0.39, -10.59) · es_marg_l0.1 14.49 (-0.40, -14.72)
es_erb 14.29 (-0.60, -10.89)
```

Exactly the §1 trade, measured live, with no fitted parameter. The trainer already prints `(naive XX.XX)`
at the end of every `step` line; the dashboard should print it too. Reading `SNR0 14.3` without it
invites precisely the wrong conclusion.

## 2. The demo audio should not be low-passed — and the real defect is timing

The samples sound like they want a low-pass. They do not. On `es_erb_l0.1` at step 13500, brick-wall
filtering the draw before scoring:

| cutoff | SNR | deficit | HB ratio, quiet frames |
|---|---|---|---|
| none | 13.88 | −10.87 | +0.99 |
| 20 kHz | 13.88 | −11.73 | −0.51 |
| 16 kHz | 13.89 | −17.54 | −3.74 |
| 12 kHz | 13.91 | −25.89 | −6.58 |
| 8 kHz | 13.98 | −35.97 | −11.76 |

Filtering moves SNR *up* and the deficit *down*. Cutting at 8 kHz buys **0.10 dB of SNR** and costs
**25 dB of deficit** — it walks the model most of the way back to `det` at −19.2, to improve a metric §1
just showed cannot rank anything. Wrong direction on both axes at once. It would also be a protocol
asymmetry of exactly the kind the companion note catalogues: if we band-limit, we band-limit every
condition — every arm, `det`, naive and the reference — and say so in the caption.

### 2.1 The high band does not gate with the speech

`band_energy_ratio` has taken a `frame_mask` argument since 1 September and nobody had used it. Splitting
the probe utterance's frames by loudness:

| frames | HB energy ratio | target HB share of that frame |
|---|---|---|
| loud, top 25 % | **−11.68 dB** | 5.794 % |
| mid, 25–75 % | −4.73 dB | 0.212 % |
| quiet, bottom 25 % | **+0.99 dB** | 0.010 % |

A **12.7 dB swing**. The model is 11.7 dB too quiet on the loud frames — the fricatives, where 5.8 % of
the frame energy lives and where the band actually matters — and 1 dB too loud in the gaps, where the
truth has essentially nothing. Instead of firing on the /s/ and vanishing between words, the high band
sits at a roughly constant level under the voice. **That is the hiss, and no low-pass removes it**,
because the offending energy is spread across the whole band.

The mean deficit over all frames is −10.87 dB and reports none of this. **The deficit metric averages
over frames and hides a 12.7 dB error.** That is the most important line in this note.

The spectral shape says the same thing in frequency. Per third octave, one draw against τ = 0:

| band | draw | τ = 0 | target share of total energy |
|---|---|---|---|
| 6735 Hz | −1.79 | −11.47 | 0.2097 % |
| 8485 | **−16.28** | −28.41 | **1.4215 %** |
| 10691 | −14.22 | −26.45 | 1.0687 % |
| 13470 | −10.08 | −23.51 | 0.2730 % |
| 16971 | −13.50 | −22.59 | 0.2091 % |
| 21382 | −9.34 | −16.71 | 0.0426 % |

Nothing is hot. Just above the cut it is nearly correct at −1.8 dB; it collapses to −16 dB exactly where
the most energy is; it comes back up at 21 kHz where there is almost none. The balance is inverted, the
level is fine.

### 2.2 Why the objective allows it, and the next arm

`d_erb` scores 32 band energies per frame and averages. That pins each band's **marginal** energy
distribution and says nothing about the correlation between high-band energy and the frame's loudness.
Spreading the energy uniformly in time satisfies the average exactly. It is the same failure as the
per-bin marginal problem of the 16 September note — an aggregate the score cannot see — one level up.
The energy score is proper for the law of φ(y); we keep choosing a φ that throws away the structure we
care about.

**The next arm is a small change.** Weight the ERB term by frame energy, or add a term on the high-band
envelope against the low-band envelope: score the conditional, not the marginal.

**Prediction, written now:** an energy-weighted ERB arm closes the loud-frame deficit from −11.7 dB to
better than −6 dB and brings the quiet-frame ratio below −6 dB, at a cost of under 0.3 dB of SNR.
Refuted if the loud-frame deficit does not improve by at least 3 dB, or if the quiet-frame ratio stays
above −2 dB.

## 3. The decoder is scale-free, and nobody asked it to be

Unplanned. The decoder only ever saw four coordinate values: `c = 2(q − i) − 1` with `q = j/4` gives
`c ∈ {−1, −0.5, 0, +0.5}`, and the anchor jitter shifts the anchor by an **integer**, so it only supplies
`c − 2m` — the same four phases displaced. `c = ±0.25` and `±0.75` were never evaluated once in 13,500
steps.

Querying the same latents at ×8 puts half the samples on exactly those unseen coordinates:

```
8x query, decimated back to 48 kHz, vs the direct 4x query    49.90 dB agreement
SNR against the truth, 4x                                     14.30 dB
SNR against the truth, 8x-then-decimated                      14.29 dB
energy the 8x query invents above 24 kHz                       0.00 %
```

and sweeping `c` densely at a real latent triple in the loudest region gives a monotone curve whose
curvature between the trained points is **0.88×** the curvature at them — smoother in the gaps, not
spikier. A 144-wide ReLU MLP is a very smooth function of one input, and the jitter forces `f(c)` and
`f(c−2)` to agree for the same latents, which regularises the whole axis rather than four points on it.
The perturbed prediction of §2.2.2 of the paper turns out to buy arbitrary output scale as a side effect.

Two consequences.

**Rendering below 48 kHz aliases.** The decoder represents one continuous function whose bandwidth was
fixed by training at 48 kHz, and it has no idea what rate it is being queried at. Query it directly at
24 kHz and everything it holds between 12 and 24 kHz folds down. To render lower, query at ×4 and
resample. Querying above ×4 is safe — that is what the 0.00 % row shows.

**LISA's arbitrary-scale claim is thinner than it reads.** Their config sets `cell_decode: False`, the
LIIF mechanism that feeds the output cell size to the decoder so it can adapt its bandwidth. Turned off.
So their decoder has no rate input either, and `gt_aug_max: 3` randomises only the **output** rate while
`input_sr` stays pinned at 8000. Their "arbitrary scale" is arbitrary output rate from one fixed input
rate — which we have, measured above, with no augmentation and no extra training. The axis both of us
are missing is the **input** rate, which is hard-fixed by the encoder: each latent sees 11 input samples,
which is 0.9 ms *at 12 kHz*, and at another rate those taps cover a different band.

Worth repeating across the seven arms after the run. No one has published whether a fixed-scale local INR
generalises off its training lattice, and the answer here is a clean yes.

## 4. What changes

1. **Add the gated deficit** — loud / mid / quiet frames — to `e4_eval.py`. Three lines on top of
   `band_energy_ratio`, and it measures the defect a listener actually hears, which the mean deficit
   hides.
2. **Print the probe's naive SNR in the dashboard**, next to `SNR0`.
3. **Report SNR only beside naive on the same audio**, and never rank on it.
4. **Do not low-pass the demo audio.** If a band limit is wanted, apply it to every condition and say so.
5. **Carry the LSD definition inline** wherever we print one: `log10(|S|² + 1e-10)`, RMS over frequency
   within a frame, mean over frames, n_fft 1024, hop 256. No two papers in this literature use the same
   one.
6. **Queue the energy-weighted ERB arm** with the §2.2 prediction pre-registered.

## 5. Prior art

Perception–distortion: Blau & Michaeli 2018. Energy score and properness: Gneiting & Raftery 2007.
Local implicit decoding and cell decoding: Chen, Liu, Wang, LIIF, CVPR 2021. The SNR-versus-LSD pattern
in bandwidth extension: [NU-Wave](https://arxiv.org/abs/2104.02321) Table 2,
[WSRGlow](https://arxiv.org/abs/2106.08507) §4.3, [AP-BWE](https://arxiv.org/abs/2401.06387) §V.
