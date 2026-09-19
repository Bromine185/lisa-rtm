# Bandwidth Extension Has an Information Ceiling: SNR Cannot Rank Models on This Task

**2026-09-18** · every number below is reproduced from a clean clone by three CPU scripts that need no
GPU, no checkpoint and no training, and that fetch their own ~15 MB of VCTK:
`audit/snr_scale.py`, `audit/protocol_fork.py`, `audit/lisa_paper_protocol.py` —
[github.com/Bromine185/lisa-rtm](https://github.com/Bromine185/lisa-rtm).

---

## 1. Setup

Bandwidth extension from 12 kHz to 48 kHz is the linear inverse problem `x = A y`, with
`A = D_R ∘ H`: an anti-alias low-pass `H` at 6 kHz, then decimation by `R = 4`. Its null space is
exactly the band being estimated, `ker(A) = range(P_H)`, because `H` annihilates everything above
6 kHz and nothing else. So the observation determines the baseband exactly — by Shannon
interpolation, with no error — and every degree of freedom a model has lives in the null space.

## 2. The identity

A point predictor reproduces the low band exactly, so its **entire** error is high-band energy it
fails to reproduce *in phase*. SNR is therefore a monotone function of one quantity: the coherently
recovered fraction `ρ` of the band above 6 kHz.

> `SNR(ρ) = −10 log₁₀( φ · (1 − ρ) )`,  `φ` = fraction of the signal's energy above 6 kHz,
> evaluated per utterance and then averaged in dB (§6 is about that choice).

Equivalently: for an ideal `H` the optimal Wiener filter for the missing band is the **zero filter**,
`W(f) = S_xy/S_xx = 0` for `f > 6 kHz`, and naive sinc upsampling is the LMMSE solution — not a weak
baseline to be beaten by a better architecture, but the answer the objective asks for.

A real `H` is not ideal, and we measure the difference rather than wave it away.
`scipy.signal.resample_poly`'s 81-tap Kaiser is only −6 dB down at 6.0 kHz, so 2.15 % of the band's
power survives it. Fitting a per-bin linear estimator of the missing band on four VCTK speakers and
scoring it on four others recovers **2.0–4.0 %** of the band (median 2.25 % over all eight rotations
of the split) — entirely from the kilohertz above the cut, where the fitted gain reaches 3.7. It is
worth **0.015 dB** above the `ρ = 0` bound. So the ceiling is *exact* for a brick wall and tight to
fifteen thousandths of a decibel for the filter everyone actually uses. §8 is what happens when the filter is removed.

## 3. Table 1 — the ceiling curve

64 held-out utterances from eight VCTK speakers, per utterance, averaged in dB. `φ = 1.943 %`.

| coherently recovered `ρ` | SNR | vs naive |
|---|---|---|
| 0 % | **19.11 dB** | +0.10 |
| 2 % — *every model we trained* | 19.19 | +0.19 |
| 10 % | 19.56 | +0.56 |
| 20 % | 20.08 | +1.07 |
| 50 % | 22.12 | +3.11 |
| naive sinc upsampling | **19.01** | — |

Naive upsampling measures 19.01 dB against a zero-recovery optimum of 19.11 dB. It *is* the
zero-recovery solution, to within 0.1 dB. Measured `ρ` above 6 kHz runs **0.004 to 0.025** across
every model we trained — deterministic, sampler, and a 22 ms dilated-context encoder whose `ρ` above
7 kHz is 0.000–0.005 — so 2 % is not a weak-model artefact, and recovering a tenth of the band buys
0.56 dB.

## 4. The two facts that end the argument

**The achievable span is about 1 dB.** Everything between "output nothing above 6 kHz" and
"recover a fifth of the band coherently" is compressed into 1.07 dB.

**The estimator's own noise is 4.32 dB.** Across those same 64 utterances the *same* estimator —
naive sinc upsampling, no model, no randomness — has a per-utterance standard deviation of 4.32 dB,
range 10.52 to 29.52. The metric's dynamic range is four times smaller than its own noise.

## 5. Table 2 — SNR actively punishes the task

Paired sweep, identical recipe and identical batches, only the spectral-loss weight `λ` differs.
36 held-out utterances (p236–p238); naive sinc upsampling scores 19.20 dB on that same audio:

| `λ` | SNR | LSD | high-band deficit | |
|---|---|---|---|---|
| 1e-3 | **19.16** | 1.546 | −17.42 dB | band empty |
| 1e-2 | 18.37 | 0.962 | −4.52 dB | |
| 1e-1 | 17.69 | 0.948 | −2.35 dB | band restored |

Monotone, and in the wrong direction. **The SNR-optimal model outputs nothing above 6 kHz**, landing
at 19.16 dB — naive to within 0.04 dB, with its high band 17.4 dB below the truth. The `λ = 1e-1`
arm is not losing; it is paying 1.47 dB of SNR to move the deficit by 15 dB, which is the right side
of the trade. Any energy a model adds above 6 kHz is
incoherent with the truth, and incoherent energy *adds* to the error. This is Blau–Michaeli, and on
this task the distortion axis has almost no range left to trade.

## 6. The ceiling is not convention-free — and that is the point

The map from an energy fraction to a decibel is not invariant to how you average. The *same*
reconstruction — exact low band, empty high band — on the *same* 168 held-out chunks:

| averaging rule | ceiling |
|---|---|
| mean of per-chunk dB | 22.06 dB |
| mean of per-utterance dB | 20.77 dB |
| pooled energy ratio | 19.82 dB |

A 2.24 dB spread from Jensen alone, twice the whole achievable span of §4. This is a strength of the
method, not a hedge: **quote every ceiling with its averaging rule, and compute it under the target's
own protocol.** A ceiling computed under your convention says nothing about a number reported under
theirs.

## 7. One worked example

LISA (ICASSP 2022, 89k parameters) reports 24.16 dB at 12 kHz → 48 kHz. Reimplementing its resampler
(torchaudio 0.6.0 `kaldi.resample_waveform`, from source), its 1-second chunking, its `calc_snr`
(`utils.py:162`) and its mean-of-per-chunk-dB, and running them on 391 chunks from eight of its own
held-out speakers (`datasets/audio_dataset.py:59`): `φ = 2.443 %`, naive scores 20.67 dB, and the
ceiling — perfect low band, empty high band — is **21.00 dB**. Its released evaluation computes the
reported number on one batch of eight 1-second chunks: the metric call sits inside an
`if ii == 3:` guard (`eval_lisa.py:206`) that also writes the paper's Figure 3, and `Averager` receives
exactly one value (`eval_lisa.py:225`, commented `# TODO : average for dB`). At that sample size a
plain sinc interpolator's batch-to-batch standard deviation is **3.63 dB**, against a claimed margin
over WSRGlow of 4.75 dB; 12.5 % of consecutive 8-chunk batches of *naive upsampling* score ≥ 24.16 dB.
By §2's identity, 24.16 dB from a band-limited 12 kHz input requires `ρ = 0.52` under their own
per-chunk averaging — half the band above 6 kHz put back in phase — and `ρ = 0.69` under the
per-utterance convention of §3. The two numbers differ because the conventions do; that is §6, and it
is why the ceiling in this section is computed their way and not ours. The largest `ρ` we could
measure for any model was 0.025.

We state this and stop. Our resampler is a reimplementation, not their binary, and we make no claim
about how the table was produced.

## 8. The protocol fork

Without anti-aliasing, `x = y[::R]` folds 6–24 kHz back into the baseband. `ker(A)` collapses, the
missing band is *present* in the observation, and the task becomes unmixing rather than synthesis.
Same estimator on both protocols, fitted on four speakers and scored on four others
(`audit/protocol_fork.py`):

| | anti-aliased `x = D_R(H y)` | aliased `x = y[::R]` |
|---|---|---|
| high-band power reaching the observation | **2.15 %** | **100.1 %** |
| recovered by an admissible per-bin linear filter, held out | 2.25 % *(2.0–4.0)* | **18.9 %** *(1.1–21.3)* |
| recovered by an oracle per-bin unfolder — *not admissible* | n/a | 50.8 % |

The first row is the whole argument and it needs no estimator: **the anti-alias filter destroys
97.8 % of the band before anything sees it; plain subsampling destroys none of it.** Everything else
is about what can be done with what is there.

The second row is the realizable answer, and it is an 8× median difference on the same corpus with
the same estimator. It is also a warning about single splits: the aliased figure spans 1.1 % to
21.3 % across the eight rotations of which four speakers are fitted, so *one* split proves nothing —
we report all eight. (The paper's own nominal split happens to be the worst of them.) The third row
is what a phase-blind per-bin unmixer reaches when the spectral split is handed to it; it is **not**
a bound in either direction, because the per-alias powers it uses are a function of `y`, not of `x`.

What we do **not** claim: a measured information ceiling for the aliased protocol. Its naive baseline
is 2.55 dB lower (the fold corrupts the baseband too), the oracle recovers 2.47 dB of that — about
half from the baseband, half from the high band — and on a second speaker set it lands *below* the
anti-aliased bound rather than above. The protocol-level, assumption-free statement is the first row.

The empirical consequence is not ours to prove: **AudioUNet's own Table 3 measures it.**
Aliased-train/aliased-test scores 33.2 dB against 30.1 dB for filtered/filtered, and either mismatch
collapses to 0.4 dB (Piano, r = 2). Its README says it outright — *"super-resolution works better on
aliased input"*, and *"the model is very sensitive to how low resolution samples are generated."*

And this is the shipped default. In `kuleshov/audio-super-res` (HEAD `45c75f4`), the standard
AudioUNet/TFiLM codebase, anti-aliasing is a flag that is **off unless you ask** — `prep_vctk.py:100`,
`if args.low_pass: x_lr = decimate(x, args.scale)` `else: x_lr = np.array(x[0::args.scale])`, with
`--low-pass` a bare `store_true` (`prep_vctk.py:32`). **`data/vctk/speaker1/Makefile` passes it;
`data/vctk/multispeaker/Makefile` does not.** The up-interpolation (a cubic spline) is identical in
both, so the anti-alias filter is the only difference between the two shipped datasets. AP-BWE states
the consequence out loud: those baselines *"performed not a strict BWE task but an SR task."* Which
Makefile produced which published table, the repository does not say — and that is the point. Half
the field's baselines may be tabulated on a different inverse problem, and nothing in the papers
lets a reader tell. This conclusion is independent of any claim about any individual paper.

## 9. The close: the literature already agrees, quietly

We checked every reported 12 kHz → 48 kHz SNR we could source. The finding is stronger than "they all
sit at the ceiling", and it is not the finding we expected.

**Exactly one of these numbers is self-reported**: WSRGlow's 18.38 dB. Every other value at this
setting is a reproduction by a later paper — AudioUNet 17.15 and MUGAN 16.87 (WSRGlow's
reimplementations); AudioUNet 18.55, TFiLM 19.51, WSRGlow 19.41 (LISA's reruns); WSRGlow 21.2,
NU-Wave 21.4, NU-Wave 2 21.6 (NU-Wave 2's retrainings); mdctGAN 21.74.

**WSRGlow alone spans 18.38 / 19.41 / 21.2 dB** — one model, one nominal task, 2.8 dB. That
between-protocol spread is *larger than the entire within-protocol achievable span of §4*. A number
without its degradation operator, its averaging convention and its split is not a measurement.

**And in NU-Wave 2's own Table 1 the unprocessed input scores 22.1 dB, higher than every model at
every input rate.** The field has been printing the ceiling next to the models for years.

It has also, quietly, stopped reporting SNR: AP-BWE, AERO, NVSR and FlowHigh report none at all.
AudioUNet said why in 2017 — *"Although, the spline baseline achieves a high SNR, its signal often
lacks higher frequencies; the LSD metric is better at identifying this problem"* — and NU-Wave 2
repeats it: *"SNR is not suitable for upsampling task."*

### What to adopt

1. **Report the ceiling beside every SNR**, computed under your own protocol, with its averaging rule
   named. It costs one CPU-minute and no checkpoint.
2. **State the degradation operator** — the filter, its cutoff, its order, whether it exists at all.
   Anti-aliased and aliased BWE are different problems with different ceilings (§8).
3. **Adjudicate elsewhere.** CRPS on high-band log-magnitude, the coherent fraction `κ`, and
   audio-mode ViSQOL reported with its floor and ceiling rows — on our 12-utterance set, a passthrough
   baseband with an empty high band scores 1.57 and the same baseband with the *true* high band scores
   4.73, which turns a raw MOS-LQO into a position on a bounded scale. Name the set: the ceiling row
   holds at 4.73 on a second 36-utterance set, but the floor moves to 1.93. Never rank arms on SNR, and
   never compare an LSD column across papers: no two of them define it the same way (LISA's
   `compute_log_distortion` logs `|X|⁴` and pins 16.6 % of the spectrogram at its epsilon floor,
   scoring 2.213 where the conventional definition scores 5.172 on the identical signal).

The ceiling is a tool, not an accusation. It takes a minute to compute, it applies to every
bandwidth-extension paper ever written, and it would have saved us three overnight GPU runs spent
trying to reproduce a number that arithmetic says is not reachable.
