# Audit — where LISA's reported 24.16 dB comes from

**Date:** 2026-09-17
**Target:** Kim, Lee, Hong, Ok, *Learning Continuous Representation of Audio for Arbitrary Scale Super
Resolution*, ICASSP 2022. [arXiv 2111.00195](https://arxiv.org/abs/2111.00195),
[ml-postech/LISA](https://github.com/ml-postech/LISA) (default branch `master`, HEAD `4f3c1cd`,
last pushed 2022-03-30).
**Why:** this repo has carried an open item since 4 September — *settle the paper's 24.16 dB by its SNR
convention and down-sampling operator against ml-postech/LISA, not by more training*. Four runs failed to
reproduce it. This audit closes the item. It is a separate note from
[`2026-09-17-snr-ceiling-gating-and-scale.md`](2026-09-17-snr-ceiling-gating-and-scale.md), which covers
what we learned about our own models.
**Reproduce:** `venv/bin/python audit/lisa_paper_protocol.py` (no GPU, fetches ~40 MB of VCTK).

---

## 0. Finding

LISA's headline 12 kHz → 48 kHz SNR of **24.16 dB** sits **3.16 dB above a hard information ceiling**
measured under the paper's own protocol, with the paper's own resampler, on eight of the paper's own
held-out speakers. No model of any size can reach it from a band-limited 12 kHz input.

The released evaluation script computes that number on **one batch of eight one-second chunks** — eight
seconds, about 0.04 % of its validation set — and the batch-to-batch standard deviation of a plain sinc
interpolator at that sample size is **3.63 dB**. A sinc interpolator with no high band at all scores
≥ 24.16 dB on **12.5 %** of consecutive batches.

The claimed margin over WSRGlow is 4.75 dB. The measurement's own noise is 3.63 dB, and n = 1.

We cannot prove the table was produced by that script rather than by the unguarded validator in
`train_lisa.py`. §7 states the uncertainty precisely and §9 gives the one experiment that would close it.
What we can state without qualification is that **24.16 dB is not reachable**, and that the released
evaluation cannot distinguish a real result from a lucky eight seconds.

The likely story is not fraud. It is a figure-generation edit left in the metric path. §3.2 explains why.

---

## 1. The claim

Table 1, verbatim. Column headers `×2 (24k→48k)` and `×4 (12k→48k)`, each with SNR↑ and LSD↓.

| Method | ×2 SNR | ×2 LSD | ×4 SNR | ×4 LSD | # params |
|---|---|---|---|---|---|
| AudioUNet [7] | 21.68 | 1.31 | 18.55 | 2.11 | 71M |
| AudioTFilm [19] | 22.23 | 1.05 | 19.51 | 2.02 | 68M |
| WSRGlow [10] | 25.29 | 0.61 | 19.41 | 1.01 | 229M |
| LISA (Ours) | 30.7 | 0.58 | **24.16** | **0.81** | 89k |

Metrics, Eq. 4 and Eq. 5:

> SNR(**x**, **x̂**) = 10 log ‖**x**‖²₂ / ‖**x** − **x̂**‖²₂
> LSD(**x**, **x̂**) = (1/L) Σ_ℓ √( (1/K) Σ_k (X(ℓ,k) − X̂(ℓ,k))² ), where X(ℓ,k) = log|STFT^(ℓ,k)(**x**)|²

Eq. 4 is the ordinary SNR and matches this repo's `snr_db`. Eq. 5 is the field's usual LSD. Neither is
the problem; §3.3 shows the code computes something else for LSD.

The paper is a 4-page ICASSP submission and is thin on protocol. Not stated anywhere: the resampling
tool, any anti-aliasing detail, the log base in Eq. 4 or Eq. 5, any STFT parameter for LSD, any
normalisation/alignment/trimming, the per-utterance versus global averaging convention, the value of λ,
the batch size, the segment length, or which speaker IDs make the split. All of it lives in the code.

## 2. Alternatives considered and ruled out

Before the finding, the things it is *not*. Each was a live hypothesis; each was tested.

**Not aliasing.** The strongest prior hypothesis was that the low-rate input was made by plain
subsampling, leaving folded copies of the 6–24 kHz band inside the input that a network could partially
unfold. That would raise the ceiling enormously — our oracle for an aliased input is 22.99 dB against
19.11 dB for a clean one, and a per-bin LMMSE unfolder recovers 72.7 % of the high band. It is also
exactly what the phrase "downsampling … with sinc interpolation [16]" could mean if implemented literally
at coincident grid points. **It is not what they do.** `datasets/wrappers.py:215`:

```python
def resize_audio_fn(audio, sr, down_sr):
    resampler = audioTransform.Resample(sr, down_sr, resampling_method='sinc_interpolation')
    return resampler(audio)
```

In torchaudio 0.6.0 that routes to `kaldi.resample_waveform`, i.e. Kaldi `LinearResample`: a
Hann-windowed sinc low-pass at `0.99 × Nyquist` of the lower rate, `lowpass_filter_width=6`. There is no
`x[::4]` in the data path (the only slice of that shape in the repo is commented out, in
`tools/print_sdr.py:36`, an offline utility). The high band is destroyed, not folded down. Hypothesis
refuted.

Their filter is looser in the transition band than `scipy.signal.resample_poly` — −12.6 dB at 6.5 kHz and
−22.4 dB at 7.0 kHz against −18.5 and −75.1 — so a little energy does leak and fold. The *oracle* value
of perfectly unfolding everything between 6 and 7.5 kHz is about **+1.6 dB**, and that assumes separating
an aliased copy buried 8–22 dB under the baseband. It cannot buy 3 dB, but it is a genuine asymmetry
between their pipeline and ours and is the one thing worth A/B-ing if we ever want strict comparability.

**Not the filtered ground truth.** Their target also passes through `resize_audio_fn` — at `gt_sr = 48000`
this is `Resample(48000, 48000)`, which is *not* identity: a 13-tap low-pass at 23.76 kHz. Measured
response: 0.00 dB at 16 kHz, −0.26 dB at 20 kHz, −0.54 dB at 23.8 kHz. It moves the naive SNR by 0.04 dB.
Real, undocumented, negligible.

**Not the averaging convention.** `calc_snr` returns a mean of per-chunk dB rather than a pooled energy
ratio. That is a genuine departure from our convention and from WSRGlow's and NU-Wave's, and it is worth
about +1.6 dB on this data (20.67 per-chunk-mean against 19.01 per-utterance-mean under our resampler).
It is folded into the measurement in §4 — the 21.00 dB ceiling is *already* computed their way — so it is
accounted for, not excluded.

**Not the corpus release.** To reach 24.16 dB with no coherent high-band prediction, the target's energy
above 6 kHz would have to be **0.384 %**. We measure 1.94 % per utterance and 2.44 % per chunk on VCTK
0.92 mic1. They use the deprecated `torchaudio.datasets.VCTK`, i.e. VCTK-Corpus 0.80 `wav48/` —
untrimmed, so *more* silence, and silence is spectrally flatter and raises the high-band fraction rather
than lowering it. A 5 dB deficit in high-frequency content between releases of the same recordings is not
credible, and their LSD of 0.81 proves their model emits a high band regardless.

**Not the `--sr` flag bug.** `eval_lisa.py:303` has `parser.add_argument('--sr', default=None)` with no
`type=int`, and `scripts/audio_eval.sh` passes `--sr 12000`, which would make input and target the same
12 kHz signal — an identity task. That is a real bug (it also raises `TypeError` inside `Resample` before
it can inflate anything), but an identity task scores far *above* 24 dB; Fig. 4's ×1 point sits near
35 dB. So the ×4 row more plausibly came from `scripts/audio_eval_multi.sh` with `--sr 48000`.

## 3. What the code does

### 3.1 The ceiling is arithmetic

A point predictor reproduces the low band exactly — sinc interpolation is the optimal reconstruction of a
band-limited signal — so its entire error is high-band energy it fails to reproduce **in phase**. With the
high band at 2.44 % of the energy, leaving it empty scores 21.00 dB. Reaching 24.16 dB means the residual
must fall to 0.384 % of the signal, which requires putting back roughly **two thirds to four fifths of
the band coherently** (68 % under mean-of-dB, 80 % under dB-of-mean — the exact figure moves with the
averaging convention, the conclusion does not).

Our measured coherent fraction above 6 kHz, for every model trained in this repo including a 22 ms
dilated-context encoder, is **≤ 2 %** (`OV2_wide`, `notes/2026-09-08-network-as-transport-map.md`). The
gap is a factor of 34 to 40.

The same arithmetic applied to their Table 2 is worse, not better. Against the energy above 4 kHz, the
8 kHz-input rows sit **+9.5 to +12.1 dB** above an empty-band ceiling, and the largest anomaly is on the
8 k→48 k row the paper calls out as out-of-distribution. An anomaly that is roughly constant across every
operating point is a property of the pipeline or the metric, not of a model that has learned something.

### 3.2 The evaluation scores one batch

`eval_lisa.py`, the repo's only evaluation entry point:

```python
    ii = 0
    pbar = tqdm(val_loader, leave=False, desc='val')
    for batch in pbar:
        ...
        pred = model(batch)
        pred.clamp_(-1, 1)
        index = 0

        if ii == 3:
            with torch.no_grad():
                snr = utils.calc_snr(pred.detach().clone(), batch['gt'])
                lsd = utils.compute_log_distortion(pred.detach().clone().cpu(), batch['gt'].cpu())
            ...
            val_res_snr.add(snr, batch_size)   # TODO : average for dB
            val_res_lsd.add(lsd, batch_size)
            ...
            plot_specgram(p_gt,   48000, f"gt{ii}_{index}.pdf")
            plot_specgram(p_pred, 48000, f"pred{ii}_{index}.pdf")
        ii += 1
```

The validation split is `int(speaker_id[1:4]) >= 350` (`datasets/audio_dataset.py:59`) minus four
excluded utterances — **8 speakers, 2,959 utterances**. The wrapper chunks at `chunk_len: 1` second with
`chunk_num = wave_len // chunk_wave_len + 1`, so the set is roughly **12,000–18,000 chunks**, about five
hours of audio. At `batch_size: 8` that is 1,500–2,300 batches. They report index 3.

`Averager` therefore receives **exactly one value**, which makes the `batch_size` weighting vacuous and
leaves no error bar, no mean, and nothing to average. The `# TODO : average for dB` comment on the very
line that adds it suggests the author knew the aggregation was unfinished.

Three details argue this is an accident rather than a shortcut, and they matter for how we describe it:

1. **It is not a speed optimisation.** The loop iterates the whole validation set and forward-passes
   every chunk through the model; only the metric call sits inside the `if`. A `break` would have saved
   the time. Keeping the full loop and scoring one batch is not what anyone writes to economise.
2. **The same block writes the figures.** `gt3_0.pdf` and `pred3_0.pdf` are Figure 3 of the paper
   (its caption: *"(b) and (c) are super resolution results of the ×4 (12kHz → 48kHz) setting in
   Table 1"*). The guard exists to produce the picture; the metric got swept inside it.
3. **A correct number existed.** `train_lisa.py::validate` has no such guard and averages the whole
   validation loader. A full-set figure was on screen every epoch of training.

Neither `scripts/audio_train.sh` nor `scripts/audio_eval.sh` runs as shipped — the former hits
`resume: /data/sss/save/final_4x/epoch-last.pth`, a checkpoint that was never released; the latter hits
the `--sr` type bug. The shipped training config is also inconsistent with the evaluated task:
`input_sr: 8000` with `gt_aug_max: 3` trains 1×–3× from 8 kHz, so `gt_sr: 48000` on line 13 is dead code
and the 4× setting of Table 1 is never trained by the released config.

### 3.3 The LSD is not the field's LSD

`utils.py:171`:

```python
def get_power(x):
    S = T.Spectrogram(n_fft=2048)(x)
    S = torch.log(torch.abs(S)**2 + 1e-8) / math.log(10)
    return S

def compute_log_distortion(x_pr, x_hr):
    x_hr = torch.flatten(x_hr)
    x_pr = torch.flatten(x_pr)
    S1 = get_power(x_hr)
    S2 = get_power(x_pr)
    lsd = torch.mean(torch.sqrt(torch.mean((S1-S2)**2 + 1e-8, dim=1)), dim=0)
    return min(lsd, 10.)
```

Four departures from Eq. 5 as printed in their own paper:

1. **It logs the fourth power.** `torchaudio.transforms.Spectrogram` defaults to `power=2.0` and already
   returns `|X|²`; squaring again gives `log10(|X|⁴) = 4 log10|X|`, where Eq. 5 and the field use
   `log10(|X|²) = 2 log10|X|`.
2. **The epsilon then floors a sixth of the spectrogram.** `|X|⁴ + 1e-8` is dominated by the epsilon
   whenever `|X| < 0.01`. On peak-normalised VCTK chunks at n_fft 2048 the median `|X|` is 5.2e-02 and
   **16.6 % of time-frequency bins are pinned at the floor**, where the conventional
   `log10(|X|² + 1e-10)` floors **0.0 %**. The metric is blind wherever it is pinned, which is
   preferentially the quiet high band — exactly the region bandwidth extension is judged on.
3. **The reduction axes are transposed.** `S` is `[freq, time]`; `mean(..., dim=1)` averages over *time*,
   then `mean(..., dim=0)` over *frequency*. Eq. 5 says RMS over frequency inside a frame, then mean over
   frames. Averaging `|·|` across frequency instead of RMS compresses the number substantially, since
   frequency is the axis with the dynamic range. (This particular error is inherited verbatim from
   Kuleshov's `audio-super-res`, so AudioUNet and TFiLM carry it as well.)
4. **It is computed on a flattened batch.** `torch.flatten` glues eight one-second chunks end to end
   before the STFT, so windows straddle the joins.

Measured on the same signal — a naive sinc upsample of the 12 kHz input, our 391 held-out chunks:

| definition | LSD |
|---|---|
| LISA's `compute_log_distortion` | **2.213** |
| conventional `log10(|S|²)`, RMS over frequency within a frame | **5.172** |

So their scale is not ours, and their 0.81 cannot be read against our 0.95. Nor against NU-Wave's,
NVSR's or AP-BWE's, which use `2 log10|X|`; nor against TFiLM's, which uses natural log on 8092-sample
frames and says so in a footnote. **No two LSD columns in this literature are in the same unit**, and
every paper prints the same equation.

Their own ablation is the tell. Table 2 reports that removing the multi-resolution spectral loss changes
LSD by **+0.00, +0.00, +0.01** across three scales. A spectral loss that moves a spectral metric by zero
means the metric is not responding to spectral content.

## 4. The ceiling, measured

`audit/lisa_paper_protocol.py`. Eight speakers from their own split (p351, p360–p364, p374, p376),
64 utterances, 391 one-second chunks. Their Kaldi resampler reimplemented from torchaudio 0.6.0 source,
their chunking, their `calc_snr`, their mean-of-per-chunk-dB.

```
energy at or above 6 kHz                        2.443 %
naive sinc upsample of the 12 kHz input         20.67 dB
perfect low band, empty high band (ceiling)     21.00 dB
LISA reports                                    24.16 dB   (+3.16)
```

The 21.00 dB row is not a baseline; it is an upper bound on every point predictor that does not
reconstruct high-band phase. It already includes their per-chunk dB averaging and their resampler. The
naive row sits 0.33 dB under it, which is the whole value of doing nothing well.

## 5. The batch of eight

Scoring the **naive** baseline — no network at all — the way `eval_lisa.py` does:

```
naive sinc upsample, 391 chunks
  consecutive 8-chunk batches (48)    sd 3.63 dB   range 13.21 to 29.04
  random 8-chunk batches (20000)      sd 2.37 dB
  P(a batch scores >= 24.16)          12.5 % consecutive,  7.1 % random
```

One consecutive batch in eight, from a sinc interpolator, beats the paper's headline. The window is not
random: `random_seed = 0`, `ii == 3` hardcoded, the loader unshuffled — it is the same few opening
utterances of the first test speaker on every run.

For scale: the reported margin over WSRGlow is 24.16 − 19.41 = **4.75 dB**, and the standard deviation of
the measurement is **3.63 dB**, at n = 1.

## 6. The baselines are not on the same task

Worth recording, because it also explains why the other three rows cluster near naive.

Kuleshov's `data/vctk/prep_vctk.py` makes the anti-aliasing filter a command-line flag:

```python
      if args.low_pass:
        x_lr = decimate(x, args.scale)
      else:
        x_lr = np.array(x[0::args.scale])
```

`data/vctk/speaker1/Makefile` passes `--low-pass`. **`data/vctk/multispeaker/Makefile` does not.** So the
widely-cited AudioUNet and TFiLM multi-speaker numbers are produced from fully aliased input. AFiLM does
the same (`codes/utils.py`). AudioUNet's own Table 3 measures what that is worth: aliased-train/
aliased-test 33.2 dB against filtered/filtered 30.1 dB, and either mismatch collapses to ~0.4 dB. Their
README says it plainly — *"super-resolution works better on aliased input"* and *"the model is very
sensitive to how low resolution samples are generated."*

AP-BWE states the consequence for comparability directly: TFiLM and AFiLM *"employed subsampling to
obtain the narrowband waveforms, which aliased high-frequency components. Thus, they performed not a
strict BWE task but an SR task."*

LISA says it re-ran baselines *"using the original authors' official source codes"* where pre-trained
models were unavailable, without saying which. If any Table 1 baseline came from the aliased recipe, that
row is measuring a different problem from LISA's own.

## 7. What we can and cannot claim

**Established, no qualification needed:**

- 24.16 dB is above the information ceiling of the 12 kHz → 48 kHz task under LISA's own resampler,
  chunking and metric, on LISA's own held-out speakers. Measured: ceiling 21.00 dB.
- The released evaluation reports a single batch of eight one-second chunks and cannot distinguish a
  result from sampling noise; the noise is 3.63 dB and the claimed margin is 4.75 dB.
- LISA's `compute_log_distortion` does not compute the LSD its paper defines, and 16.6 % of its
  spectrogram is pinned at its epsilon floor.
- The downsampling operator is honest and anti-aliased; the aliasing explanation is refuted.

**Not established:**

- That Table 1 was produced by `eval_lisa.py` rather than by `train_lisa.py`'s unguarded validator. The
  Figure 3 coincidence is circumstantial evidence, not proof.
- Any claim about intent. A guard that also writes the paper's figures is far better explained by an
  edit nobody took back out.

**Circumstantial, and worth stating once:** no independent reproduction of LISA exists. Sixteen stars,
one unmodified mirror fork, two issues from 2022 — both asking for audio demos — unanswered, and no demos
published. Of sixteen citations, almost all are image and MRI INR work; NVSR names LISA once in a list
and does not benchmark it. `ssr_eval`, the field's shared harness, has no LISA row. A paper claiming to
beat a 229M-parameter flow by 4.75 dB with 89k parameters, and four years of silence from the
bandwidth-extension literature.

## 8. Why this took four runs to find

Recorded so it does not happen again. The 1 September note put the paper's numbers *in evidence against*
our own measurement — hypothesis H1′ reads *"Unknown; the paper's numbers argue against a large one …
SNR 24.16 dB with a ~2–3 % high-band energy fraction implies most high-band energy is recovered
coherently."* That inference was sound given the premise and the premise was wrong. The 4 September note
got closer, naming *"a different SNR convention, a different downsampling operator … or their evaluation
including the input band differently"* and correctly flagging that all three baselines land within ~1 dB
of naive on our data, but stopped at *"Not resolved here."*

The lesson is narrow and practical: **when a published number sits above a ceiling you can compute, the
number is the thing to audit, not the ceiling.** Three failed reproductions cost more GPU hours than
reading `eval_lisa.py` would have cost in minutes.

## 9. What would settle it definitively

One run, no training, no checkpoint. In a clone of `ml-postech/LISA`: delete the `if ii == 3:` guard,
replace `pred = model(batch)` with a plain sinc upsample of `batch['inp']`, and report `utils.calc_snr`
and `utils.compute_log_distortion` three ways — (a) full validation set, (b) batch 3 only, (c) full set
with the input made by `scipy.signal.resample_poly(x, 1, 4)` instead of torchaudio.

- (a) ≈ 20–21 dB and (b) ≈ 24 dB → the headline is a sinc baseline on a lucky eight seconds.
- (a) ≈ 24 dB → the ceiling is in their data pipeline, not the model, and no network is needed to beat
  the published literature.
- (a) − (c) → prices their looser transition band, which is the one legitimate asymmetry against us.

`audit/lisa_paper_protocol.py` already computes the equivalent of (a) and (b) outside their codebase,
which is why this note states its conclusions now. Running it inside their code would remove the last
reimplementation caveat.

## 10. How to report this

Not as a takedown. The finding that matters for our paper is the positive one: **the task has a ceiling,
we can compute it, and every honest published number sits at it.** AudioUNet 17.15/18.55, MUGAN 16.87,
WSRGlow 18.38/19.41, TFiLM 19.51 — every independently reported 12 kHz → 48 kHz SNR in the literature
falls in 17–19.5 dB, which is where we land, and which is where the arithmetic says everyone must land.

So: state the ceiling, show that SNR cannot rank models on this task
([the companion note](2026-09-17-snr-ceiling-gating-and-scale.md) §1), and move the adjudication to CRPS,
the band-energy deficit, the coherent fraction and audio-mode ViSQOL. One paragraph on LISA's number, in
the protocol section, factual and without adjectives: the operator is anti-aliased, the metric is
computed on one batch of eight seconds, the ceiling is 21.00 dB. Cite the line numbers. Let the reader
draw the conclusion.

And apply the same standard to ourselves. Every number we publish states its degradation operator, its
averaging convention, its LSD definition inline, and the size of the set it was measured on.

## 11. Sources

Primary: [arXiv 2111.00195](https://arxiv.org/abs/2111.00195) (read as ar5iv full text),
[ml-postech/LISA](https://github.com/ml-postech/LISA) at HEAD `4f3c1cd`, and
[torchaudio v0.6.0](https://github.com/pytorch/audio/blob/v0.6.0/torchaudio/compliance/kaldi.py)
`compliance/kaldi.py` for the resampler semantics. Comparators:
[AudioUNet](https://arxiv.org/abs/1708.00853) and
[kuleshov/audio-super-res](https://github.com/kuleshov/audio-super-res),
[TFiLM](https://arxiv.org/abs/1909.06628), [AFiLM](https://github.com/ncarraz/AFILM),
[WSRGlow](https://arxiv.org/abs/2106.08507), [NU-Wave](https://arxiv.org/abs/2104.02321),
[NVSR](https://arxiv.org/abs/2203.14941) and [ssr_eval](https://github.com/haoheliu/ssr_eval),
[AP-BWE](https://arxiv.org/abs/2401.06387).
