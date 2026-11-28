# EARS: predictions before the run (2026-11-27)

Written and committed before any model has seen an EARS file. The camera-ready reports the outcome
against these, pass or fail.

## The test

- **Corpus.** EARS (Richter et al., Interspeech 2024): 48 kHz, anechoic, GRAS 40HH at about 1 m.
  CC BY-NC 4.0. None of the four released models lists it as training data. Our 88k arms saw VCTK only.
- **Speakers.** The official EARS test speakers, p102 to p107. Chosen by the dataset's own split, not by us.
- **Utterances.** Each speaker's `*_regular.wav` files (read speech, regular style), sorted by name, cut
  into consecutive non-overlapping 4 s windows. A window is kept if at least 80% of its 20 ms frames are
  within 40 dB of the file's loudest frame. From each speaker's kept windows, 16 evenly spaced ones
  (`np.linspace`, as `sota/fetch_wide.py` picks VCTK files). 6 × 16 = 96 utterances, 384 s.
- **Input.** As on VCTK: peak 0.95, order-8 Chebyshev I low-pass, `resample_poly` to 12 kHz
  (`sota/make_inputs.py`).
- **Draws.** 8 per utterance for every stochastic model, on all 96. Levels and calibration come from
  the same set.
- **Scorer.** `sota/score.py` unchanged, then `sota/bootstrap.py` (10,000 utterance resamples, 95%) and
  `sota/level_checks.py`-style slopes.

## Predictions

"Gap" is `gap_hb`, "indep" is `gap_indep_hb`, "diff" is gap minus indep, with its 95% interval.
Tails are shares of HB bins with the truth below (lower) or above (upper) all 8 draws; ideal 11.1%.

1. **FLowHigh, prior restored, is still a regressor.** Gap below half of indep. Lower + upper tail
   above 50%.
2. **NU-Wave 2 samples at its own level.** |diff| below 0.2 dB. Band-averaged deficit below −4 dB.
3. **AudioSR is under-dispersed.** Diff below −0.5 dB with the interval excluding 0. Upper tail above
   30%.
4. **Our sampler (`es_dec_erb_l0.1`) is close to independent and short in the upper tail.** Diff between
   −0.6 and 0 dB. Upper tail above lower tail. Lower tail between 7% and 15%.
5. **The loss domain still sets the deficit.** `det_paper` deficit below −15 dB. `det` above it by at
   least 8 dB. `es_marg` and `es_dec_erb_l0.1` within 3 dB of zero.
6. **Level regresses per utterance.** Slope of deficit on the truth's HB energy share is negative for
   every model, interval below 0.
7. **FLowHigh as released is a point estimate.** Gap below 0.01 dB.

## What would count as a failure

Any of 1 to 7 not holding. A failure is reported in the paper, not dropped. The likeliest failure is
5 or 4 for our arms: EARS is anechoic and recorded differently from VCTK, and our network has seen only
VCTK.
