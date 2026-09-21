# Research note — the deficit is an envelope, and the envelope is predictable

*21 September 2026. Scripts: `audit/envelope_regression.py`, `audit/envelope_rescale.py`. Data:
`lisa_rtm_cache/results/envelope_{OV3_fast,rescale_OV3_fast}.json`. Twelve utterances, six held-out
speakers, the OV3_fast checkpoints at step 16,000, evaluation STFT (n\_fft 1024, hop 256).*

The 16 September note ended with a pre-registered arm: *score the high-band envelope against the
low-band envelope. Score the conditional, not the marginal.* This note measures whether that arm is
worth running. It is, but not for the reason I expected, and one of the three obvious ways to build
it is already refuted.

## 1. The question

Every arm is too quiet on loud frames and too loud in the gaps. Two different failures make that
shape and they need opposite fixes.

* **Compressed variance.** The model knows which frames need energy and under-delivers in proportion.
  Regression to the mean, one level up from the waveform. Signature: slope below 1, correlation high.
* **Wrong assignment.** The model emits about the right spread of energies and does not know which
  frame gets which. Signature: correlation near zero.

So regress the model's log high-band frame energy on the truth's, across frames, and read the slope,
the correlation and the ratio of standard deviations. A correct sampler sits at 1, 1, 1. Note that
slope = r × sd ratio, so the two right-hand columns decompose the first.

## 2. It is compressed variance, in every arm

| arm | slope | r | sd ratio |
|---|---|---|---|
| `det` | 0.860 | 0.927 | 0.927 |
| `es_marg` | 0.746 | 0.898 | 0.829 |
| `es_marg_l0.1` | 0.800 | 0.905 | 0.878 |
| `es_split_l0.1` | 0.747 | 0.885 | 0.835 |
| `es_erb_l0.1` | 0.858 | 0.858 | 0.996 |
| `es_dec_l0.1` | 0.838 | 0.881 | 0.947 |
| `es_dec_erb_l0.1` | 0.861 | 0.869 | 0.985 |

Slope under 1 everywhere, correlation between 0.86 and 0.93 everywhere. The models know perfectly
well where the energy goes. They will not commit to how much.

The decomposition separates the ERB arms from the rest. `d_ERB` fixed the compression — sd ratio
0.829 to 0.996 — and slightly *worsened* the tracking, r 0.898 to 0.858. That is what a per-cell
score should do. It pins the marginal distribution of band energy. It pays nothing for getting the
frame right.

`audit/envelope_regression.py` labels the ERB arms "calibrated" because its threshold is slope < 0.8.
The threshold is too lenient. Slope 0.86 with r 0.86 is not calibrated; it is a model whose spread is
the right size and partly in the wrong places.

## 3. The envelope is 88 % predictable from the low band

On the truth alone, with no model involved:

```
r(log E_hb, log E_lb)                            +0.377
R^2 of log E_hb on a 16-band log low-band profile +0.891   (in-sample, per utterance)
R^2 of 16 high ERB band energies on the same,
    least squares, leave-one-speaker-out          +0.882   (0.85-0.90 across the six)
residual sd                                        6.19 dB
```

One scalar of low-band energy is worth little. The low band's **shape** is worth almost everything.
A linear map, seventeen numbers per band, transfers across speakers.

So the high band is two objects, not one:

| | predictable | what it wants |
|---|---|---|
| envelope — energy per ERB band per frame | R² 0.88 across speakers | to be predicted |
| fine structure — detail within a band, phase | ≤ 2 % coherent (8 Sep note) | to be sampled |

We sample both. That is the error.

## 4. Six ways to fix the envelope after the fact

`audit/envelope_rescale.py` takes each arm's draw, applies a gain per (ERB band, frame) to the high
band, leaves phase and within-band structure alone, and remeasures. `es_marg`, the middle of the pack:

| gain | deficit | loud | quiet | swing | LSD | HB-LSD |
|---|---|---|---|---|---|---|
| raw | −8.14 | −8.85 | −1.57 | −7.27 | 0.931 | 1.023 |
| `global` — one gain per band per utterance | +0.06 | −0.70 | **+7.04** | −7.75 | 1.044 | 1.161 |
| `fitted` — leave-one-speaker-out least squares | −2.97 | −3.21 | −0.43 | −2.78 | **0.918** | **1.009** |
| `fit+inf` — the same, variance-inflated | −1.62 | −1.79 | −0.65 | **−1.14** | 0.930 | 1.023 |
| `fit+samp` — the same, residual sampled | **+0.13** | **−0.13** | +2.54 | −2.67 | 0.978 | 1.081 |
| `oracle` — the truth's own band energy | −0.13 | −0.14 | +0.10 | −0.24 | 0.744 | 0.798 |

Four things come out of that table.

**`global` is a fraud detector and it fires.** One gain per band drives the mean deficit to zero and
does not touch the swing, because it buys the zero by putting 7 dB of surplus into the silences. LSD
gets worse. The deficit is not a level problem, and band-energy deficit on its own can be gamed —
the same lesson the transport ladder's shaped-noise control taught.

**`fitted` flattens the field.** Every arm lands within 0.026 LSD of every other: 0.899 to 0.924,
HB-LSD 1.00 to 1.05. `det` comes in at 0.924 from a raw 1.159. A linear map on the low band, fitted
on five speakers and applied to a sixth, takes the *deterministic* model past every trained sampler's
raw LSD. The seven arms differ, on this evidence, mostly in one scalar per band per frame.

**Sampling the envelope residual as i.i.d. noise is wrong.** `fit+samp` is the honest Bayesian move
— predict the mean, add a draw from the residual spread — and it drives the deficit to zero while
leaving the swing where it was and costing 0.06 LSD. It fills the silences. The residual is 6.19 dB
and it is *not* homoscedastic: its spread must be conditional on loudness. Deterministic inflation
(`fit+inf`) halves the swing instead, and costs a third as much LSD.

**The predictor inherits the disease.** `fitted` still shows a −3 dB loud deficit while predicting
the truth's own energy, because least squares returns a conditional mean, and a conditional mean of a
log energy regresses to the mean exactly the way the network does. Fixing regression to the mean with
a regression is circular. This is the whole argument for a proper score, restated one level up.

## 5. Which half of the high band is broken

`swap` runs the exchange the other way: the truth's fine structure and phase carrying the model's
envelope. Its low band is the truth's, so its full LSD and SNR are meaningless; read only HB-LSD and
the band-energy columns, which are restricted to bins above 6 kHz.

| what is correct | HB-LSD | deficit | swing |
|---|---|---|---|
| neither (`raw`) | 1.023 | −8.14 | −7.27 |
| envelope only (`oracle`) | 0.798 | −0.13 | −0.24 |
| fine structure only (`swap`) | **0.607** | −8.49 | −7.73 |
| both | 0 | 0 | 0 |

The two metrics disagree about which half is worse, and they disagree because they are measuring
different halves. Fixing the fine structure buys 41 % of HB-LSD and nothing at all of the deficit.
Fixing the envelope buys 22 % of HB-LSD and **all** of the deficit, the gating and the fricatives.

This is the perception–distortion split showing up as two halves of one band. LSD counts bins, so it
counts fine structure. The deficit counts energy in time, so it counts the envelope — and the
envelope is the half you hear as muffle. LSD was already known to reward regression to the mean
(`lisa_rtm.ipynb` §5, `notes/2026-09-17-lisa-reported-numbers-audit.md` §3.3); this says it is also
looking at the wrong half.

SNR falls monotonically with every dB restored, in every row of every table. Established, expected,
not a regression (`notes/2026-09-17-snr-ceiling-gating-and-scale.md` §1).

## 6. The arm this pre-registers

Not "score the envelope against the low-band envelope". That was the right target and the wrong verb.

**`es_env`: split the score by what is predictable.**

1. A head predicts the high band's ERB energy per frame from the low band's ERB profile — the thing
   with R² 0.88. Train it under a proper score on the *band-energy vector*, not L2, so its spread is
   conditional. §4 shows both halves of why: L2 would regress to the mean, and a homoscedastic
   residual would fill the silences.
2. The waveform path keeps the energy score, on a **per-band normalised** high band, so it only has
   to get the shape right and cannot trade shape against level.
3. Multiply at the output.

**Predictions, before the run.** Deficit above −1.5 dB and swing above −1.5 dB, from `fit+inf`.
HB-LSD at or under 0.95 — better than `fitted`'s 1.01, because the network sees the whole low band
and not sixteen numbers, and worse than `oracle`'s 0.80. Audio-mode ViSQOL up on `es_erb_l0.1`, and
that is the one that decides it: `global` proved the deficit alone can be bought.

**The falsifier.** If `es_env` improves the deficit but moves neither HB-LSD nor audio ViSQOL, it is
`global` with more parameters, and the split was not worth it.

**The cheap thing to do first.** `fitted` is post hoc. It needs no retraining, it is 16 × 17 numbers,
and it already beats every arm's raw LSD. Score it under ViSQOL and CRPS as an eighth readout
alongside `draw` / `mean16` / `logmean16`. If a linear map on held-out speakers survives ViSQOL, that
is a result on its own, and it sets the bar the trained arm has to clear.

## 7. What this does not establish

The fit is twelve utterances from six speakers. R² holds at 0.85–0.90 across all six, which is what
makes it credible, but twelve utterances is twelve utterances. Nothing here has been through ViSQOL
or any listener — `global` is the standing reminder that a deficit driven to zero can sound worse.
The 16 ERB bands above 6 kHz and the 16-band low profile are unswept choices. And `oracle` is an
oracle: it says where the ceiling is, not that a network reaches it.
