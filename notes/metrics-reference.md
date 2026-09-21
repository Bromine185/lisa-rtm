# Every metric used to compare the eight arms

Written 2026-09-21 because the comparison had got hard to hold in one head. Definitions are quoted
from the code, not paraphrased. Line references are to the notebook cell that defines them
(`lisa_rtm.ipynb` code cell 4 unless stated) and to `overnight3/e4_eval.py` / `e5_visqol.py`.

Numbers throughout are the OV50 run on EVAL12 (12 held-out utterances, p236-p238, M = 16 draws).

---

## 0. The one thing to get right first

**Three of these metrics are anti-correlated with the others.** Rank the 43 evaluation conditions by
LSD and measure Spearman rho against each judge:

| column | rho vs LSD rank | reading |
|---|---|---|
| ViSQOL audio48k | **-0.927** | agrees strongly |
| NSIM audio | -0.882 | agrees |
| ViSQOL speech16k | -0.272 | barely related |
| NSIM speech | -0.371 | weak |
| **PESQ wb** | **+0.336** | disagrees |
| **SNR** | **+0.471** | disagrees |

SNR and PESQ reward the opposite of what LSD and audio-mode ViSQOL reward. Any sentence of the form
"arm X is better" has to name the metric, and a table sorted by one of them is running backwards in
two columns.

---

## 1. Training-record metrics

From `history_OV50_<arm>.json`, written by `fast/train_arm.py`. Logged every 25 steps (train) and
every 500 steps (val), on the **same fixed validation batches for every arm** -- 8 x 64 one-second
segments drawn by `stream(VAL_TAG)`, identical across the eight VMs by construction.

| field | what it is | cross-arm? |
|---|---|---|
| `loss` | EMA of the arm's own training objective | **no** |
| `wave` | waveform term of the training batch | yes in principle |
| `spec` | spectral term of the training batch | yes in principle |
| `spread` | `dwf(y1, y2)`, the energy score's ensemble spread | yes |
| `lr` | learning rate | -- |
| `val_loss` | the arm's **own objective** on the fixed val batches | **no** |
| **`val_wave`** | **the two-draw energy score of the waveform; plain L1 for a det arm** | **yes, as a proper score -- see below** |
| `val_spec` | the spectral term | yes, with one exception below |

**`val_loss` is not a leaderboard.** The eight objectives are different functions. `det_paper` has
lambda = 0; `det` adds 0.01 x a spectral term; the `-erb` arms fold `lambda x (ERB term)` *inside*,
so a larger lambda inflates the number by construction. `es_erb_l0.1`'s `val` is large because
lambda is 0.1, not because the arm is losing. `fast/compare.py` exists to print `val_wave` instead,
and `fast/plot_histories.py` deliberately never overlays `val_loss` on a shared axis.

`val_loss` **is** comparable within a shared `(kind, lambda)` pair, which here means exactly two
pairs: `{es_marg, es_dec_l0.01}` and `{es_erb_l0.1, es_dec_erb_l0.1}`.

**`val_wave` is a proper score, not a reconstruction error, and the difference is the whole
column.** `ArmStack.losses` (`e2b_fast.py:339, 358`) returns plain `|yh - y|` for a deterministic
arm and `0.5(|y-y1| + |y-y2|) - 0.5|y1-y2|` for a sampler. The energy score of a point mass IS its
MAE, so these are one proper score with the det branch as its degenerate case, and comparing them
across arms is legitimate in exactly the way CRPS is. But a sampler's number is its mean single-draw
L1 *minus half its ensemble spread*, and on OV50 the validation-time spread is 63-154% of `val_wave`
itself. Measured on the run's own validation batches by `fast/val_wave_decompose.py` -- the
identities `dw == 0.5(d1+d2) - 0.5 d12` and `spread == d12` verified at tensor level to 9e-10 from
an independent forward, the `val_wave` anchored to the recorded curves to 3e-10, seed noise on the
single-draw L1 under 0.13%:

| arm | val_wave | val spread | spread / val_wave | single-draw L1 |
|---|---|---|---|---|
| det_paper | 0.004027 | 0 | -- | **0.004027** |
| det | 0.004323 | 0 | -- | 0.004323 |
| es_dec_erb_l0.1 | 0.003564 | 0.002236 | 0.63 | 0.004682 |
| es_erb_l0.1 | 0.003561 | 0.002308 | 0.65 | 0.004715 |
| es_marg | 0.003249 | 0.003971 | 1.22 | 0.005234 |
| es_dec_l0.01 | 0.003250 | 0.003970 | 1.22 | 0.005235 |
| es_erb_l0.01 | 0.003262 | 0.004035 | 1.24 | 0.005279 |
| es_erb_l0.001 | 0.003212 | 0.004939 | 1.54 | **0.005681** |

The ranking reverses end to end. The arm with the best `val_wave` has the worst waveform error,
because it has the largest spread and is credited the most. Read the column as "which predictive
distribution scores best", never as "which arm reconstructs the waveform best" -- the latter is the
right-hand column, and `det_paper` wins it, which is the same fact as its -27.6 dB deficit seen from
the other side: emitting nothing above 6 kHz is the L1-optimal answer to an unpredictable band.

**`spread` is exactly 0 at every logged step for `det` and `det_paper`** and non-zero for all six
samplers. That is how the det/es split in `ARMS` is confirmed from the training record rather than
assumed from the arm name -- and it is the independent evidence that `tau = 0` is right for those
two, which is the silent failure `fast/convert_ckpt.py` exists to fix.

**Trap: `det_paper`'s recorded `val_spec` is not comparable.** It is 27.6 dB down in the high band,
so its own output there sits at the order of bf16's rounding noise, and `log(H + 1e-7)` counts that
noise as signal. Training ran under bf16 autocast; in fp32 the same checkpoint scores **1.944** where
the run recorded **1.603**. See `notes/2026-09-21-eval-results.md`.

---

## 2. Waveform and spectrum metrics (EVAL12)

### SNR -- `snr_db`, cell 4 line 1

    10 * log10(sum(y^2) / max(sum(e^2), 1e-20))

Genuine dB. **Effectively inert on this task and should not be used to rank arms.**

| | value |
|---|---|
| floor (passthrough + **empty** high band) | 18.99 dB |
| `naive` upsample | 19.01 dB |
| every arm | 17.83 - 19.01 dB |
| ceiling (passthrough + true high band) | 41.60 dB |

The floor -- a condition with *no high band at all* -- outscores six of the eight arms. Every dB of
high band a model restores costs it SNR, because the high band's phase is unpredictable. The
`max(..., 1e-20)` floor also means a perfect reconstruction returns a finite ~240 dB, not infinity;
anything in that band is a passthrough or identity bug, not a result.

### LSD -- `lsd_db`, cell 4 line 8

    d  = log10(Sy^2 + 1e-10) - log10(Sh^2 + 1e-10)
    LSD = mean_over_frames( sqrt( mean_over_freq( d^2 ) ) )

**NOT IN dB.** There is no factor of 10. The values are **decades of power**; true dB is 10x. The
name lies. Do not "fix" it -- every number in this repo uses this form, and changing it silently
breaks comparability with every prior run. Lower is better.

Three LSD variants live in the tree and they are not interchangeable:

| function | transform | scale |
|---|---|---|
| `lsd_db` (cell 4) -- used by `e4_eval` and `e5_visqol` | log10(\|X\|^2) | 1x |
| `lsd_standard` (`audit/lisa_paper_protocol.py:87`) | log10(\|X\|^2) | 1x |
| `lsd_lisa` (`audit/lisa_paper_protocol.py:81`) -- LISA's released `utils.py` | log10(\|X\|^4) | **2x** |

`lsd_lisa` squares an already-squared spectrogram, RMSes over time-within-bin rather than
frequency-within-frame, adds 1e-8 inside the sqrt, and caps at 10.0. LISA's published LSD numbers
carry all four differences. `det_paper` exists so the comparison can be internal, where the
convention cancels.

Floor 5.606, `naive` 4.776, ceiling 0.105.

### HB-LSD -- `lsd_db(..., k_from=CFG.eval_k_cut)`

The same thing restricted to bins above the input Nyquist (bin 128 of 513, i.e. 6 kHz). This is the
band the model must invent, so it is the stricter number. Floor 6.470, ceiling 0.045.

### deficit / baseband / curve -- `band_energy_ratio`, cell 4 line 23

    rho_b = 10 * log10( (predicted energy in band b) / (true energy in band b) )

Genuine dB, per third-octave band. **0 dB = correct energy, negative = over-smoothed.** `deficit` is
the mean of `rho_b` over bands above `fs_lo/2`; `baseband` is the mean below it; `curve` is the full
per-band vector (the y-axis of `ov3_spectrum_OV50.png`).

This is the most directly interpretable number in the set: it says how much high-band energy is
missing. Floor -52.65 dB, ceiling -0.00 dB.

**Trap:** it is a mean over *frames* as well as bands, so it hides gating. The high band does not
switch on and off with the speech (-11.7 dB on loud frames, +1.0 dB in the gaps, per the 17 Sep
audit), and no current number reports that. `band_energy_ratio` already takes a `frame_mask`; the
gated version is an open TODO.

### hb_coh / hb_kappa / hb_frac / bb_coh -- `coherent()`, `e4_eval.py:31`

Per high-band third-octave, from the complex STFTs:

    coherent fraction  = Re<Y,P> / <Y,Y>     predictable fraction; ~0 means unpredictable
    kappa              = Re<Y,P> / <P,P>     1 = informative output energy, ~0 = hallucinated

`hb_frac` is the high band's share of the truth's total energy. `bb_coh` is the same coherence on
the baseband and is an **alignment check** -- it should be ~1, and if it is not, the two signals are
time-shifted and every other number is suspect.

Measured `hb_coh` is 0.005 - 0.012 for every arm. **The high band is essentially incoherent with the
truth for all of them**, which is why SNR cannot separate them and why calibration, not accuracy, is
the interesting axis.

---

## 3. Probabilistic metrics (EVAL12, stochastic arms only)

All computed on `lm_hb(w)` -- the log-magnitude STFT above the cut, `e4_eval.py:20` -- so they score
the **high-band log-magnitude**, not the waveform.

### CRPS -- `crps_ensemble`, cell 4 line 40

    term1 = mean_m |x_m - y|
    diff  = mean over ALL M^2 pairs |x_i - x_j|
    CRPS  = mean( term1 - 0.5 * diff )

Lower is better. Strictly proper for the predictive distribution.

**Trap 1 -- the docstring is wrong.** It says "Fair estimator". It is not: `diff` averages over M^2
pairs *including the diagonal*, where the fair estimator divides by M(M-1). The spread term is
under-credited by (M-1)/M = 15/16, so CRPS is biased slightly **upward** for the samplers. It is the
same estimator the OV2 tables used, so it is comparable within this repo; it penalises the samplers,
not the baselines, so the sampler advantage it reports is conservative.

**Trap 2 -- the deterministic arms.** With M = 1, `diff` is 0 and CRPS collapses to the mean absolute
error of a point forecast. `det` and `det_paper` are the reference line on this column, not
competitors: a point forecast is a degenerate predictive distribution and must lose a proper
probabilistic score. Quoting "CRPS -41% vs det" as the headline result is partly definitional.

### Sliced CRPS -- `e4_eval.py:111`

`crps_ensemble(ens @ THETA, truth @ THETA)` with 32 fixed random directions (`stream("ov2/theta")`,
the same directions as OV2 so the numbers compare). Scores the **joint** law of a frame rather than
the per-bin marginals.

### PIT histogram and `pit_end` -- `pit_ranks`, cell 4 line 48

    rank of the truth among the M draws, per time-frequency bin, in 0..M

Flat histogram <=> calibrated. `pit_end` is the mass in the two end bins.

| | value |
|---|---|
| per-bin ideal | 1/(M+1) = 0.0588 |
| `pit_end` ideal | 2/(M+1) = **0.1176** |
| OV3_fast (7.6 ep) | 0.211 = 1.79x |
| OV50 best (`es_dec_erb_l0.1`) | 0.182 = 1.55x |

**Trap 1 -- `pit_end` hides the asymmetry, and the asymmetry is the finding.** Split the two ends:
the bottom bin is at its ideal value within 15% for every arm, and *all* the excess is in the top bin
(1.94x to 5.64x). That is a one-sided energy bias, not a narrow ensemble. It matters because the
remedy differs: widening the ensemble to drive `pit_end` to 0.118 would push the bottom bin past
ideal while the top stayed high.

**Trap 2 -- error bars.** Ranks are per time-frequency bin, so N is huge but heavily correlated. A
binomial error bar on the bin count is far too tight. Do not call a small deviation significant on
sample size alone.

### spread-skill -- `spread_skill`, cell 4 line 53

Bins the bins by ensemble spread (8 bins) and plots mean spread against RMSE of the ensemble mean.
**Unit slope is the target**; above the diagonal means RMSE > spread, i.e. under-dispersed.

Measured: near the diagonal at high spread (1.17 vs 1.19) and off it by 2x at low spread (0.41 vs
0.83). The ensembles are calibrated where they are unsure and overconfident where they are confident.

### SNR gap -- `e4_eval.py:83, 134`

    snr_gap             = SNR(mean of M draws) - SNR(one draw)
    snr_gap_calibrated  = 10 * log10( 2 / (1 + 1/M) ) = 2.75 dB at M = 16

**The cleanest single calibration number in the set**, because the target is analytic. Observed
0.69 - 1.04 dB, i.e. every arm carries **25% to 38%** of the spread a calibrated 16-member ensemble
would have.

**Trap:** it measures spread in the *waveform* domain while PIT measures it in the high-band
log-magnitude domain, and they disagree in ordering. `es_erb_l0.001` has the *largest* SNR gap (1.04)
and the *worst* PIT (0.398). More waveform spread, worse log-magnitude calibration.

### corr_err -- `e4_eval.py:122`

    || corrcoef(truth groups) - corrcoef(predicted groups) ||_F

over 16 frequency groups of the high band. Lower is better. Measures whether the bands co-vary the
way the truth's do -- the thing a per-bin score cannot see. This is where the ERB term shows its
largest single effect (`es_marg` 0.900 -> `es_erb_l0.01` 0.705, -22%).

**Trap: it is computed from draw 0 only** (`pred_frames.append(grouped(ens[0]))`), not from the
ensemble, so it is a single-draw statistic even for the samplers.

---

## 4. Perceptual metrics (`e5_visqol.py`)

Requires a separate install: `pip install "visqol-python[lattice]" pesq torchmetrics torchaudio`.
The `[lattice]` extra matters -- without `ai_edge_litert` the speech MOS silently falls back from the
lattice mapping to polynomial and stops matching OV2. The cell prints which is live and records it in
the JSON. **Check that line before quoting a MOS.**

**The decisive fact is each judge's dynamic range on this task:**

| judge | floor | ceiling | range | `naive` (no high band at all) sits at |
|---|---|---|---|---|
| **ViSQOL audio48k** | 1.567 | 4.725 | **3.158** | **0.3%** |
| ViSQOL speech16k | 3.947 | 4.508 | 0.561 | 4.4% |
| **PESQ wb** | 4.285 | 4.615 | **0.330** | **27.9%** |
| NSIM audio | 0.724 | 0.994 | 0.270 | 1.0% |
| NSIM speech | 0.961 | 0.995 | 0.034 | 7.3% |

### ViSQOL audio mode, 48 kHz -- **lead with this one**

32 ERB bands to 24 kHz, so it sees 100% of the band the model must invent. Report as **share of
range**, `(x - 1.567) / 3.158`, which is the only scale on which it is readable. Best condition in
OV50: `es_dec_erb_l0.1 logmean16 + passthrough`, 3.021 = 46.0%.

### ViSQOL speech mode, 16 kHz -- **report for continuity, do not lead with it**

Resamples to 16 kHz, so its Nyquist is 8 kHz while the band in question is 6-24 kHz: it sees about
11% of the problem. Its entire range for this task is 0.561 MOS.

**It is mostly measuring baseband leakage.** Turning on passthrough -- which touches *only* the
baseband -- moves it by **+0.459**, which is 82% of everything it can express, while moving audio
mode by +0.009 (0.3% of range).

### NSIM (both modes)

The **mapping-free** similarity score underneath MOS-LQO. Safer than MOS, which saturates: the
`ceiling` condition scores 4.725 audio / 4.508 speech, not 5.0. Read the ceiling row for what
"perfect" scores, never 5.

### PESQ wideband -- **retire it for this task**

Ranks `det_paper` 2nd and `naive` 3rd of 43 conditions, above every sampler, while both sit within
0.15 of the empty-band floor on audio ViSQOL. It cannot see the band, so it scores whatever leaves
the baseband cleanest -- and doing nothing leaves it perfectly clean.

---

## 5. Two cross-cutting dimensions

Every metric above is computed against a **condition**, and a condition is (arm x readout x
passthrough). Getting these wrong is what makes two tables disagree.

### Readout -- one model, three answers

| readout | what it is | wins |
|---|---|---|
| **one draw** (tau=1, seed 0) | an actual sample | the only one CRPS/PIT can use; best **deficit** (-6.26 dB) |
| **mean16** | mean of 16 waveforms | best **SNR** and **PESQ**; worst LSD |
| **logmean16** | per-bin mean of log\|STFT\| over 16 draws, phase of draw 0 | best **LSD** and **audio ViSQOL** |

Averaged over the six samplers, passthrough on: LSD 0.956 / 1.036 / 0.875; audio ViSQOL 2.696 /
2.683 / 2.849; PESQ 4.029 / 4.318 / 4.301; SNR 18.24 / 18.85 / 18.76.

**A deterministic arm has only one readout.** Comparing `det` against a sampler's *single draw* on
LSD compares `det`'s only output against the sampler's worst readout for that metric, and is not a
fair comparison. Compare each model at its best readout for the metric in question.

### Passthrough

Low band of the naive upsample + high band of the model, brick-walled at `fs_lo/2`
(`e4_eval.py:71`). This is how the system would ship -- the low band is *given*, so there is no
reason to regenerate it. Mean effect: LSD -0.019, audio ViSQOL +0.009, speech ViSQOL +0.459.

**The 16 Sep note reports every condition WITH passthrough. The `LSD` column of the OV50 EVAL12
table is RAW.** Comparing the two straight makes OV50 look worse than it is. It already did once.

### Anchors

`floor` (passthrough + empty high band), `ceiling` (passthrough + true high band) and `naive` are
model-independent: they depend only on the twelve utterances and the protocol. **Use them to check
that two runs are evaluating the same thing** -- OV50 and OV3_fast agree on all of them to three
decimals, which is what makes the cross-run comparison legitimate.

---

## 6. Not in this repo

**FAD.** No Fréchet Audio Distance code anywhere. It needs an embedding network -- VGGish, PANNs and
CLAP disagree with each other -- and it is unstable below a few hundred clips. EVAL12 is twelve. That
is a study-design decision before it is a coding task.

**Gated deficit.** See the trap under `band_energy_ratio`. Open TODO from 17 Sep.
