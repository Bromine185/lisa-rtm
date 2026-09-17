# Research note — the score's geometry, noise where the fine structure lives, and the right readout for each judge

**Date:** 2026-09-16 (run `OV3_es`, Colab A100-80GB; predictions written before the run, results filled in after)
**Question:** the 8 Sep sampler wins the proper score and loses the perceptual judges. Is that a fact about samplers, or about *this* sampler's distance, its noise pathway, and the statistic we read out of it?
**Rule of the day (user, 16 Sep):** a winner must improve CRPS and calibration *and* LSD + audio-mode ViSQOL together. Stochastic sampling is the object of study.

---

## 0. Three moves, one model class

The energy score is strictly proper for the law of $\phi(y)$ whenever $d(a,b) = \|\phi(a) - \phi(b)\|$. Nothing in that theorem says $\phi$ has to be the log-magnitude STFT that LSD happens to use; and nothing says the noise has to enter at the encoder. The 8 Sep sampler made both choices by default. Tonight varies each one at a time, on identical batches and one init.

1. **The geometry of $d$** (what the score is proper for, and what the judges measure). `es_marg` uses $\ell_1$ on log-magnitude bins: proper for per-bin marginals only. Two alternatives: the Euclidean norm over the whole log-magnitude spectrogram (`es_ged`, Gritsenko et al. 2020's choice, strictly proper for the joint law), and $\ell_1$ on log ERB-band energies (`es_erb`: 32 gammatone-like bands to 24 kHz, the neurogram ViSQOL's NSIM is computed on). Plus the weight of the spectral term inside $d$ raised from $10^{-2}$ to $10^{-1}$ (`es_marg_l0.1`).
2. **Where the noise enters.** LISAS feeds 8 Gaussian channels at the 12 kHz input; every 48 kHz output sample inside one input interval is then a deterministic function of the same three latents. The band above 6 kHz has period shorter than four output samples, so its randomness has to be smuggled through the latents. `LISASD` adds 4 Gaussian channels per *output* sample at the decoder input (+~600 weights): noise at the rate of the fine structure it has to generate.
3. **The readout.** LSD is squared error in log-power, so its minimiser is the conditional mean of the *log*-magnitude. The waveform ensemble mean estimates the mean waveform (its log-magnitude is far below), and a single draw estimates nothing. The sampler can produce the LSD-optimal object directly: per-bin mean of $\log|Y|$ across $M$ draws, phase of draw 0, resynthesised (`logmean16`). One model, three readouts: draw for CRPS and listening, waveform mean for SNR, log-magnitude mean for LSD.

## 1. Arms (identical batches, one init per class, batch 32 × 1 s, $\lambda$ inside $d$)

The literature synthesis (`notes/lit/2026-09-16-synthesis-experiment-menu.md`) sharpened the plan before launch: the deficit is a bias in an *aggregate* (band energy) that a per-bin proper score never constrains, and the waveform term dominates $d$ in a band that is 98 % incoherent with the input. So the spectral weight is raised ten-fold in every new arm, one arm restricts the waveform term to the low band (the sampler's `det_split`), and the ERB term is kept as an *added* aggregate-proper term on log band energies rather than a replacement.

| arm | class | $d$ | $\lambda$ | tests |
|---|---|---|---|---|
| `det` | LISAS, $\tau = 0$ | paper loss | $10^{-2}$ | reference |
| `es_marg` | LISAS | $d_w + \lambda d_m$ | $10^{-2}$ | reference sampler (8 Sep recipe) |
| `es_marg_l0.1` | LISAS | $d_w + \lambda d_m$ | $10^{-1}$ | spectral geometry dominant |
| `es_split_l0.1` | LISAS | $d_w(\Pi_{\mathrm{lo}}\cdot) + \lambda d_m$ | $10^{-1}$ | no waveform term above 6 kHz |
| `es_erb_l0.1` | LISAS | $d_w + \lambda (d_m + d_{\mathrm{ERB}})/2$ | $10^{-1}$ | plus the aggregate-proper term on log ERB-band energies |
| `es_dec_l0.1` | LISASD | $d_w + \lambda d_m$ | $10^{-1}$ | noise at the output rate |
| `es_dec_erb_l0.1` | LISASD | $d_w + \lambda (d_m + d_{\mathrm{ERB}})/2$ | $10^{-1}$ | both |

$d_{\mathrm{ERB}}(a,b) = \frac13\sum_s \mathrm{mean}_{b,t}\,\lvert \log\sqrt{E_{b,t}(a)} - \log\sqrt{E_{b,t}(b)}\rvert$ with $E_{b,t} = \sum_k W_{bk}\lvert Y_{t,k}\rvert^2$ over 32 triangular bands equally spaced on the ERB-rate scale from 50 Hz to 24 kHz; inside the two-draw estimator this is a sum of CRPS terms on each band's log energy, which is exactly the aggregate a bin-marginal score leaves free. Dropped from the first draft: `det_l0.1` (the 4 Sep $\lambda = 10^{-1}$ checkpoint already gives that ceiling) and `es_ged` (lowest expected payoff).

Validation loss: each arm's own objective on 8 fixed held-out batches (speakers p236–p237, utterances disjoint from the evaluation set), every 500 steps, fixed noise seeds, no anchor jitter. Training loss: EMA 0.98 of the per-step loss.

## 2. Predictions, written before the run

- **P1 (weight).** `es_marg_l0.1` beats `es_marg` on LSD and audio ViSQOL of a single draw, keeps a flat PIT and equal or lower CRPS. Refuted if CRPS rises by more than 5 % or the PIT end bins exceed 0.15.
- **P2 (low-band waveform term).** `es_split_l0.1` has a smaller single-draw energy deficit than `es_marg_l0.1` at equal or better CRPS, because no term asks for zero above 6 kHz. Refuted if its deficit is not smaller.
- **P3 (judge's geometry).** `es_erb_l0.1` gains at least 0.10 audio-mode ViSQOL over `es_marg_l0.1` at equal CRPS, and its single-draw high-band energy sits within 2 dB of the truth on the Hub set; LSD moves less than ViSQOL does.
- **P4 (decoder noise).** `es_dec_l0.1` closes the in-distribution energy gap: single-draw Hub deficit from −7.9 dB to within ±2 dB, and the SNR gap between mean-of-16 and one draw within 0.5 dB of the calibrated 2.75 dB. Refuted if the deficit stays below −5 dB.
- **P5 (combination).** `es_dec_erb_l0.1` is the only arm that improves on `es_marg` in both families at once.
- **P6 (deterministic ceiling).** No deterministic arm can win overall: `det`'s CRPS stays at the MAE of a point forecast, about 1.0, whatever its LSD.
- **P7 (readout).** `logmean16` of any calibrated sampler beats that sampler's single draw and the deterministic `det` on LSD, because it estimates the LSD minimiser directly. Refuted if `logmean16` LSD is not below both.

## 3. Results

Run `OV3_fast`, finished 2026-09-17: 7 arms × 16,000 steps, batch 64 × 1 s, 7.6 epochs of the 37.3 h
corpus, A100-80GB, 891→966 ms/step. Evaluation on EVAL12 (12 held-out Hub utterances, p236–p238),
M = 16 draws. Every condition is reported **with the baseband passed through**, which is how it would
ship; the raw numbers are in `lisa_rtm_cache/results/visqol_OV3_fast.json`.

### 3.1 The scoreboard, each arm at its best readout

| arm | CRPS | LSD | ViSQOL audio | share of range | deficit, 1 draw | SNR | readout |
|---|---|---|---|---|---|---|---|
| `det` | 1.318 | 1.198 | 2.483 | 29.0 % | −14.39 dB | 18.90 | one draw (τ=0) |
| `es_marg` | 0.697 | 0.946 | 2.760 | 37.8 % | −12.37 | 18.81 | one draw |
| `es_marg_l0.1` | 0.805 | 1.004 | 2.701 | 35.9 % | −13.00 | 18.75 | one draw |
| `es_split_l0.1` | 0.913 | 1.045 | 2.558 | 31.4 % | −15.18 | 18.85 | one draw |
| **`es_erb_l0.1`** | **0.612** | **0.871** | **2.773** | **38.2 %** | −8.08 | 18.76 | **logmean16** |
| `es_dec_l0.1` | 0.702 | 0.965 | 2.738 | 37.1 % | −10.79 | 18.63 | one draw |
| `es_dec_erb_l0.1` | 0.684 | 0.933 | 2.646 | 34.2 % | **−6.97** | 18.76 | logmean16 |
| naive | — | 4.776 | 1.576 | 0.3 % | −52.65 | 19.01 | — |
| floor: passthrough + empty HB | — | 5.606 | 1.567 | 0 % | −52.65 | 18.99 | — |
| ceiling: passthrough + true HB | — | 0.105 | 4.725 | 100 % | −0.00 | 41.60 | — |

"Share of range" is where the arm sits between an empty high band and the true one on audio-mode
ViSQOL — the only scale on which that judge is readable.

**`es_erb_l0.1` with the `logmean16` readout is the first arm in this repo to take both families at
once.** Against `det`: CRPS −53.6 %, LSD −27.3 %, audio ViSQOL +0.290, deficit +6.31 dB, for
**−0.14 dB of SNR**. That was the rule set on 16 Sep, and it is met.

### 3.2 The readout interaction, which is the real finding

The same six samplers, LSD and audio ViSQOL, readout by readout, all with passthrough:

| arm | draw LSD | logmean16 LSD | draw ViSQOL | logmean16 ViSQOL | Δ ViSQOL |
|---|---|---|---|---|---|
| `es_marg` | 0.946 | 0.920 | 2.760 | 2.684 | **−0.076** |
| `es_marg_l0.1` | 1.004 | 0.998 | 2.701 | 2.579 | **−0.122** |
| `es_split_l0.1` | 1.045 | 1.038 | 2.558 | 2.472 | **−0.086** |
| `es_erb_l0.1` | 0.972 | **0.871** | 2.473 | **2.773** | **+0.300** |
| `es_dec_l0.1` | 0.965 | 0.935 | 2.738 | 2.704 | **−0.034** |
| `es_dec_erb_l0.1` | 0.999 | 0.933 | 2.530 | 2.646 | **+0.116** |

`logmean16` improves LSD for **all six** arms — P7, confirmed without exception. But it improves
audio ViSQOL for **only the two ERB arms**, and makes the other four worse. The ERB term and the
log-magnitude ensemble readout are not two independent wins; they are one mechanism. Score the band
energies and the draws agree about band energy, so averaging their log-magnitudes sharpens the
spectrum. Leave the band energies free and the draws disagree, so the average smears — better on
LSD, which is a per-bin squared error, worse on a neurogram that reads bands.

The plain waveform mean (`mean16`) is worse than either on both judges for every arm — LSD 1.099 to
1.215, ViSQOL 2.358 to 2.501 — while being the SNR-optimal readout. One model, three readouts, three
different answers, exactly as §0.3 argued.

### 3.3 Calibration

| arm | PIT end bins (ideal 0.118) | SNR gap (calibrated 2.75) | κ | coherent fraction |
|---|---|---|---|---|
| `det` | — | — | +0.107 | 0.73 % |
| `es_marg` | 0.412 | 0.32 | +0.053 | 0.69 % |
| `es_marg_l0.1` | 0.538 | 0.24 | +0.026 | 0.45 % |
| `es_split_l0.1` | 0.609 | 0.05 | +0.066 | 0.86 % |
| `es_erb_l0.1` | **0.211** | 0.65 | **−0.009** | 0.00 % |
| `es_dec_l0.1` | 0.375 | 0.36 | +0.011 | 0.43 % |
| `es_dec_erb_l0.1` | 0.274 | 0.75 | +0.001 | −0.09 % |

The ERB arms are the best calibrated by a distance, and their κ is zero to three decimals: their high
band is honestly incoherent with the input, not a hallucinated coherent component. `det`'s κ of
+0.107 is the disease named in the 8 Sep note, still there. Every arm's SNR gap is far below the
calibrated 2.75, so no arm is fully dispersed — the draws are still too alike.

### 3.4 Gating, which no metric here was reporting

From `audit/band_gating.py` over 12 utterances (p236–p238, p360, p361, p374 — a different set from
EVAL12, so the levels differ). Deficit in dB, split by frame loudness:

| arm | draw | τ = 0 | loud | mid | quiet | swing (loud − quiet) |
|---|---|---|---|---|---|---|
| `det` | −10.45 | −10.45 | −11.30 | −6.75 | −3.77 | **−7.53** |
| `es_marg` | −7.77 | −10.44 | −8.41 | −4.39 | −1.56 | −6.85 |
| `es_marg_l0.1` | −7.13 | −9.98 | −7.67 | −4.59 | −3.84 | −3.83 |
| `es_split_l0.1` | −9.41 | −9.28 | −9.93 | −7.90 | −3.93 | −6.01 |
| `es_erb_l0.1` | −3.54 | −16.29 | −4.45 | +0.38 | +0.16 | −4.60 |
| `es_dec_l0.1` | −5.49 | −13.25 | −6.15 | −2.18 | −2.30 | −3.85 |
| `es_dec_erb_l0.1` | −3.44 | −15.33 | −4.53 | +0.83 | −3.20 | **−1.33** |

Two things. The **τ = 0 column is the mechanism**: the ERB arms are the emptiest of all with the noise
off (−16.29, −15.33) and the fullest with it on (−3.54, −3.44). A 12.7 dB swing between noise-off and
noise-on is the noise channel doing the work it was built for; `es_split_l0.1`'s 0.13 dB swing is a
noise channel that is not connected to anything.

And `det` has the **worst** gating of all seven (−7.53 dB), not the best. The single-utterance
measurement in the 17 Sep note suggested the sampler's gating was the defect; across twelve
utterances and all seven arms, every sampler gates better than the deterministic model, and
`es_dec_erb_l0.1` gates best. That earlier reading is corrected here.

### 3.5 Predictions

| | verdict | |
|---|---|---|
| **P1** weight | **refuted** | `es_marg_l0.1` is worse than `es_marg` on LSD (1.004 vs 0.946), audio ViSQOL (2.701 vs 2.760) and CRPS (+15.5 %, the 5 % refutation threshold). PIT end bins 0.538, over the 0.15 bar. Every clause failed. |
| **P2** low-band waveform term | **refuted** | `es_split_l0.1` deficit −15.18 against `es_marg_l0.1`'s −13.00 — larger, not smaller — at worse CRPS (0.913 vs 0.805). Removing the waveform term above 6 kHz removed the only thing pushing on that band. |
| **P3** judge's geometry | **partly confirmed** | On the matched readout `es_erb_l0.1` gains **+0.194** audio ViSQOL over `es_marg_l0.1` (2.773 vs 2.579), clearing the 0.10 bar, at better CRPS (0.612 vs 0.805). The clause that its high-band energy would sit within 2 dB of the truth fails: −8.08 dB. |
| **P4** decoder noise | **refuted** | `es_dec_l0.1` deficit −10.79 dB, below the −5 dB refutation line. SNR gap 0.36 against a calibrated 2.75. |
| **P5** combination | **refuted as stated** | `es_dec_erb_l0.1` is not the only arm to take both families — it is not even the best. `es_erb_l0.1`, the cheaper model with no decoder noise, beats `es_marg` on CRPS, LSD and audio ViSQOL simultaneously and beats `es_dec_erb_l0.1` on all three. |
| **P6** deterministic ceiling | **confirmed** | `det` CRPS 1.318, against 0.612 for the best sampler. No readout and no λ moves a point forecast's CRPS. |
| **P7** readout | **confirmed, all six arms** | `logmean16` beats that arm's own draw and beats `det` on LSD for every stochastic arm: 0.871/0.920/0.933/0.935/0.998/1.038 against `det`'s 1.198. |

Two confirmed, one partial, four refuted.

## 4. What it means

**The geometry of the score was the lever; the noise pathway was not.** Three of the four refuted
predictions were about where the noise enters or how hard the waveform term pushes. Raising λ hurt.
Splitting the waveform term hurt. Decoder-side noise helped a little on its own (`es_dec_l0.1`
deficit −10.79 against `es_marg_l0.1`'s −13.00) and nothing at all on top of the ERB term
(`es_dec_erb_l0.1` is worse than `es_erb_l0.1` on CRPS, LSD and ViSQOL, for 576 extra weights). What
moved every judge at once was changing **what the distance measures**: adding a proper score on log
ERB-band energies, the space the perceptual judge's neurogram lives in.

The mechanism is now legible. A per-bin log-magnitude score constrains each bin's marginal and leaves
the *aggregate* band energy free, so the network can satisfy it without ever using its noise channel —
`es_marg`'s noise-off/noise-on swing is 2.7 dB. Pin the band energies with `d_ERB` and the only way to
match them is to actually draw: `es_erb_l0.1`'s swing is 12.7 dB. That is also why it is the best
calibrated arm (PIT end bins 0.211 against 0.412–0.609) and why its κ is −0.009: it is not
hallucinating a coherent high band, it is sampling an incoherent one, which is what the truth is.

**And the readout is not a free choice.** P7 held for all six arms on LSD, which was expected —
`logmean16` estimates LSD's minimiser directly. What was not predicted is that it improves audio
ViSQOL *only* for the two arms trained with the ERB term, and degrades it for the other four. The
readout and the objective have to agree about what the band structure is. Averaging log-magnitudes
across draws that disagree about band energy smears the bands; across draws that agree, it sharpens
them. So "train under any proper score, then read out whatever the judge wants" is false. The right
statement is narrower: **train under a score proper for the aggregate the judge reads, then take that
aggregate's ensemble mean.**

The cost of all of it is 0.14 dB of SNR, on a metric whose entire achievable range on this task is
about one decibel and which is maximised by doing nothing
(`notes/2026-09-17-snr-ceiling-gating-and-scale.md` §1). That is the trade, and it is the right side
of it.

**What this does not settle.** No arm is properly dispersed — every SNR gap is far below the
calibrated 2.75, so the draws are still too alike and CRPS has room. The deficit is still −8 dB at
best, against a ±2 dB target nobody reached. 16,000 steps is 7.6 epochs against the paper's 50, and
the validation losses were still falling. And the gated deficit says every arm, `det` worst of all,
puts its high band on too flat in time. The next arm is the one the 17 Sep note pre-registered: weight
the ERB term by frame energy, or score the high-band envelope against the low-band envelope. Score the
conditional, not the marginal — one level up from the move that worked here.

## 5. Prior art this builds on

Energy score and its properness: Gneiting & Raftery 2007. Two-draw feed-forward training: DISCO Nets (Bouchacourt et al. 2016), Gritsenko et al. 2020 (spectral energy distance, the $\ell_2$ log-magnitude geometry), Engression (Shen & Meinshausen 2025). Noise at every layer: EnScale (Schillinger et al. 2025). The factor of two between a draw and the mean: Blau & Michaeli 2018. The readout argument is the perception–distortion reading of the ensemble: the optimal estimator under a distortion is the conditional mean *in that distortion's coordinates*, and the sampler gives it for any coordinates you name.
