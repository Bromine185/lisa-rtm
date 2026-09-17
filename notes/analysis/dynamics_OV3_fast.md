# Training dynamics, seven arms — `OV3_fast`, 16 000 steps

**Date:** 2026-09-17. **Run:** `OV3_fast`, 16 000 steps, batch 64 × 1 s at 48 kHz, 2 099 steps/epoch = **7.62 epochs**. **Optimiser:** lr 1e-3, MultiStep γ = 0.5 at steps **3200, 6400, 8000, 9600, 11200, 12800** (0.2 / 0.4 / 0.5 / 0.6 / 0.7 / 0.8 of the run); final lr 1.5625e-5 = lr0/64. **Val:** 8 fixed batches × 64 × 1 s, every 500 steps. **Train EMA:** α = 0.02 on the per-step training loss, sampled at the val points. **Dev probe:** `p236_002`, one utterance, one draw at τ = 1 (`snr1`, `def1`) and the noiseless pass at τ = 0 (`snr0`, `def0`), every 500 steps.

**Data:** `lisa_rtm_cache/results/ov3_history_OV3_fast.json` → `audit/dynamics_ov3.py` → `lisa_rtm_cache/results/dynamics_OV3_fast.json` and the four figures in `figs/`.

## 0. What is comparable and what is not

**`val_loss`, `val_wave`, `val_spec` and `train_loss_ema` are each arm's OWN objective.** For `det` they are plain distances (L1 waveform; spectral convergence + log-magnitude). For the samplers they are the two-draw energy-score form ½[d(y,y₁) + d(y,y₂)] − ½ d(y₁,y₂). `es_split_l0.1`'s waveform term is the low band only. The ERB arms' spectral term is ½(d_logmag + d_erb), the others' is d_logmag alone, and λ is 0.01 for `det` and `es_marg`, 0.1 for the rest. **Never compare these columns across arms.** They are read along time within one arm and nothing else.

`snr0`, `snr1`, `def0`, `def1` and `spread` **are** comparable across arms. `spread` = d_wave(y₁, y₂) full band on the training batch, the same estimator for every stochastic arm.

**The probe has a ceiling.** `p236_002`'s own naive sinc upsample scores **14.89 dB** SNR. Every arm sitting at 14.3–14.8 dB is at that utterance's ceiling, not failing. SNR is maximised by adding nothing, so it ranks nothing here; it is in the tables to show no arm broke the baseband.

**Dev-probe noise floor** (sd of the step-to-step difference over the last 8 probes, dB): def0 0.04–0.12, def1 0.08–0.16, snr0 0.0006–0.0024. Treat any deficit move under ~0.2 dB as nothing.

## 1. Final values, step 16 000

Last val point and last dev point. Left block is each arm's own objective — read down a row, never down a column. Right block is comparable.

| arm | val_loss | val_wave | val_spec | train EMA | val/train | snr0 | snr1 | def0 | def1 | spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| det | 0.016243 | 0.005421 | 1.0821 | 0.013780 | 1.179 | 14.85 | — | -19.09 | — | — |
| es_marg | 0.010059 | 0.004636 | 0.5423 | 0.008327 | 1.208 | 14.84 | 14.76 | -17.58 | -15.95 | 0.00285 |
| es_marg_l0.1 | 0.064285 | 0.005458 | 0.5883 | 0.051391 | 1.251 | 14.52 | 14.44 | -16.97 | -14.56 | 0.00259 |
| es_split_l0.1 | 0.068798 | 0.000849 | 0.6795 | 0.047877 | 1.437 | 14.70 | 14.70 | -16.93 | -16.99 | 0.00152 |
| es_erb_l0.1 | 0.040096 | 0.006184 | 0.3391 | 0.037399 | 1.072 | 14.31 | 13.85 | -21.84 | -11.12 | 0.00251 |
| es_dec_l0.1 | 0.058386 | 0.005181 | 0.5321 | 0.050368 | 1.159 | 14.64 | 14.48 | -19.98 | -13.23 | 0.00247 |
| es_dec_erb_l0.1 | 0.040737 | 0.005402 | 0.3534 | 0.036343 | 1.121 | 14.52 | 14.24 | -22.77 | -10.72 | 0.00235 |

SNR and deficit in dB. `snr_naive` = 14.89 dB for every row (same utterance). `spread` is the mean over the last 8 logs (200 steps). `es_split_l0.1`'s `val_wave` is a low-band distance and is small for that reason alone.

Best val point: step 16 000 for six arms. `es_split_l0.1` bottomed at step 10 500 with 0.068686 against 0.068798 at the end — 0.16 % worse, inside its own wobble.

## 2. When `def1` first crossed −12 dB

| arm | first step with def1 > −12 dB | def1 max (step) | def0 max (step) | def0 ever > −12 dB |
|---|---:|---:|---:|---|
| det | n/a (no noise) | — | -19.05 (15 500) | never |
| es_marg | **never** | -15.93 (15 500) | -16.46 (10 500) | never |
| es_marg_l0.1 | **never** | -14.49 (15 500) | -16.01 (5 000) | never |
| es_split_l0.1 | **never** | -16.86 (13 000) | -16.79 (13 000) | never |
| es_erb_l0.1 | **3 000** | -10.34 (3 000) | -18.07 (3 000) | never |
| es_dec_l0.1 | **never** | -13.14 (13 000) | -15.79 (5 500) | never |
| es_dec_erb_l0.1 | **3 000** | -9.74 (7 500) | -19.85 (7 500) | never |

Two arms crossed, both ERB, both at step 3 000 (1.43 epochs), and neither fell back below −12 dB at any later probe. No noiseless pass ever crossed, in any arm.

## 3. The `def1 − def0` gap

| arm | 4 000 | 8 000 | 12 000 | 16 000 | Δ 4k→16k |
|---|---:|---:|---:|---:|---:|
| es_marg | -0.47 | -0.08 | +0.63 | +1.63 | +2.10 |
| es_marg_l0.1 | -0.05 | +1.24 | +2.16 | +2.41 | +2.46 |
| es_split_l0.1 | +0.13 | -0.06 | -0.07 | -0.06 | -0.19 |
| es_erb_l0.1 | +8.59 | +9.24 | +10.28 | +10.72 | +2.13 |
| es_dec_l0.1 | +0.13 | +5.61 | +6.36 | +6.75 | +6.62 |
| es_dec_erb_l0.1 | +9.51 | +10.75 | +11.56 | +12.05 | +2.54 |

The two halves, since the gap can widen from either end:

| arm | def0 4k | def0 16k | Δ | def1 4k | def1 16k | Δ |
|---|---:|---:|---:|---:|---:|---:|
| det | -20.71 | -19.09 | +1.63 | — | — | — |
| es_marg | -18.94 | -17.58 | +1.36 | -19.41 | -15.95 | +3.46 |
| es_marg_l0.1 | -17.08 | -16.97 | +0.11 | -17.13 | -14.56 | +2.57 |
| es_split_l0.1 | -18.65 | -16.93 | +1.72 | -18.52 | -16.99 | +1.54 |
| es_erb_l0.1 | -20.34 | -21.84 | **-1.50** | -11.75 | -11.12 | +0.62 |
| es_dec_l0.1 | -16.76 | -19.98 | **-3.22** | -16.63 | -13.23 | +3.40 |
| es_dec_erb_l0.1 | -20.74 | -22.77 | **-2.03** | -11.23 | -10.72 | +0.50 |

Three arms emptied their noiseless output as training went on. Those are the three whose gap is largest.

## 4. Slope of `val_loss` over the last 4 000 steps

Linear fit on steps 12 000–16 000, expressed as % of the fitted value at 12 000, per 1 000 steps. Negative = still falling.

| arm | val_loss | train EMA | val_wave | val_spec | val_loss 8k–12k | val_loss 4k–8k |
|---|---:|---:|---:|---:|---:|---:|
| det | **-0.34** | -0.28 | +0.02 | -0.51 | -1.41 | -1.89 |
| es_marg | **-0.52** | -0.35 | -0.32 | -0.68 | -2.07 | -1.06 |
| es_marg_l0.1 | **-0.28** | -0.25 | -0.43 | -0.26 | -0.96 | -3.36 |
| es_split_l0.1 | **-0.05** | -0.17 | -1.59 | -0.03 | -0.21 | -1.85 |
| es_erb_l0.1 | **-0.24** | -0.29 | -0.32 | -0.23 | -1.12 | -3.10 |
| es_dec_l0.1 | **-0.24** | -0.22 | -0.23 | -0.24 | -1.26 | -4.16 |
| es_dec_erb_l0.1 | **-0.22** | -0.34 | -0.20 | -0.22 | -1.31 | -1.59 |

All seven are still learning, all seven have almost stopped. The rate fell between the 4k–8k window and the last 4 000 steps by 2.0× for `es_marg`, 5.6× for `det`, 7–17× for the four λ = 0.1 samplers, and 40× for `es_split_l0.1`. `es_split_l0.1` is the flattest at −0.05 %/1k, and its remaining motion is all in the low-band waveform term (−1.59 %/1k) while its spectral term sits at −0.03. `det` is the mirror image: its waveform term is dead (+0.02 %/1k) and only the spectral term still moves.

## 5. Did any val loss rise in the last quarter?

**No.** Steps 12 000 → 16 000:

| arm | val 12 000 | val 16 000 | change | worst rise above the running min | upticks (of 8) | rose |
|---|---:|---:|---:|---:|---:|---|
| det | 0.016460 | 0.016243 | -1.32 % | 0.11 % | 1 | no |
| es_marg | 0.010285 | 0.010059 | -2.20 % | 0.00 % | 0 | no |
| es_marg_l0.1 | 0.065037 | 0.064285 | -1.16 % | 0.27 % | 1 | no |
| es_split_l0.1 | 0.068843 | 0.068798 | -0.07 % | 0.32 % | 5 | no |
| es_erb_l0.1 | 0.040584 | 0.040096 | -1.20 % | 0.03 % | 2 | no |
| es_dec_l0.1 | 0.059391 | 0.058386 | -1.69 % | 1.10 % | 3 | no |
| es_dec_erb_l0.1 | 0.041155 | 0.040737 | -1.02 % | 0.53 % | 4 | no |

Every arm ended at its lowest val of the quarter. The worst excursion anywhere was `es_dec_l0.1` at 1.10 % above its running minimum — a single 500-step blip, not a trend. `es_split_l0.1` ticked up 5 times out of 8 and still ended down 0.07 %: that arm is flat, not diverging. No arm needs early stopping.

## 6. What the six lr cuts did

`val_change_pct` is the val point after the cut against the val point before it. `prev_window_change_pct` is the same-length window immediately before, i.e. the drift the cut has to beat. Mean over the seven arms:

| step | lr after | mean val change | drift in the window before | mean train-wave change |
|---|---:|---:|---:|---:|
| 3 200 | 5.00e-4 | **-10.76 %** | -5.55 % | -21.4 % |
| 6 400 | 2.50e-4 | **-4.74 %** | +0.73 % | -14.1 % |
| 8 000 | 1.25e-4 | **-2.62 %** | -0.38 % | -1.9 % |
| 9 600 | 6.25e-5 | -0.31 % | -0.56 % | -0.2 % |
| 11 200 | 3.13e-5 | -0.38 % | -0.09 % | -3.8 % |
| 12 800 | 1.56e-5 | -0.51 % | -0.13 % | -6.9 % |

Per arm, `val_change_pct`:

| step | det | es_marg | es_marg_l0.1 | es_split_l0.1 | es_erb_l0.1 | es_dec_l0.1 | es_dec_erb_l0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 200 | -3.34 | -9.73 | -11.60 | -11.67 | -14.01 | -5.70 | **-19.29** |
| 6 400 | -3.76 | -4.12 | **-11.19** | -2.47 | -3.58 | -5.09 | -2.98 |
| 8 000 | -3.66 | -3.49 | +0.21 | -1.35 | -3.35 | -2.05 | -4.64 |
| 9 600 | -0.94 | -0.82 | +0.88 | +0.36 | -1.61 | -1.86 | +1.80 |
| 11 200 | -1.65 | -0.64 | -0.23 | -0.01 | -1.07 | -0.60 | +1.52 |
| 12 800 | -0.48 | -0.39 | -0.44 | -0.24 | -0.16 | -0.73 | -1.12 |

Cut 1 (3 200) is the only one that changed the run. It beat the drift it had to beat in five arms of seven, by 1.9× to 5.4×, and took `es_dec_erb_l0.1` down 19.3 % in 500 steps; `det` (0.75×) and `es_dec_l0.1` (0.54×) were already falling faster before the cut than after it. Cut 2 (6 400) still bought 4.7 % against a +0.7 % mean drift — four of the seven arms were drifting *up* into it and the cut reversed them. Cut 3 (8 000) bought 2.6 %. Cuts 4, 5 and 6 bought 0.3–0.5 % on val while the training-wave EMA fell 0.2 %, 3.8 % and 6.9 %. That divergence is the tell: after step 9 600 the cuts were taking SGD noise out of the batch loss, not finding anything. Three cuts would have done the work of six.

## 7. Figures

- `figs/dyn_loss_grid.png` — train EMA and val loss per arm, seven panels, log y, lr cuts marked. Each panel is its own objective; heights do not compare across panels. Panel titles carry the final val and the last-4k slope of §4.
- `figs/dyn_deficit.png` — high-band deficit on `p236_002`, τ = 1 solid and τ = 0 dashed, with the 0 dB line and the −12 dB threshold of §2. The two ERB solids sit alone in the −10 to −11 dB band from step 3 000 on, and their dashes fall away after 7 500.
- `figs/dyn_snr.png` — dev-probe SNR against the 14.89 dB naive line. Every arm is pinned under it. The spread across arms is 1.0 dB and it means nothing about quality.
- `figs/dyn_spread.png` — two-draw spread d_wave(y₁, y₂) on the training batch, log y, thin raw at every 25 steps, bold 200-step mean. `es_split_l0.1` peels off the pack at about step 5 800.

## 8. Reading

The ERB arms separated at step 2 500 and were gone by 3 000: at 2 500 the worse ERB arm led the best non-ERB sampler by 3.5 dB of `def1`, at 3 000 by 5.1 dB, and no arm ever closed that. Both crossed −12 dB at step 3 000, which is 1.43 epochs, and `es_erb_l0.1` posted its all-time best `def1` of −10.34 dB at that exact probe — the ERB term did its whole job in the first sixth of the run. Decoder noise did not compound with it: against `es_marg_l0.1`, the ERB term alone bought +3.44 dB of `def1`, decoder noise alone bought +1.33 dB, and the two together bought +3.84 dB instead of the +4.77 dB that would add. Stacked on ERB, decoder noise was worth +0.40 dB, under a third of its standalone value, and the 12-utterance gated note puts the same quantity at +0.1 ± 0.2 dB — two ways of giving the model somewhere to put incoherent energy, filling one hole. 16 000 steps was plenty for the ERB arms and short for everyone else: from step 6 500 to 16 000 `es_erb_l0.1` gained 0.32 dB and `es_dec_erb_l0.1` 0.13 dB of `def1`, while `es_marg_l0.1` gained 2.54 dB and `es_dec_l0.1` 2.44 dB and both were still climbing when the lr ran out. Every arm's val loss was still falling at the end, between −0.05 and −0.52 % per 1 000 steps, and none rose in the last quarter, so nothing here is an overfitting story — the val/train ratios of 1.07–1.44 say the same. The run did not converge, it ran out of step size: by 12 800 the lr was lr0/64, and the last three cuts moved val by 0.2–1.1 % while moving the training EMA 3–8 %, which is SGD noise leaving the batch loss rather than learning. A 50-epoch run is 104 950 steps, 6.6× this one, and the honest prediction is that it changes the losers and not the winners: stretch the schedule so the lr stays above 2.5e-4 out to roughly step 40 000 and the non-ERB samplers should pick up 1–3 dB of `def1` and close part of the gap, while the ERB arms move under 1 dB because their `def1` has been flat since step 6 000 and `es_dec_erb_l0.1` has not beaten its step-7 500 peak in 8 500 steps. The ranking would survive; the margin would shrink. The spread trace looks at first like collapse — every arm ends far below its early peak — but that peak (7–17× the final value, hit between steps 225 and 400) is an untrained network thrashing, and the right test is `def1 − def0`, which grew in five arms of six and reached 12.05 dB for `es_dec_erb_l0.1`. Most of that growth came from the wrong end on purpose: between steps 4 000 and 16 000 `es_dec_erb_l0.1`'s noiseless output *lost* 2.03 dB of high-band energy while its draw gained 0.50 dB, and `es_dec_l0.1` lost 3.22 dB while gaining 3.40 dB. The models are moving the high band out of the conditional mean and into the noise channel, which is the opposite of collapsing toward determinism, and the τ = 0 pass getting worse is the price and the proof. In absolute terms the channel is still loud at the end: two draws of `es_erb_l0.1` differ by 0.00251 against a draw-to-truth distance of about 0.0074 (its `val_wave` plus half its spread, one figure crossing a train batch with a val batch), a third of the way, and `es_marg`'s spread is the one that is still *rising* (+0.33 %/1k over the last 4 000 steps). One arm did collapse. `es_split_l0.1` has the lowest spread of the seven at 0.00152, still falling at −1.09 %/1k, and a `def1 − def0` gap of −0.06 dB — its τ = 1 draw and its τ = 0 pass are the same signal. Its waveform term never looks above 6 kHz, so nothing in its objective pays for two draws differing up there, and its noise became decoration.
