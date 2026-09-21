# OV50 — every metric, all eight arms

EVAL12, n=12, M=16. LSD is `lsd_db`: decades of power, ×10 for dB. `pt` = baseband passthrough.

## 1. Training record (from the histories, step 104950/104950)

| arm | kind | λ | class | val_wave | val_spec | own objective | spread (final) |
|---|---|---|---|---|---|---|---|
| `det_paper` | det | 0 | LISAS | 0.004063 | 1.6034 | 0.004063 | 0.00000 |
| `det` | det | 0.01 | LISAS | 0.004341 | 0.8140 | 0.012481 | 0.00000 |
| `es_marg` | es_marg | 0.01 | LISAS | 0.003262 | 0.4360 | 0.007622 | 0.00377 |
| `es_erb_l0.001` | es_marg_erb | 0.001 | LISAS | 0.003219 | 0.3861 | 0.003606 | 0.00465 |
| `es_erb_l0.01` | es_marg_erb | 0.01 | LISAS | 0.003259 | 0.2822 | 0.006082 | 0.00384 |
| `es_erb_l0.1` | es_marg_erb | 0.1 | LISAS | 0.003539 | 0.2772 | 0.031262 | 0.00230 |
| `es_dec_l0.01` | es_marg | 0.01 | LISASD | 0.003272 | 0.4340 | 0.007612 | 0.00376 |
| `es_dec_erb_l0.1` | es_marg_erb | 0.1 | LISASD | 0.003567 | 0.2768 | 0.031244 | 0.00225 |

`val_wave` is the only column comparable across all eight. `spread` is exactly 0 for both det arms at every logged step.

## 2. EVAL12 — fidelity and spectrum

| arm | SNR | LSD raw | LSD pt | HB-LSD | deficit dB | mean-16 deficit | HB κ | HB coherent | corr_err |
|---|---|---|---|---|---|---|---|---|---|
| `det_paper` | 19.01 | 2.122 | 2.122 | 2.446 | -27.63 | — | +0.269 | +0.0058 | 1.721 |
| `det` | 18.84 | 0.916 | 0.910 | 1.044 | -12.61 | — | +0.087 | +0.0078 | 0.820 |
| `es_marg` | 18.01 | 0.999 | 0.969 | 1.111 | -7.68 | -17.08 | +0.007 | +0.0068 | 0.900 |
| `es_erb_l0.001` | 17.83 | 1.052 | 0.998 | 1.143 | -12.34 | -19.26 | +0.008 | +0.0073 | 0.733 |
| `es_erb_l0.01` | 17.95 | 0.972 | 0.946 | 1.085 | -7.02 | -17.45 | +0.017 | +0.0054 | 0.705 |
| `es_erb_l0.1` | 18.07 | 0.953 | 0.936 | 1.073 | -6.26 | -15.62 | +0.031 | +0.0118 | 0.910 |
| `es_dec_l0.01` | 17.95 | 0.982 | 0.956 | 1.095 | -7.05 | -17.39 | +0.013 | +0.0052 | 0.877 |
| `es_dec_erb_l0.1` | 18.22 | 0.950 | 0.932 | 1.069 | -6.85 | -15.93 | +0.028 | +0.0083 | 0.880 |
| floor (pt + empty HB) | 18.99 | 5.606 | — | 6.470 | -52.65 | — | — | — | — |
| ceiling (pt + true HB) | 41.60 | 0.105 | — | 0.045 | -0.00 | — | — | — | — |

## 3. EVAL12 — probabilistic

| arm | CRPS | sliced CRPS | PIT end (ideal 0.1176) | PIT bin0 /ideal | PIT binM /ideal | SNR gap | of calibrated 2.75 |
|---|---|---|---|---|---|---|---|
| `det_paper` | 2.5576 | 2.1312 | — (deterministic) | — | — | — | — |
| `det` | 0.9597 | 0.9536 | — (deterministic) | — | — | — | — |
| `es_marg` | 0.6115 | 0.5914 | 0.2070 | 1.01× | 2.51× | 0.90 | 33% |
| `es_erb_l0.001` | 0.7284 | 0.6776 | 0.3978 | 1.13× | 5.64× | 1.04 | 38% |
| `es_erb_l0.01` | 0.5748 | 0.5675 | 0.1859 | 1.12× | 2.04× | 0.97 | 35% |
| `es_erb_l0.1` | 0.5767 | 0.5685 | 0.1873 | 1.11× | 2.07× | 0.82 | 30% |
| `es_dec_l0.01` | 0.6002 | 0.5808 | 0.2021 | 1.00× | 2.44× | 0.97 | 35% |
| `es_dec_erb_l0.1` | 0.5700 | 0.5636 | 0.1819 | 1.15× | 1.94× | 0.69 | 25% |

## 4. EVAL12 — readouts (LSD, passthrough)

| arm | one draw | mean16 | logmean16 | draw/logmean16 | implied v/b² |
|---|---|---|---|---|---|
| `det_paper` | 2.122 | — | — | — | — |
| `det` | 0.910 | — | — | — | — |
| `es_marg` | 0.969 | 1.193 | 0.883 | 1.098 | 0.21 |
| `es_erb_l0.001` | 0.998 | 1.344 | 0.976 | 1.022 | 0.04 |
| `es_erb_l0.01` | 0.946 | 1.160 | 0.841 | 1.124 | 0.26 |
| `es_erb_l0.1` | 0.936 | 1.146 | 0.844 | 1.110 | 0.23 |
| `es_dec_l0.01` | 0.956 | 1.200 | 0.873 | 1.095 | 0.20 |
| `es_dec_erb_l0.1` | 0.932 | 1.141 | 0.835 | 1.117 | 0.25 |

## 5. ViSQOL / PESQ (lattice mapping; audio48k floor 1.567, ceiling 4.725)

| condition | audio48k | share of range | NSIM au | speech16k | NSIM sp | PESQ wb |
|---|---|---|---|---|---|---|
| ceiling: passthrough + true HB | 4.725 | 100.0% | 0.994 | 4.508 | 0.995 | 4.615 |
| es_dec_erb_l0.1 logmean16 + passthrough | 3.021 | 46.0% | 0.856 | 4.195 | 0.980 | 4.309 |
| det + passthrough | 3.001 | 45.4% | 0.851 | 4.168 | 0.978 | 4.270 |
| es_erb_l0.1 logmean16 + passthrough | 2.991 | 45.1% | 0.854 | 4.196 | 0.980 | 4.283 |
| es_erb_l0.01 logmean16 + passthrough | 2.942 | 43.5% | 0.851 | 4.199 | 0.982 | 4.294 |
| es_dec_l0.01 logmean16 + passthrough | 2.927 | 43.1% | 0.849 | 4.251 | 0.983 | 4.321 |
| es_marg logmean16 + passthrough | 2.877 | 41.5% | 0.846 | 4.249 | 0.983 | 4.333 |
| es_dec_erb_l0.1 tau=1 + passthrough | 2.815 | 39.5% | 0.834 | 4.045 | 0.974 | 4.000 |
| es_dec_l0.01 tau=1 + passthrough | 2.776 | 38.3% | 0.832 | 4.104 | 0.976 | 4.059 |
| es_erb_l0.1 tau=1 + passthrough | 2.766 | 38.0% | 0.828 | 4.040 | 0.971 | 3.930 |
| es_marg tau=1 + passthrough | 2.707 | 36.1% | 0.828 | 4.084 | 0.977 | 4.074 |
| es_erb_l0.01 tau=1 + passthrough | 2.694 | 35.7% | 0.826 | 4.102 | 0.975 | 3.995 |
| es_erb_l0.001 tau=1 + passthrough | 2.420 | 27.0% | 0.803 | 3.949 | 0.971 | 4.118 |
| es_erb_l0.001 logmean16 + passthrough | 2.338 | 24.4% | 0.803 | 4.253 | 0.983 | 4.265 |
| det_paper + passthrough | 1.717 | 4.8% | 0.742 | 4.037 | 0.970 | 4.385 |
| naive | 1.576 | 0.3% | 0.727 | 3.972 | 0.964 | 4.377 |
| floor: passthrough + empty HB | 1.567 | 0.0% | 0.724 | 3.947 | 0.961 | 4.285 |

## 6. Checkpoint verification (val_wave check)

| arm | term | recorded | fp32 mean | eps sd | AMP | predicted | residual | bands |
|---|---|---|---|---|---|---|---|---|
| `det_paper` | val_wave | 0.00406263 | 0.00402731 | — | +0.95% | 0.00406568 | -0.075% | 0.15 |
| `det_paper` | val_spec | 1.60338 | 1.9444 | — | -17.52% | 1.6038 | -0.026% | 0.05 |
| `det` | val_wave | 0.00434071 | 0.00432293 | — | +0.48% | 0.0043438 | -0.071% | 0.14 |
| `det` | val_spec | 0.814028 | 0.807589 | — | +0.74% | 0.813575 | +0.056% | 0.11 |
| `es_marg` | val_wave | 0.00326217 | 0.00324977 | 0.028% | +0.41% | 0.00326309 | -0.028% | 0.28 |
| `es_marg` | val_spec | 0.435952 | 0.454811 | 0.248% | -4.49% | 0.434411 | +0.353% | 0.38 |
| `es_erb_l0.001` | val_wave | 0.00321946 | 0.00321411 | 0.074% | +0.34% | 0.00322491 | -0.169% | 0.64 |
| `es_erb_l0.001` | val_spec | 0.386113 | 0.420703 | 0.094% | -8.34% | 0.385621 | +0.127% | 0.35 |
| `es_erb_l0.01` | val_wave | 0.00325941 | 0.0032597 | 0.049% | +0.13% | 0.00326404 | -0.142% | 0.81 |
| `es_erb_l0.01` | val_spec | 0.282234 | 0.288156 | 0.070% | -1.94% | 0.28258 | -0.123% | 0.48 |
| `es_erb_l0.1` | val_wave | 0.0035391 | 0.00356256 | 0.046% | -0.65% | 0.00353935 | -0.007% | 0.04 |
| `es_erb_l0.1` | val_spec | 0.277234 | 0.28247 | 0.115% | -1.80% | 0.277391 | -0.057% | 0.14 |
| `es_dec_l0.01` | val_wave | 0.00327247 | 0.00324728 | 0.084% | +0.94% | 0.0032778 | -0.163% | 0.55 |
| `es_dec_l0.01` | val_spec | 0.433973 | 0.445012 | 0.113% | -1.87% | 0.436703 | -0.629% | 1.53 |
| `es_dec_erb_l0.1` | val_wave | 0.00356711 | 0.00356033 | 0.085% | +0.52% | 0.00357901 | -0.334% | 1.11 |
| `es_dec_erb_l0.1` | val_spec | 0.276769 | 0.279551 | 0.112% | -0.90% | 0.277038 | -0.097% | 0.24 |

Verdict: identity passes by 13–1800 bands. All eight `val_wave` residuals negative (sign test p = 0.0078) — the AMP proxy, not the checkpoints.
