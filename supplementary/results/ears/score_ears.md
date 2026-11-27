# Released SOTA vs ours, one scorer (ears: 96 utterances, 384 s, p102, p103, p104, p105, p106, p107)

12 kHz -> 48 kHz. Input: Chebyshev I order 8, 0.05 dB, sosfiltfilt, resample_poly (the FLowHigh / NU-Wave 2 test input). Every draw gain-aligned on its energy below 5.5 kHz. Per-draw numbers are the mean over draws, then over utterances. ViSQOL speech mapping: lattice.

## 1. Levels

Ideal: 0 for every column. LSD in decades of power; deficits in dB, negative = too quiet.

| condition | M | LSD 2048 | LSD-HF 2048 | LSD-LF 2048 | LSD 1024 | HB-LSD 1024 | deficit | deficit bb | loud | mid | quiet |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ceiling: true LB, empty HB | 1 | 5.015 | 5.789 | 0.060 | 4.602 | 5.311 | -34.28 | -21.48 | -67.41 | -35.74 | -25.13 |
| naive (resample_poly) | 1 | 4.086 | 4.716 | 0.194 | 3.821 | 4.409 | -34.55 | -20.06 | -37.37 | -33.52 | -26.99 |
| FLowHigh (as shipped) | 8 | 1.610 | 1.854 | 0.205 | 1.606 | 1.850 | 1.08 | -2.66 | 0.80 | 1.39 | 3.45 |
| FLowHigh, trained prior (std 1) | 8 | 1.596 | 1.838 | 0.204 | 1.593 | 1.835 | 0.24 | -3.40 | 0.01 | 0.57 | 3.10 |
| NU-Wave 2 | 8 | 1.928 | 2.213 | 0.366 | 1.926 | 2.210 | -4.86 | -9.15 | -5.82 | -3.10 | 10.81 |
| AudioSR (speech) | 8 | 1.208 | 1.357 | 0.529 | 1.181 | 1.325 | 1.08 | 1.93 | 0.23 | 2.30 | 5.23 |
| AP-BWE | 1 | 2.147 | 2.474 | 0.232 | 2.148 | 2.476 | 3.34 | 0.11 | 2.89 | 3.47 | 9.74 |
| ours_det_paper | 1 | 1.769 | 2.037 | 0.229 | 1.736 | 1.998 | -18.13 | -14.65 | -17.40 | -19.06 | -14.20 |
| ours_det_paper \| pt | 1 | 1.767 | 2.037 | 0.193 | 1.735 | 1.999 | -18.19 | -14.80 | -17.44 | -19.17 | -13.44 |
| ours: det | 1 | 1.730 | 1.990 | 0.275 | 1.726 | 1.985 | -2.70 | -7.28 | -3.50 | -0.94 | 5.66 |
| ours: det \| pt | 1 | 1.727 | 1.991 | 0.194 | 1.724 | 1.987 | -2.72 | -7.34 | -3.52 | -0.96 | 5.86 |
| ours_es_marg | 8 | 1.873 | 2.143 | 0.469 | 1.848 | 2.114 | 3.27 | -0.45 | 2.15 | 5.32 | 8.60 |
| ours_es_marg \| pt | 8 | 1.860 | 2.143 | 0.202 | 1.836 | 2.116 | 3.26 | -0.49 | 2.14 | 5.31 | 9.08 |
| ours_es_marg logmean8 \| pt | 1 | 1.705 | 1.965 | 0.199 | 1.706 | 1.965 | -1.74 | -4.67 | -2.67 | 0.13 | 4.96 |
| ours_es_erb_l0.01 | 8 | 1.857 | 2.125 | 0.454 | 1.834 | 2.099 | 2.42 | -1.48 | 1.12 | 4.74 | 8.65 |
| ours_es_erb_l0.01 \| pt | 8 | 1.845 | 2.126 | 0.200 | 1.823 | 2.101 | 2.40 | -1.51 | 1.10 | 4.73 | 9.07 |
| ours_es_erb_l0.01 logmean8 \| pt | 1 | 1.701 | 1.960 | 0.198 | 1.699 | 1.958 | -2.01 | -5.43 | -3.02 | -0.03 | 5.19 |
| ours_es_erb_l0.001 | 8 | 1.835 | 2.079 | 0.662 | 1.821 | 2.064 | -1.92 | -3.62 | -2.83 | -0.05 | 6.26 |
| ours_es_erb_l0.001 \| pt | 8 | 1.805 | 2.080 | 0.204 | 1.794 | 2.067 | -1.95 | -3.74 | -2.86 | -0.08 | 6.79 |
| ours_es_erb_l0.001 logmean8 \| pt | 1 | 1.722 | 1.985 | 0.200 | 1.710 | 1.970 | -4.66 | -6.49 | -5.45 | -2.91 | 3.89 |
| ours: es_dec_erb_l0.1 | 8 | 1.868 | 2.143 | 0.387 | 1.843 | 2.114 | 3.37 | -0.84 | 2.06 | 5.56 | 8.75 |
| ours: es_dec_erb_l0.1 \| pt | 8 | 1.860 | 2.144 | 0.198 | 1.835 | 2.115 | 3.36 | -0.86 | 2.05 | 5.55 | 9.05 |
| ours: es_dec_erb_l0.1 logmean8 \| pt | 1 | 1.709 | 1.969 | 0.196 | 1.709 | 1.969 | -1.39 | -4.96 | -2.25 | 0.34 | 4.84 |

## 2. SNR against the ceiling, and coherence

Ceiling = truth brick-walled at 6 kHz. The HB error is E(1 + p - 2 coh): an incoherent high band at the right level costs about 3 dB against the ceiling. coh: the predictable fraction (0 = unpredictable). kappa: ideal 1 (0 = hallucinated energy). p implied: the HB level the SNR allows if coh = 0 and the LB is perfect (an upper bound on the true level). LB coh: alignment check, ideal 1.

| condition | SNR | vs ceiling | coh | kappa | p implied dB | deficit bb dB | LB coh | gain |
|---|---|---|---|---|---|---|---|---|
| ceiling: true LB, empty HB | 21.16 ± 0.58 | 0.00 | -- | -- | -60.00 | -21.48 | 1.000 | 1.000 |
| naive (resample_poly) | 20.86 ± 0.56 | -0.30 | 0.011 | 1.394 | -13.40 | -20.06 | 0.994 | 1.000 |
| FLowHigh (as shipped) | 18.37 ± 0.49 | -2.78 | -0.000 | 0.001 | -0.84 | -2.66 | 1.000 | 0.962 |
| FLowHigh, trained prior (std 1) | 18.58 ± 0.49 | -2.57 | -0.000 | 0.001 | -1.34 | -3.40 | 1.000 | 0.961 |
| NU-Wave 2 | 20.36 ± 0.53 | -0.79 | 0.018 | 0.066 | -8.41 | -9.15 | 1.000 | 1.007 |
| AudioSR (speech) | 14.08 ± 0.34 | -7.07 | 0.002 | 0.008 | 5.83 | 1.93 | 0.992 | 1.886 |
| AP-BWE | 17.39 ± 0.48 | -3.76 | 0.013 | 0.012 | 0.94 | 0.11 | 1.000 | 1.010 |
| ours_det_paper | 20.97 ± 0.57 | -0.19 | 0.017 | 0.194 | -17.81 | -14.65 | 1.000 | 1.009 |
| ours_det_paper \| pt | 20.82 ± 0.55 | -0.34 | 0.016 | 0.188 | -13.31 | -14.80 | 0.994 | 1.009 |
| ours: det | 20.33 ± 0.54 | -0.83 | 0.022 | 0.064 | -7.71 | -7.28 | 1.000 | 1.008 |
| ours: det \| pt | 20.20 ± 0.53 | -0.96 | 0.019 | 0.062 | -7.06 | -7.34 | 0.994 | 1.008 |
| ours_es_marg | 17.63 ± 0.48 | -3.52 | 0.020 | 0.014 | 0.55 | -0.45 | 0.999 | 1.008 |
| ours_es_marg \| pt | 18.06 ± 0.49 | -3.09 | 0.019 | 0.014 | -0.30 | -0.49 | 0.994 | 1.008 |
| ours_es_marg logmean8 \| pt | 19.59 ± 0.53 | -1.57 | 0.017 | 0.025 | -4.28 | -4.67 | 0.994 | 1.008 |
| ours_es_erb_l0.01 | 18.08 ± 0.48 | -3.08 | 0.021 | 0.017 | -0.34 | -1.48 | 0.999 | 1.008 |
| ours_es_erb_l0.01 \| pt | 18.51 ± 0.49 | -2.65 | 0.018 | 0.016 | -1.29 | -1.51 | 0.994 | 1.008 |
| ours_es_erb_l0.01 logmean8 \| pt | 19.76 ± 0.53 | -1.40 | 0.016 | 0.021 | -4.85 | -5.43 | 0.994 | 1.008 |
| ours_es_erb_l0.001 | 17.56 ± 0.49 | -3.59 | 0.021 | 0.019 | 0.67 | -3.62 | 0.997 | 1.004 |
| ours_es_erb_l0.001 \| pt | 19.32 ± 0.51 | -1.83 | 0.019 | 0.018 | -3.47 | -3.74 | 0.994 | 1.004 |
| ours_es_erb_l0.001 logmean8 \| pt | 20.00 ± 0.53 | -1.15 | 0.017 | 0.029 | -5.97 | -6.49 | 0.994 | 1.004 |
| ours: es_dec_erb_l0.1 | 18.18 ± 0.49 | -2.98 | 0.021 | 0.022 | -0.54 | -0.84 | 1.000 | 1.006 |
| ours: es_dec_erb_l0.1 \| pt | 18.24 ± 0.49 | -2.92 | 0.020 | 0.021 | -0.68 | -0.86 | 0.994 | 1.006 |
| ours: es_dec_erb_l0.1 logmean8 \| pt | 19.70 ± 0.53 | -1.46 | 0.017 | 0.033 | -4.66 | -4.96 | 0.994 | 1.006 |

## 3. Calibration (M = 8)

Ideals: gap 2.50 dB; spread 1 (<1 = under-dispersed); PIT low and high 0.111 each, end 0.222; RMSE/spread in each spread-skill bin ~1.06; corr_err 0.

| condition | gap | gap HB | spread | spread HB (pooled) | PIT low | PIT high | PIT end | RMSE/spread, low bin | high bin | corr_err draw 0 | corr_err mean | SNR of mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FLowHigh (as shipped) | 0.00 | 0.00 | 0.000 | 0.000 | 0.716 | 0.276 | 0.992 | >100 | 45.37 | 0.498 | 0.498 | 18.37 |
| FLowHigh, trained prior (std 1) | 0.08 | 0.06 | 0.023 | 0.014 | 0.655 | 0.232 | 0.887 | 82.71 | 4.55 | 0.520 | 0.518 | 18.66 |
| NU-Wave 2 | 0.53 | 0.50 | 0.179 | 0.085 | 0.511 | 0.120 | 0.631 | 8.13 | 2.01 | 0.516 | 0.590 | 20.88 |
| AudioSR (speech) | 2.71 | 2.37 | 1.164 | 0.797 | 0.289 | 0.195 | 0.484 | 7.10 | 1.29 | 0.472 | 0.488 | 16.35 |
| ours_es_marg | 2.67 | 2.37 | 1.159 | 0.659 | 0.460 | 0.069 | 0.529 | 8.76 | 1.38 | 1.091 | 0.783 | 20.28 |
| ours_es_marg \| pt | 2.23 | 2.37 | 0.913 | 0.659 | 0.462 | 0.069 | 0.530 | 8.78 | 1.38 | 1.077 | 0.777 | 20.27 |
| ours_es_erb_l0.01 | 2.34 | 2.04 | 0.975 | 0.519 | 0.472 | 0.069 | 0.541 | 8.58 | 1.65 | 0.974 | 0.748 | 20.41 |
| ours_es_erb_l0.01 \| pt | 1.90 | 2.04 | 0.747 | 0.519 | 0.474 | 0.069 | 0.543 | 8.61 | 1.65 | 0.963 | 0.744 | 20.40 |
| ours_es_erb_l0.001 | 2.63 | 1.25 | 1.135 | 0.277 | 0.471 | 0.171 | 0.642 | 9.39 | 2.44 | 0.759 | 0.521 | 20.19 |
| ours_es_erb_l0.001 \| pt | 1.16 | 1.25 | 0.411 | 0.277 | 0.473 | 0.171 | 0.644 | 9.41 | 2.45 | 0.739 | 0.507 | 20.48 |
| ours: es_dec_erb_l0.1 | 2.14 | 2.11 | 0.862 | 0.567 | 0.479 | 0.064 | 0.543 | 8.82 | 1.58 | 1.071 | 0.742 | 20.28 |
| ours: es_dec_erb_l0.1 \| pt | 1.99 | 2.11 | 0.785 | 0.567 | 0.480 | 0.064 | 0.544 | 8.84 | 1.58 | 1.063 | 0.739 | 20.19 |

## 4. Proper scores on the HB log-magnitude

Lower is better. A point forecast (M = 1) scores its MAE: it is the reference line, not a competitor. CRPS is the datasheet's estimator (spread over all M^2 pairs, which under-credits a sampler's spread by (M-1)/M); CRPS fair uses M(M-1). corr_err ideal 0.

| condition | M | CRPS | CRPS fair | sliced CRPS | corr_err draw 0 |
|---|---|---|---|---|---|
| naive (resample_poly) | 1 | 5.8697 | -- | 5.4201 | 2.340 |
| FLowHigh (as shipped) | 8 | 1.6063 | 1.6055 | 1.6610 | 0.498 |
| FLowHigh, trained prior (std 1) | 8 | 1.5171 | 1.5057 | 1.5642 | 0.520 |
| NU-Wave 2 | 8 | 1.6843 | 1.6293 | 1.5534 | 0.516 |
| AudioSR (speech) | 8 | 0.8968 | 0.8491 | 0.9200 | 0.472 |
| AP-BWE | 1 | 2.2137 | -- | 2.1753 | 1.246 |
| ours_det_paper | 1 | 1.9694 | -- | 2.1487 | 1.005 |
| ours_det_paper \| pt | 1 | 1.9702 | -- | 2.1501 | 1.004 |
| ours: det | 1 | 1.7542 | -- | 1.7913 | 0.568 |
| ours: det \| pt | 1 | 1.7566 | -- | 1.7938 | 0.568 |
| ours_es_marg | 8 | 1.5022 | 1.4443 | 1.4864 | 1.091 |
| ours_es_marg \| pt | 8 | 1.5058 | 1.4479 | 1.4921 | 1.077 |
| ours_es_marg logmean8 \| pt | 1 | 1.7312 | -- | 1.7721 | 1.110 |
| ours_es_erb_l0.01 | 8 | 1.4953 | 1.4403 | 1.4941 | 0.974 |
| ours_es_erb_l0.01 \| pt | 8 | 1.4990 | 1.4440 | 1.4991 | 0.963 |
| ours_es_erb_l0.01 logmean8 \| pt | 1 | 1.7160 | -- | 1.7747 | 0.965 |
| ours_es_erb_l0.001 | 8 | 1.5412 | 1.4973 | 1.5419 | 0.759 |
| ours_es_erb_l0.001 \| pt | 8 | 1.5450 | 1.5010 | 1.5478 | 0.739 |
| ours_es_erb_l0.001 logmean8 \| pt | 1 | 1.7550 | -- | 1.8043 | 0.708 |
| ours: es_dec_erb_l0.1 | 8 | 1.5189 | 1.4628 | 1.4853 | 1.071 |
| ours: es_dec_erb_l0.1 \| pt | 8 | 1.5215 | 1.4654 | 1.4890 | 1.063 |
| ours: es_dec_erb_l0.1 logmean8 \| pt | 1 | 1.7388 | -- | 1.7634 | 1.064 |

## 5. Perceptual (draw 0; the readout waveform for readouts)

Ceilings: ViSQOL audio 4.73 (NSIM 1), ViSQOL speech ~4.5 (NSIM 1), PESQ wb 4.64. Audio mode spans 1.57-4.73 on this task; speech mode and PESQ stop at 8 kHz, so they see only 6-8 kHz of the missing band.

| condition | ViSQOL audio | NSIM audio | ViSQOL speech | NSIM speech | PESQ wb |
|---|---|---|---|---|---|
| ceiling: true LB, empty HB | 2.20 | 0.774 | 3.86 | 0.959 | 4.20 |
| naive (resample_poly) | 2.20 | 0.778 | 3.95 | 0.962 | 4.37 |
| FLowHigh (as shipped) | 2.99 | 0.852 | 4.24 | 0.984 | 4.43 |
| FLowHigh, trained prior (std 1) | 3.01 | 0.854 | 4.25 | 0.984 | 4.44 |
| NU-Wave 2 | 2.05 | 0.731 | 3.51 | 0.942 | 4.02 |
| AudioSR (speech) | 3.09 | 0.839 | 3.69 | 0.947 | 3.88 |
| AP-BWE | 2.30 | 0.784 | 4.02 | 0.971 | 4.24 |
| ours_det_paper | 2.38 | 0.795 | 4.01 | 0.967 | 4.43 |
| ours_det_paper \| pt | 2.38 | 0.796 | 4.02 | 0.968 | 4.43 |
| ours: det | 2.45 | 0.805 | 4.10 | 0.975 | 4.35 |
| ours: det \| pt | 2.45 | 0.805 | 4.12 | 0.976 | 4.35 |
| ours_es_marg | 2.36 | 0.778 | 3.70 | 0.953 | 3.86 |
| ours_es_marg \| pt | 2.37 | 0.793 | 3.94 | 0.965 | 4.04 |
| ours_es_marg logmean8 \| pt | 2.66 | 0.818 | 4.06 | 0.973 | 4.31 |
| ours_es_erb_l0.01 | 2.39 | 0.787 | 3.83 | 0.960 | 3.99 |
| ours_es_erb_l0.01 \| pt | 2.39 | 0.795 | 3.98 | 0.967 | 4.13 |
| ours_es_erb_l0.01 logmean8 \| pt | 2.69 | 0.821 | 4.08 | 0.974 | 4.31 |
| ours_es_erb_l0.001 | 2.15 | 0.729 | 3.18 | 0.921 | 3.51 |
| ours_es_erb_l0.001 \| pt | 2.16 | 0.774 | 3.90 | 0.965 | 4.09 |
| ours_es_erb_l0.001 logmean8 \| pt | 2.23 | 0.785 | 4.02 | 0.972 | 4.24 |
| ours: es_dec_erb_l0.1 | 2.39 | 0.795 | 3.99 | 0.968 | 4.05 |
| ours: es_dec_erb_l0.1 \| pt | 2.38 | 0.797 | 4.02 | 0.969 | 4.10 |
| ours: es_dec_erb_l0.1 logmean8 \| pt | 2.68 | 0.823 | 4.11 | 0.975 | 4.34 |
