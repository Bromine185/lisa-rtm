| arm | SNR | LSD | HB-LSD | deficit dB | CRPS | sliced CRPS | HB kappa | PIT end-bins | mean-of-16 SNR | mean deficit | SNR gap (calibrated 2.75) | one-draw passthrough LSD | logmean16 LSD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| det | 18.84 | 0.916 | 1.044 | -12.61 | 0.9597 | 0.9536 | +0.087 | - | - | - | - | 0.910 | - |
| det_paper | 19.01 | 2.122 | 2.446 | -27.63 | 2.5576 | 2.1312 | +0.269 | - | - | - | - | 2.122 | - |
| es_dec_erb_l0.1 | 18.22 | 0.950 | 1.069 | -6.85 | 0.5700 | 0.5636 | +0.028 | 0.182 | 18.91 | -15.93 | 0.69 | 0.932 | 0.835 |
| es_dec_l0.01 | 17.95 | 0.982 | 1.095 | -7.05 | 0.6002 | 0.5808 | +0.013 | 0.202 | 18.92 | -17.39 | 0.97 | 0.956 | 0.873 |
| es_erb_l0.001 | 17.83 | 1.052 | 1.143 | -12.34 | 0.7284 | 0.6776 | +0.008 | 0.398 | 18.87 | -19.26 | 1.04 | 0.998 | 0.976 |
| es_erb_l0.01 | 17.95 | 0.972 | 1.085 | -7.02 | 0.5748 | 0.5675 | +0.017 | 0.186 | 18.93 | -17.45 | 0.97 | 0.946 | 0.841 |
| es_erb_l0.1 | 18.07 | 0.953 | 1.073 | -6.26 | 0.5767 | 0.5685 | +0.031 | 0.187 | 18.90 | -15.62 | 0.82 | 0.936 | 0.844 |
| es_marg | 18.01 | 0.999 | 1.111 | -7.68 | 0.6115 | 0.5914 | +0.007 | 0.207 | 18.91 | -17.08 | 0.90 | 0.969 | 0.883 |
| floor: passthrough + empty HB | 18.99 | 5.606 | 6.470 | -52.65 | - | - | - | - | - | - | - | - | - |
| ceiling: passthrough + true HB | 41.60 | 0.105 | 0.045 | -0.00 | - | - | - | - | - | - | - | - | - |

EVAL12 n=12; PIT end-bins ideal 0.118; one-draw passthrough LSD is draw 0 (seed 0) with the baseband passed through; logmean16 LSD is the 16-draw log-magnitude readout with baseband passthrough (raw readout in the JSON). CRPS uses the notebook's crps_ensemble: all-pairs spread term over M^2 pairs incl. the diagonal (standard, not the fair M(M-1) estimator; spread under-credited by (M-1)/M), the same estimator as the OV2 tables.