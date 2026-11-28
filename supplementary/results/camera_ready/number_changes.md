# Camera-ready: every number that moved from VCTK to EARS

Old = the Sep 29 paper (VCTK: levels on 240 utterances, calibration on 32 × 8). New = EARS (96 utterances × 8 draws, test speakers p102–p107).
Both columns come from `sota/paper_numbers.py` (`vctk` / `ears`), which reproduces every number in the Sep 29 paper from the VCTK files. Sources are keys in `supplementary/results/ears/` (EARS) or `lisa_rtm_cache/sota/work/` (VCTK).

| Quantity | Old (VCTK) | New (EARS) | EARS source |
|---|---|---|---|
| T1 FLowHigh deficit | -1.5999 | 1.0763 | `ears agg.flowhigh.deficit` |
| T1 FLowHigh LSD | 0.7612 | 1.6095 | `ears agg.flowhigh.lsd` |
| T1 FLowHigh ViSQOL | 3.6070 | 2.9864 | `ears agg.flowhigh.visqol_audio` |
| T1 FLowHigh gap | 0.0846 | 0.0644 | `ears agg.flowhigh_std1.gap_hb` |
| T1 FLowHigh gap indep | 1.7713 | 1.4819 | `ears agg.flowhigh_std1.gap_indep_hb` |
| T1 FLowHigh CRPS | 0.6824 | 1.5057 | `ears agg.flowhigh_std1.crps_fair` |
| T1 AP-BWE deficit | -1.3769 | 3.3368 | `ears agg.apbwe.deficit` |
| T1 AP-BWE LSD | 0.7800 | 2.1469 | `ears agg.apbwe.lsd` |
| T1 AP-BWE ViSQOL | 3.4764 | 2.3014 | `ears agg.apbwe.visqol_audio` |
| T1 AP-BWE CRPS | 0.7768 | 2.2137 | `ears agg.apbwe.crps (MAE)` |
| T1 NU-Wave 2 deficit | -7.9628 | -4.8616 | `ears agg.nuwave2.deficit` |
| T1 NU-Wave 2 LSD | 0.9892 | 1.9285 | `ears agg.nuwave2.lsd` |
| T1 NU-Wave 2 ViSQOL | 2.5695 | 2.0524 | `ears agg.nuwave2.visqol_audio` |
| T1 NU-Wave 2 gap | 0.5732 | 0.5022 | `ears agg.nuwave2.gap_hb` |
| T1 NU-Wave 2 gap indep | 0.6283 | 0.5569 | `ears agg.nuwave2.gap_indep_hb` |
| T1 NU-Wave 2 CRPS | 0.5834 | 1.6293 | `ears agg.nuwave2.crps_fair` |
| T1 AudioSR deficit | -0.6003 | 1.0840 | `ears agg.audiosr.deficit` |
| T1 AudioSR LSD | 1.5608 | 1.2084 | `ears agg.audiosr.lsd` |
| T1 AudioSR ViSQOL | 2.4073 | 3.0904 | `ears agg.audiosr.visqol_audio` |
| T1 AudioSR gap | 2.6188 | 2.3733 | `ears agg.audiosr.gap_hb` |
| T1 AudioSR gap indep | 3.9151 | 3.6877 | `ears agg.audiosr.gap_indep_hb` |
| T1 AudioSR CRPS | 1.2431 | 0.8491 | `ears agg.audiosr.crps_fair` |
| T1 ours, point deficit | -6.1795 | -2.6956 | `ears agg.ours_det.deficit` |
| T1 ours, point LSD | 0.8641 | 1.7304 | `ears agg.ours_det.lsd` |
| T1 ours, point ViSQOL | 3.1254 | 2.4490 | `ears agg.ours_det.visqol_audio` |
| T1 ours, point CRPS | 0.8892 | 1.7542 | `ears agg.ours_det.crps (MAE)` |
| T1 ours, sampler deficit | -0.6991 | 3.3746 | `ears agg.ours_es_dec_erb_l0.1.deficit` |
| T1 ours, sampler LSD | 0.8951 | 1.8678 | `ears agg.ours_es_dec_erb_l0.1.lsd` |
| T1 ours, sampler ViSQOL | 2.8255 | 2.3856 | `ears agg.ours_es_dec_erb_l0.1.visqol_audio` |
| T1 ours, sampler gap | 2.0712 | 2.1129 | `ears agg.ours_es_dec_erb_l0.1.gap_hb` |
| T1 ours, sampler gap indep | 2.3673 | 2.3794 | `ears agg.ours_es_dec_erb_l0.1.gap_indep_hb` |
| T1 ours, sampler CRPS | 0.5137 | 1.4628 | `ears agg.ours_es_dec_erb_l0.1.crps_fair` |
| T2 det_paper deficit | -21.7803 | -18.1317 | `ears agg.ours_det_paper.deficit` |
| T2 det_paper CRPS | 2.8525 | 1.9694 | `ears agg.ours_det_paper.crps (MAE)` |
| T2 det deficit | -6.1795 | -2.6956 | `ears agg.ours_det.deficit` |
| T2 det CRPS | 0.8892 | 1.7542 | `ears agg.ours_det.crps (MAE)` |
| T2 es_marg deficit | -1.0493 | 3.2725 | `ears agg.ours_es_marg.deficit` |
| T2 es_marg gap | 2.2132 | 2.3741 | `ears agg.ours_es_marg.gap_hb` |
| T2 es_marg CRPS | 0.5372 | 1.4443 | `ears agg.ours_es_marg.crps_fair` |
| T2 es_erb deficit | -1.0825 | 2.4168 | `ears agg.ours_es_erb_l0.01.deficit` |
| T2 es_erb gap | 2.0316 | 2.0367 | `ears agg.ours_es_erb_l0.01.gap_hb` |
| T2 es_erb CRPS | 0.5115 | 1.4403 | `ears agg.ours_es_erb_l0.01.crps_fair` |
| T2 es_erb l0.001 deficit | -5.7300 | -1.9181 | `ears agg.ours_es_erb_l0.001.deficit` |
| T2 es_erb l0.001 gap | 1.1585 | 1.2530 | `ears agg.ours_es_erb_l0.001.gap_hb` |
| T2 es_erb l0.001 CRPS | 0.6825 | 1.4973 | `ears agg.ours_es_erb_l0.001.crps_fair` |
| T2 es_dec_erb deficit | -0.6991 | 3.3746 | `ears agg.ours_es_dec_erb_l0.1.deficit` |
| T2 es_dec_erb gap | 2.0712 | 2.1129 | `ears agg.ours_es_dec_erb_l0.1.gap_hb` |
| T2 es_dec_erb CRPS | 0.5137 | 1.4628 | `ears agg.ours_es_dec_erb_l0.1.crps_fair` |
| ceiling SNR (12 kHz input) | 22.0162 | 21.1551 | `ears agg.ceiling.snr` |
| det_paper SNR below ceiling | 0.1507 | 0.1862 | `ears agg.ours_det_paper.vs_ceiling` |
| det_paper band short | 21.7803 | 18.1317 | `ears agg.ours_det_paper.deficit` |
| log-mag term recovers | 15.6008 | 15.4361 | `ears deficit ours_det - ours_det_paper` |
| energy score recovers more | 5.1303 | 5.9681 | `ears deficit ours_es_marg - ours_det` |
| NU-Wave 2 broadband deficit | -8.9284 | -9.1494 | `ears agg.nuwave2.deficit_bb` |
| max coherent fraction c | 0.0164 | 0.0216 | `ears max agg.*.coh` |
| NU-Wave 2 loud frames | -8.9576 | -5.8209 | `ears agg.nuwave2.loud` |
| NU-Wave 2 quiet frames | 4.7209 | 10.8135 | `ears agg.nuwave2.quiet` |
| NU-Wave 2 spread | 0.0937 | 0.0850 | `ears agg.nuwave2.spread_pooled_hb` |
| AudioSR level at 10.7 kHz | 4.9486 | 2.2295 | `ears agg.audiosr.curve[2]` |
| AudioSR level at 21.4 kHz | -11.1080 | -1.0029 | `ears agg.audiosr.curve[5]` |
| ERB corr err es_marg | 0.3739 | 1.0905 | `ears agg.ours_es_marg.corr_err_d0` |
| ERB corr err es_erb | 0.2890 | 0.9741 | `ears agg.ours_es_erb_l0.01.corr_err_d0` |
| flowhigh_std1 pit_lo % | [40.78, 38.66, 42.92] | [65.5, 64.38, 66.58] | `bootstrap flowhigh_std1 pit_lo point, 95% CI` |
| flowhigh_std1 pit_hi % | [45.95, 43.78, 48.12] | [23.21, 22.19, 24.28] | `bootstrap flowhigh_std1 pit_hi point, 95% CI` |
| audiosr pit_hi % | [51.98, 50.17, 53.97] | [19.55, 18.31, 20.83] | `bootstrap audiosr pit_hi point, 95% CI` |
| ours_es_dec_erb_l0.1 pit_hi % | [21.21, 18.35, 24.26] | [6.429, 6.088, 6.79] | `bootstrap ours_es_dec_erb_l0.1 pit_hi point, 95% CI` |
| ours_es_dec_erb_l0.1 pit_lo % | [10.71, 9.245, 12.19] | [47.87, 46.84, 48.91] | `bootstrap ours_es_dec_erb_l0.1 pit_lo point, 95% CI` |
| nuwave2 pit_lo % | [13.18, 10.04, 16.48] | [51.1, 49.4, 52.78] | `bootstrap nuwave2 pit_lo point, 95% CI` |
| nuwave2 pit_hi % | [21.71, 18.25, 25.35] | [11.98, 11.16, 12.81] | `bootstrap nuwave2 pit_hi point, 95% CI` |
| flowhigh_std1 gap - indep | [-1.687, -1.994, -1.398] | [-1.418, -1.535, -1.303] | `bootstrap flowhigh_std1 gap_minus_indep point, 95% CI` |
| nuwave2 gap - indep | [-0.05512, -0.07335, -0.03865] | [-0.05468, -0.06214, -0.04759] | `bootstrap nuwave2 gap_minus_indep point, 95% CI` |
| audiosr gap - indep | [-1.296, -1.507, -1.086] | [-1.314, -1.443, -1.189] | `bootstrap audiosr gap_minus_indep point, 95% CI` |
| ours_es_dec_erb_l0.1 gap - indep | [-0.296, -0.389, -0.2185] | [-0.2665, -0.294, -0.2396] | `bootstrap ours_es_dec_erb_l0.1 gap_minus_indep point, 95% CI` |
| ours_es_marg gap - indep | [-0.1532, -0.2245, -0.0957] | [-0.1268, -0.1438, -0.1103] | `bootstrap ours_es_marg gap_minus_indep point, 95% CI` |
| ours_es_erb_l0.01 gap - indep | [-0.1153, -0.1655, -0.07391] | [-0.09221, -0.1038, -0.08111] | `bootstrap ours_es_erb_l0.01 gap_minus_indep point, 95% CI` |
| ours_es_erb_l0.001 gap - indep | [-0.2312, -0.3003, -0.1744] | [-0.2165, -0.238, -0.1954] | `bootstrap ours_es_erb_l0.001 gap_minus_indep point, 95% CI` |
| FLowHigh as released: draw spread, waveform rel. RMS | 2.32e-06 | 0.0003334 | `draw_agreement flowhigh wave_rel median` |
| FLowHigh prior restored: HB rel. RMS | 0.1920 | 0.1897 | `draw_agreement flowhigh_std1 hb_rel median` |
| NU-Wave 2: HB rel. RMS | 1.9037 | 1.7168 | `draw_agreement nuwave2 hb_rel median` |
| slope FLowHigh | -0.0753 | -0.1433 | `level_checks slope.flowhigh` |
| slope AP-BWE | -0.1188 | -0.2004 | `level_checks slope.apbwe` |
| slope NU-Wave 2 | -0.3149 | -0.4077 | `level_checks slope.nuwave2` |
| slope AudioSR | -0.3383 | -0.2677 | `level_checks slope.audiosr` |
| slope ours range | [-0.1964, -0.3457] | [-0.2724, -0.4061] | `level_checks slope.ours_<six Table 2 arms>` |
| n levels / calibration | [240, 32] | [96, 96] | `_meta.n_utts` |

## Numbers that did not come from EARS

- VCTK harness reproduction (Setup): 0.761/3.61 FLowHigh, 0.780/3.48 AP-BWE, 0.989/2.57 NU-Wave 2. Unchanged; still VCTK.
- NU-Wave 2's own SNR bound: C − S = 0.5 dB gives at most −8.5 dB, and −8.9 dB measured broadband on its VCTK split. Unchanged; VCTK by nature. EARS adds −9.1 dB.
- 2.50 dB at M = 8: theory, unchanged.
- 104,950 training steps, 88k parameters, 8 + 4 noise channels: unchanged.

## Wording changes that follow from EARS

- Abstract: "FLowHigh ranks first on LSD and ViSQOL" removed. AudioSR leads on EARS (1.208 / 3.09 against FLowHigh 1.610 / 2.99).
- Abstract: "8–9 dB too quietly" → "9 dB too quietly" (broadband −9.1 dB; band-averaged −4.86).
- Abstract: "AudioSR is biased by band" removed. On EARS its third-octave levels run +2.8 to −1.0 dB.
- Abstract: "closed by an energy score" → "an energy score fills the band". On EARS it overshoots: +3.37 dB band-averaged, −0.84 dB broadband.
- "Every model has c < 0.02" → "c ≤ 0.022". Max is ours_det, 0.0216.
- FLowHigh: "agree to 2×10⁻⁶" → "3×10⁻⁴". The released script on EARS draws on CUDA.
- FLowHigh: "a tenth as much as NU-Wave 2's" → "a ninth". HB relative RMS 0.190 against 1.717.
- AudioSR paragraph: band bias removed. It now states the lead on LSD/ViSQOL and both tails (29% below, 20% above).
- What Sets the Deficit: new sentence on the overshoot. Every Table 1 model but AudioSR over-fills the top third octave: ours +11.3 dB and AP-BWE +14.9 dB at 21 kHz, FLowHigh +9.7, NU-Wave 2 +7.6.
- Our sampler's tails flip. Was 21% above all draws with the lower tail ideal; now 48% below and 6% above. Stated as against the prediction.
- Setup: EARS described. "We wrote seven predictions before running EARS; four hold."
- Conclusion: rankings move with the corpus (FLowHigh leads on VCTK, AudioSR on EARS). Limitations now say: six test speakers, network trained on VCTK only, three of seven predictions failed.
- refs.bib: Richter et al. 2024 (EARS) added.

## Predictions (written before the run; `supplementary/results/ears/ears_predictions.json`)

1. **PASS**: FLowHigh (prior restored) is still a regressor: gap < indep/2, tails sum > 50%. gap 0.064, indep 1.482, tails 65.5% + 23.2%
2. **PASS**: NU-Wave 2 samples at its own level: |diff| < 0.2 dB, deficit < -4 dB. diff -0.055 [-0.062, -0.048], deficit -4.86
3. **FAIL**: AudioSR is under-dispersed: diff < -0.5 dB with interval below 0, upper tail > 30%. diff -1.314 [-1.443, -1.189], upper tail 19.5%
4. **FAIL**: Our sampler: -0.6 < diff < 0 dB, upper tail > lower tail, lower tail in [7%, 15%]. diff -0.266 [-0.294, -0.240], tails 47.9% / 6.4%
5. **FAIL**: Loss domain sets the deficit: det_paper < -15 dB, det >= det_paper + 8, es_marg and es_dec_erb within 3 dB of 0. det_paper -18.13, det -2.70, es_marg 3.27, es_dec_erb 3.37
6. **PASS**: Level regresses per utterance: every slope negative, interval below 0. flowhigh -0.143 [-0.236, -0.068]; flowhigh_std1 -0.159 [-0.247, -0.086]; nuwave2 -0.408 [-0.507, -0.300]; audiosr -0.268 [-0.366, -0.177]; apbwe -0.200 [-0.293, -0.112]; ours_det_paper -0.406 [-0.485, -0.337]; ours_det -0.285 [-0.379, -0.194]; ours_es_marg -0.298 [-0.388, -0.212]; ours_es_erb_l0.01 -0.303 [-0.398, -0.202]; ours_es_erb_l0.001 -0.272 [-0.376, -0.183]; ours_es_dec_erb_l0.1 -0.282 [-0.379, -0.187]
7. **PASS**: FLowHigh as released is a point estimate: gap < 0.01 dB. gap 3.35e-05
