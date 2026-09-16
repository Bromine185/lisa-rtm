# LISA energy-score sampler: tonight's experiment menu

Scope: 88k-param LISA, 12→48 kHz VCTK, energy score at M=2, d = L1 waveform + 0.01·multi-scale log-mag L1. Symptom: per-bin PIT flat, CRPS-optimal τ=1, but high-band (HB, >6 kHz) energy −7.9 dB in-distribution; single draw loses to deterministic on LSD (1024/256, log10 power) and ViSQOL-audio.

Diagnosis the three reports converge on: the M=2 estimator is unbiased, so the deficit is not the estimator. It is a **bias in an aggregate** (band energy) that a marginal-proper score on bins does not constrain (FCN3's argument), driven by a distance d that is waveform-L1-dominated in a band that is 98 % incoherent with the input: the attractive term is minimised by a low-energy median-like output, and the 0.01 spectral weight cannot pull energy back. Temperature moves spread, not bias, so τ cannot fix it. Both metric losses are mostly level, not texture: a −7.9 dB bias over 75 % of bins alone contributes ≈0.6 to LSD, and ViSQOL's NSIM luminance term punishes under-levelled bands.

## A. Candidate interventions

Legend: CRPS column refers to CRPS on log-magnitude bins (current key metric) plus PIT flatness. "+" better, "0" none, "−" worse, "?" unknown. Cost: retrain ≈ one A100 run.

| # | Intervention | Source | What changes | CRPS / calibration | LSD + ViSQOL-a | Cost | Falsifiable prediction |
|---|---|---|---|---|---|---|---|
| 1 | Spectral-dominant d; waveform L1 only below 6 kHz | [Gritsenko 2008.01160](https://arxiv.org/abs/2008.01160) eq. 9; weights from [LACE 2307.06610](https://arxiv.org/abs/2307.06610), [BAE-Net 2312.13722](https://arxiv.org/abs/2312.13722) | objective | 0/+ on log-mag CRPS; waveform CRPS may drop | HB gap −7.9→±1.5 dB; LSD −0.05..−0.10; ViSQOL +0.2 | retrain | Gap >3 dB after this ⇒ d is not the cause |
| 2 | Fair-CRPS term on per-frame HB energy (dB) | [2407.00650](https://arxiv.org/abs/2407.00650) Prop. 1–2; [FCN3 2507.12144](https://arxiv.org/abs/2507.12144) | objective (add proper term) | + on energy PIT; per-bin PIT stays flat | LSD −0.1 (level); ViSQOL +0.3 | retrain | Energy-ratio PIT flattens; if per-bin PIT bends, weight too high |
| 3 | ERB/gammatone log-band-energy term in d | [ViSQOL v3 2004.09584](https://arxiv.org/abs/2004.09584); LACE L_env | objective | 0 | ViSQOL 2.6–2.9→≥3.1; LSD flat | retrain | ViSQOL up but CRPS down ⇒ level gaming |
| 4 | Noise into implicit MLP via FiLM/AdaLN | [AIFS-CRPS 2412.15832](https://arxiv.org/abs/2412.15832); [SLED 2505.13181](https://arxiv.org/abs/2505.13181); [EnScale 2509.26258](https://arxiv.org/abs/2509.26258) | architecture (+≈2k params) | + spread; τ*=1 kept | gap closes ≥3 dB | retrain | Spread unchanged ⇒ input noise was not the bottleneck |
| 5 | Variogram term p=0.5 across bins per frame, log-mag | [2407.00650](https://arxiv.org/abs/2407.00650); [Scheuerer–Hamill 2015](https://journals.ametsoc.org/view/journals/mwre/143/4/mwr-d-14-00269.1.xml) | objective (proper, not strict) | 0 | LSD −0.03; ViSQOL 0/+ | retrain | No LSD gain ⇒ inter-bin texture already right |
| 6 | M=4 draws, fair estimator | [Engression 2307.00835](https://arxiv.org/abs/2307.00835) eq. 13; [Pacchiardi 2112.08217](https://arxiv.org/abs/2112.08217) | estimator variance | <3 % CRPS gain | 0 | 2× fwd/step | Gap unchanged ⇒ rules out estimator variance |
| 7 | afCRPS α=0.95 (repulsive coeff 0.4875) | [2412.15832](https://arxiv.org/abs/2412.15832); [2506.10868](https://arxiv.org/abs/2506.10868) | objective | stability only; slightly less spread | 0 | free | Only matters if a member collapses onto y |
| 8 | Patched energy score (local windows + global term) | [2112.08217](https://arxiv.org/abs/2112.08217) | objective | + (WB calib err 0.086→0.055) | ? | retrain | Calibration on band energy improves, LSD flat |
| 9 | Folded/rectified baseband as extra input channel | [Opus BWE 2412.11392](https://arxiv.org/abs/2412.11392); [1602.08215](https://arxiv.org/abs/1602.08215) | architecture (0 params) | 0 | LSD-HF −0.05; ViSQOL +0.1 | retrain | Coherent fraction stays ≤2 %; no metallic MOS penalty |
| 10 | SBR-style gain head: g_tonal·fold + g_noise·noise from ERB envelope | Opus BWE; SBR (Dietz 2002) | architecture (small) | ? | ViSQOL +0.1 over #9 | retrain | Gains saturate at 0/1 ⇒ already learned implicitly |
| 11 | 2D-conv log-mag discriminator (MRAD), hinge 0.1 | [AP-BWE 2401.06387](https://arxiv.org/abs/2401.06387) Tab. VIII; [AERO 2211.12232](https://arxiv.org/abs/2211.12232) | objective (adversarial) | − risk: sharpens, PIT humps | LSD −0.04; ViSQOL +0.1 | ≈2× train time | Waveform MSD instead ⇒ LSD +0.06 (AERO) |
| 12 | Differentiable NSIM or λ·LSD aux loss | [FSC-Net 2606.06962](https://arxiv.org/abs/2606.06962); [NOMAD 2309.16284](https://arxiv.org/abs/2309.16284) | objective (improper) | − risk | ViSQOL +0.1..0.2 | retrain | ViSQOL up, CRPS/MUSHRA down ⇒ AERO failure mode |
| 13 | Linkwitz-Riley polynomial crossover | [Vocos-BWE 2603.07285](https://arxiv.org/abs/2603.07285) | inference | 0 | LSD −0.01..−0.03 | free | — |
| 14 | Snake + 2× anti-aliased nonlinearity | [BigVGAN 2206.04658](https://arxiv.org/abs/2206.04658); AERO Tab. 4 | architecture | 0 | ≤0.02 LSD | retrain | Worth it only if aliasing >20 kHz is visible |
| 15 | Post-hoc per-ERB-band gain on HB, fit on validation | SBR envelope adjustment; LACE | inference (0 params) | 0 if applied to all members | LSD −0.15 in-dist; ViSQOL +0.3; <0.02 on +0.6 dB set | zero | Bias is input-dependent ⇒ needs #2/#18 |
| 16 | Geometric-mean magnitude readout | [Gneiting 0912.0902](https://arxiv.org/abs/0912.0902); Ephraim–Malah LSA 1985 | readout | ensemble untouched | LSD < deterministic, falls with M | M fwd | Waveform mean's LSD *rises* ≈10·log10 M dB |
| 17 | Envelope from ensemble, fine structure from one draw | [Blau–Michaeli 1711.06077](https://arxiv.org/abs/1711.06077); Jax 2004; NSIM | readout | ensemble untouched | ViSQOL +0.3; LSD between #16 and single draw | M fwd | — |
| 18 | Per-band MBM affine on log-mag fitted by fair CRPS | [2106.09512](https://arxiv.org/abs/2106.09512); [Van Schaeybroeck–Vannitsem qj.2397](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.2397); [EMOS](https://journals.ametsoc.org/view/journals/mwre/133/5/mwr2904.1.xml) | calibration repair | + strictly (proper fit) | gap→±1 dB; LSD −0.1 | one fit | a_k≈0 above 6 kHz on the +0.6 dB set |
| 19 | MBR medoid under d among M draws | [ArtiFree 2509.19495](https://arxiv.org/abs/2509.19495); [2510.19471](https://arxiv.org/abs/2510.19471) | readout | selected PIT hump-shaped | LSD −0.05..−0.1; ViSQOL +0.1..0.2, saturates M≈8 | M fwd | Report CRPS on unselected ensemble |
| 20 | Per-band τ_k so spread/skill = √(M/(M+1)) | [FlowDec 2503.01485](https://arxiv.org/abs/2503.01485); [ArchesWeatherGen 2412.12971](https://arxiv.org/abs/2412.12971) | readout | <3 % | 0 | free | Gap unchanged |
| 21 | Global τ sweep (control) | [Glow 1807.03039](https://arxiv.org/abs/1807.03039); [WaveGlow 1811.00002](https://arxiv.org/abs/1811.00002) | readout | CRPS min at τ=1 | τ<1 worsens LSD | free | Neither direction fixes the bias |
| 22 | PMRF-style dial: (1−λ)·#16 + λ·draw | [Freirich 2107.02555](https://arxiv.org/abs/2107.02555); [PMRF 2410.00418](https://arxiv.org/abs/2410.00418) | readout | — | traces D(P) curve | M fwd | Flat curve on HB (see D) |
| 23 | Light GAN on top of GED | Gritsenko | objective | ? | +0.19 MOS (their TTS) | high | Not tonight |

## B. Six paired arms, priority order

Each arm: control = current model, treatment = one change, same batches, same seeds, same steps. Record predictions before launch. Notation: S_k = magnitude STFT with window k ∈ {64,128,…,2048}, 50 % overlap; ε = 1e-5 inside logs; y = target, y1,y2 = two draws; HB = 6–24 kHz.

**Arm 1. Spectral-dominant d.**
d(a,b) = ‖h_lp∗(a−b)‖₁ + Σ_k [ ‖S_k(a)−S_k(b)‖₁ + √(k/2)·‖log(S_k(a)+ε)−log(S_k(b)+ε)‖₂ ] / N_k,
h_lp = 6 kHz low-pass, N_k = number of STFT cells, spectral sum normalised so it is ≈5× the waveform term at step 0. Loss = ½[d(y,y1)+d(y,y2)] − ½d(y1,y2), unchanged.
Prediction: HB energy gap −7.9 dB → within ±1.5 dB on both sets; single-draw LSD ≤0.90; ViSQOL-a +0.2; log-mag CRPS equal or better; waveform CRPS below 6 kHz unchanged. If the gap stays above 3 dB, d is not the cause and arm 4 becomes top.

**Arm 2. Aggregate-proper term: fair CRPS on band energy.** Keep current d, add
L_E = w·Σ_t Σ_b { ½[|E_tb(y)−E_tb(y1)| + |E_tb(y)−E_tb(y2)|] − ½|E_tb(y1)−E_tb(y2)| },
E_tb = 10·log10 of energy in ERB band b (8 bands, 6–24 kHz) at frame t (20 ms). Proper by the transformation principle (Prop. 1) and the weighted-sum principle (Prop. 2). w set so L_E ≈ 0.3·L at step 0.
Prediction: PIT of E_tb flattens (U-shape gone); per-bin PIT stays flat; gap → ±1 dB; LSD −0.10; ViSQOL +0.3. If per-bin PIT bends, w too high.

**Arm 3. ViSQOL-aligned feature term.** Keep d; add d_G(a,b) = ‖log(G(a)+ε) − log(G(b)+ε)‖₁ with G = 32-band gammatone (ERB-spaced, 48 kHz) energy envelope at 10 ms hop, inside the same fair estimator. Weight so d_G ≈ d at step 0.
Prediction: ViSQOL-a single draw ≥3.1; LSD flat (±0.02); CRPS unchanged. ViSQOL up with CRPS down ⇒ level gaming; discard.

**Arm 4. Noise via FiLM into the implicit decoder.** Keep the 8 input noise channels; add z ~ N(0, I₈) per frame → 2-layer MLP (8→32→2·H_l per layer) → (γ_l, β_l); decoder layer l becomes h_l = γ_l ⊙ ReLU(W_l h_{l−1} + b_l) + β_l, γ_l initialised at 1, β_l at 0. Adds ≈2k params.
Prediction: single-draw HB spread (ensemble std of log-mag) rises ≥30 %; CRPS-optimal τ stays at 1; gap closes ≥3 dB. Spread unchanged ⇒ noise path is not the bottleneck; drop.

**Arm 5. Folded baseband input channel.** Add one channel x_fold = x·sin(log(|x|+ε)) (Opus BWE) beside the 8 noise channels; nothing else changes.
Prediction: LSD-HF −0.05; ViSQOL +0.1; coherent fraction stays ≤2 %; no CRPS change. If gains appear only in voiced frames, add rectified |x| for unvoiced (arm 5b).

**Arm 6. Variogram term across bins.** Add
L_V = w_V·Σ_t Σ_{i<j, |i−j|≤4} ( ½Σ_m |X_{m,i}−X_{m,j}|^{0.5} − |y_i−y_j|^{0.5} )²,
X_{m,i} = log|STFT| (1024/256) of draw m, bin i, frame t; bins above 6 kHz only; w_V so L_V ≈ 0.1·L at step 0.
Prediction: LSD −0.03 with no CRPS loss; ViSQOL flat. This targets inter-bin texture the energy score barely sees.

Stacking rule for night 2: 1+2, then 1+2+3. Run #6 from the table (M=4) as a control only if GPU time remains; its expected gain is <3 %.

## C. Three readouts from an M=16 ensemble, no retraining

Compute STFT X_m (1024/256) of each draw m. Keep the raw ensemble for CRPS and PIT; the readouts are point estimates and must not replace the ensemble in calibration reporting.

**R1. Geometric-mean magnitude (LSA readout).**
|X̂_k| = exp( (1/M) Σ_m log|X_{m,k}| ), phase φ_k from draw 1, iSTFT, then baseband passthrough. Bayes-optimal for squared error in log power, which is what LSD is.
Prediction: LSD below the deterministic model, decreasing in M as Var(log|X|)/M; the waveform ensemble mean's LSD *rises* with M by ≈10·log10 M dB of HB energy loss. Report both curves for M ∈ {1,2,4,8,16}.

**R2. Envelope from ensemble, fine structure from one draw.**
Ê_b(t) = (1/M) Σ_m log G_b(X_m) over 32 gammatone bands; output = draw 1 with per-band gain exp(Ê_b − log G_b(X_1)) applied on the STFT, smoothed across band edges (Linkwitz-Riley polynomial mask, #13).
Prediction: ViSQOL-a ≥ single draw + 0.3 (luminance term fixed, structure term untouched); LSD between R1 and a single draw; the output remains a single sample-like texture, so a listening test should not report the AERO "smooth" artefact.

**R3. Per-band member-by-member calibration, fitted by fair CRPS.**
For band b: X̃_{m,b} = a_b + β_b·X̄_b + γ_b·(X_{m,b} − X̄_b), on log-mag, X̄ = ensemble mean; fit (a_b, β_b, γ_b) on a held-out split by minimising fair CRPS of E_tb. Apply to all 16 members.
Prediction: HB gap → ±1 dB; CRPS strictly decreases (proper fit); PIT stays flat; on the +0.6 dB set a_b ≈ 0 above 6 kHz, so the correction is set-specific bias, not a universal gain.

Also log, as diagnostics: per-band spread/skill vs √(M/(M+1)), rank histogram of E_tb, and the PMRF dial (#22) at λ ∈ {0, 0.25, 0.5, 1}.

## D. Most novel defensible idea

**Claim: for an input-incoherent band, the perception–distortion trade-off is free, and the sampler should be trained and read out to exploit that.**

Blau–Michaeli and Freirich prove a convex, non-increasing D(P) with posterior sampling costing up to 2× the MMSE distortion. That result assumes the distortion measure sees the same components as the perceptual index. Above 6 kHz in speech BWE the fine structure is ≤2 % coherent with the input, so LSD and NSIM (both envelope-type statistics on log-magnitude) are almost invariant to the fine structure, while the texture carries all the perceptual index. The conditional law is then well approximated as p(envelope | x) × p(texture | envelope), and the two can be scored and read out separately:

1. Train with a proper score on the transform T(y) = (ERB log-envelope at 100 Hz, multi-scale log-STFT): fair energy score on the second (Gritsenko) plus fair CRPS on the first (arm 2). The sum is proper (Prop. 2) and the aggregate term constrains exactly the functional a bin-marginal score leaves free (FCN3's argument, moved from spherical harmonics to gammatone bands).
2. At inference, output envelope = conditional mean of the log-envelope over M draws (Bayes readout for LSD and NSIM luminance, Gneiting consistency, Ephraim–Malah LSA) and texture = one draw (keeps NSIM structure and the calibrated ensemble). This is R2 applied to a model trained under (1).

Why new relative to prior art in the reports: FCN3's aggregate-CRPS lives in weather; Gritsenko's GED has no aggregate term and no readout theory; PMRF/Freirich assume a distortion that sees everything; SBR/Jax decompose envelope and excitation but without a proper score; no report found a paper that derives the ensemble readout for NSIM or trains an audio sampler with an aggregate-proper term.

Quantitative prediction: on the >6 kHz band, the PMRF dial (#22) traces a D(P) curve that is flat within 0.02 LSD and 0.1 ViSQOL from λ=0 to λ=1, while below 6 kHz it is convex as in the theorem. The combined model plus R2 should reach LSD ≤0.85 and ViSQOL-a ≥3.3 at CRPS ≤ baseline.

What refutes it: (a) the D(P) curve on the HB is not flat (LSD of a single draw worse than R1 by >0.05 after arm 2); (b) R2 does not lift ViSQOL by ≥0.2, showing NSIM's structure term does depend on texture; (c) arm 2's energy-PIT flattens but per-bin PIT bends, meaning the decomposition is not proper in practice; (d) a MUSHRA panel prefers the ensemble-mean waveform to R2, meaning texture is not what carries perception here.

## E. Attributions any write-up must cite

- Energy score, propriety, transformation and weighted-sum principles: Gneiting & Raftery 2007 (JASA); [2407.00650](https://arxiv.org/abs/2407.00650); [Pacchiardi et al. 2112.08217](https://arxiv.org/abs/2112.08217) (patched/kernel scores, calibration numbers).
- Consistent scoring functions / which functional a metric elicits: [Gneiting 2011, 0912.0902](https://arxiv.org/abs/0912.0902).
- Two-draw unbiased energy-score estimator: [Gritsenko et al. 2020, 2008.01160](https://arxiv.org/abs/2008.01160) (spectral energy distance, eq. 9, ablations); [Engression, 2307.00835](https://arxiv.org/abs/2307.00835) eq. 13; [DISCO Nets, 1606.02556](https://arxiv.org/abs/1606.02556).
- Fair / almost-fair CRPS and noise via conditional norms: [AIFS-CRPS, 2412.15832](https://arxiv.org/abs/2412.15832); multi-scale afCRPS [2506.10868](https://arxiv.org/abs/2506.10868); fair-score over-dispersion [2602.15830](https://arxiv.org/abs/2602.15830).
- Aggregate/spectral CRPS motivation: [FourCastNet 3, 2507.12144](https://arxiv.org/abs/2507.12144).
- Variogram score: Scheuerer & Hamill 2015, MWR 143(4).
- Perception–distortion: [Blau & Michaeli 1711.06077](https://arxiv.org/abs/1711.06077); [Freirich, Michaeli, Meir 2107.02555](https://arxiv.org/abs/2107.02555); [PMRF 2410.00418](https://arxiv.org/abs/2410.00418); [Ohayon et al. 2211.08944](https://arxiv.org/abs/2211.08944); [PSCGAN 2103.04192](https://arxiv.org/abs/2103.04192).
- Log-spectral-amplitude estimator: Ephraim & Malah 1985 (IEEE TASSP 33(2)).
- Ensemble spread–skill law: Fortin et al. 2014 (J. Hydrometeor. 15(4)); Leutbecher 2019 (QJRMS, qj.3387).
- Post-hoc calibration: Gneiting et al. 2005 EMOS (MWR 133(5)); Van Schaeybroeck & Vannitsem 2015 MBM (qj.2397); [2106.09512](https://arxiv.org/abs/2106.09512); Kuleshov et al. 2018 (ICML).
- Noise temperature / per-band noise: [Glow 1807.03039](https://arxiv.org/abs/1807.03039); [WaveGlow 1811.00002](https://arxiv.org/abs/1811.00002); [ArchesWeatherGen 2412.12971](https://arxiv.org/abs/2412.12971); [FlowDec 2503.01485](https://arxiv.org/abs/2503.01485); [PriorGrad 2106.06406](https://arxiv.org/abs/2106.06406).
- Sample selection: [ArtiFree 2509.19495](https://arxiv.org/abs/2509.19495).
- Base model: [LISA, Kim et al. ICASSP 2022, 2111.00195](https://arxiv.org/abs/2111.00195).
- BWE baselines and LSD basis: [NU-Wave 2, 2206.08545](https://arxiv.org/abs/2206.08545) (same 1024/256 basis); [AP-BWE 2401.06387](https://arxiv.org/abs/2401.06387); [AERO 2211.12232](https://arxiv.org/abs/2211.12232); [FlowHigh 2501.04926](https://arxiv.org/abs/2501.04926); [Vocos-BWE 2603.07285](https://arxiv.org/abs/2603.07285); [NVSR 2203.14941](https://arxiv.org/abs/2203.14941); [DDSP-BWE 2311.07363](https://arxiv.org/abs/2311.07363).
- Envelope/excitation and folding: [LACE 2307.06610](https://arxiv.org/abs/2307.06610); [Opus BWE 2412.11392](https://arxiv.org/abs/2412.11392); Makhoul & Berouti 1979 via [1602.08215](https://arxiv.org/abs/1602.08215); Dietz et al. 2002 SBR (AES 112); Jax & Vary 2004.
- ViSQOL / NSIM: [ViSQOL v3, 2004.09584](https://arxiv.org/abs/2004.09584); Hines & Harte 2012 (Speech Communication); [NOMAD 2309.16284](https://arxiv.org/abs/2309.16284).
- Noise placement in one-step samplers: [SLED 2505.13181](https://arxiv.org/abs/2505.13181); [EnScale 2509.26258](https://arxiv.org/abs/2509.26258); [EAR 2505.07812](https://arxiv.org/abs/2505.07812).

Unverified items carried over from the reports (do not cite as read): FCN3 App. E.1 exact spectral-CRPS equation; Ephraim–Malah, Fortin, Leutbecher, Kuleshov, Scheuerer–Hamill full texts; SBR/IGF/PNS primary texts; Jax 2004 quote; ViSQOL 2015 window constants.