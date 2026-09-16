# Proper-score training tricks for the LISA sampler: verified findings

## 0. Where your current loss sits

Your loss `L = 1/2[d(y,y1)+d(y,y2)] - 1/2 d(y1,y2)` is exactly the **fair (unbiased) energy-score estimator at M=2**. AIFS-CRPS eq. (2): `fCRPS = 1/M sum_j |x_j-y| - 1/(2M(M-1)) sum_j sum_k |x_j-x_k|`; for M=2 the double sum is `2|x1-x2|`, giving coefficient 1/2 ([2412.15832](https://arxiv.org/html/2412.15832)). Gritsenko's eq. (7)-(8), `L_GED = E[2 d(x,y) - d(y,y')]`, is the same thing times 2 and is stated to be "an unbiased estimator of the energy score" ([2008.01160](https://ar5iv.labs.arxiv.org/html/2008.01160)). Engression eq. (13) gives the general-m unbiased form `1/m sum_j ||Y-g(X,eps_j)|| - 1/(2m(m-1)) sum_{j!=j'} ||g(X,eps_j)-g(X,eps_j')||` ([2307.00835](https://arxiv.org/html/2307.00835)). So the estimator is not biased toward under-dispersion. The under-dispersion must come from (a) the distance `d`, (b) what the score is insensitive to, or (c) the noise path.

## 1. AIFS-CRPS and almost-fair CRPS

- Biased CRPS (eq. 1): `CRPS = 1/M sum|x_j-y| - 1/(2M^2) sum_j sum_k |x_j-x_k|`. Fair (eq. 2) as above. Almost-fair (eq. 3): `afCRPS_alpha = alpha*fCRPS + (1-alpha)*CRPS`, alpha in (0,1], **alpha=0.95** used; 4 members at O96, **2 members at N320** "to mitigate costs" ([2412.15832](https://arxiv.org/html/2412.15832)).
- Closed form from the multi-scale paper: `afCRPS_alpha = 1/M sum_j |x_j-y| - (M-1+alpha)/(2M^2(M-1)) sum_j sum_k |x_j-x_k|` ([2506.10868](https://arxiv.org/html/2506.10868)). For M=2, alpha=0.95 the repulsive coefficient on `|x1-x2|` is **0.4875 vs 0.5 fair vs 0.25 biased**.
- Why fair degenerates: "the fCRPS suffers from a degeneracy in the case where all members apart from one have the same value as the verifying observation. In this case the remaining member is unconstrained" ([2412.15832](https://arxiv.org/html/2412.15832)). With M=2 this is: if y1=y then `1/2|y2-y| - 1/2|y1-y2| = 0` for any y2. afCRPS *reduces* spread pressure slightly; it is a stability fix, not an anti-under-dispersion fix. Do not expect it to lift your -7.9 dB.
- Dispersion result: AIFS-CRPS members show "no visible blurring", "no dampening of smaller scales with lead time", and are "over-dispersive for some variables, such as geopotential at 500 hPa" ([2412.15832](https://arxiv.org/html/2412.15832)). Noise enters not at the input but via "conditional layer normalisations ... which replace all standard layer normalisations in the processor" after a 2-layer MLP ([2412.15832](https://arxiv.org/html/2412.15832)).
- Multi-scale afCRPS (2506.10868) is **not spectral**: eq. (2) `L = sum_i zeta_i * afCRPS(smoothed members, smoothed truth)` with two scales, zeta_i=1, Gaussian kernel sigma = 8 grid spacings; it "better constrains small scale variability without negatively impacting forecast skill" ([2506.10868](https://arxiv.org/html/2506.10868)). Numbers on spread not given.

## 2. FourCastNet 3

FCN3 trains with 16 members and "a composite probabilistic loss ... combin[ing] the spatial, point-wise CRPS loss term with a loss term in the spectral domain", weighting "spectral coefficients according to their multiplicity" and enforcing "a good match of their distributions across all wavelengths". Motivation, verbatim: point-wise CRPS aggregated over marginals is not proper for the joint, and "the CRPS can be minimized in a point-wise manner by an unphysical ensemble". Result: spread-skill "approaching 1", slightly over-dispersive at short leads, and FCN3 "retains perfectly the correct slopes in the power spectra" at 30 days ([2507.12144](https://arxiv.org/html/2507.12144)). **Could not verify** the exact spectral-CRPS equation (Appendix E.1, eq. 87): three fetches of the HTML did not return it. Treat "CRPS on each SH coefficient, weighted by multiplicity, added to spatial CRPS" as the verified content.

This is the direct analogue of your symptom: PIT flat per log-mag bin while band energy (an aggregate) is wrong. A marginal-proper score does not constrain aggregates.

## 3. Gritsenko et al. spectral energy distance

- Eq. (6) `D^2_GED(p|q) = E[2d(x,y) - d(x,x') - d(y,y')]`; training loss drops the data term: eq. (7)-(8) `L*_GED = sum_i [2 d(x_i,y_i) - d(y_i,y_i')]`, two independent draws per conditioning, unbiased ([2008.01160](https://ar5iv.labs.arxiv.org/html/2008.01160)).
- Eq. (9) distance: `d(x_i,x_j) = sum_{k in {2^6..2^11}} sum_t [ ||s^k_t(x_i) - s^k_t(x_j)||_1 + alpha_k ||log s^k_t(x_i) - log s^k_t(x_j)||_2 ]`, magnitude STFT, 50% overlap, `alpha_k = sqrt(k/2)` chosen to "approximately equalize the influence of all the different L1 and L2 terms" ([2008.01160v2](https://arxiv.org/html/2008.01160v2)). No epsilon inside the log is stated (unverified). **No waveform term at all.**
- Properness: strictly proper "with respect to p(s^k_t)" but "not necessarily" for p(x|c) because of long-range dependencies (Appendix A).
- Ablations: removing the repulsive term: MOS 3.00+-0.07 vs 4.06+-0.06 (r+m), cFDSD 0.047 vs 0.040, speech "metallic"; adding an unconditional GAN: 4.25+-0.06, cFDSD 0.033; GAN-TTS re-implementation 4.16, cFDSD 0.077 ([2008.01160](https://ar5iv.labs.arxiv.org/html/2008.01160)).
- Noise placement: only `y_i = f_theta(c_i, z_i)` is stated; the appendix does not say where z enters the 77-block GAN-TTS-style generator (unverified).

Inference for your model (mine, flagged): your `d` is waveform-L1 + 0.01 * spectral. Above 6 kHz the target is incoherent with the input (coherent fraction <= 2 %), so the waveform-L1 attractive term is minimised by a low-energy median-like output in that band, while the tiny spectral weight and the repulsive term cannot pull the energy back. That predicts exactly an energy deficit with a still-flat per-bin PIT.

## 4. Score families

- Energy score of order alpha/beta: `ES_alpha(F,y) = E||X-y||^alpha - 1/2 E||X-X'||^alpha`, alpha in (0,2), strictly proper; alpha=2 only proper ([2407.00650](https://arxiv.org/html/2407.00650); [2112.08217](https://arxiv.org/html/2112.08217)). EAR ablation (ImageNet FID/IS): alpha=1.0 3.55/230.3, 1.25 3.73/223.1, 1.5 4.10/212.1, 1.75 4.32/204.2, **2.0 188.1/6.4**; alpha<1 "invariably induce[s] rapid training collapse due to gradient instability" ([2505.07812](https://arxiv.org/html/2505.07812)). So lowering alpha below 1 is not a sharpening tool.
- DISCO Nets eq. (4): `F = DIV(P,Q) - gamma DIV(Q,Q)`, gamma in [0,1]; gamma=1 with `Delta_beta = ||y-y'||_2^beta`, beta in (0,2), is the strictly proper energy score. Hand pose MeJEE: gamma=0 21.6, 0.25 21.2, **0.5 20.9** mm; gamma=0.5 "better captures the true distribution (lower ProbLoss)" ([1606.02556](https://ar5iv.labs.arxiv.org/html/1606.02556)). Note gamma<1 is *less* repulsive; gamma>1 is improper. K used in training could not be verified.
- Kernel/MMD score: `S_k(P,y) = E k(X,X') - 2 E k(X,y)`, Gaussian bandwidth tuned on validation ([2112.08217](https://arxiv.org/html/2112.08217)).
- Variogram score: `VS_p(F,y) = sum_{i,j} w_ij (E_F|X_i-X_j|^p - |y_i-y_j|^p)^2`, p=0.5 commonly suggested, proper but not strictly ("invariant to change of sign and shift"), built to remedy the energy score's weak sensitivity to dependence structure ([2407.00650](https://arxiv.org/html/2407.00650); [2112.08217](https://arxiv.org/html/2112.08217); [Scheuerer-Hamill 2015](https://journals.ametsoc.org/view/journals/mwre/143/4/mwr-d-14-00269.1.xml), AMS page 403'd, statement taken from search snippet). Pinson-Tastu: energy score "discrimination ability may be limited when focusing on the dependence structure", worse "as dimension increases" ([orbit.dtu.dk](https://orbit.dtu.dk/en/publications/discrimination-ability-of-the-energy-score)); the "more sensitive to mean than variance" phrasing appears only in secondary snippets (unverified verbatim).
- Patched energy score: energy score on a local patch, shifted across the field and summed; "non-strictly proper as it does not evaluate long-range dependencies", strictness restored by adding a global term. WeatherBench calibration error: Energy 0.0863, Kernel 0.0797, Energy-Kernel 0.0794, **Patched-8 0.0550**, Patched-16 0.0690; GAN 0.3625. Training used 10 draws, robust "using as few as 3" ([2112.08217](https://arxiv.org/html/2112.08217)).
- Transformation principle (Prop. 1): `S_T(F,y) = S(T(F),T(y))` is proper for any measurable T, strictly proper iff S strictly proper on T(F) and T injective; weighted sums of proper scores stay proper, strictly if one strict term has w>0 (Prop. 2) ([2407.00650](https://arxiv.org/html/2407.00650)). This licenses adding CRPS on band energy, on gammatone bands, or a variogram term without losing propriety.

## 5. Noise placement

- EnScale: coarse MLP has "noise ... concatenated to hidden units in each hidden layer"; super-res stage concatenates "several channels of Gaussian noise in each pixel" before a shared per-pixel MLP; rationale "this facilitates learning complex representations of the random inputs and yields spatially correlated variability" ([2509.26258](https://arxiv.org/html/2509.26258)). No controlled input-only vs deep comparison found (unverified).
- SLED: noise enters six residual MLP blocks via AdaLN, "scale and shift values are predicted by applying a linear transformation to the input noise"; RMSE-only (no repulsive term) WER-C 40.60 vs 1.59 with energy distance ([2505.13181](https://arxiv.org/html/2505.13181)).
- CALM: noise `U[-0.5,0.5]` refined by residual MLP blocks; N=8 model samples and M=100 target draws because "a single target sample introduces high variance" ([2510.27688](https://arxiv.org/html/2510.27688)).
- AudioDEAR: noise is the input of a residual-MLP energy head, context via FiLM ([2605.00329](https://arxiv.org/html/2605.00329)).
- Engression pre-ANM `Y = g(WX + eta)` vs post-ANM `g(WX)+eta`: noise before the nonlinearity gives "horizontal" perturbation; linear pre-ANM "is not identifiable" ([2307.00835](https://arxiv.org/html/2307.00835)). No result on noise dimension limiting the conditional law (unverified).
- Learned/heteroscedastic noise scale in energy-score nets: nothing verifiable found. Closest: EAR's `tau_train` on the repulsive term (0.99) and `tau_infer` (0.7) scaling only the FiLM shift, which "trade diversity for accuracy" ([2505.07812](https://arxiv.org/html/2505.07812)).

## 6. Draws per step

Unbiased for any M>=2; variance drops with M. Practice: AIFS 2-4, FCN3 16, Pacchiardi 10 (3 ok), CALM 8, AudioDEAR m=2 best of {2,3,4} on IS ("IS peaks at m=2"), DISCO K unverified. No paper gives a gradient-variance formula vs M (unverified).

## 7. Metric-aligned feature-space scores

Gritsenko is the only audio case with ablations (log-L2 + mag-L1 multi-scale). A bone-conduction restoration paper uses "generalized energy distance loss based on multi-scale Mel spectrograms" because "time-domain l1 loss may be insufficient for the generation of high-frequency information" ([IEEE SPL](https://ieeexplore.ieee.org/document/10374184/), abstract only). No gammatone/NSIM-space energy score found.

## 8. 2025-26 one-step audio with proper scores

AudioDEAR: `L = ||x1-y|| + ||x2-y|| - ||x1-x2||` (latent L2, m=2), plus distillation lambda=1000; AudioCaps FD 18.67, FAD 2.79, KL 1.06, IS 9.66 vs SoundCTM 19.83/2.51/1.36/7.98 ([2605.00329](https://arxiv.org/html/2605.00329)). SLED as above. "Score-Based Training for EBM TTS" ([2505.13771](https://arxiv.org/html/2505.13771v1)) is score matching, not a proper score.

## Ranked cheapest interventions

1. **Make `d` spectral-dominant; confine waveform L1 to <6 kHz.** Use Gritsenko eq. (9) form (mag-L1 + `sqrt(k/2)` * log-L2, windows 64-2048). Prediction: high-band energy gap goes from -7.9 dB to within +-1.5 dB; LSD single-draw <= 0.90; CRPS unchanged or better. If the gap stays, the cause is not `d`.
2. **Add a fair-CRPS term on per-frame high-band energy in dB** (transform principle). Prediction: energy-ratio PIT flattens; per-bin PIT stays flat; ViSQOL-audio +0.2.
3. **Move noise into the implicit MLP via FiLM/AdaLN** (AIFS-CRPS, SLED, EnScale). Prediction: single-draw spread rises so the CRPS-optimal tau stays 1 while energy gap closes >=3 dB; if spread does not change, input noise was not the bottleneck.
4. **Variogram term (p=0.5) across frequency bins within a frame on log-mag**, small weight. Prediction: LSD falls >=0.03 with no CRPS loss, because it constrains inter-bin texture the energy score barely sees.
5. **Gammatone/ERB log-band energy score term** (ViSQOL-aligned features). Prediction: ViSQOL-audio single draw from 2.6-2.9 to >=3.1; LSD flat.
6. **M=4 draws, fair estimator; afCRPS alpha=0.95 only if a member collapses onto the target.** Prediction: small CRPS gain (<3 %), no change in energy gap; under-dispersion surviving M=4 rules out estimator variance.

Not cheap but verified: adding a light GAN on top of GED gave +0.19 MOS ([2008.01160](https://ar5iv.labs.arxiv.org/html/2008.01160)).