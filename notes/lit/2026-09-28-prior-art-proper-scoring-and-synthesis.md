# Prior art: proper scoring of BWE models, and whether "the missing band is a distribution" is a new synthesis

Written 2026-09-28 for the student-track submission (`lisa_rtm_1.pdf`, "The Missing Band Is a
Distribution"). Two questions:

1. Has any bandwidth-extension (BWE) paper evaluated models with a proper scoring rule?
2. Has anyone synthesised the field the way the submission does, and is that synthesis useful?

Evidence markers: **[P]** quoted from the primary full text; **[P✓]** quoted from the primary text
and re-extracted by hand during this review; **[A]** abstract or bibliographic metadata only;
**[S]** search snippet, or a later paper's description of the work. A search that finds nothing
is not a priority proof; section 1 says what was and was not covered.

---

## 0. Answers

### Q1. Has anyone evaluated BWE models with a proper score?

**Not that I can find, in audio BWE or in neighbouring audio restoration.**

- 119 arXiv full texts on BWE, audio/speech/music super-resolution and speech restoration
  (2017 to Sept 2026) were regex-scanned for proper-score, likelihood, calibration and diversity
  terms, and every hit was read in context.
- The evaluation sections of about 30 named systems were read.
- Classical artificial BWE (1994–2016) was checked through theses and author PDFs.
- Adjacent tasks were checked: speech enhancement, TTS, vocoders and codecs.

None reports CRPS, an energy/kernel/variogram score, a rank histogram, PIT, interval coverage or
spread–skill for a BWE model.

Not even the log score appears. WaveNet-BWE, WSRGlow, UDM+ and CVAE-BWE are trained by likelihood
or ELBO, and none reports a held-out value.

The nearest things that do exist:

| What exists | Where | Why it is not the submission's evaluation |
|---|---|---|
| Spread across draws proposed as "a natural measure of uncertainty" | Lemercier et al., IEEE SPM 2024 review [P✓] | Suggested, never checked against error |
| LSD-variance and uncertainty maps over N AudioSR draws | Jin et al., AAAI 2026 [P] | Spread treated as a nuisance to search over (best-of-N), never calibrated |
| FAD / Fréchet distance | CQT-Diff 2023, BABE 2024, BABE-2 2024 [P] | Set-level, not conditional on the input; not a proper score of p(y\|x) |
| Real-vs-super-resolved classifiers | Silaev, Drossos & Virtanen 2026 [P] | Marginal distribution only |
| Mutual information between narrow band (NB) and high band (HB) features under a fitted GMM | Jax & Vary 2002; Nour-Eldin et al. 2006 / 2013; Geiser 2012 [P] | When averaged over test vectors, a log-score gain; but reported as a property of the data (a bound), never used to score systems |
| Held-out log-likelihood of conditional audio models | WaveFlow (vocoder) 2020; Uria et al. (TTS) 2015; OverFlow 2023 [P] | Not BWE |
| Uncertainty-vs-error checks | Fang & Gerkmann 2023 (sparsification/AUSE); Emura 2024 (sample variance predicts SI-SDR); ArtiFree 2026 [P] | Speech enhancement; no proper score, ranks or coverage |

Outside audio the protocol is standard.

- **Stochastic climate/weather downscaling** — Leinonen et al. 2021; Harris et al. 2022; Price &
  Rasp 2022; CorrDiff 2025 [P]. This is literally super-resolution. These papers use CRPS, rank
  histograms and spread–skill, and report under-dispersed generative models.
- **General inverse problems** — Baattrup et al. (May 2026) [P✓] propose the same protocol:
  CRPS, a spectrum-fidelity check and coverage calibration.

**Defensible wording:** "to our knowledge, the first evaluation of audio BWE models with a
strictly proper scoring rule and rank-based calibration diagnostics, following practice in
probabilistic downscaling [Leinonen 2021; Harris 2022; Mardani 2025]."

**Not defensible:** "the first ever proper scoring of BWE models" without that scope and
credit, or any suggestion that the protocol itself is new.

### Q2. Has anyone synthesised the field this way?

**The narrative, yes, many times; the remedy, not in audio.**

| Link in the submission's story | Status for BWE | Strongest prior statements |
|---|---|---|
| (a) BWE is ill-posed / one-to-many | Established; quantified since 2002 | Jax & Vary 2002 (mutual-information (MI) bound); Nour-Eldin 2006/2013; Bauer 2016; Bruna, Sprechmann & LeCun 2016 (names audio BWE) [P✓]; WSRGlow 2021 [P✓]; CQT-Diff 2023; Yang et al. survey 2026 [P✓] |
| (b) Point objectives return the conditional mean → muffled / over-smoothed | Established | Classical: Jax & Vary 2003; Nour-Eldin 2013 ("oversmoothing") [P]. Speech synthesis and voice conversion: Toda & Tokuda 2005/07; Zen et al. 2009 ("muffled") [P]. Modern: StoRM 2023 (names BWE) [P✓]; Fu et al. 2026 ("a blend of modes, which sounds muffled", for bandwidth limitation) [P✓]; Yang et al. 2026 [P✓] |
| (c) Point metrics mislead; SNR prefers an empty HB | Established; the empty-HB point is explicit | Kuleshov et al. 2017 (spline); Su et al. 2021: "Any processing to the narrowband input simply lowers its PSNR" [P✓]; NU-Wave 2 2022 (input row beats every model) [P]; WSRGlow 2021 (lower temperature → better SNR and LSD, worse sound) [P✓]; Lemercier et al. 2023 (identity map wins instrumental metrics "especially for bandwidth extension") [P✓]; EBEN 2023; AudioSR 2024; Silaev et al. 2026 |
| (d) Evaluate p(y\|x) with proper scores + calibration | **Not found in BWE or audio restoration.** Standard in downscaling | Leinonen 2021; Harris 2022; CorrDiff 2025; Baattrup 2026 (general) |
| (e) Released "generative" BWE systems are near-deterministic | **Partly new** | FlowHigh's inference noise override: not reported anywhere (verified in code, §4.1). One Euler step returns the mean: known for rectified flow (Liu et al. 2022) and flow-based speech enhancement (Lay 2024; Cross & Ragni 2025; Zhou 2025). Counter-evidence: AudioSR is dispersed (Jin 2026, and the submission's own Table 2), as are multi-step score-based speech-enhancement models (Emura 2024; ArtiFree 2026) |

### Is the synthesis useful?

**As a message, little. As a measurement, yes, for a specific job.**

- **As a message.** Links (a)–(c) are one to two decades old in classical BWE and speech
  synthesis, and were restated in 2023–2026 by StoRM, Fu et al. and the BWE survey. Retelling them
  earns nothing and invites "not novel".
- **As a measurement.** Link (d) is missing from about 120 BWE papers, is routine in a
  neighbouring super-resolution field, and costs M forward passes. It has already caught something
  nobody had reported (FlowHigh).
  - What it can do: test whether a system samples p(y|x). For BWE with low-band passthrough, theory
    says that is the right target (Ohayon et al. 2023, §4.1).
  - What it cannot do: rank perceptual quality (§4.2). Classical BWE and speech-synthesis evidence
    shows listeners sometimes prefer the quiet mean.

---

## 1. What was searched, and the limits

Five parallel search passes covered modern BWE, classical BWE, adjacent audio, other
super-resolution fields, and surveys/metrics. Hand spot-checks followed.

- **Modern BWE / audio SR (2016 – Sept 2026).**
  - arXiv API field searches of "bandwidth extension" / "audio super-resolution" / "speech
    super-resolution" crossed with: likelihood, NLL, "bits per", CRPS, "energy score",
    "energy distance", "scoring rule", calibration, uncertainty, probabilistic, diversity,
    stochastic, "posterior sampling", "multiple samples", variance, ensemble, "one-to-many",
    "ill-posed", "over-smoothing".
  - 119 full-text PDFs downloaded and regex-scanned; hits read in context.
  - Evaluation sections read for Kuleshov 2017, TFNet, Ling 2018, Gupta 2019, CVAE-BWE, Su 2021,
    NU-Wave, NU-Wave 2, WSRGlow, NVSR, VoiceFixer, UDM+, CQT-Diff, BABE, BABE-2, AERO, mdctGAN,
    AudioSR, AP-BWE, FlowHigh, Bridge-SR, FlashSR, AudioLBM, A2SB, UniverSR, Jin 2026 and
    Silaev 2026.
  - Every one of these uses point or perceptual metrics, FAD/FD at most, or diversity without
    calibration (§2.1).
- **Classical BWE (1994–2016).** Primary texts reached through author and thesis repositories:
  RWTH (Jax, Geiser), McGill (Nour-Eldin, Kabal), Aalto (Pulakka, Kallio), EURASIP theses (Bauer),
  J-STAGE, and NAIST/Nitech.
- **Adjacent audio.** Speech enhancement, dereverberation, TTS / statistical parametric synthesis,
  vocoders, codecs and inpainting.
- **Other fields.** Climate downscaling, image SR (SRFlow, NTIRE "Learning the SR space" 2021/22),
  general ML evaluation, and the scoring-rule methodology literature.

**Not covered well:**
- IEEE- or ISCA-only papers with no arXiv version. The ISCA archive failed TLS (expired
  certificate), and IEEE Xplore is paywalled.
- Primary texts not reached:
  - Nilsson et al. 2000 and 2002; Nilsson & Kleijn 2001.
  - Jax's 2002 dissertation; the Iser, Minker & Schmidt book.
  - Pulakka et al. 2015; Abel et al. 2017.
  - Toda & Tokuda 2007 (IEICE) and Toda, Black & Tokuda 2007 (TASLP) full texts.
  - The Pirklbauer et al. 2023 metrics paper.
- A proper score reported under an unusual name in a paywalled evaluation section would have been
  missed.

---

## 2. Q1 in detail

### 2.1 BWE systems checked, and what they evaluate with

Abbreviations: S = SNR; L = LSD (± = HF/LF split); V = ViSQOL; P = PESQ; M = MOS/MUSHRA/listening
test; F = FAD/FD.

| System | Metrics | Probabilistic evaluation? |
|---|---|---|
| Kuleshov 2017; TFNet 2018 | S, L (+M) | none; trained as p(y\|x) with Gaussian noise, i.e. MSE |
| Ling 2018; Gupta 2019 (WaveNet BWE) | accuracy/S/L/P/M; M only | trained by maximum likelihood, no held-out NLL |
| CVAE-BWE (Bachhav et al. 2019/20) | LSD, COSH, segSNR, P, WER | trained by conditional NLL/ELBO, not reported |
| Su 2021 (HiFi-GAN BWE) | PSNR, L, M | none |
| NU-Wave 2021; NU-Wave 2 2022 | S, L(±), M | run-to-run std of test averages (≈1e-5), not per input |
| WSRGlow 2021 | S, L, M | exact-likelihood flow; no held-out NLL; temperature sweep |
| NVSR; VoiceFixer; AERO; mdctGAN; AudioSR | L, V, P, M ... | none |
| UDM+ 2023 | L, LSD-LF, P | trained on the variational bound (VLB), not reported |
| CQT-Diff 2023; BABE 2024; BABE-2 2024 | L, F, M | set-level FD/FAD only; one draw per input |
| AP-BWE 2024; FlowHigh 2025; Bridge-SR; FlashSR; AudioLBM; A2SB | L(±), V, ... | none |
| UniverSR 2026 | LSD-HF, 2f-model, M | none |
| Jin et al. 2026 | L, WER, speaker sim., aesthetics, CLAP, LSD-variance | spread measured, not calibrated |
| Silaev et al. 2026 | S, L, M, classifier separability | marginal distributional test |

The 2026 survey (Yang et al., arXiv 2605.16681 v2) was searched in full. It contains no CRPS,
scoring rule or calibration. Its metrics section lists SNR, SegSNR, SI-SDR, LSD, ViSQOL, PESQ, STOI,
MOS and preference tests. Its recommendation is "reference-free metrics and standardized
evaluation protocols" [P].

### 2.2 Spread measured but never calibrated

- **Lemercier et al., IEEE SPM 41(6):72–84, 2024** [P✓]:
  - "these models typically produce deterministic outputs, disregarding the inherent uncertainty in
    their results"
  - "running several realizations of the reverse diffusion process and measuring the empirical
    standard deviation of the obtained estimates can provide the user with a natural measure of
    uncertainty".

  This is the closest audio statement of the idea. The submission's rank histogram and spread
  ratio supply the missing check: is that standard deviation the right size?
- **Jin et al., "Inference-time Scaling for Diffusion-based Audio Super-resolution", AAAI 2026**
  (arXiv 2508.02391).
  - They measure "the variance across time-frequency bins in the STFT domain over multiple
    generations from the same LR input" [P].
  - They treat the variance as a defect: "high-variance and quality-limited outputs" [P✓ abstract].
  - They then pick the best of N draws with verifiers, including an oracle that uses the reference.

  This is the opposite reading of spread from the submission's, and worth citing as such.
- **Speech enhancement**:
  - Emura, EUSIPCO 2024 ("Estimation of output SI-SDR solely from enhanced speech signals in
    diffusion-based generative speech enhancement"): the inverse relative variance of multiple
    outputs correlates linearly with output SI-SDR [P]. A spread–skill relation in all but name.
  - Fang & Gerkmann, ICASSP 2023 and TASLP 2023: sparsification plots and AUSE [P].
  - ArtiFree, ICASSP 2026: embedding variance flags phoneme artifacts [P].

### 2.3 Classical BWE: a log score hidden inside the information bounds

- **Jax & Vary, ICASSP 2002.** They bound the error of any memoryless *deterministic* estimator
  ("the estimate ỹ is calculated deterministically from the feature vector x") by the MI between
  NB and HB features, measured at I = 2.24–2.61 bit/frame [P].
- **Nour-Eldin, Shabestary & Kabal, ICASSP 2006.** The NB explains 5.55% of HB entropy (I = 1.52
  bits vs H(Y) = 27.42 bits) [P].
- **Nour-Eldin, PhD thesis, McGill 2013.** "we can not reduce GMM-based BWE distortion to less than
  dLSD(RMS) = 4.62dB" [P].
- **Geiser, PhD thesis, RWTH 2012.** "the actual amount of mutual information that is shared
  between the baseband and the extension band is relatively low" [P].

Nour-Eldin (2013) estimates MI by "stochastic integration of test features vectors": the mean
of log[p(y|x)/p(y)] over held-out data. That is exactly the log-score gain of the model's
conditional density over the HB marginal. (Jax & Vary 2002 instead drew 10⁶ samples from the fitted
GMM.) So classical BWE computed a proper-score quantity twenty years ago. It used it to bound the
task, never to score a system.

- **What this means for the paper:** cite it as the quantitative precedent for (a). Present the
  submission's waveform coherence c < 0.02 as a waveform-level counterpart of these
  feature-level MI numbers.

### 2.4 Where the protocol already lives

- **Leinonen, Nerini & Berne, IEEE TGRS 59(9):7211–7223, 2021** (stochastic super-resolution GAN).
  - Evaluation: CRPS, rank statistics, and outlier fraction above the calibrated value [P].
  - "The RMSE, in particular, is minimized at the mean of possible solutions, and therefore is of
    limited use in assessing the performance of GANs" [P].
  - They suggest the method "may be useful in applications beyond the geoscience domain" [P].
- **Harris et al., JAMES 14, 2022.**
  - Evaluation: CRPS, 100-member rank histograms, and a radially averaged log-spectral distance.
  - "These are deterministic methods, hence the CRPS reduces to the mean absolute error";
    "...implying that the GAN and VAE-GAN are slightly underdispersive" [P].
- **Price & Rasp, AISTATS 2022**: CRPS, Brier score, reliability diagrams, rank histograms [P].
- **Mardani et al. (CorrDiff), Commun. Earth Environ. 6:124, 2025.**
  - Evaluation: CRPS against the baselines' MAE, spread–skill with the √(1+1/n) correction, rank
    histograms, and paired significance tests.
  - Finding: the predictions are "under-dispersive for most channels" [P].
- **Baattrup et al., "Pointwise Metrics Mislead", arXiv 2605.22891, May 2026** (particle physics)
  [P✓ abstract]:
  - "By the law of total variance, point estimators trained to minimize MSE or MAE produce a
    marginal spectrum strictly narrower than the truth whenever the posterior has nonzero width."
    This is the submission's "the mean of high bands that do not agree is quiet", stated in
    general form.
  - "model rankings reverse between pointwise and distributional metrics, and calibration further
    separates architectures indistinguishable under CRPS."
- **Image SR does not use proper scores.** SRFlow, the NTIRE 2021/22 "Learning the SR space"
  challenges and explorable SR make the one-to-many argument. They evaluate with LPIPS, LR-PSNR
  and diversity, and NTIRE penalises deterministic methods with a zero diversity score [P].

---

## 3. Q2 in detail

### 3.1 The closest whole-chain statements

**In audio, nobody has the chain through (d).** The nearest statements:

- **Bruna, Sprechmann & LeCun, ICLR 2016** [P✓]:
  - "Inverse problems in image and audio, and super-resolution in particular, can be seen as
    high-dimensional structured prediction problems, where the goal is to characterize the
    conditional distribution of a high-resolution output given its low-resolution corrupted
    observation."
  - Point estimates "suffer from the regression-to-the-mean problem".
  - The method "could be used in other challenging ill-posed problems such as audio bandwidth
    extension."
- **StoRM, Lemercier et al., IEEE/ACM TASLP 2023** [P✓]: "an optimal predictive model learns the
  mapping to the posterior mean y → E[x|y] ... This phenomenon is known as regression to the mean".
  The same passage names bandwidth extension as a case.
- **Fu et al., arXiv 2603.02641, 2026** [P✓]: under "bandwidth limitation, and low SNR ... the
  regression output is biased towards prior mean ... the mean may be a blend of modes, which sounds
  muffled and unnatural". They then cite Blau & Michaeli (2× MMSE) and Freirich et al.
- **Yang et al. survey, 2026** [P✓]: BWE is "inherently ill-posed ... prone to regression-to-the-mean
  effects and spectral over-smoothing"; "high SNR does not guarantee good HF reconstruction
  quality" [P].

**Outside audio, the full chain including (d) exists:** Leinonen 2021, Harris 2022, CorrDiff 2025
and Baattrup 2026 (§2.4).

**So the synthesis is a transfer**, and it should say so. Citing the downscaling papers as the
methodological source strengthens the paper: it shows the protocol is established, and moves the
novelty onto the audio application and the audit, where it can be defended.

### 3.2 Where earlier BWE work disagrees with the submission

This is the part a knowledgeable reviewer will raise, and the manuscript does not address it.

1. **Classical BWE made the output quieter on purpose.** It treated HB energy as a posterior and
   deliberately biased the decision quieter.
   - Nilsson & Kleijn, ICASSP 2001 [S]: an asymmetric cost "that penalizes over-estimates more
     than under-estimates", with attenuation that "depends on the broadness of the a-posteriori
     distribution".
   - Pulakka, PhD thesis, Aalto 2013 [P]: "asymmetric error measures have been proposed so that
     energy overestimation is penalized more heavily than underestimation".
   - Nour-Eldin 2013 [P]: over-estimation "can make the extended wideband signal often sound more
     annoying than the original narrowband signal".
   - Jax & Vary, ICASSP 2003 [P]: the MMSE (conditional-expectation) estimator beat ML/MAP
     classification in listening tests.

   The submission's "muffled = bad" is the opposite design verdict.
2. **Speech synthesis: samples beat the mean only when the model is good.**
   - Uria et al., ICASSP 2015 [P]: "mean acoustic trajectories sound muffled", yet listeners
     "showed preference for means instead of samples generated by RNADE".
   - Henter, King, Merritt & Degottex, arXiv 1807.10941 [P✓]: "for poor models, random examples
     sound worse than the mean, while for accurate models the reverse is true"; restoring variance
     lowers likelihood, so "an objectively inferior model produces perceptually superior output".
3. **The muffling mechanism has named precedents the paper should cite.**
   - Toda & Tokuda 2005 [P]: "Using the over-smoothed trajectory causes the muffled sound."
   - Zen, Tokuda & Black 2009 [P]: synthesised speech sounds "evidently muffled ... because the
     generated speech-parameter trajectories are often over-smoothed".
   - Global-variance compensation was applied to BWE by Fujitsuru et al. (NCSP 2008) and Liu et al.
     (Interspeech 2015) [S].

**What to say in the paper:** sampling beats the mean perceptually only once the sampler is
accurate, which matches the submission's own finding that averaging draws rebuilds the deficit.
And the classical systems chose quiet deliberately, for perceptual reasons.

---

## 4. Is the synthesis useful?

### 4.1 What the measurement gives the field that it lacks

1. **A test of a claim the field makes and never checks.** Generative BWE papers present
   themselves as models of p(y|x). WSRGlow: "we use normalizing flow to address the one-to-many
   problem by modeling the conditional distribution of HR audio" [P✓]. Nothing in about 120 papers
   tests that claim. A conditional proper score and a rank histogram are the minimal test.

2. **It detects collapse that LSD and ViSQOL reward.** FlowHigh's released code was checked
   (github.com/jjunak-yun/FLowHigh_code, `main`):
   - **The code.**
     - Training path for `independent_cfm_adaptive` (the paper's path):
       `w = t·x1 + (1−t)·x0 + σ_t·ε` with `σ_t = 1 − (1 − σ_min)t`, so σ₀ = 1, and target
       `(x1 − x0) − (1 − σ_min)ε` (`cfm_superresolution.py`, around lines 794–822).
     - `inference.py` line 63 calls `sample(..., std_2 = 1.)` but passes no `std_1`.
     - `sample()` (lines 615–618) then runs `if std_1 is None or std_2 is None: std_1 = 1.0;
       std_2 = self.sigma`, so the start is `y0 = cond + 1e-4·ε` (lines 657–660).
     - The README tells users to run `--time_step 1 ... --sigma 0.0001`, and says to "use the same
       value for sigma as was used during training". The value really is σ_min, the path parameter;
       the override also makes it the start-noise scale.
     - The explicit `std_2 = 1.` suggests the override is accidental.
     - No prior report was found in the paper, its three GitHub issues, UniverSR, the survey or
       Silaev et al.
   - **The theory: one step returns the mean whatever the start noise.** At t = 0 the network sees
     both `x = cond + sε` (s = start-noise scale) and `cond`, so ε is identified. The optimal field
     is `v*(x,0) = E[x1|cond] − cond − (1−σ_min)(x − cond)`. One Euler step gives
     `E[x1|cond] + σ_min·sε`: the conditional-mean mel spectrogram, then a deterministic vocoder.
     - This is my derivation, from the rectified-flow result that v(x,t) = E[X1 − X0 | Xt = x]
       (Liu et al. 2022). Speech-enhancement analogues exist: Lay et al. 2024; Cross & Ragni 2025;
       Wang et al. AAAI-26, where "one-step sampling is nearly equivalent to a predictive model"
       [P].
     - This explains why the submission's "prior restored" FlowHigh still has spread 0.023.
     - **What to say in the paper:** the noise override is a real, reportable bug. The collapse,
       however, comes from single-step sampling, and the manuscript should say that.
   - **Supporting sightings.** UniverSR notes FlowHigh's "overly smooth high-frequency components".
     The survey notes "FLowHigh maintains nearly unchanged ViSQOL from NFE 2 to 50" [P].

3. **Theory that fits BWE better than the paper's current citation.** Ohayon, Adrai, Elad &
   Michaeli (ICML 2023) prove that "any restoration algorithm that attains perfect perceptual
   quality and whose outputs are consistent with the input must be a posterior sampler" [P].
   - A BWE system with low-band passthrough is consistent with its input by construction. So for
     BWE, the perceptually ideal output really is a draw from p(y|x). That is the argument the
     conclusion needs.
   - Blau & Michaeli 2018, which the paper cites for "score the distribution", is about matching the
     *marginal* p(ŷ) to p(y), not about conditional scoring.
   - Silaev et al. 2026 find classifiers separate FlowHigh and FlashSR outputs from real audio
     almost perfectly. That is consistent with this: a deterministic, consistent system cannot
     match the data distribution.

4. **Uncertainty that downstream users can trust.** Lemercier 2024 and ArtiFree 2026 want to use
   sample spread to flag unreliable output. Calibration is the property that makes that use valid.

### 4.2 What the measurement cannot do, and what a reviewer will ask

1. **It does not rank sound quality.**
   - Theis, van den Oord & Bethge 2016: evaluation criteria "are largely independent", and models
     "need to be evaluated directly with respect to the application(s)" [P].
   - Freirich, Michaeli & Meir 2021: without consistency, "there often exist perfect perceptual
     quality estimators that achieve lower distortion" than posterior sampling [P].
   - Section 3.2 above: listeners punish over-estimation, and samples from imperfect models sound
     worse than the mean.
   - **Hence:** the manuscript's CRPS ordering, NU-Wave 2 (0.583) ahead of FlowHigh (0.682),
     inverts LSD, ViSQOL and an 8 dB deficit. It needs an explanation, not only a table.
     - Candidates to test: log-magnitude CRPS weights quiet bins most (point 3), and NU-Wave 2 is
       "4.7 dB over in quiet" frames. NU-Wave 2 also has some spread (0.094).
     - Decompose CRPS into reliability, resolution and uncertainty (Hersbach 2000), and report it
       per band and per frame-energy stratum.
2. **A sampler beating a regressor on CRPS is partly built in.**
   - CRPS reduces to absolute error for a point forecast (Gneiting, Balabdaoui & Raftery 2007;
     Harris 2022). ½E|Y−Y′| ≤ E|Y−m| for any m, so a calibrated sampler beats every point forecast.
   - The repo already says this (`notes/metrics-reference.md` §3: "partly definitional").
   - The informative comparisons are:
     - among stochastic models;
     - against the regressor "dressed" with held-out residuals (Gneiting et al. 2007's persistence
       dressing) as a fair probabilistic baseline.
3. **What is scored decides what is seen.** The paper scores per-bin HB log-magnitude marginals.
   - **Summed marginal CRPS is not proper for the joint.** Shuffling ensemble members independently
     per point leaves it unchanged (FourCastNet 3, arXiv 2507.12144 [P]).
   - **The energy score is the obvious joint fix, but it barely detects correlation errors**
     (Pinson & Tastu 2013 [S]). The variogram score does better (Scheuerer & Hamill 2015 [A]).
   - **The log transform acts as a weighting.** Scoring in the log domain weights low-energy bins
     heavily (the chaining-function view of transformed scores, Allen, Ginsbourger & Ziegel 2023
     [P]). Ties at the floor need randomised ranks (Leinonen; Harris).
   - **For an incoherent band, per-bin log power is dominated by fine structure.** For a circular
     Gaussian bin, |X|² is exponential and Var(ln|X|²) = π²/6, about 5.6 dB standard deviation
     (standard result). So per-bin CRPS partly measures "are the draws noise-like and independent",
     not "is the envelope right".
   - **Fixes:**
     - CRPS on ERB-band energies (proper by the transformation principle);
     - the repo's sliced CRPS and `corr_err`;
     - a variogram term across bins.
4. **The numbers need statistics.**
   - **Estimator.** Every `crps_ensemble` here (`build_notebook.py:682`,
     `sampler/build_sampler_notebook.py:693`) averages over all M² pairs, including the diagonal,
     although the docstrings say "Fair estimator".
     - That is the standard (unfair) estimator, biased by +E|X−X′|/(2M) (Ferro, Richardson &
       Weigel 2008 [P]; Ferro 2014; Zamo & Naveau 2018).
     - At M = 8 the penalty grows with spread, so it favours collapsed models such as FlowHigh when
       generative models are compared with each other.
     - Switch to the fair form: multiply the pair term by M/(M−1).
     - **Correction, 2026-09-29.** This applies to the older EVAL12 code on `main`, not to the paper.
       The paper's CRPS values come from `sota/score.py` (branch `claude/sota-model-diagnosis-822734`),
       which reports both estimators. Tables 1–2 print its `crps_fair` (FlowHigh 0.682, NU-Wave 2
       0.583, the ERB arms 0.512–0.515), so the published ranking does not carry this bias. See
       `supplementary/results/number_check.md`.
   - **Sample size.** 32 utterances × 8 draws is small, and bins are correlated.
     - Use an utterance-level block bootstrap or paired tests, not bin counts (Hamill 2001; Bröcker
       2018; CorrDiff used paired Wilcoxon tests).
     - Baattrup et al. found CRPS stabilised only by about M = 100 in their setting.
   - **Rank tails.** P(truth above all 8 draws) = 1/9 ≈ 11.1% for one tail; both tails together give
     2/9 ≈ 22.2%. The manuscript's "21 to 24% ... against 11%" is one tail: it is `pit_hi` on `core`
     (0.211–0.241 for the λ ≥ 0.01 arms; `pit_lo` 0.098–0.111), confirmed 2026-09-29 against the SOTA
     branch's results. The paper should still say "one tail" and report
     both. A top-only excess is a level bias, a different defect from a narrow ensemble (Hamill
     2001; Heinrich 2021). Uniform ranks are "necessary but not sufficient" for calibration
     (Gneiting et al. 2007 [P]).
   - **Spread ratio.** Define how the "calibrated ensemble" reference is built. If it comes from
     the error of the ensemble mean, apply the finite-ensemble correction: a calibrated n-member
     ensemble has E[MSE of mean] = (1 + 1/n)·E[spread²] (Fortin et al. 2014). A calibrated
     8-member ensemble therefore has spread/RMSE ≈ √(8/9), not 1.
5. **Under-dispersion is per model and per sampler setting, not a property of "SOTA".**
   - AudioSR is dispersed (spread 0.765 in the paper's own Table 2; Jin 2026).
   - Multi-step score-based speech-enhancement models are dispersed, and their spread predicts
     error (Emura 2024; ArtiFree 2026).
   - SRFlow samples below temperature 1 by design.
   - Report steps, temperature and guidance for each audited model.

### 4.3 Verdict

| Question | Answer |
|---|---|
| Is the narrative new? | No. (a)–(c) are established for BWE, (b) in the exact word "muffled". The full chain including (d) exists in downscaling and in general form (Baattrup 2026). |
| Is the measurement new for audio BWE? | Apparently yes. Nothing found in about 120 BWE papers, the 2026 survey, classical BWE or adjacent audio. Scope the claim and credit the source. |
| Is it useful? | Yes, framed as "testing whether generative BWE systems sample p(y\|x)". Ohayon et al. 2023 makes that the perceptually right target for consistent BWE, and it has already exposed an unreported collapse. Not as a replacement leaderboard for sound quality. |
| Is a stand-alone synthesis/position paper worth writing? | Low marginal value. The survey (2026), StoRM (2023), Fu et al. (2026) and Baattrup et al. (2026) already cover it. The value is in the audit and the protocol, provided the audit is reproducible (estimator, sample sizes, confidence intervals, sampler settings, code). |

---

## 5. Concrete edits to the manuscript

1. **Abstract, "first ever proper scoring of BWE models".** Replace with the scoped wording in §0,
   crediting Leinonen 2021, Harris 2022 and Mardani 2025.
2. **"Point prediction is the key cause of muffling [cite]".** Cite:
   - Bruna et al. 2016; StoRM 2023; Fu et al. 2026; the Yang et al. 2026 survey;
   - classical: Jax & Vary 2003; Nour-Eldin 2013;
   - speech synthesis: Toda & Tokuda 2007; Zen et al. 2009.
3. **SNR ceiling.**
   - Cite Su et al. 2021 ("Any processing to the narrowband input simply lowers its PSNR") and
     NU-Wave 2's input row.
   - Present E(1 + p − 2c) and c < 0.02 as the quantification: a model beats the sinc ceiling only
     if c > p/2.
   - Qualify it: the ceiling is the *ideal* (sinc) upsampler at a given rate. Kuleshov 2017 and
     NU-Wave report models above spline or linear interpolation.
4. **"SOTA models have low output diversity".** Make it per model; AudioSR is dispersed.
5. **FlowHigh.** State two separate facts:
   - (i) the inference override, with file and line references;
   - (ii) one Euler step returns E[x1|x] whatever the start noise (Liu et al. 2022). Fact (ii) is
     why restoring the prior still gives a spread of 0.023.
6. **Conclusion, "score the distribution (Blau and Michaeli 2018)".**
   - Cite Ohayon et al. 2023 for why a consistent BWE system should output a posterior draw.
   - Cite Freirich et al. 2021 for the caveat.
   - Address the classical quiet-bias and speech-synthesis evidence (§3.2).
7. **Methods.**
   - Fair CRPS estimator.
   - Both rank tails.
   - How the calibrated reference ensemble is built.
   - Utterance-level bootstrap confidence intervals.
   - A dressed-regressor baseline.
   - CRPS on ERB-band energies next to per-bin CRPS.
   - Sampler settings for every audited model.

---

## 6. Key references (for the bibliography)

- **BWE, modern**
  - Bruna, Sprechmann, LeCun. Super-resolution with deep convolutional sufficient statistics.
    ICLR 2016. [arXiv:1511.05666](https://arxiv.org/abs/1511.05666)
  - Kuleshov, Enam, Ermon. Audio super resolution using neural networks. ICLR-W 2017.
    [arXiv:1708.00853](https://arxiv.org/abs/1708.00853)
  - Su, Wang, Finkelstein, Jin. Bandwidth extension is all you need. ICASSP 2021.
    [PDF](https://gfx.cs.princeton.edu/pubs/Su_2021_BEI/ICASSP2021_Su_Wang_BWE.pdf)
  - Zhang, Ren, Xu, Zhao. WSRGlow. Interspeech 2021.
    [arXiv:2106.08507](https://arxiv.org/abs/2106.08507)
  - Han, Lee. NU-Wave 2. Interspeech 2022. [arXiv:2206.08545](https://arxiv.org/abs/2206.08545)
  - Lemercier, Richter, Welker, Gerkmann. Analysing diffusion-based generative approaches versus
    discriminative approaches for speech restoration. ICASSP 2023.
    [arXiv:2211.02397](https://arxiv.org/abs/2211.02397)
  - Moliner, Elvander, Välimäki. BABE. IEEE/ACM TASLP 32, 2024.
    [arXiv:2306.01433](https://arxiv.org/abs/2306.01433)
  - Lemercier et al. Diffusion models for audio restoration: a review. IEEE SPM 41(6):72–84, 2024.
    [arXiv:2402.09821](https://arxiv.org/abs/2402.09821)
  - Jin et al. Inference-time scaling for diffusion-based audio super-resolution. AAAI 2026.
    [arXiv:2508.02391](https://arxiv.org/abs/2508.02391)
  - Silaev, Drossos, Virtanen. ICASSP 2026 workshops.
    [arXiv:2601.03443](https://arxiv.org/abs/2601.03443)
  - Yang et al. A survey of advancing audio super-resolution and bandwidth extension from
    discriminative to generative models. arXiv, v2 Aug 2026.
    [arXiv:2605.16681](https://arxiv.org/abs/2605.16681)
  - Yun, Kim, Lee. FLowHigh. ICASSP 2025. [arXiv:2501.04926](https://arxiv.org/abs/2501.04926);
    code: github.com/jjunak-yun/FLowHigh_code
- **BWE, classical**
  - Jax, Vary. An upper bound on the quality of artificial bandwidth extension of narrowband
    speech signals. ICASSP 2002.
    [PDF](https://www.iks.rwth-aachen.de/fileadmin/publications/jax02b.pdf)
  - Jax, Vary. MMSE/HMM-based ABE. ICASSP 2003.
    [PDF](https://www.iks.rwth-aachen.de/fileadmin/publications/jax03.pdf)
  - Nilsson, Kleijn. Avoiding over-estimation in bandwidth extension of telephony speech.
    ICASSP 2001. [S]
  - Nour-Eldin, Shabestary, Kabal. ICASSP 2006.
    [PDF](https://www.ece.mcgill.ca/~pkabal/papers/2006/Nour-EldinC2006.pdf)
  - Nour-Eldin. PhD thesis, McGill 2013.
    [PDF](https://www.mmsp.ece.mcgill.ca/Theses/2013/Nour-EldinT2013.pdf)
  - Geiser. PhD thesis, RWTH 2012.
    [PDF](https://www.iks.rwth-aachen.de/fileadmin/publications/geiser12.pdf)
  - Pulakka. PhD thesis, Aalto 2013.
    [aaltodoc](https://aaltodoc.aalto.fi/server/api/core/bitstreams/59d20eb8-0cb8-450b-a92b-8a459d10e83d/content)
- **Adjacent audio**
  - Lemercier et al. StoRM. IEEE/ACM TASLP 2023.
    [arXiv:2212.11851](https://arxiv.org/abs/2212.11851)
  - Fu et al. Rethinking training targets, architectures and data quality for universal speech
    enhancement. 2026. [arXiv:2603.02641](https://arxiv.org/abs/2603.02641)
  - Emura. EUSIPCO 2024.
    [PDF](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000236.pdf)
  - Fang, Gerkmann. ICASSP 2023. [arXiv:2212.04831](https://arxiv.org/abs/2212.04831)
  - Lay et al. Single and few-step diffusion for generative speech enhancement. ICASSP 2024.
    [arXiv:2309.09677](https://arxiv.org/abs/2309.09677)
  - Ping et al. WaveFlow. ICML 2020. [arXiv:1912.01219](https://arxiv.org/abs/1912.01219)
  - Uria et al. Trajectory-RNADE. ICASSP 2015.
    [PDF](https://homepages.inf.ed.ac.uk/imurray2/pub/15tts_rnade/tts_rnade.pdf)
  - Henter, King, Merritt, Degottex. Analysing shortcomings of statistical parametric speech
    synthesis. arXiv 2018. [arXiv:1807.10941](https://arxiv.org/abs/1807.10941)
  - Zen, Tokuda, Black. Statistical parametric speech synthesis. Speech Communication 2009.
  - Toda, Tokuda. Global variance. IEICE Trans. E90-D(5), 2007.
  - Gritsenko et al. A spectral energy distance for parallel speech synthesis. NeurIPS 2020.
    [arXiv:2008.01160](https://arxiv.org/abs/2008.01160)
  - Li, Yang, Yang. Restoration of bone-conducted speech with U-Net-like model and energy distance
    loss. IEEE SPL 31:166–170, 2024. doi:10.1109/LSP.2023.3347149. [A] An energy-distance
    *training* loss on a BWE-like task; evaluated with PESQ/STOI.
- **Other fields and theory**
  - Leinonen, Nerini, Berne. IEEE TGRS 2021. [arXiv:2005.10374](https://arxiv.org/abs/2005.10374)
  - Harris et al. JAMES 2022. [arXiv:2204.02028](https://arxiv.org/abs/2204.02028)
  - Price, Rasp. AISTATS 2022. [arXiv:2203.12297](https://arxiv.org/abs/2203.12297)
  - Mardani et al. CorrDiff. Commun. Earth Environ. 2025.
    [arXiv:2309.15214](https://arxiv.org/abs/2309.15214)
  - Baattrup et al. Pointwise metrics mislead. arXiv 2026.
    [arXiv:2605.22891](https://arxiv.org/abs/2605.22891)
  - Ohayon, Adrai, Elad, Michaeli. ICML 2023. [arXiv:2211.08944](https://arxiv.org/abs/2211.08944)
  - Freirich, Michaeli, Meir. NeurIPS 2021. [arXiv:2107.02555](https://arxiv.org/abs/2107.02555)
  - Blau, Michaeli. CVPR 2018. [arXiv:1711.06077](https://arxiv.org/abs/1711.06077)
  - Theis, van den Oord, Bethge. ICLR 2016. [arXiv:1511.01844](https://arxiv.org/abs/1511.01844)
  - Liu, Gong, Liu. Rectified flow. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- **Scoring-rule methodology**
  - Gneiting, Raftery. JASA 2007.
  - Gneiting, Balabdaoui, Raftery. JRSS-B 69:243–268, 2007.
  - Ferro, Richardson, Weigel. Met. Appl. 15:19–24, 2008.
  - Ferro. QJRMS 140:1917–1923, 2014.
  - Zamo, Naveau. Math. Geosci. 50:209–234, 2018.
  - Hamill. MWR 129:550–560, 2001.
  - Hersbach. Weather Forecast. 15:559–570, 2000.
  - Fortin et al. JHM 15:1708–1713, 2014.
  - Scheuerer, Hamill. MWR 143:1321–1334, 2015.
  - Allen, Ginsbourger, Ziegel. SIAM/ASA JUQ 11:906–940, 2023.
  - FourCastNet 3. [arXiv:2507.12144](https://arxiv.org/abs/2507.12144)
