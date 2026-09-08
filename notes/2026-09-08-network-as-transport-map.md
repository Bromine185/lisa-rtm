# Research note — the network is the transport map: proper scoring rules end regression to the mean at zero inference cost

**Date:** 2026-09-08 (overnight run, unattended)
**Question:** the deterministic ladder failed on an honest source (T1 moved HB-LSD 1.7 %), while the
zero-parameter stochastic rung moved CRPS 21 %. If a hand-built conditional transport map on noise is
the only thing that works, what happens when the *network itself* is trained to be that map?
**Headline:** trained under the energy score, the same 88k-parameter LISA becomes a conditional
sampler at zero inference cost (1.5 ms per second of audio on an A100, 16.5 ms on a laptop CPU). On
held-out speakers its CRPS is **43 % below the deterministic model and 23 % below the best hand-built
transport rung**; its ensemble mean is a *better point predictor* than the point-trained model (SNR
18.90 vs 18.86 dB, 18.78 vs 18.35 on the second set); and the cross-power coherence it makes measurable
shows that **the band above 6 kHz is unpredictable from LISA's input for every model trained here** —
coherent fraction ≤ 2 %, zero above 7 kHz — so the deterministic SNR ceiling is 0.3 dB above naive
upsampling and at least 88 % of the energy the paper's spectral loss puts there is hallucinated. A
graph experiment reproduces the mechanism in the graph Fourier basis and gives a one-model test that
separates objective-induced from architecture-induced over-smoothing.

---

## 0. The idea, and why it is the right next rung

Over-smoothing in a deterministic model trained on a one-to-many problem is a geometric fact, not a
bug: the loss minimiser is a barycentre of the conditional law. Under an ℓ1/ℓ2 geometry on waveforms,
the barycentre of a distribution with random phase collapses toward zero, so the output leaves the
manifold of natural signals. The ladder T1–T4 pushes *statistics* back onto that manifold; it cannot
push *information* back, because the information lives in the conditional law and a point has none.

The stochastic rung S was the first rung that carried a distribution: reference noise pushed through
a closed-form conditional map fitted to two moments. Tonight's move is to let the network learn that
map. Eight Gaussian channels enter the encoder alongside the waveform; the same 5-layer ReLU decoder
reads the same three latents. With the noise set to zero this is LISA exactly (same parameter count
to within 896 weights, same init, same latency). The training objective is the **energy score**

$$\mathrm{ES}(P, y) = \mathbb{E}\,d(Y, y) - \tfrac12\,\mathbb{E}\,d(Y, Y'), \qquad Y, Y' \sim P,$$

estimated with two draws per step: $\tfrac12[d(y,\hat y_1) + d(y,\hat y_2)] - \tfrac12 d(\hat y_1, \hat y_2)$.
It is strictly proper whenever $d$ is a metric of strong negative type (Sejdinovic et al. 2013), so its
unique minimiser is the conditional law itself — the sampler is rewarded for spreading exactly as much
as the truth spreads, no more. It is the multivariate CRPS, i.e. the metric §12 already uses to judge
the rung. No discriminator, no diffusion steps, no second network.

**What the choice of $d$ decides.** Properness holds for the law of $\phi(y)$ whenever $d(a,b) =
\|\phi(a) - \phi(b)\|$. Three geometries were trained, differing only in $d$:

| arm | $d(a,b)$ | proper for |
|---|---|---|
| `es_wave` | mean $\lvert a_t - b_t \rvert$ over samples | the per-sample marginals of the waveform |
| `es_marg` | above $+ \lambda\cdot$ mean $\lvert \log\lvert A\rvert - \log\lvert B\rvert \rvert$ over 3 STFT scales | plus the per-bin marginals of log-magnitude |
| `es_slice` | above with the log-magnitude term replaced by $\mathbb{E}_\theta \lvert \theta\cdot(A_t - B_t)\rvert$, $\theta$ uniform on the sphere in $\mathbb{R}^F$, 64 directions per step | the **joint** law of a whole frame (Cramér–Wold) |

`es_marg` is a sum of univariate CRPSs and is therefore blind to cross-bin dependence — the same
blindness the frame-shuffle control was built to expose. `es_slice` is the cheapest strictly proper
score for the joint law of a frame: one matmul.

Two deterministic controls trained on the same batches: `det` (the paper's L1 + λ·MS-STFT at
λ = 1e-2, the frontier run's best) and `det_split` (L1 on the *low-passed* waveform + λ·MS-STFT, so no
waveform term ever asks for zero above 6 kHz — the cheapest deterministic fix a practitioner would try).

**Two falsifiable predictions, written down before the run.**
1. The ensemble mean of the sampler has the *same* high-band deficit as the deterministic model. The
   deficit is the conditional mean reporting how much of the band is unpredictable; it is a
   measurement, not a defect.
2. A single draw has no deficit.

If (1) holds, `deficit(mean) − deficit(single draw)` is a **predictability spectrum**: at each
frequency, 10·log10 of the fraction of target power that is predictable from the input. That number
also bounds the SNR any deterministic model can reach (§4).

---

## 1. Result

Five arms, identical batches and initialisation, 38 000 steps of batch 32 × 1 s (9.1 epochs of the
37.3 h Hub corpus), 4.25 h on an A100-40GB at 402 ms/step for all five. Two evaluation sets, because
the 4 September note showed the same checkpoint measures 5 dB apart on them: **Hub** (`test_utts[:12]`,
p236–238 from the corpus the models trained on, M = 16 draws) and **DataShare** (the original
`test_FULL.npz`, 12 utterances per held-out speaker, M = 32, evaluated independently on the Mac from
the Drive checkpoints). Every comparison below is within one set.

### 1.1 The table

Hub held-out, 12 utterances, ensemble of 16 (CRPS of a deterministic arm = MAE of its point forecast):

| arm | one draw: SNR | HB-LSD | deficit | **CRPS** | sliced CRPS | cross-bin corr err | HB κ | ensemble mean: SNR | deficit |
|---|---|---|---|---|---|---|---|---|---|
| naive upsampling | 19.0 | – | −∞ | – | – | – | – | – | – |
| `det` (paper loss, λ=1e-2) | 18.86 | 1.097 | −13.0 | 1.015 | 1.012 | 0.91 | 0.12 | – | – |
| `det_split` | 18.71 | 1.062 | −11.0 | 0.973 | 0.984 | 0.96 | 0.04 | – | – |
| **`es_marg`** | 18.14 | **1.075** | −7.9 | **0.583** | **0.576** | **0.62** | 0.02 | **18.90** | −16.6 |
| `es_slice` | 18.26 | 1.147 | −12.6 | 0.782 | 0.734 | 0.65 | 0.05 | 18.82 | −16.7 |
| `es_wave` | 17.64 | 1.161 | −9.7 | 0.733 | 0.677 | 1.75 | −0.01 | 18.74 | −17.0 |

DataShare held-out, 36 utterances, ensemble of 32, plus the 4 September deterministic λ-frontier
evaluated on the same utterances:

| arm | one draw: SNR | LSD | HB-LSD | deficit | **CRPS** | HB κ | coherent HB | mean-of-32: SNR | deficit | SNR gap (calibrated 2.88) |
|---|---|---|---|---|---|---|---|---|---|---|
| naive upsampling | 19.20 | – | – | −∞ | – | – | 0 | – | – | – |
| λ=1e-3 (4 Sep, 36k) | 19.16 | 1.546 | 1.778 | −17.4 | 1.779 | – | – | – | – | – |
| λ=1e-2 (4 Sep, 36k) | 18.37 | 0.962 | 1.097 | −4.5 | 1.012 | – | – | – | – | – |
| λ=1e-1 (4 Sep, 36k) | 17.69 | 0.948 | 1.084 | −2.4 | 1.000 | – | – | – | – | – |
| `det` | 18.35 | 0.960 | 1.095 | −4.5 | 1.011 | 0.06 | 1.8 % | – | – | – |
| `det_split` | 17.45 | 0.973 | 1.108 | −2.3 | 1.022 | 0.03 | 2.5 % | – | – | – |
| **`es_marg`** | 15.93 | 1.029 | 1.156 | **+0.6** | **0.686** | 0.01 | 1.6 % | **18.78** | −10.0 | **2.85** |
| `es_slice` | 16.23 | 1.025 | 1.108 | −3.7 | 0.704 | 0.02 | 1.4 % | 18.35 | −9.1 | 2.12 |
| `es_wave` | 14.20 | 1.320 | 1.328 | −0.6 | 0.884 | 0.00 | 0.4 % | 17.88 | −10.0 | 3.68 |

Figures: `overnight2/predictability_OV2_es.png` (energy ratio per band, samples vs ensemble mean; coherent
fraction above 6 kHz), `overnight2/frontier_OV2_es.png` (τ sweep against the λ frontier in the (SNR,
CRPS) and (SNR, deficit) planes), `overnight2/calibration_OV2_es.png` (PIT, spread–skill), and the Colab
counterparts in Drive `lisa_rtm/figures/ov2_*`.

### 1.2 What the table says, item by item

**The proper score moves by a third to a half, at zero inference cost.** CRPS on high-band
log-magnitude: 1.015 → 0.583 on Hub (−43 %), 1.011 → 0.686 on DataShare (−32 %). The sliced CRPS,
which sees cross-bin structure, moves by the same amount. This is the same 88k-parameter network, one
forward pass per draw; `det_split` — the practitioner's deterministic fix — moves CRPS by 4 %.

**The sampler contains a better point predictor than the point-trained model.** The ensemble mean
scores 18.90 dB against `det`'s 18.86 on Hub and 18.78 against 18.35 on DataShare, and above the
λ = 1e-2 and 1e-1 members of the deterministic frontier. Only the near-pure-L1 model (λ = 1e-3, 19.16)
beats it, and that model is naive upsampling with a −17 dB high band.

**Prediction (2) — a single draw has no deficit — holds on DataShare (+0.6 dB) and fails
in-distribution (−7.9 dB on Hub, −2.3 dB on the paper split).** The waveform calibration test of §4(iii)
says the same thing from the other side: the mean-minus-draw SNR gap is 2.85 dB against the required
2.88 on DataShare and 0.76 against 2.75 on Hub. The model is under-dispersed *in energy* on the
distribution it trained on, by roughly 6–8 dB in the 7–14 kHz bands — yet its PIT histogram on Hub is
nearly flat (end bins 0.076 and 0.117 against 0.059 uniform) and its τ sweep bottoms out exactly at
τ = 1 (CRPS 0.610 at τ = 1 vs 0.755 at 0.75 and 0.624 at 1.25). Calibrated in log-magnitude, short in
power: the draws get the typical bin right and under-produce the loud frames that carry the energy.
Nine epochs is also not convergence — `es_marg`'s probe deficit was still improving at step 38 000.

**Prediction (1) — the ensemble mean has the deterministic model's deficit — is false, and the reason
is the finding of the night.** The mean of 16 or 32 draws sits at −16.6 dB (Hub) and −10.0 dB
(DataShare), 3.5–5.5 dB *below* the deterministic arm. It cannot equal it, because the deterministic
arm's high band is not a conditional mean: its coherent fraction is 1.8 % and κ = 0.06–0.12, so **at
least 88 % of the energy the spectral loss puts above 6 kHz is uncorrelated with the truth.** The
deterministic model at λ = 1e-2 is a sampler with one frozen sample. The ensemble mean, by contrast, is
pure residual: its energy ratio equals the single-draw ratio minus 10·log10 M, which is what a mean of
independent draws with no coherent component does. The conditional mean of this band, for this model
class, is zero.

**The high band is unpredictable at the waveform level, for every arm, and the deterministic ceiling
is 0.3 dB above naive upsampling.** Coherent fraction by third-octave from 6 kHz, Hub EVAL12: 0.050,
0.002, 0.002, 0.001, −0.000, 0.002 for `det`; every other arm is within ±0.003 of the same numbers. The
5 % in the first band is the anti-aliasing filter's transition region (5.7–7 kHz), not prediction. The
bound of §4(ii) evaluates to 19.5 dB on DataShare (naive 19.20); the three λ-frontier models and both
deterministic arms tonight sit at 19.16, 18.37, 17.69, 18.35 and 17.45. **No deterministic LISA — at
any λ, any of the four training runs of this project — has beaten naive upsampling, and none can by
more than 0.3 dB.** Reaching the paper's 24.16 dB would require coherently predicting three quarters of
the power above 6 kHz from an 11-sample window; §1.4 tests whether context is the missing ingredient.

**The geometry of the score matters on speech, unlike on the graph.** `es_wave` (waveform L1 alone)
gets the energy right (−0.6 dB on DataShare) and the structure wrong: LSD 1.32 against 1.03, cross-bin
correlation error 1.75 against 0.62, PIT skewed to the low ranks. A per-sample proper score is proper for
per-sample marginals and nothing else, and the network's inductive bias did not rescue it here — a
per-sample MLP, unlike the graph's root-weight GNN, constrains nothing across bins. `es_slice`, proper
for the joint law of a frame, converged more slowly (under-dispersed: PIT top bin 41 % on Hub, SNR gap
2.12) and ends level with `es_marg` on DataShare (0.704 vs 0.686) but behind it on Hub (0.782 vs
0.583). At equal training budget the marginal log-magnitude score is the better bargain; the sliced
score needs either more steps or a larger weight than 1e-2 — one matmul is cheap, the gradient signal
through 64 random directions is diluted.

**The deterministic band-split fix restores energy and nothing else.** `det_split` reaches −2.3 dB
with κ = 0.03 and the same CRPS as `det`. This is PairNorm's fate in §5, observed.

**The τ sweep dominates the λ frontier.** In the (SNR, CRPS) plane (`frontier_OV2_es.png`, left) the
deterministic family runs flat at CRPS ≈ 1.0 from 17.7 to 18.4 dB and then climbs to 1.78 at 19.2 dB;
`es_marg`'s single knob traces 0.66–0.73 across the same SNR range. In the (SNR, energy) plane the
deterministic family is about 1 dB better at equal mean deficit, because its hallucinated energy sits
more in the high bands where speech has little power and the SNR penalty is smaller. Both are honest
readings of the same fact: incoherent energy costs SNR wherever it is put, and the only question is
whether it carries a distribution (the sampler) or a single frozen draw (the deterministic model).

### 1.3 Against the post-hoc stochastic rung

The notebook's own ladder (§10–§12 of `lisa_rtm.ipynb`) was run on tonight's `det` arm, same 12 Hub
utterances, same CRPS definition, fitted on 200 training-speaker utterances. The controls behave: the
mis-specified map hurts (HB-LSD 1.176 vs T0 1.097), the frame shuffle is far worse (1.504), shaped
noise worse still (1.547).

| | HB-LSD | deficit | SNR | CRPS |
|---|---|---|---|---|
| T0 identity (`det`) | 1.097 | −13.2 | 18.86 | 1.015 |
| T1 quantile, λ=1 (best deterministic rung) | 1.061 | −8.0 | 18.36 | – |
| T3 conditional, λ=1 | 1.070 | −9.9 | 18.60 | 0.982 |
| **S: noise through the closed-form conditional map** (16 draws) | 1.044 | −9.6 | 18.58 | **0.753** |
| **`es_marg`: noise through the learned map** (16 draws, one draw for HB-LSD/SNR) | 1.075 | −7.9 | 18.14 | **0.583** |

The hand-built rung takes 23 % off the point forecast; the learned one takes 43 %, i.e. a further 23 %
off the rung. The deterministic rungs are where the 4 September note left them: T1 buys 3 % of
HB-LSD. Every rung of the ladder except S is now strictly dominated by a network that costs nothing
extra to run, and S is dominated on the only metric it was built to win.

### 1.4 The receptive-field test: context does not make the band coherent

`LISASW` adds a residual stack of dilated convolutions (k = 3, dilations 2…64) on LISA's 32-d latents:
receptive field 263 input samples = 22 ms, two to four pitch periods, +11k parameters, everything else
unchanged. Two arms on the same batches for 26 000 steps (6.2 epochs, 1.3 h): the paper's loss and the
marginal energy score. DataShare set, 36 utterances, M = 32:

| arm | SNR | LSD | HB-LSD | deficit | CRPS | HB κ | coherent fraction by band from 6 kHz | mean-of-32 SNR |
|---|---|---|---|---|---|---|---|---|
| `det` (11-sample context, 38k steps) | 18.35 | 0.960 | 1.095 | −4.5 | 1.011 | 0.06 | .050 .002 .002 .001 .000 .002 | – |
| `wide_det` (22 ms context, 26k steps) | 18.39 | 0.931 | 1.062 | −4.4 | 0.976 | 0.06 | .102 .001 .005 .000 .000 .000 | – |
| `es_marg` (11-sample) | 15.93 | 1.029 | 1.156 | +0.6 | 0.686 | 0.01 | .050 −.001 .001 −.002 −.002 −.001 | 18.78 |
| `wide_es_marg` (22 ms) | 14.38 | 1.055 | 1.171 | +3.4 | 0.654 | 0.00 | .078 .000 .002 .000 .000 .002 | 18.84 |

**Falsified.** Twenty-two milliseconds of context leaves the coherent fraction above 7 kHz at
0.000–0.005 for both objectives. The only movement is in the first band (5.7–7 kHz, the anti-aliasing
transition region), where the deterministic arm's κ rises from 0.31 to 0.38 — it reproduces slightly
more of the filter's roll-off, which is baseband information. LSD improves by 0.03 and CRPS by 0.03–0.04
from a better *envelope*; SNR does not move (18.39 vs 18.35, ceiling 19.5). The sampler with context is
over-dispersed on this set (+3.4 dB, SNR gap 4.46 against 2.88, τ* = 0.75) where the narrow one was
calibrated (+0.6 dB, 2.85) — the extra capacity went into spread, not coherence.

The Colab evaluation on the Hub sets (`overnight2/colab_results/results_OV2_wide.json`) says the same:
`wide_det` coherent fraction 0.008 / 0.014 / 0.019 and κ 0.05–0.08 on the three splits, SNR 18.82 vs
18.86 for the narrow model; `wide_es_marg` single-draw deficit −6.1 dB in-distribution (narrow −7.9),
+0.2 dB on the paper split, CRPS 0.606 (narrow 0.583).

So within this model family, the high band of speech is incoherent with the input regardless of
context, at least out to 22 ms and 6 epochs. That is consistent with what the band physically contains
— fricative and breath noise, and harmonics whose period-to-period phase is not stable — and it
sharpens the reading of the paper's 24.16 dB: the number is not reachable by coherent prediction from
the 12 kHz input on this data with this family, with or without context. The remaining explanations are
the SNR convention and the down-sampling operator, as the 4 September note already listed.


## 2. The graph experiment — the same fact, in the graph Fourier basis

Run on the Mac while the A100 trained (`overnight2/gnn_toy.py`, 2 minutes, CPU). A 600-node 8-NN
geometric graph; inputs live in the 40 lowest Laplacian modes; targets are a smooth nonlinear
function of the inputs plus an **unpredictable** component in modes ≥ 200 whose amplitude depends on
the input. Two architectures × two objectives, one seed:

| model | deficit by graph-frequency band (dB), single draw | ensemble mean of 32 | CRPS |
|---|---|---|---|
| shallow GNN (root weight), **MSE** | +0.0 / −0.2 / −3.3 / **−15.5** / −16.1 | same | 0.468 |
| shallow GNN, **energy score** | +0.1 / +1.4 / +6.1 / **−1.4** / −1.3 | −0.0 / −0.2 / −2.5 / **−13.0** / −13.3 | **0.356** |
| deep GNN (8 propagation layers, no skip), MSE | −0.6 / −6.2 / −12.0 / −21.8 / −19.2 | same | 0.610 |
| deep GNN, energy score | +1.8 / −0.9 / −5.2 / **−16.4** / −15.4 | −0.4 / −7.4 / −10.8 / −21.2 / −19.2 | 0.469 |

Bands: modes 0–40, 40–120, 120–200, 200–400, 400–600.

Read the two right-hand bands. **Shallow + MSE** loses 15.5 dB of high-graph-frequency energy — the
conditional mean has none, and this is *objective-induced* over-smoothing: nobody in the GNN
over-smoothing literature measures it because it does not depend on depth. **Shallow + energy score**
puts it back (−1.4 dB) in a single draw, and the ensemble mean of the same sampler reopens the hole to
−13.0 dB: prediction (1) holds on a graph. **Deep + energy score** cannot put it back (−16.4 dB in a
single draw): eight propagation layers are a low-pass filter and no objective can undo that. That is
*architecture-induced* over-smoothing, the classical kind.

So one model gives a **two-part decomposition of any measured over-smoothing**:

* deficit of a single draw = what the *architecture* cannot express;
* deficit of the ensemble mean minus that = what the *objective* removed by taking a conditional
  expectation.

In LISA the Fourier-feature control had to be a second architecture to reach the same conclusion. A
sampler trained under a proper score reaches it with one.

### 2.1 Does the geometry of the score matter on the graph? A clean negative

`overnight2/gnn_toy_scores.py` trains the shallow GNN under three objectives that are each strictly
proper for something different: the sum of per-node CRPS (marginals only), the Euclidean energy score
(joint law of all nodes), and the sliced energy score (joint law, one-dimensional power). Prediction
before running: the marginal score would put its noise in the wrong graph-frequency bands (spatially
white), the joint scores in the right ones.

| objective | single-draw band ratio (dB) | per-node CRPS | CRPS of graph-Fourier coefficients, modes ≥ 200 |
|---|---|---|---|
| per-node CRPS | +0.0 / +1.1 / +5.9 / −1.4 / −1.0 | 0.343 | 0.481 |
| Euclidean energy score | +0.1 / +1.4 / +6.1 / −1.4 / −1.3 | 0.356 | 0.481 |
| sliced energy score | +0.1 / +1.7 / +6.2 / −1.5 / −2.1 | 0.380 | 0.487 |

**The prediction failed.** All three put the noise in the same bands to within a decibel, and the
joint-structure score (CRPS of the graph-Fourier coefficients) is identical across them. On this
toy the *architecture's* inductive bias — a root weight that can compute node-minus-neighbourhood,
i.e. a high-pass — decided where the noise went, and the scoring rule only decided how much. The
+6 dB leak into modes 120–200 (a band where the target carries almost nothing) is shared by all
three, which says the same thing from the other side: a joint-proper score did not prevent it either,
because a band with no energy contributes nothing to any of these scores.

Kept because it sharpens the audio question rather than settling it: whether `es_marg` and `es_slice`
differ on speech is now a genuine empirical question, not a foregone conclusion.

## 3. Latency

Same architecture as LISA plus eight input channels, so nothing changes at inference. One stochastic
draw, `LISAS` forward including noise sampling:

| device | 1 s of audio | 20 ms chunk | real-time factor |
|---|---|---|---|
| A100-40GB (Colab) | 1.51 ms (p95 1.53) | 1.34 ms (launch-bound) | 660× |
| Apple M4 CPU, 10 threads | 16.5 ms (p95 16.6) | 0.67 ms (p95 0.70) | 60× |
| Apple M4 GPU (MPS) | 14.2 ms | 1.02 ms | 70× |
| wide-context `LISASW`, M4 CPU | 18.8 ms | – | 53× |

Algorithmic look-ahead is 5 input samples for the encoder plus 1 for the decoder's right neighbour:
0.5 ms at 12 kHz. With a 20 ms streaming buffer the end-to-end latency is ~21 ms on a laptop CPU; the
50 ms budget is met with a factor of two to spare and no GPU. The post-hoc transport rung it replaces
cost 33 ms per second of audio in numpy.

## 4. Three identities that turn the deficit into a measurement

All three follow from one line. Write $y = \mu(x) + \varepsilon$ with $\mu = \mathbb{E}[y\mid x]$ and
$\Sigma = \mathrm{Cov}(y \mid x)$, and let $Y$ be a draw from a *calibrated* sampler, $Y \sim p(y\mid x)$
independently of $y$.

**(i) The deficit of the conditional mean is the predictable fraction.** Per frequency band $b$,
$\mathbb{E}\,P_\mu(b) / \mathbb{E}\,P_y(b) = \rho(b) := 1 - \operatorname{tr}\Sigma_b / \mathbb{E}P_y(b)$. So the
energy-ratio curve of the ensemble mean, in dB, is $10\log_{10}\rho(b)$: a **predictability spectrum**.
The $M$-draw mean is biased upward by the residual $(1-\rho)/M$; the bias-corrected estimate is
$\hat\rho = (M r - 1)/(M - 1)$ where $r$ is the measured ratio.

**(ii) The predictable fraction bounds every deterministic model.** Any point predictor's error power
is at least $\sum_b (1-\rho(b))\,P_y(b)$, so $\mathrm{SNR}_{\det} \le 10\log_{10}\big(P_y / \sum_b (1-\rho(b)) P_y(b)\big)$.
This is a property of the data and the input band, not of any architecture, and it is the number to
compare a published SNR against before trying to reproduce it.

**(iii) A calibrated sampler's single draw has exactly twice the error of the mean.**
$\mathbb{E}\|y - Y\|^2 = \mathbb{E}\|y-\mu\|^2 + \mathbb{E}\|Y-\mu\|^2 = 2\operatorname{tr}\Sigma$. Hence
$\mathrm{SNR}(\text{mean}) - \mathrm{SNR}(\text{one draw}) = 3.01$ dB when the error is entirely
unpredictable, or $10\log_{10}\!\big(2/(1+1/M)\big)$ = 2.75 dB against an $M{=}16$ mean. A smaller gap
means the sampler is under-dispersed, a larger one over-dispersed. This is a calibration test that
needs nothing but two SNR numbers — the waveform-domain twin of the PIT histogram.

The same arithmetic is why SNR, LSD and any other RMSE-type score *must* rank a perfect sampler below a
perfect point predictor (LSD by a factor $\sqrt2$ in the noise-only limit: two independent Rayleigh
draws differ by $\sqrt2 \times 0.557$ log10-power units, against $0.557$ for the mean). A metric that
punishes correctness by a known constant is not a judge; it is a ruler that has to be read with the
constant subtracted.

Numbers for this data are in §1.

### 4.1 The coherent fraction, and why the deterministic deficit is not a predictability estimate

A dry run of the analysis on the step-9000 checkpoints exposed a flaw in prediction (1) as first
stated. The deterministic arm's high-band energy is set by λ, not by predictability: the 4 September
frontier run moved the deficit from −17 to −2 dB by turning λ from 1e-3 to 1e-1 with *no* gain in SNR.
So the energy ratio of a model trained with a spectral term is not $\rho$; part of that energy is
hallucinated. The estimator that separates the two is the **coherent fraction**

$$\hat\rho(b) = \frac{\mathrm{Re}\langle Y_b, P_b\rangle}{\langle Y_b, Y_b\rangle}, \qquad
\kappa(b) = \frac{\mathrm{Re}\langle Y_b, P_b\rangle}{\langle P_b, P_b\rangle},$$

with $Y, P$ the complex STFTs of target and prediction summed over band $b$. For the exact conditional
mean $\hat\rho = \rho$ and $\kappa = 1$ (orthogonality of the residual); for a predictor that adds
uncorrelated energy $\kappa < 1$; and for an $M$-draw ensemble mean $\hat\rho$ is unbiased in $M$
because the residual noise is uncorrelated with $y$. The baseband gives the alignment check: coherence
and $\kappa$ both 1.00 ± 0.02 for every arm.

**Mid-training reading (step 9000, 12 held-out utterances):** $\kappa$ above 6 kHz is **0.05** for the
deterministic arm and **0.01** for every sampler; the coherent fraction of the high band is **2 %**
(deterministic) and **1.5 %** (ensemble mean of `es_marg`). Ninety-five percent of the energy the
spectral loss places above 6 kHz is uncorrelated with the truth. The deterministic SNR ceiling from
(ii) is then **18.2 dB against naive upsampling's 17.9 dB on the same utterances** — a 0.3 dB ceiling
that every deterministic arm this project has trained (λ = 1e-3, 1e-2, 1e-1, band-split) sits under.
(Final-checkpoint numbers in §1.)

This has a structural reading. LISA's encoder sees 11 input samples at 12 kHz — 0.9 ms, less than one
pitch period — so coherent continuation of harmonics above 6 kHz is impossible *by construction*, and
the deficit is the model correctly reporting that. It also puts the paper's 24.16 dB in perspective:
reaching it would require coherently predicting roughly three quarters of the high-band power from a
0.9 ms window. The receptive-field hypothesis is tested directly in §1.4 (`LISASW`: the same model with
a dilated residual stack on the latents, 22 ms of context, +11k parameters).

## 5. What this means for geometric deep learning

The audio problem is a laboratory: the "high band" is defined exactly (above the input Nyquist), its
energy is measurable in dB, and the ground truth is available at 48 kHz. What the laboratory shows is
not about audio.

**1. Over-smoothing has two causes, and the standard diagnosis only sees one.** The GNN literature
treats over-smoothing as an *architectural* pathology: repeated propagation is a low-pass filter on
the graph Laplacian, node features converge, and the remedies (residuals, PairNorm, DropEdge, gradient
gating, shallow depth) all act on the architecture. §2 shows a second cause that has nothing to do
with depth: any deterministic model trained with a pointwise loss on a target that is not a function
of its input outputs a conditional barycentre, and a barycentre has no energy where the target is
unpredictable — which, on a graph, is precisely the high graph-frequency band. A two-layer GNN with a
root weight is not over-smoothed by depth and still loses 15 dB there. Both causes produce the same
symptom in the graph Fourier basis; only the sampler tells them apart.

**2. PairNorm-type fixes are transport maps on statistics, and they restore energy, not information.**
PairNorm recentres and rescales features to a fixed total pairwise distance — it is the diagonal Bures
map T2 applied per layer. The ladder result on the honest audio source (T1 buys 1.7 % of HB-LSD) is a
warning about what such fixes can and cannot do: they can put variance back, they cannot put the
*conditional structure* back, because the information was destroyed by the expectation, not by the
normalisation. If a graph task is one-to-many at the node level (conformer generation, trajectory
forecasting, inverse design, any regression whose target has an unpredictable component), a
normalisation layer is treating the symptom.

**3. The fix is an objective, not an architecture, and it costs nothing at inference.** Concatenate a
few Gaussian channels to the node features, train under the energy score with two forward passes per
step, and sample once at inference. The same GNN, the same latency, and now a calibrated conditional
sampler whose ensemble mean is the L2-optimal point predictor (§1: SNR of the mean vs the
deterministic arm). For a GDL practitioner this is the recipe: `x -> [x, eps]`, `loss = ES`, done.

**4. The scoring rule's geometry is a modelling choice, and the toy says the architecture may
override it.** On the graph, per-node CRPS, Euclidean and sliced energy scores produced the same
spatial structure (§2.1); the root-weight architecture's inductive bias set where the noise went. This
is worth knowing before spending effort on a fancy joint score: check first whether the architecture
already constrains the joint structure. The audio comparison `es_marg` vs `es_slice` (§1) is the same
question on a problem where the architecture (a per-sample MLP) does *not* constrain cross-bin
structure.

**5. A predictability spectrum is a stopping rule.** The energy ratio of the ensemble mean, per
frequency band, is the fraction of target power that is predictable from the input *within this model
class*. It bounds the SNR any deterministic model can reach (§4), and it tells you when a lower loss is
no longer possible without hallucinating. On a graph, the same curve over Laplacian bands says which
graph frequencies of the target are predictable from the input at all — a quantity that is
independent of depth, and a much sharper question than "does my GNN over-smooth".

## 6. Artefacts

| what | where |
|---|---|
| five checkpoints, step 38 000 | Drive `lisa_rtm/checkpoints/OV2_es/{det,det_split,es_marg,es_slice,es_wave}.pt` |
| wide-context checkpoints | Drive `lisa_rtm/checkpoints/OV2_wide/{wide_det,wide_es_marg}.pt` |
| training histories | Drive `lisa_rtm/ov2_history_OV2_es.json`, `train_OV2_es.log` |
| Colab evaluation (Hub sets, τ sweep, latency, ladder) | `overnight2/colab_results/*.json`, Drive `lisa_rtm/ov2/` |
| local independent analysis (DataShare set, coherence, bound, frontier) | `overnight2/analysis_OV2_es.json`, `analysis_OV2_es.log`, reproduced by `overnight2/analysis_local.py OV2_es 32 36 --extra=XL4_b64_1s` |
| figures | `overnight2/predictability_OV2_es.png`, `frontier_OV2_es.png`, `calibration_OV2_es.png`, `gnn_toy.png`; Drive `lisa_rtm/figures/ov2_*` |
| **listening examples** — truth, naive, `det`, `det_split`, one draw and a second draw of each sampler, and each sampler at τ = 0, for three held-out utterances | Drive `lisa_rtm/ov2/audio/u{0,5,10}_*.wav` |
| graph experiments | `overnight2/gnn_toy.py`, `gnn_toy_scores.py`, results `*.json`, log |
| code | `overnight2/c0_boot.py` … `c5_launch_wide.py` (Colab cells, exec'd from Drive `lisa_rtm/ov2/`), `c1_model.py` (LISAS, losses, trainer), `c1b_wide.py` (LISASW) |
| CPU latency | `overnight2/cpu_latency.json` |

Colab session: GitHub-opened `lisa_rtm.ipynb`, A100-SXM4-40GB. The corpus lives in host RAM
(`HostCorpus`); on this card a GPU-resident 37 h corpus is 32 GB and OOMs the two-draw arms.

## 7. Judged by LSD + ViSQOL (evening of 8 September)

The user's primary metrics are LSD and ViSQOL, with SNR two rungs down. Every condition of the night
was re-scored with `evaluation.py` (ViSQOL speech mode at 16 kHz via the pure-Python `visqol-python`
port with the upstream lattice mapping, plus wideband PESQ), and additionally with ViSQOL audio mode at
48 kHz and the mapping-free NSIM of each. Twenty-nine conditions × 48 utterances (Hub 12, DataShare 36);
per-utterance values in `overnight2/visqol_results/visqol_per_utt.json`, paired deltas with bootstrap
CIs in `visqol_tables.md`, figure `lsd_vs_visqol.png`.

### 7.1 Speech-mode ViSQOL and PESQ cannot judge this task

ViSQOL speech mode analyses 50 Hz–8 kHz in 21 ERB bands whatever the input rate (one band, centred at
6.8 kHz, overlaps the reconstructed 6–24 kHz); wideband PESQ stops at 8 kHz too. On DataShare the
**naive upsampler with an empty high band scores best on both** (ViSQOL-sp 4.245, PESQ 4.39), the
λ = 1e-3 model with a −17 dB high band is second (4.231), and the deterministic arms sit within 0.05 of
one another. On Hub, naive is 0.15 below `det` because the decimation filter's transition band
(5.7–6 kHz, the metric's top band) is rolled off and every model restores it. A metric that prefers the
band left empty, or cannot tell −17 dB from −4 dB, is not measuring the thing this project changes.
What it *does* see is the baseband, and there it is sharp: the sampler's single draw loses 1.06 MOS on
Hub and 0.26 on DataShare, PESQ 0.5, from the −23 dB incoherent floor its noise pathway leaves below
6 kHz (baseband κ 0.995 vs 0.999). §7.4 tests whether passthrough of the given baseband removes that.

### 7.2 Audio-mode ViSQOL at 48 kHz sees the band, and rewards energy

| condition | Hub: LSD | ViSQOL-au | DataShare: LSD | ViSQOL-au |
|---|---|---|---|---|
| naive | 4.776 | 1.58 | 4.304 | 1.93 |
| λ = 1e-3 | 1.715 | 1.92 | 1.546 | 2.24 |
| `det` (λ = 1e-2) | 0.963 | 2.82 | 0.960 | 2.73 |
| **λ = 1e-1** | **0.899** | **3.02** | 0.948 | 2.85 |
| **`wide_det`** (22 ms context) | 0.941 | 2.93 | **0.931** | 2.89 |
| `det_split` | 0.934 | 2.73 | 0.973 | 2.66 |
| ladder T1 on det | 0.921 | 2.93 | 0.961 | 2.88 |
| ladder T3 on det | 0.929 | 2.89 | 0.965 | 2.86 |
| ladder S on det | 0.910 | 2.91 | 0.994 | 2.83 |
| `es_marg` one draw, τ = 1 | 0.963 | 2.74 | 1.028 | 2.61 |
| `es_marg` one draw, τ = 0.5 | 1.132 | 2.46 | 0.942 | 2.90 |
| `es_marg` mean of 16 | 1.128 | 2.59 | 0.942 | 2.88 |
| `wide_es_marg` mean of 16 | 1.083 | 2.58 | 0.962 | **2.90** |

Audio-mode ViSQOL climbs monotonically with deterministic high-band energy (1.6 → 1.9 → 2.8 → 3.0 from
naive to λ = 1e-1) and LSD falls with it. **On the user's two metrics the winners are deterministic:
λ = 1e-1 on Hub, the wide-context deterministic model on DataShare, and the post-hoc T1 rung adds
+0.11 / +0.15 audio ViSQOL to `det` at no LSD cost on both sets** — the ladder, dead on CRPS, is alive
on ViSQOL. The learned sampler's single draw at τ = 1 is 0.08–0.12 below `det` in audio mode and no
better on LSD; its over-smoothed ensemble mean and its τ = 0.5 draws reach the top of the DataShare
table (2.88–2.90, LSD 0.942) and the bottom half of the Hub table, the two sets again disagreeing because
of their 5–8 dB difference in high-band level.

### 7.3 What this means

LSD and NSIM are both maximised by a conditional mean of the log-spectrogram, so a metric suite built on
them ranks the arms in the reverse order of the proper score: the sampler wins CRPS by 43 % and loses
speech ViSQOL by a MOS point; its ensemble mean, which is the *most* over-smoothed object in the table,
is the sampler's best entry on both LSD and ViSQOL. This is §4's arithmetic observed on a perceptual
proxy, and it is the decision the project has to make explicitly: **optimise for LSD + ViSQOL and the
answer is a deterministic model with a large spectral weight, more context, and a T1 rung on top; optimise
for a calibrated distribution and the answer is the sampler.** They are different objectives and no
single model tonight is best on both. The retraining that follows from the first choice is a λ sweep
above 1e-1 with baseband passthrough (`overnight2/c8_launch_lambda.py`, written, needs the A100).

### 7.4 Baseband passthrough

For bandwidth extension the band below the input Nyquist is given, so the honest system output is the
input's own baseband plus the model's band above it (brick-wall at 6 kHz). Hub set, original → passthrough:

| condition | LSD | ViSQOL-sp | PESQ | ViSQOL-au |
|---|---|---|---|---|
| `det` | 0.963 → 0.956 | 4.12 → 4.18 | 4.24 → 4.25 | 2.82 → 2.82 |
| λ = 1e-1 | 0.899 → 0.893 | 4.17 → 4.23 | 4.25 → 4.27 | 3.02 → 3.02 |
| `wide_det` | 0.941 → 0.934 | 4.10 → 4.15 | 4.25 → 4.27 | 2.93 → 2.93 |
| ladder T1 on det | 0.921 → 0.914 | 4.12 → 4.11 | 4.14 → 4.15 | 2.93 → 2.92 |
| **`es_marg`, one draw τ = 1** | 0.963 → **0.933** | **3.06 → 4.11** | **3.72 → 4.04** | 2.74 → 2.77 |
| `es_marg`, τ = 0.75 | 1.018 → 0.997 | 3.22 → 4.17 | 4.02 → 4.25 | 2.65 → 2.68 |
| `es_marg`, mean of 16 | 1.128 → 1.122 | 3.90 → 4.16 | 4.29 → 4.32 | 2.59 → 2.59 |
| passthrough + empty high band (floor) | 5.61 | 3.95 | 4.29 | 1.57 |
| passthrough + true high band (ceiling) | 0.105 | 4.51 | 4.62 | **4.73** |

**The sampler's speech-ViSQOL and PESQ penalty was baseband leakage, not high-band texture.** With the
given baseband restored, one draw at τ = 1 sits level with the deterministic arms on speech ViSQOL
(4.11 vs 4.11–4.23) and PESQ, and *beats* `det` on LSD (0.933 vs 0.956). Its audio-mode gap to `det`
shrinks to 0.05 and to λ = 1e-1 stays at 0.25: what audio-mode ViSQOL still prefers is the
deterministic model's larger, smoother high-band energy. The deterministic arms gain ~0.06 speech
ViSQOL and 0.007 LSD from passthrough themselves — the baseband LISA re-synthesises is very slightly
worse than the input it was given. Passthrough should be the default for every model in this project.

The two bound rows calibrate the whole table: an empty high band scores 1.57 in audio mode, the true high
band 4.73, and the best model tonight (λ = 1e-1, 3.02) has recovered **46 % of that range**. On LSD the
picture is the same: 5.6 → 0.105 available, 0.89 achieved. The room above every model here is large, and
§7.2 says it is climbed with deterministic energy plus context, not with sampling — on these two metrics.

DataShare set, original → passthrough (36 utterances):

| condition | LSD | ViSQOL-sp | PESQ | ViSQOL-au |
|---|---|---|---|---|
| `det` | 0.960 → 0.954 | 4.20 → 4.25 | 4.26 → 4.27 | 2.73 → 2.73 |
| λ = 1e-1 | 0.948 → 0.945 | 4.21 → 4.25 | 4.29 → 4.30 | 2.85 → 2.85 |
| `wide_det` | 0.931 → **0.926** | 4.23 → 4.28 | 4.30 → 4.30 | 2.89 → 2.89 |
| ladder T1 on det | 0.961 → 0.957 | 4.14 → 4.16 | 4.17 → 4.18 | 2.88 → 2.87 |
| `es_marg`, one draw τ = 1 | 1.028 → 1.008 | **3.94 → 4.19** | **3.82 → 4.01** | 2.61 → 2.62 |
| `es_marg`, one draw τ = 0.75 | 0.945 → **0.930** | 4.06 → 4.24 | 4.09 → 4.21 | 2.76 → 2.77 |
| `es_marg`, mean of 16 | 0.942 → 0.938 | 4.20 → 4.25 | 4.34 → 4.35 | 2.88 → 2.88 |
| passthrough + empty high band (floor) | 4.89 | 4.25 | 4.25 | 1.93 |
| passthrough + true high band (ceiling) | 0.110 | 4.46 | 4.58 | 4.73 |

Same story on the second set: the sampler's speech-mode penalty is gone with passthrough (4.19 vs 4.25
for `det`), PESQ recovers, and a τ = 0.75 draw with passthrough reaches LSD 0.930, within 0.004 of the
best deterministic model. The floor row is the sharpest statement of §7.1: on this set the empty high
band scores the *same* speech ViSQOL and PESQ as the true high band to within 0.2 and 0.3 MOS, so a
16 kHz speech metric has essentially nothing to say about a 6–24 kHz reconstruction.
