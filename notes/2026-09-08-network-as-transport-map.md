# Research note — the network is the transport map: proper scoring rules end regression to the mean at zero inference cost

**Date:** 2026-09-08 (overnight run, unattended)
**Question:** the deterministic ladder failed on an honest source (T1 moved HB-LSD 1.7 %), while the
zero-parameter stochastic rung moved CRPS 21 %. If a hand-built conditional transport map on noise is
the only thing that works, what happens when the *network itself* is trained to be that map?
**Headline:** (filled in §1)

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

(filled in after training)

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

## 3. Latency

(filled in)

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

(filled in)

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
