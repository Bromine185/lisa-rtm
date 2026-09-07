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

## 4. The deficit as a bound on deterministic SNR

For an L2-optimal predictor, $\mathbb{E}\|y\|^2 = \|\mathbb{E}y\|^2 + \operatorname{tr}\mathrm{Cov}$, so
the energy ratio of the conditional mean at frequency $f$ is the predictable fraction $\rho(f)$ of the
target's power there. Any deterministic model's error power is at least the unpredictable power
$\sum_f (1-\rho(f)) P_y(f)$. On this data naive polyphase upsampling scores 19.2 dB, i.e. the whole
band above 6 kHz plus baseband imperfection is 1.2 % of total power. (numbers filled in from the
measured curve)

## 5. What this means for geometric deep learning

(filled in)

## 6. Artefacts

(filled in)
