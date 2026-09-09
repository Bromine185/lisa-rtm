# Model names — complete expansions

Every model, arm, rung, condition, run tag, evaluation set and metric abbreviation used anywhere in
this repository (README, TODO, the three research notes, `overnight/`, `overnight2/`,
`build_notebook.py`, `evaluation.py`), with the full expansion of the name and the exact definition
behind it. Every entry is verified against the source it points to. When a note uses a prose alias
("wide-context det", "λ = 1e-1 (4 Sep, 36k)"), the alias is listed under the canonical name.

**Project name.** `lisa-rtm` = **LISA, Regression To the Mean**: the deficit of a deterministic
one-to-many model is the conditional mean collapsing toward zero energy in the unpredictable band.

## 0. How to read a name

| pattern | reads as | example |
|---|---|---|
| `<kind>_l<λ>` | 4 Sep arm: decoder kind + spectral weight | `relu_l1e-2` = ReLU decoder, λ = 10⁻² |
| `det`, `det_split`, `es_<geometry>` | 8 Sep arm: objective family | `es_marg` = energy score, marginal geometry |
| `wide_<arm>` | same objective on the wide-context encoder `LISASW` | `wide_det` |
| `det_l<λ>` | planned λ sweep, decimal λ | `det_l0.3` = paper loss, λ = 0.3 |
| `lam<λ>` | 4 Sep checkpoint reloaded as a ViSQOL condition | `lam1e-1` = `relu_l1e-1` |
| `<arm> tau=<τ>` | one stochastic draw at noise temperature τ, seed 0 | `es_marg tau=0.75` |
| `<arm> mean16` | mean of 16 draws at τ = 1, seeds 0–15 | `es_marg mean16` |
| `ladder <rung> on det` | post-hoc transport rung applied to the `det` arm's output | `ladder T1 on det` |
| `<condition> \| passthrough` | given baseband + that condition's band above 6 kHz | `det \| passthrough` |
| `T<n>`, `S` | rung of the post-hoc transport ladder | `T3` |
| `XL…`, `OV2…`, `FULL…`, `SMOKE…` | run tag = checkpoint directory | `OV2_es` |

## 1. Architectures

| name | expansion | definition |
|---|---|---|
| **LISA** | **L**ocal **I**mplicit representation for **S**uper resolution of **A**rbitrary scale. Kim, Lee, Hong, Ok, *Learning Continuous Representation of Audio for Arbitrary Scale Super Resolution*, ICASSP 2022, arXiv 2111.00195; official code `ml-postech/LISA`. | Reimplemented in `build_notebook.py` (`class LISA`, line 1123). Conv encoder with kernels (7, 3, 3, 1) and channels (16, 32, 64, 32), so each 32-d latent sees 11 input samples = 0.9 ms at 12 kHz. Decoder: 5-layer ReLU MLP, hidden width 144, that reads a relative coordinate in [−1, 1] plus the latents of the anchor sample and its two neighbours and emits one amplitude at any continuous time coordinate. 86,881 parameters (the paper reports ~89k). ×4 upsampling, 12 kHz → 48 kHz. Trained on L1 waveform loss + λ · multi-scale STFT loss. |
| `LISAEncoder`, `LISADecoder` | the two halves of LISA | `build_notebook.py` lines 1094 and 1109. `LISADecoder` is shared by every model below. |
| `MultiScaleSTFTLoss` (MS-STFT) | **M**ulti-**S**cale **S**hort-**T**ime **F**ourier **T**ransform loss | Spectral convergence + log-magnitude L1 at three FFT sizes (n_fft, n_fft/2, n_fft/4). `build_notebook.py` line 1147. Its weight is `lambda_spec` (λ); the paper's released code uses 1e-3. |
| **LISAFF** | LISA with **F**ourier-**F**eature coordinates (Tancik et al. 2020) | `overnight/cell2_trainer.py` line 128. Identical encoder, latents, loss and batches; only the decoder's view of the relative coordinate changes, `c → [c, sin(πkc), cos(πkc)]` for k = 1…6. The architecture control: if it emits more high band than the ReLU arm at the same λ, over-smoothing is partly spectral bias rather than the loss. Arm kind `"ff"`. |
| **LISAS** | LISA, **S**tochastic | `overnight2/c1_model.py` line 19. LISA with `N_NOISE = 8` Gaussian noise channels concatenated to the waveform at the encoder input, at the input sample rate. With the noise at zero it is LISA exactly (same init, same latency); 87,777 parameters, i.e. +896 weights (8 extra input channels × 16 first-layer filters × kernel 7). The single class used for every 8 Sep arm, deterministic and stochastic. Its noise temperature is `tau` (τ). |
| **LISASW** | LISA, **S**tochastic, **W**ide context | `overnight2/c1b_wide.py` line 11. LISAS plus a residual stack of dilated 1-D convolutions (kernel 3, dilations 2, 4, 8, 16, 32, 64) on the 32-d latents, followed by a 1×1 projection. Receptive field 11 + 2·(2+4+8+16+32+64) = 263 input samples = 22 ms, two to four pitch periods. 107,457 parameters (the 8 Sep note says "+11k"; the measured count in `overnight2/analysis_OV2_wide.log` is +19,680 over LISAS). The receptive-field test of note §1.4. |
| `MODEL_CLS` | — | Global in `train_ov2` that selects LISAS or LISASW (`overnight2/c5_launch_wide.py` line 6). |
| `HostCorpus`, `GPUCorpus` | not models | The concatenated training corpus held in host RAM (`overnight2/c1_model.py` line 81) or on the GPU (`overnight/cell2_trainer.py` line 22). A 37 h corpus on the GPU is 32 GB and OOMs the two-draw arms on an A100-40GB, hence the host version. |
| shallow GNN | shallow **G**raph **N**eural **N**etwork with root weight | `overnight2/gnn_toy.py` line 110, depth 2, `skip=True`. The self/root weight lets it compute node-minus-neighbourhood, a high-pass on the graph. Not over-smoothed by depth. |
| deep GNN | 8-layer graph neural network, no skip | Same file, depth 8, `skip=False`. Eight pure propagation layers are a low-pass filter on the graph Laplacian: architecture-induced over-smoothing. |

## 2. Training objectives (arm kinds)

The loss decides what the network converges to; the name of an arm is its loss.

| kind | expansion | loss |
|---|---|---|
| `relu` | ReLU-decoder LISA, the paper's loss | `L1(waveform) + λ · MS-STFT`. 4 Sep naming; kind passed to `make_arm` in `overnight/cell2_trainer.py`. |
| `ff` | Fourier-feature decoder (LISAFF), the paper's loss | Same loss, LISAFF architecture. |
| `det` | **det**erministic, the paper's loss | `L1(waveform) + λ · MS-STFT(spectral convergence + log-magnitude)`, λ = 1e-2 unless the name says otherwise. `overnight2/c1_model.py` line 8; `arm_loss` at line 156. |
| `det_split` | deterministic, band-**split** waveform term | `L1(low-passed waveform) + λ · MS-STFT`. The waveform term is computed on the low-passed output, so no term ever asks for zero above 6 kHz. "The cheapest deterministic fix a practitioner would try." |
| `ES` | **E**nergy **S**core (Gneiting & Raftery 2007) | `ES(P, y) = E d(Y, y) − ½ E d(Y, Y′)`, `Y, Y′ ~ P`, estimated with two draws per step as `½[d(y, ŷ₁) + d(y, ŷ₂)] − ½ d(ŷ₁, ŷ₂)`. Strictly proper whenever `d` is a metric of strong negative type; equals the multivariate CRPS. The three `es_*` arms differ only in `d`. |
| `es_wave` | energy score, **wave**form geometry | `d = L1(waveform)` only, no spectral term. Proper for the per-sample marginals of the waveform. |
| `es_marg` | energy score, **marg**inal log-magnitude geometry | `d = L1(waveform) + λ · L1(log|STFT|)` over three STFT scales. Proper for per-sample and per-bin marginals; blind to cross-bin dependence. The best sampler of the 8 Sep run. |
| `es_slice` | energy score, **slice**d log-magnitude geometry | Log-magnitude term replaced by `E_θ |θ · (A_t − B_t)|`, θ uniform on the unit sphere in ℝ^F, 64 directions per step. Proper for the joint law of a whole frame (Cramér–Wold). |
| `d_wave`, `d_logmag`, `d_sliced` | the three distances `d` | `overnight2/c1_model.py` lines 133, 136, 139. |
| `spread` | logged term | `d_wave(ŷ₁, ŷ₂)`, the sampler's own dispersion, logged every step. |
| graph toy: `mse` | **M**ean **S**quared **E**rror | Deterministic objective in `gnn_toy.py`. |
| graph toy: `es` / `es_eucl` | Euclidean energy score | Energy score with the Euclidean norm over all nodes: proper for the joint law. |
| graph toy: `crps_node` | sum of per-node **C**ontinuous **R**anked **P**robability **S**cores | Proper for marginals only. `gnn_toy_scores.py` line 33. |
| graph toy: `es_sliced` | sliced energy score | 64 random unit directions in ℝ^N. `gnn_toy_scores.py` line 37. |

## 3. Trained models, by run

### 3.1 Notebook runs (1 Sep)

| checkpoint | expansion |
|---|---|
| `checkpoints/FULL/lisa.pt` | The `FULL` preset (48 kHz, ×4, batch 16, 12,288-sample segments, 20k steps) trained at **λ = 1.0**, the original mis-weighting. Correct magnitudes, random phase, SNR −5.8 dB. Kept as the failure reference. |
| `checkpoints/FULL_lam0.001/lisa.pt` | Same preset retrained at **λ = 1e-3** (the paper's value), step 20,000, on the 0.54 h DataShare cache. SNR 18.13 vs naive 18.16. Since 1 Sep the λ is written into the checkpoint path (`RUN = CKPT / f"{CFG.name}_lam{λ}"`) so a differently weighted run can never be silently resumed. |
| `SMOKE`, `FULL` | The two `Config` presets in `build_notebook.py` (lines 239, 251). SMOKE is 16 kHz, 400 steps, for `validate.py`. |

### 3.2 The λ frontier, run tag `XL4_b64_1s` (4 Sep)

**XL** = the extra-large corpus stage: the `overnight/` cells are numbered XL-1 … XL-6 and the run
was the first on the full 37.3 h Hub corpus. `XL4` = four arms; `b64` = batch 64; `1s` = 1-second
training chunks. 36,000 steps = 17.1 epochs, one A100-80GB, identical batches for all four arms.
Defined in `overnight/cell4_launch.py` lines 6–7.

| arm | expansion | aliases used in prose |
|---|---|---|
| `relu_l1e-3` | Faithful LISA: ReLU decoder, λ = 10⁻³, the official spectral weight. The H1′ measurement arm. Deficit −17.4 dB on DataShare. | "λ = 1e-3 (4 Sep, 36k)", "the official λ", "near-pure-L1 model", `lam1e-3`, `XL_relu_l1e-3` |
| `relu_l1e-2` | ReLU decoder, λ = 10⁻². The "muffled-but-coherent" regime: deficit −4.5 dB, phase coherent, LSD 0.962. The best deterministic model of that run and the source the ladder was run on. | "λ = 1e-2 (4 Sep, 36k)", `XL_relu_l1e-2`; the 8 Sep `det` arm is this recipe retrained on batch 32 |
| `relu_l1e-1` | ReLU decoder, λ = 10⁻¹. Best LSD and audio-mode ViSQOL of everything trained so far (Hub LSD 0.899, ViSQOL-audio 3.02), worst SNR (17.69). | "λ = 1e-1 (4 Sep, 36k)", `lam1e-1`, "the 4 Sep λ = 1e-1 model" |
| `ff_l1e-3` | LISAFF (Fourier-feature decoder), λ = 10⁻³. The architecture control; behaves like `relu_l1e-3`. | "fourier λ=1e-3 (control)" |
| `XL_relu_l1e-2`, `XL_relu_l1e-3` | Not new models: the `CFG.name` stamped on the notebook's auto-generated research logs when the ladder (XL-6) ran on that arm, giving `overnight/RESEARCH_XL_relu_l1e-2.md` and `…l1e-3.md`. | — |
| `t0`, `t1`, … | Not arms: throwaway names in the timing harness `time_steps`. Do not confuse with rungs T0–T4. | — |

### 3.3 The proper-scoring run, tag `OV2_es` (8 Sep)

**OV2** = **ov**ernight run **2** (the `overnight2/` directory; cells are numbered OV2-0 … OV2-8);
`es` = energy score. Five arms on LISAS, identical batches and initialisation, 38,000 steps of
batch 32 × 1 s = 9.1 epochs of the 37.3 h Hub corpus, 4.25 h on an A100-40GB. Defined in
`overnight2/c2_launch.py` lines 2–8. Checkpoints: Drive `lisa_rtm/checkpoints/OV2_es/<arm>.pt`.

| arm | expansion |
|---|---|
| `det` | Deterministic LISAS (noise channels present but τ = 0), the paper's loss at λ = 1e-2. The reference arm: every Δ in `visqol_tables.md` is against it, and the post-hoc ladder and the ViSQOL rung conditions are fitted on and applied to it. Prose: "`det` (paper loss, λ=1e-2)", "the deterministic arm", "the point-trained model". |
| `det_split` | Deterministic LISAS, band-split waveform loss (see §2), λ = 1e-2. Restores high-band energy (−2.3 dB) and nothing else: κ 0.03, same CRPS as `det`. |
| `es_marg` | Stochastic LISAS trained under the marginal-geometry energy score, λ = 1e-2. The headline model: CRPS −43 % vs `det` on Hub, ensemble mean a better point predictor than `det`. Prose: "the sampler", "the learned map", "the learned sampler". |
| `es_slice` | Stochastic LISAS under the sliced (joint-frame) energy score, λ = 1e-2, 64 directions. Converged more slowly; level with `es_marg` on DataShare, behind on Hub. |
| `es_wave` | Stochastic LISAS under the waveform-only energy score. Energy right, structure wrong (LSD 1.32). |

### 3.4 The receptive-field test, tag `OV2_wide` (8 Sep)

Two arms on LISASW, same batches, 26,000 steps = 6.2 epochs, 1.3 h. Defined in
`overnight2/c5_launch_wide.py` line 7. Checkpoints: Drive `lisa_rtm/checkpoints/OV2_wide/`.

| arm | expansion |
|---|---|
| `wide_det` | LISASW (22 ms context), deterministic, paper loss, λ = 1e-2. Prose: "the wide-context det", "wide-context deterministic model", "`wide_det` (22 ms context, 26k steps)". Best LSD on DataShare (0.931 → 0.926 with passthrough). |
| `wide_es_marg` | LISASW, marginal energy score, λ = 1e-2. Prose: "the sampler with context". Over-dispersed on DataShare (+3.4 dB). |

### 3.5 The planned λ sweep, tag `OV2_lambda` (written, not run)

`overnight2/c8_launch_lambda.py` line 8, needs an A100 (≥ 40 GB host RAM for the corpus). Paper loss
on LISAS, deterministic, λ extended upward from 1e-1, judged on LSD + audio-mode ViSQOL with
baseband passthrough.

| arm | expansion |
|---|---|
| `det_l0.1` | deterministic, λ = 0.1 (= a re-run of the 4 Sep `relu_l1e-1` recipe on batch 32) |
| `det_l0.3` | deterministic, λ = 0.3 |
| `det_l1.0` | deterministic, λ = 1.0 (the 1 Sep failure weight, now survivable because passthrough makes the model responsible only for the band above 6 kHz) |

### 3.6 Other run tags

| tag | expansion |
|---|---|
| `OV2_det` | `CFG.name` when the post-hoc ladder (`overnight2/c4_ladder.py`) runs on the `det` arm. |
| `SMOKE_OV2`, `SMOKE_WIDE` | CPU smoke-test tags (`overnight2/smoke_local.py`, `smoke_wide_local.py`). |
| `LADDER_ARM` | Which trained arm the ladder is fitted on; default `"det"` in `overnight2/c4_ladder.py`, chosen by the gate rule in `overnight/cell6_ladder.py` on 4 Sep. |
| `wrap_lisa` | Loads a plain LISA (XL4) checkpoint into a LISAS with the noise-channel weights zeroed, so ε = 0 is bit-identical (`overnight2/analysis_local.py` line 109). |

## 4. Inference-time variants of one model

| name | expansion |
|---|---|
| `tau`, τ | Noise temperature: the scalar multiplying the Gaussian input channels at inference. τ = 0 gives the deterministic network; τ = 1 is the calibrated draw the score was trained for. Sweep values 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5 in `overnight2/c3_eval.py`. |
| `<arm> tau=<τ>` | One draw at that temperature, seed 0 (`overnight2/c6_visqol_eval.py` line 96). |
| `<arm> mean16` | Mean of 16 draws at τ = 1, seeds 0–15. Prose: "ensemble mean", "mean of 16". The most over-smoothed object in the tables and the sampler's best entry on LSD and ViSQOL. |
| "mean-of-32", `M = 32` | The same on the DataShare set, 32 draws, in the local Mac analysis (`analysis_local.py`). Hub uses `M_DRAWS = 16`. |
| `eval12_tau0`, "τ = 0" | A sampler run with its noise switched off: the "same network as LISA" control. Listening files `u{0,5,10}_<arm>_tau0.wav`. |
| "one draw", "single draw" | One forward pass at τ = 1. |
| "a second draw" | Same at seed 1, for the listening set. |

## 5. The post-hoc transport ladder (rungs)

Maps fitted on the deterministic model's high-band log-magnitude statistics and applied after the
fact. Definitions in `build_notebook.py`; the summary table is in the README.

| rung | expansion |
|---|---|
| `T0` | Identity: the deterministic model's own output, the baseline row (`class Identity`, line 539). |
| `T1` | Per-bin **quantile** map, `F_t⁻¹ ∘ F_s`: exact one-dimensional optimal transport. Matches every marginal, ignores cross-bin structure (`class QuantileMap`, line 385). |
| `T2 diag` | **Bures–Wasserstein** map `m_t + A(l − m_s)` with diagonal covariance, i.e. per-bin σ_t/σ_s scaling, ratio clipped at 8 (`class BuresMap`, line 460, `mode="diag"`). |
| `T2 block` | Block-diagonal Bures, 12 blocks, Ledoit–Wolf shrinkage ρ = 0.1. |
| `T2 full` | Full-covariance Bures. Implemented and gate-tested but not in `MAPS`, so never in a results table. |
| `T3` | **Conditional** affine-in-covariates location-scale map. Covariates are computed from the input only: log low-band frame energy and low-band spectral flatness (`class ConditionalMap`, line 549). Answers the pooled-marginal objection. |
| `T4`, `(lam=…)` | **McCann displacement interpolation** `T_λ = (1−λ) Id + λ T` between identity and any rung (`class Interpolated`, line 525). Reported as `<rung> (lam=1)` and `<rung> (best lam=0.90)`; this λ is unrelated to the spectral weight. |
| `S` | The **stochastic** rung: reference complex Gaussian noise pushed through a closed-form conditional transport map, `M_det · e^{iφ} + σ_k(c) · ε`, with σ²(c) from `ConditionalPower` (`CPOW`, line 590). The only rung that produces samples; zero learned parameters. Prose: "S: noise through the closed-form conditional map", "the hand-built rung". |
| `Conditioned` | Freezes a `ConditionalMap` against one utterance's covariates so it has the plain `T(L)` signature (line 614). |
| `ladder T1 on det`, `ladder T3 on det`, `ladder S on det` | The three rungs as conditions in the ViSQOL run, applied on top of the `det` arm's output (`overnight2/c6_visqol_eval.py` line 113). S uses one draw. |
| `HI` | The high band as a slice of bins above `k_cut` in the operating STFT basis. |

## 6. Baselines, controls and bounds

| name | expansion |
|---|---|
| `naive`, "naive upsampling", `naive_upsample` | Polyphase (windowed-sinc) interpolation of the decimated 12 kHz input to 48 kHz, so the high band is empty (`build_notebook.py` line 1229). The trivial baseline gate 2a must beat; no deterministic LISA here has beaten it by more than 0.3 dB SNR. |
| "shaped noise", `control: shaped noise` | 1970s noise-filling bandwidth extension: white noise with per-third-octave gains fitted so high-band energy tracks low-band frame energy, voice-activity gated (`build_notebook.py` line 1589). If the ladder cannot beat this, the OT machinery bought vocabulary not value. |
| "mis-specified map", `control: mis-specified T1`, `MISSPEC` | T1 with the target quantile table rolled a third of the bins across frequency (line 1631). If it still improves HB-LSD, LSD is rewarding generic energy inflation. |
| "frame shuffle", `control: frame shuffle`, `shuffled` | T1's corrected high-band frames permuted in time (line 1635). Pooled marginals are permutation-invariant; a metric suite that does not score this much worse cannot see conditional structure. |
| "basis check", `alt` | HB-LSD recomputed at twice the FFT size and hop (line 1643). A gain that lives only in the operating basis is an artefact. |
| "passthrough", "baseband passthrough", "hybrid", `<condition> \| passthrough` | The given baseband plus the model's band above it: low-pass of the naive upsampling of the input + high-pass of the condition's output, brick-wall at 6 kHz (`overnight2/c7_hybrid_eval.py`, `hybrid`). The honest system output for bandwidth extension, since the low band is given. Computed for the ten conditions in `PICK`. |
| `passthrough + zero HB`, "passthrough + empty high band (floor)" | Lower bound row: given baseband and nothing above 6 kHz (= brick-wall naive). Audio ViSQOL 1.57 on Hub, 1.93 on DataShare. |
| `passthrough + TRUE HB (oracle)`, "passthrough + true high band (ceiling)" | Upper bound row: given baseband plus the ground-truth high band. Audio ViSQOL 4.73 on both sets. |
| "pred-magnitude + target-phase", "target-magnitude + pred-phase", `tgt-mag+pred-phase` | Phase-swap resyntheses (`overnight/cell3_eval.py` lines 23–37): the decisive 1 Sep test that showed the λ = 1.0 model had correct magnitudes and random phase. |
| `probe` | `test_utts[0]`, the single held-out utterance scored at every checkpoint during training. |
| GATE 1 | Closed-form validation of the transport code before any data: T1 against the analytic log-Rayleigh shift to 1e-12, Bures identity `A Σ_s A = Σ_t` to 1e-10, McCann endpoints. |
| GATE 2a, GATE 2b | Fidelity (`SNR > SNR_NAIVE`) and headroom (`DEFICIT < −1 dB`). |
| `H1`, `H1′`, `H2`, `H3`, `H4`, `H5` | Hypothesis labels (1 Sep note §4, updated 4 Sep §4, settled in TODO): H1 mechanism (deterministic loss ⇒ conditional mean ⇒ deficit); H1′ magnitude at paper scale; H2 a deterministic transport rung fixes it; H3 only a sampler moves a proper score; H4 LSD is a bad judge; H5 transport beats classical BWE. |

## 7. Models cited from the literature

| name | expansion |
|---|---|
| **WSRGlow** | *WSRGlow: A Glow-based Waveform Generative Model for Audio Super-Resolution*, Zhang, Ren, Xu, Zhao, Interspeech 2021. WaveNet + Glow normalising flow, 229M parameters, the first model to generate 48 kHz from 12 kHz. Cited from LISA's Table 1 (SNR 19.41, LSD 1.01) as the generative model that a deterministic 89k model "beats" on LSD. |
| **TFiLM** (written `TFilm` in the notes) | **T**emporal **F**eature-w**i**se **L**inear **M**odulation, Birnbaum, Kuleshov, Enam, Koh, Ermon, NeurIPS 2019. Convolutional super-resolution network with RNN-driven feature modulation. LISA Table 1: SNR 19.51, LSD 2.02. |
| **AudioUNet** | The U-Net audio super-resolution model of Kuleshov, Enam, Ermon, *Audio Super Resolution using Neural Networks*, ICLR 2017 workshop. LISA Table 1: SNR 18.55, LSD 2.11. All three baselines land within ~1 dB of naive upsampling on this data. |
| **PairNorm**, **DropEdge** | GNN over-smoothing remedies (Zhao & Akoglu 2020; Rong et al. 2020). Not run here; argued in the 8 Sep note §5 to be the diagonal Bures map T2 applied per layer: they restore energy, not information. |
| "GAN baseline" | Proposed in the TODO, never run: the community's default answer to over-smoothing. |
| NU-Wave, NVSR, AERO | Not mentioned anywhere in this repository. |

## 8. Corpora, evaluation sets and splits

| name | expansion |
|---|---|
| **VCTK** | The CSTR **V**oice **C**loning **T**ool**K**it corpus, version 0.92, University of Edinburgh Centre for Speech Technology Research: 110 English speakers, 48 kHz. The dataset LISA trains and tests on. Speaker IDs `p225` … are VCTK's. |
| **DataShare** (route) | Edinburgh DataShare, `datashare.ed.ac.uk`, the original VCTK zip fetched by HTTP range requests (`VCTK_URL`, `build_notebook.py` line 929). Now returns 403. |
| `train_FULL.npz`, `test_FULL.npz` | The DataShare-derived caches: train = 400 utterances, speakers p225–p234, 0.538 h; test = 120 utterances, p236/p237/p238. Speaker-disjoint. |
| **Hub** (route) | The Hugging Face Hub copy `sanchit-gandhi/vctk`: full VCTK 0.92 mic1, 27 parquet shards, 48 kHz (`overnight/cell1_corpus.py`). 97 training speakers, 39,639 utterances, 37.3 h. Every model since 4 Sep trained on it. |
| **Hub set**, `hub12`, `eval12`, `EVAL12` | `test_utts[:12]`: 12 held-out utterances from speakers p236–p238 as loaded from the Hub route. The in-distribution evaluation set (same acquisition route as training). M = 16 draws. |
| **DataShare set**, `datashare36`, `spread36` | 36 utterances from `test_FULL.npz`, 12 per held-out speaker (`i % 40 < 12`). The same three speakers via a different acquisition route; the independent set, evaluated on the Mac with M = 32. |
| "the two-test-set discrepancy" | The same checkpoint measures a high-band deficit 5–8 dB deeper on the Hub set than on DataShare. Every number must name its set. |
| `first12` | First 12 utterances of `test_FULL.npz` (local eval, superseded by `spread36`). |
| `ours120` | All 120 Hub utterances of p236–p238. |
| `ours12`, `paper12`, `train12` | 4 Sep Colab splits: 12 of ours, 12 of the paper's split, 12 from seen training speakers ("train (seen spk)"). |
| `paper100`, "paper split", "id ≥ 350" | 100 utterances spread across the paper's own held-out split: LISA trains on speaker id < 350, so id ≥ 350 (8 speakers) is its test set (`PAPER_TEST_MIN = 350`). |
| `OUR_TEST` | `{"p236", "p237", "p238"}`, held out for continuity with `test_FULL.npz`. |
| `allsplits` | The 4-arm × 4-split table `overnight/table_allsplits.md`. |
| "light corpus", `TRAIN_EVERY = 200` | 1-in-200 train subsample plus both test splits, ~2 GB host RAM, for eval-only runtimes (`overnight2/c0b_boot_light.py`). |
| "fit subset", "200 training-speaker utterances" | The utterances the post-hoc ladder statistics are fitted on. |
| `MANIFEST`, `xl_manifest.json` | Speaker/hour record of the Hub corpus. |
| listening set `u{0,5,10}_*.wav` | Truth, naive, `det`, `det_split`, two draws of each sampler and each sampler at τ = 0, for three held-out utterances, Drive `lisa_rtm/ov2/audio/`. |
| graph toy data | 600-node 8-nearest-neighbour random geometric graph; inputs in the 40 lowest Laplacian modes (`K_LO`); targets = smooth f(x) + input-dependent noise in modes ≥ 200 (`K_HI`). Bands 0–40, 40–120, 120–200, 200–400, 400–600. |
| synthetic harmonic-plus-fricative corpus | What `validate.py` substitutes for VCTK to run offline. |

## 9. Metrics and other abbreviations

| abbreviation | expansion |
|---|---|
| **SNR** | **S**ignal-to-**N**oise **R**atio, `10 log10(‖y‖² / ‖y − ŷ‖²)` on the waveform. Phase-sensitive; the paper's Eq. (4). |
| **LSD** | **L**og-**S**pectral **D**istance: mean over frames of the RMS over bins of the difference of log10 power spectra, evaluation basis n_fft 1024, hop 256. |
| **HB-LSD** | LSD restricted to the **h**igh **b**and, bins above the input Nyquist (6 kHz). |
| LSD (paper basis) | LSD at n_fft 2048 / hop 1024, the released code's basis; orientation only. |
| band energy ratio, ρ_b | Per-third-octave `10 log10(Σ|Ŝ|² / Σ|S|²)` in dB; 0 = correct energy, negative = over-smoothed. |
| **deficit**, "HB deficit" | Mean band energy ratio over the bands above 6 kHz. The direct measure of regression to the mean. |
| baseband (dB) | The same over the bands below 6 kHz: an alignment sanity check, should be ≈ 0. |
| **CRPS** | **C**ontinuous **R**anked **P**robability **S**core, fair-ensemble form, on high-band log-magnitude. Proper: minimised by the true conditional law. For a deterministic arm it equals the MAE of its point forecast. |
| sliced CRPS, `sCRPS` | CRPS of the ensemble projected on 32 random unit directions across bins; sees cross-bin structure. |
| **PIT** | **P**robability **I**ntegral **T**ransform histogram: rank of the truth among M draws. Flat = calibrated, U-shaped = under-dispersed, dome = over-dispersed. |
| spread–skill | Binned ensemble spread against RMSE of the ensemble mean; unit slope is the target. |
| corr err, `corr_err` | Frobenius norm of the difference of cross-bin correlation matrices, truth vs prediction, over 16 frequency groups. |
| coherent fraction, ρ̂(b) | `Re⟨Y, P⟩ / ⟨Y, Y⟩` with Y, P the complex STFTs of target and prediction: share of target power reproduced coherently. |
| **κ** (kappa), "HB κ" | `Re⟨Y, P⟩ / ⟨P, P⟩`: κ = 1 means every unit of output power is informative, κ ≈ 0 means hallucinated. |
| `bb_coh`, `bb_kappa` | The same pair on the baseband; both ≈ 1.00 for every arm. |
| predictability spectrum, `rho`, `rho_energy` | `10 log10 ρ(b)`: fraction of target power predictable from the input, per band; `rho_energy` is the M-draw bias-corrected `(M r − 1)/(M − 1)`. |
| `snr_bound_db`, deterministic SNR ceiling | `10 log10(P_y / Σ_b (1 − ρ(b)) P_y(b))`, the SNR any point predictor can reach on this data (19.5 dB on DataShare). |
| `snr_gap_db`, `snr_gap_calibrated_db` | `SNR(mean) − SNR(one draw)`; a calibrated sampler shows `10 log10(2 / (1 + 1/M))` = 2.75 dB at M = 16, 2.88 at M = 32. |
| **ViSQOL** | **Vi**rtual **S**peech **Q**uality **O**bjective **L**istener (Google). Output is **MOS-LQO**, **M**ean **O**pinion **S**core, **L**istening **Q**uality **O**bjective, 1–5. Two modes below. Run through the pure-Python `visqol-python` port, not Google's Bazel package. |
| ViSQOL speech mode, `visqol_speech16k`, "ViSQOL-sp" | Speech mode at 16 kHz: 21 **ERB** (**E**quivalent **R**ectangular **B**andwidth) bands, 50 Hz–8 kHz, with the upstream "lattice" MOS mapping. Sees only one band of the reconstructed 6–24 kHz, so it cannot judge this task (note §7.1). `evaluation.py`. |
| ViSQOL audio mode, `visqol_audio48k`, "ViSQOL-au" | Audio mode at 48 kHz: 32 bands to 24 kHz, sees the whole reconstructed band. Rewards deterministic high-band energy monotonically. |
| **NSIM** | **N**eurogram **S**imilarity **I**ndex **M**easure: the mapping-free similarity behind each ViSQOL MOS (`nsim_speech16k`, `nsim_audio48k`). |
| **PESQ**, `pesq_wb` | **P**erceptual **E**valuation of **S**peech **Q**uality, ITU-T P.862.2 wideband, at 16 kHz. Stops at 8 kHz, so it cannot see the reconstructed band either. |
| **MOS** | **M**ean **O**pinion **S**core, the 1–5 scale ViSQOL and PESQ report in. |
| **STFT**, **MS-STFT** | **S**hort-**T**ime **F**ourier **T**ransform; **M**ulti-**S**cale STFT loss. |
| **OT**, W2 | **O**ptimal **T**ransport; 2-Wasserstein distance. |
| **BWE** | **B**andwidth **E**xtension, the task. |
| **MAE**, **MSE**, **RMSE** | Mean absolute / squared / root-mean-squared error. |
| **MLP**, **ReLU** | Multi-layer perceptron; rectified linear unit. |
| **GNN**, **GDL** | Graph neural network; geometric deep learning. |
| **VAD**, `VAD_FLOOR` | Voice activity detection; frames more than 60 dB below the loudest are silence and never excited. |
| **RF** | Receptive field (11 input samples for LISA, 263 for LISASW). |
| A100-40GB / A100-80GB, **MPS** | The Colab GPUs (SXM4 40 GB; 80 GB on 4 Sep); Apple **M**etal **P**erformance **S**haders, the M4 GPU backend used for latency. |
| real-time factor, p50 / p95 | Seconds of audio per second of compute; median and 95th-percentile wall time for 1 s of audio and a 20 ms chunk (`overnight2/cpu_latency.py`). |
| "epsilon floor", "eps floor" | T0's high band sitting at the STFT ε, which made any energy injection score as an HB-LSD win on the 1 Sep model. |
| "no-headroom zone", "grey zone" | The ±2 dB deficit band where the premise fails. |
| "muffled-but-coherent regime" | High-band magnitudes systematically low but structure and phase real: what the transport thesis needs, found at λ = 1e-2. |
| "objective-induced" vs "architecture-induced" over-smoothing | The two-part decomposition from §2 of the 8 Sep note: deficit of a single draw = what the architecture cannot express; deficit of the ensemble mean minus that = what the objective removed. |
