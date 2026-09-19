# Algorithm specification — the seven-arm energy-score LISA sampler

**Date:** 2026-09-19. **Purpose:** the contract for the 50-epoch GPU-saturating rewrite. A competent
engineer who has never seen this repo must be able to reimplement the seven arms from this document
alone and reproduce the numbers to the tolerances of §6.3. Where the repo's own comments disagree with
its own measurements, §9 says so and this document binds to the measurement.

**Sources** (role, then what this document takes from it):

| file | role |
|---|---|
| `overnight2/c1_model.py` | `LISAS`, `HostCorpus`, `reconstruct`, `SCALES`, `lowpass`, `d_wave`, `d_logmag`, `d_sliced`, `arm_loss`, `probe_metrics`, `load_arm` |
| `overnight3/e1_model.py` | `LISASD`, the sub-pixel decode path, `copy_shared`, `erb_filterbank`, `erb_feats`, `d_erb`, `d_logmag_l2`, `arm_loss3` — the single-arm REFERENCE objective |
| `overnight3/e2b_fast.py` | `ArmStack`, `TargetFeats`, `_fwd_bwd`, `val_loss_fast`, `train_ov3_fast` — **the code the 50-epoch run executes** |
| `overnight3/e2_trainer.py` | `val_batches`, `plot_curves`, `_nan`, `_f` (imported by e2b, which asserts their presence at `e2b_fast.py:53`); `train_ov3`/`val_loss` are the sequential reference, NOT used |
| `overnight3/e3_launch.py` | **`:7-15` ONLY** — the seven `ARMS` literal. The rest of that file is a different, SEQUENTIAL launcher and exec'ing it is harmful; see I2. |
| `overnight3/e8_launch_fast.py` | the measured A100 throughput table (`:2-5`) and the *reference* stacked launcher. **Not the executed one** — see the next row. |
| `build_train_notebook.py` | **the launcher that actually runs** (`:432-494`), `make_dashboard` (`:319-406`), and the inline order of every source file (`:178,213-215,228,239,264`) |
| `overnight3/e7_fastsettings.py` | the batch-size sweep (`:7`). **Never executes on the notebook path** — it is not inlined by `build_train_notebook.py`; the flags come from `e2b_fast.py:63-68,94-105`. |
| `overnight/cell1_corpus.py` | corpus acquisition, roles, `load_bytes`, `test_utts` |
| `build_notebook.py` | `Config`/`FULL` (`:251-264`), `stream`/`seed_everything` (`:157-175`), `LISADecoder` (`:1109-1120`), `MultiScaleSTFTLoss` (`:1147-1161`), `decimate` (`:1034-1036`), `save_ckpt` (`:1241`), **and the probe readout `snr_db`/`third_octave_edges`/`band_energy_ratio`/`naive_upsample`/`stft`/`_hann` (`:295-307,643-647,658-679,1229-1232`) — see §4.5** |
| `overnight3/gate_subpixel.py` | the exactness gate (measured numbers quoted in §6.3) |
| `notes/analysis/dynamics_OV3_fast.md` | the 16,000-step reference run this one extends |
| `notes/2026-09-16-score-geometry-decoder-noise-readouts.md` | what each arm is FOR, scientifically |

---

## 0. What the algorithm is, in one page

LISA is an implicit-neural-representation audio super-resolver: a 4-layer 1-D convolutional encoder
reads a 12 kHz waveform and produces one 32-dimensional latent per input sample; a 5-layer 144-wide
coordinate MLP then reads, for each of the four 48 kHz output samples inside an input cell, the triple
of latents `[z_{i-1}, z_i, z_{i+1}]` around an anchor `i` plus a scalar coordinate, and emits one
sample. 87,777 parameters (§1).

Trained on L1 + multi-scale-STFT, this network regresses to the mean. The band above 6 kHz is almost
entirely incoherent with the input — it has to be *invented*, not *inferred* — and a point predictor
minimising a mean-seeking loss answers by inventing nothing. The measured consequence is a high-band
energy deficit of −19.09 dB for the `det` arm at the end of the reference run
(`notes/analysis/dynamics_OV3_fast.md:§1`).

The fix under test is to keep the network exactly as it is, inject Gaussian noise channels at its
input (`LISAS`, `c1_model.py:16,23`) or additionally at its decoder (`LISASD`, `e1_model.py:21-32`),
and train it under the **energy score**, a strictly proper scoring rule for the whole conditional law.
Two draws per step, one forward pass:

    ES_d(y; y1, y2) = ½·[ d(y,y1) + d(y,y2) ] − ½·d(y1,y2)

unbiased for `E d(y,Y) − ½ E d(Y,Y')` (Gneiting & Raftery 2007; implemented identically at
`c1_model.py:167`, `e1_model.py:257`, `e2b_fast.py:358,372`). Because `eps = 0` recovers LISA exactly
(`c1_model.py:43-44` substitutes explicit zeros rather than skipping the concatenation), the
deterministic and stochastic arms share one class, one parameter count and one initialisation. At
inference the sampler costs exactly one forward pass — there is no iterative denoising.

Seven arms are trained **simultaneously on identical batches from one initialisation**, varying one
thing each: the spectral weight λ, the geometry of the distance `d`, whether the waveform term is
restricted to the low band, and where the noise enters. The comparison is the experiment; anything
that breaks "identical batches, identical init, comparable losses" destroys it.

The 50-epoch re-run uses the **stacked** trainer `train_ov3_fast` (`e2b_fast.py:570`) with
`GROUP_MAX = 1` (`build_train_notebook.py:452`, the executed launcher; = `e8_launch_fast.py:11`),
i.e. seven single-arm stacks executed sequentially inside
one `_fwd_bwd` on one shared batch. Every "stacking" mechanism (grouped conv, `baddbmm`) is therefore
inert at A = 1 and the arm axis is a length-1 leading dimension. This is the measured-fastest
configuration on an A100 (`e8_launch_fast.py:4-6`) and it is **not** re-validated on the target
hardware (§9).

---

## 1. Configuration constants

FULL preset, `build_notebook.py:251-264` (`:251` is `FULL = Config(`, `:264` the closing paren; `:265-268`
are the `PRESET` switch), with the launcher's overrides.

| symbol | value | where it is set |
|---|---|---|
| `fs_hi` | 48 000 Hz | `build_notebook.py:252` |
| `R` (`upsample`) | 4 | `build_notebook.py:252` |
| `fs_lo` | 12 000 Hz | derived, `fs_hi / R` |
| `SEG` (`corpus.seg_hi`) | 48 000 samples = 1 s | `build_train_notebook.py:448` (= `e8_launch_fast.py:13`) (**overrides** `CFG.seg_samples = 12288`) |
| `BATCH` (B) | 64 | `build_train_notebook.py:447` (= `e8_launch_fast.py:12`) (**overrides** `CFG.batch_size = 16`) |
| `L` | 12 000 | derived, `SEG / R` |
| `N` = `T` | 48 000 | derived, `L · R` |
| `enc_channels` | (16, 32, 64, 32) | `build_notebook.py:257` |
| `enc_kernels` | (7, 3, 3, 1) | `build_notebook.py:257` |
| `C` (latent width) | 32 | `enc_channels[-1]`, assigned at `c1_model.py:29` (`self.enc, self.dim = nn.Sequential(*layers), c_in`) |
| `dec_hidden` (H) | 144 | `build_notebook.py:258` |
| `dec_layers` | 5 | `build_notebook.py:258` |
| `N_NOISE` | 8 | `c1_model.py:16` |
| `N_DEC` | 4 | `e1_model.py:18` |
| `d_in` | 97 (LISAS) / 101 (LISASD) | `1 + 3C` (`build_notebook.py:1112`) `+ N_DEC` (`e1_model.py:30-31`) |
| `n_fft`, `hop` | 2048, 512 | `build_notebook.py:261` |
| `SCALES` | [(2048,512), (1024,256), (512,128)] | `c1_model.py:112` |
| `eval_n_fft`, `eval_hop` | 1024, 256 | `build_notebook.py:261` (probe band-energy basis only) |
| ERB bands, `f_lo` | 32, 50 Hz | `e1_model.py:177` |
| `lr` | 1e-3 | `build_train_notebook.py:490` (= `e8_launch_fast.py:39`) |
| `milestones` | (0.2, 0.4, 0.5, 0.6, 0.7, 0.8) × steps | `build_train_notebook.py:491` (= `e8_launch_fast.py:40`) |
| `gamma` | 0.5 | `build_train_notebook.py:491` (= `e8_launch_fast.py:40`) |
| `grad_clip` | 1e-3, **per arm** | `build_train_notebook.py:491`, `e2b_fast.py:386-395` |
| `SEED` | 0 | `build_notebook.py:157` — the NUMPY-stream seed. The torch generators are seeded separately by `seed_everything()` (`:164-172`); see **§1.1**. |
| `tag` | `"OV3_fast"` | `build_train_notebook.py:449` (= `e8_launch_fast.py:14`) — **part of the numerics**, see I2. **Must be set and asserted explicitly**, because `e3_launch.py:20` would set it to `"OV3_es"`. |
| `val_every`, `n_val_batches`, `log_every` | 500, 8, 25 | `e2b_fast.py:571` (defaults, not overridden) |
| `ckpt_every` | 500 | `build_train_notebook.py:451` (= `e8_launch_fast.py:16`) |
| `tau` (training) | 1.0 for es arms, eps = None for det | `e2b_fast.py:329-330` (no tau factor ⇒ implicitly 1.0); `e2b_fast.py:338` |
| LISAS / LISASD parameters | 87,777 / 88,353 | **verified by construction**, see below |
| LISA (deterministic, 1 input channel) | 86,881 | verified; the `+896` is 8 extra input channels × 16 × k=7 |

**Parameter count arithmetic** (derived, recomputed here from `build_notebook.py:1109-1120` and
`c1_model.py:23-29`): encoder `9·16·7+16 + 16·32·3+32 + 32·64·3+64 + 64·32·1+32 = 10,880`;
decoder `97·144+144 + 3·(144·144+144) + 144·1+1 = 76,897`; total **87,777**. LISASD adds
`N_DEC · dec_hidden = 4 · 144 = 576` ⇒ **88,353**.

**Decoys.** `CFG.lambda_spec = 1e-3`, `CFG.batch_size = 16`, `CFG.steps = 20000`,
`CFG.seg_samples = 12288` and `CFG.train_speakers = (p225…p234)` (`build_notebook.py:253-260`) are
**not** used by this run. λ comes from the ARMS dict per arm, batch/seg from the launcher, and the
speakers from the Hub loader (§5). `CFG.eval_n_fft/eval_hop` are used only by `probe_metrics`.

### 1.1 Seeding, parameter initialisation, and the execution order that fixes them

**The three seed calls.** `seed_everything(seed=0)` (`build_notebook.py:164-172`) runs **once at boot**
(`build_notebook.py:175`) and calls, in this order:

```
np.random.seed(0)                # the legacy numpy global — NOT the batch stream
torch.manual_seed(0)             # the CPU default generator AND every CUDA device generator
torch.cuda.manual_seed_all(0)    # redundant after the line above, but it is what the file does
torch.backends.cudnn.deterministic = True ; cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
```

`stream(label)` (`:159-162`) is a **separate** mechanism: it derives an independent
`np.random.default_rng` (PCG64) from `blake2b(label, 8) ^ SEED`, so the batch stream does not touch
the torch generators and the torch generators do not touch it.

**No re-seed happens after boot.** `e2b_fast.py:94-105` *reverses* `seed_everything`'s determinism
flags at import time (`use_deterministic_algorithms(False)`, `cudnn.deterministic=False`,
`cudnn.benchmark=True`, TF32 on, `set_float32_matmul_precision("high")`) **without re-seeding
anything**. The generator states at the training launch are therefore whatever the preceding cells
left them at.

**Parameter initialisation distribution — the thing `§6.3`'s "bit-for-bit" depends on.** Every weight
comes from PyTorch's `nn.Conv1d` / `nn.Linear` defaults, applied inside each module's `__init__` in
`nn.Sequential` construction order (`c1_model.py:24-30`, `build_notebook.py:1112-1117`):

```
weight ~ kaiming_uniform_(weight, a = sqrt(5))
         i.e. U(-b, +b) with b = sqrt(6 / ((1 + 5) * fan_in)) = 1/sqrt(fan_in)
         fan_in = in_channels * prod(kernel_size) for Conv1d, in_features for Linear
bias   ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in))         # same fan_in, drawn AFTER the weight
```

Construction order for `LISAS(CFG)`: `enc.0` (Conv1d 9→16, k=7), `enc.2` (16→32, k=3), `enc.4`
(32→64, k=3), `enc.6` (64→32, k=1), then `dec.net.0` (Linear 97→144), `dec.net.2`, `dec.net.4`,
`dec.net.6` (144→144), `dec.net.8` (144→1) — 18 tensors, weight then bias for each (§2.6).

**`LISASD` consumes an EXTRA draw.** `LISASD.__init__` first runs `LISAS.__init__` in full (all 18
tensors), then **replaces** `dec.net[0]` with a fresh `nn.Linear(101, 144)` (`e1_model.py:30-31`),
which draws its own `kaiming_uniform_` weight and bias from the same CPU generator. `copy_shared`
(`e1_model.py:140-153`) then zeroes that weight and copies the LISAS 97 columns into the first 97, so
the drawn values are discarded — **but the generator has still advanced by 144·101 + 144 = 14,688
draws.** Any reimplementation that builds LISASD without that replacement, or that builds it before
LISAS, produces a different downstream stream.

**Execution order between boot and `_make_models` (all on the CPU default generator).** On the
executed notebook path, in order:

| # | site | what it constructs | draws |
|---|---|---|---|
| 1 | `build_notebook.py:1164` | `LISA(CFG)` (1 input channel) | 86,881 |
| 2 | `c1_model.py:286` | a whole `LISAS(CFG)` **inside a `print`** (parameter-count banner) | 87,777 |
| 3 | `e1_model.py:301-302` | `LISASD(CFG)` **then** `LISAS(CFG)`, also inside a `print` | 88,353 + 14,688 + 87,777 |
| 4 | `time_ov3_fast` → `_make_models` (`e2b_fast.py:507-516,737`) | `LISAS(CFG)` + `LISASD(CFG)` for the **pre-run benchmark**, then 23 fwd/bwd steps of CUDA draws | 87,777 + 88,353 + 14,688, and `n+3 = 23` steps of CUDA noise |
| 5 | `train_ov3_fast` → `_make_models` (`e2b_fast.py:510-512`) | **the run's real `LISAS(CFG)` and `LISASD(CFG)`** | — |

Only #4 was previously flagged (CHANGES-TRAJECTORY-1). #1–#3 are equally load-bearing: they are
side effects of *print statements* and a rewrite that drops the banners silently changes the
initialisation of all seven arms.

**The rewrite's obligation.** Choose one and write it in the run log:

- **(A) reproduce the sequence** — construct #1–#4 in exactly that order before `_make_models`, or
- **(B) re-seed** — call `torch.manual_seed(0)` (and `torch.cuda.manual_seed_all(0)`) **immediately
  before** `_make_models` in `train_ov3_fast`, and record it as **CHANGES-TRAJECTORY-2**.

**(B) is recommended**, because (A) is unverifiable without the reference run's weights and because
(B) makes the init reproducible from this document alone. Either way, distinguish two claims:

- *"the seven arms are identical to each other"* — I3, **testable now**, holds under (A) and (B);
- *"the seven arms are identical to the OV3_fast reference run"* — **not achievable**: the reference
  run's initial `state_dict` was never committed (§9.28), and §6.3's "initial parameters, bit-for-bit"
  row means **across arms within one run**, not across runs.

### 1.2 The launcher that actually runs

`build_train_notebook.py:432-494` is the cell the notebook executes. It is **not** `e8_launch_fast.py`,
and it differs in four ways that matter:

| | `e8_launch_fast.py` | `build_train_notebook.py:432-494` (executed) |
|---|---|---|
| `BUDGET_H` default | 4.0 (`:15`) | **3.0** (`:450`) |
| `ARMS` | read from the kernel (set by `e3_launch.py`) | **inlined literally** (`:434-442`) — no `e3_launch.py` exec |
| `FAST["GROUP_MAX"]`, `FAST["STREAMS"]` | set unconditionally to 1, False (`:11`) | `globals().get(...)` with defaults 1, False (`:452-453`) |
| corpus build | asserts a corpus exists (`:24`) | **builds it** if absent, then trims `train_utts` to 200 (`:467-472`) |

`steps` is computed identically in both (`build_train_notebook.py:478` / `e8_launch_fast.py:27`):

```
steps = max(2000, (int(BUDGET_H * 3600 / dt) // 1000) * 1000)     # always a multiple of 1000
if isinstance(globals().get("STEPS_OVERRIDE"), int): steps = int(STEPS_OVERRIDE)
```

**Consequence: 104,950 is unreachable from the budget path** — it is not a multiple of 1000. The
50-epoch run MUST set `STEPS_OVERRIDE = 104950` in a cell above the launcher (or the launcher must be
rewritten to take `steps` directly). See §8 for why 104,950 and not 104,906.

---

## 2. The model

### 2.1 Encoder

`LISAS.encode` (`c1_model.py:41-45`). Input `x_lo (B, L)` fp32. `N_NOISE = 8` Gaussian channels are
**concatenated** to the single signal channel, never added, so the first conv has `1 + 8 = 9` input
channels (`c1_model.py:23`):

```
h = cat([x_lo.unsqueeze(1), eps], 1)                  # (B, 9, L)
h = ReLU(Conv1d(9,  16, k=7, pad=3)(h))               # (B, 16, L)
h = ReLU(Conv1d(16, 32, k=3, pad=1)(h))               # (B, 32, L)
h = ReLU(Conv1d(32, 64, k=3, pad=1)(h))               # (B, 64, L)
z =      Conv1d(64, 32, k=1, pad=0)(h)                # (B, 32, L)   NO trailing ReLU
return z.transpose(1, 2)                              # (B, L, C=32)
```

The ReLU is skipped after the last conv (`if i < len(cfg.enc_channels) - 1`, `c1_model.py:25-26`), so
the latent is unrectified. `padding = k//2` holds `L` fixed at every layer. When `eps is None` the
method substitutes an explicit **zero tensor** of shape `(B, 8, L)` (`c1_model.py:43-44`): a
deterministic arm is the stochastic network evaluated at eps = 0, with identical parameter count and
identical init. **CHANGES MATH:** gating the noise off by skipping the concatenation instead changes
the conv input width, the parameter count and the initialisation.

### 2.2 Decoder

`LISADecoder` (`build_notebook.py:1109-1120`), `dec_layers = 5`:

```
Linear(d_in, 144) → ReLU → [Linear(144,144) → ReLU] × 3 → Linear(144, 1)
```

`relus = (True, True, True, True, False)` (`e2b_fast.py:210-213`). It is fed **per output sample** with

```
X_j = [ coord_j | z_{i-1} | z_i | z_{i+1} | eps_dec_j ]          (97 or 101 wide)
```

where `i` is the anchor index for output `j`. **The column order is contractual**: coordinate first,
the three latent neighbours in the order `[i-1, i, i+1] × C`, decoder noise last. `_layer1` slices
`W1[:,:,1:1+3C]` at `e2b_fast.py:297`, `W1[:,:,0]` (the coordinate column `w_c`) at `e2b_fast.py:301`
and `W1[:,:,1+3C:]` (the decoder-noise block `w_d`) at `e2b_fast.py:323`, all **by position**
(`:294-296` are the explanatory comment); reordering the feature vector corrupts the sub-pixel
decomposition **silently, with no error**.

Neighbour boundary handling is **clamp-to-edge**: `ii.clamp(0, L-1)` in the gather path
(`e1_model.py:81`), equivalently a replicate pad `cat([z[:,:1], z, z[:,-1:]])` in the sub-pixel path
(`e1_model.py:112`, `e2b_fast.py:250`). Zero-padding instead **CHANGES MATH** at the two edge cells.

### 2.3 The noise pathways (LISAS vs LISASD)

| | LISAS | LISASD |
|---|---|---|
| encoder noise | `eps_enc (B, 8, L)` at the **12 kHz input rate** | same |
| decoder noise | none | `eps_dec (B, L·R, 4)` at the **48 kHz output rate** |
| layer-1 width | 97 | 101 |
| parameters | 87,777 | 88,353 (+576) |
| where defined | `c1_model.py:19-62` | `e1_model.py:21-52` |

`LISASD` widens **only** `dec.net[0]`, by replacing the module in place
(`self.dec.net[0] = nn.Linear(first.in_features + n_dec, first.out_features)`, `e1_model.py:30-31`).
Everything else is inherited.

**Initialisation is shared.** `copy_shared(src, dst)` (`e1_model.py:140-153`) copies every
same-shaped weight and, for `dec.net.0.weight`, **zeroes the tensor and copies the LISAS 97 columns
into the first 97**, so the four decoder-noise columns start at exactly 0 and LISASD ≡ LISAS at
initialisation, at eps = 0 **and at any `eps_dec`**. `smoke_ov3_local.py:63-71` asserts
`max|LISAS(x) − LISASD(x)| < 1e-6` at eps = 0 and that decoder noise then moves the output by more
than 1e-6.

**Draw order inside a LISASD sample:** `eps_enc` FIRST, `eps_dec` SECOND, from the same generator
(`e1_model.py:34-39`; `e2b_fast.py:329-330`). One 4-vector per output sample — 48,000 per second
against the encoder's 12,000.

**Side-channel hazard.** `LISASD.encode` stashes the decoder noise on the mutable attribute
`self._eps_dec` and `decode` slices `d[:, j0:j1]` (`e1_model.py:41-49`). This exists so that chunked
full-utterance `reconstruct()` (encode once, decode in 32768-sample chunks, `c1_model.py:66-78`) gets
the right noise rows. It is unlocked and untested shared state: a rewrite that parallelises probe
reconstruction across arms or threads must not share a module instance.

`ArmStack.sample_eps` (`e2b_fast.py:327-331`) applies **no tau factor** — training always draws at
scale 1.0. The `tau` attribute is set on the exported plain modules only (`e2b_fast.py:514-515`:
0.0 for det kinds, 1.0 otherwise) and is consulted by `reconstruct()`/`probe_metrics` alone.

### 2.4 Anchor jitter

During training the anchor is jittered (`perturb=True`). For each **output sample** `j`:

```
q      = j / R                                        # fp32, exact for R = 4
anchor = q + eta,   eta ~ N(0, 0.5²)                  # ONE draw per OUTPUT sample
i      = clamp(floor(anchor), 0, L-1)
coord  = 2·(q − i) − 1                                # computed from the JITTERED index
```

(`c1_model.py:50-53`; `e1_model.py:107-109` for `anchor`/`idx` and `e1_model.py:121` for `coord` on the
sub-pixel path — or `e1_model.py:75-78`, where all four lines are contiguous, on the gather path;
`e2b_fast.py:311-313`.) The standard deviation is **half an input cell**.

Two consequences that a rewrite must not "fix":

1. **`coord` is computed from the jittered index, not from `floor(q)`.** Writing `i = n + m` with
   `n = floor(q)` gives `coord = c_p − 2m` where `c_p = 2p/R − 1`. So under jitter the coordinate is
   **not confined to [−1, 1)** — it takes values outside that range whenever the jitter moves the
   anchor. Computing `coord` any other way **CHANGES MATH**: the jitter would stop being a pure
   choice of which latent triple is read and would become a perturbation of the coordinate itself.
   Without jitter `coord` takes exactly the R values {−1, −0.5, 0, +0.5} at R = 4.
2. **Jitter is applied to the det arm too.** `_fwd_bwd` passes `perturb=True` to every stack
   including the deterministic one (`e2b_fast.py:469`, det forward at `e2b_fast.py:338`); the
   sequential reference hard-codes the same (`c1_model.py:159`). No launch path reaches the det arm
   with `perturb=False` during training.

Validation and `reconstruct()` use `perturb=False` (`e2b_fast.py:539`, `c1_model.py:66-78`). The
training-time sampler law therefore includes the jitter while the deployed sampler does not — see §9.

`FAST["JITTER_PER_CELL"]` / `LISA_JITTER_PER_CELL` move the draw to one per **input cell**
(`e2b_fast.py:302-309`, `e1_model.py:102-106`). Both default False and both are flagged CHANGES MATH
in the source. The two implementations even differ in draw shape — `randn(B, L)` in e1 versus
`randn(A, S, L)` in e2b — so a rewrite must not silently unify them.

### 2.5 The sub-pixel reformulation and its exactness argument

`FAST["SUBPIXEL"] = True` (`e2b_fast.py:65`) and `LISA_SUBPIXEL = True` (`e1_model.py:66`) are the
defaults, and `e1_model.py:137` rebinds `LISAS.decode = _decode`, so the gather-only decode written
at `c1_model.py:47-59` is **dead code** once `e1_model.py` is exec'd — every plain-module decode,
including probe `reconstruct()` at checkpoints, goes through the sub-pixel path.

Decoder layer 1 is linear in its input blocks, so `W1 = [w_c | W_z | w_d]` splits by column block, and
the latent block depends **only on the anchor `i`**. It can therefore be evaluated once per INPUT
cell as a length-3 convolution over the replicate-padded latents at 12 kHz:

```
zp  = cat([z[:,:1], z, z[:,-1:]], dim=L)                    # (…, L+2, C)
W_z = W1[:, 1:1+3C].reshape(H, 3, C).transpose(1, 2)        # (H, C, 3), taps (A_-1, A_0, A_+1)
u   = conv1d(zp^T, W_z)                                     # (…, L, H)   at 12 kHz
```

**The `(H,3,C) → transpose → (H,C,3)` step is load-bearing and numerically silent if wrong.**
`conv1d`'s `out[o,n] = Σ_{c,k} W[o,c,k]·in[c,n+k]` then reads `zp[n], zp[n+1], zp[n+2] =
z[n-1], z[n], z[n+1]` under the replicate padding (`e2b_fast.py:294-297`). Getting the transpose
wrong reverses the neighbour order and raises nothing.

Three decode paths exist:

| path | when | cost | backward |
|---|---|---|---|
| **(a) gather, `ArmStack`** | `SUBPIXEL=False` (A/B only) | **ONE** gather of `3N` rows from the replicate-padded latent (`e2b_fast.py:267-268`, header note `:34-35`) + the full 97/101-wide Linear | `scatter_add` atomics on the `(A,S,L+2,C)` padded latent |
| **(a′) gather, e1 reference** | `LISA_SUBPIXEL=False` | **three separate** `take()` gathers, one per neighbour (`e1_model.py:80-84`) + the same Linear | `scatter_add` atomics, three smaller scatters |
| **(b) sub-pixel, jittered** | `SUBPIXEL=True`, `perturb=True` — **training** | one gather of rows of `u` + rank-1 coordinate term | `scatter_add` atomics on `u` |
| **(c) sub-pixel, gather-free** | `SUBPIXEL=True`, `perturb=False` — **validation, reconstruct** | no gather at all: `u.unsqueeze(3) + c_p·w_c + b1`, the periodic shuffle | pure sum-reduction (no atomics, no index tensor) |

(a) and (a′) are **arithmetically identical** but are different kernels with different backward
traffic and different scatter shapes; a rewrite of the `SUBPIXEL=False` A/B must reproduce
`ArmStack`'s single-gather form, not e1's three-gather form, or the A/B is not measuring what it
claims. Path (b): `X = gather(u, dim_L, i) + coord·w_c + b1`. Path (c) applies when `i = floor(q) = n`, so the
R outputs of cell `n` share row `u[n]` and differ only by `c_p·w_c` (`e2b_fast.py:316-321`). In
`e1_model` path (c) additionally requires `j0 % R == 0 and (j1-j0) % R == 0` (`e1_model.py:115`);
`ArmStack` has no `j0/j1` arguments and always decodes the full sequence, so the condition there is
just `not perturb` (`e2b_fast.py:236,316`).

For LISASD both sub-pixel paths then add `eps_dec @ w_d^T` (`e2b_fast.py:322-323`). Note
`ArmStack` **skips the term entirely when `eps_dec is None`** ("a zero block: exactly a no-op") while
the reference materialises an explicit zero tensor (`e1_model.py:124-126`) — arithmetically identical.

**FLOP effect.** Layer-1 MACs per output sample fall from `97·144 = 13,968` to
`(3·32·144)/4 + 144 = 3,600` — a **3.88× reduction** (derived, verified) — and the 97-wide high-rate
feature tensor never enters the graph.

**Exactness — the honest claim.** `e2b_fast.py:31` and `e2b_fast.py:288` say the jittered sub-pixel
path "stays bit-for-bit the gather path's". **That is wrong as stated for the layer as a whole.** It
is true only of the *coordinate arithmetic* (`c = c_p − 2m` selects which row of `u` is read). The
layer-1 *sum* is reassociated: the gather path computes a 97-term dot product per output sample; the
sub-pixel path computes a 96-term dot product per input cell and adds the coordinate term separately.
`e1_model.py:93-94` states the accurate claim — "exactly `_decode_gather`, up to the summation order
of layer 1 (~1e-7 relative in fp32)" — and the repo's own gate measures it:

| case | max\|diff\| | relative | source |
|---|---|---|---|
| LISAS, `perturb=False` | 7.45e-9 | **2.207e-07** | `gate_subpixel.py:68-72` (run 2026-09-19) |
| LISAS, `perturb=True`, identical jitter | 7.45e-9 | **1.961e-07** | ibid. |
| LISASD, `perturb=False` | 9.31e-9 | **4.243e-07** | ibid. |
| LISASD, `perturb=True`, identical jitter | 1.12e-08 | **3.975e-07** | ibid. |

Gate tolerance is `rel < 1e-5`; all four pass. **This document binds to `e1_model.py:93-94`'s
wording.** The gate runs in fp32 with autocast OFF (`gate_subpixel.py:3,38-39`), so this is an fp32
claim only — under the bf16 autocast the run actually uses, the deviation is larger and is dominated
by bf16 rounding (§6.3, §9).

### 2.6 The stacked execution path (what the run actually runs)

`ArmStack` (`e2b_fast.py:192-403`) stores every parameter with a **leading arm axis**:
`nn.Parameter(v.unsqueeze(0).repeat(A, …).clone())` for every key of the base model's `state_dict`
(`e2b_fast.py:205`). At A = 1, LISASD: `enc.0.weight (1,16,9,7)`, `enc.2.weight (1,32,16,3)`,
`enc.4.weight (1,64,32,3)`, `enc.6.weight (1,32,64,1)`, `dec.net.0.weight (1,144,101)`,
`dec.net.{2,4,6}.weight (1,144,144)`, `dec.net.8.weight (1,1,144)`.

**There are exactly 18 parameter tensors per arm**, not 16: four encoder `Conv1d` at `enc.{0,2,4,6}` ×
(weight, bias) = 8, plus five decoder `Linear` at `dec.net.{0,2,4,6,8}` × (weight, bias) = 10. That 18
sets the length of `ArmStack.P`, the number of `p.grad = None` assignments per step (7 × 18 = **126**),
and the length of the reduction loop inside `clip_` (`e2b_fast.py:387-395`).

The arm axis is leading on parameters, `eps_enc (A,S,8,L)`, `eps_dec (A,S,N,4)`, `jitter (A,S,N)` and
the output `(A,S,N)`. It is **folded into the channel axis** inside `conv1d` (`groups=A`) and is the
**batch axis** of `baddbmm` inside the MLP. A rewrite that moves the arm axis anywhere else must
re-derive the grouped-conv channel ordering, currently `[arm0 ch0..ch8, arm1 ch0..ch8, …]` from
`h.transpose(0,1).reshape(S, A·(1+n_noise), L)` (`e2b_fast.py:241-242`).

```python
# ArmStack.forward, e2b_fast.py:236-274
h = cat([x.expand(A,S,L).unsqueeze(2), eps_enc], 2)        # (A, S, 1+8, L)   fp32
h = h.transpose(0,1).reshape(S, A*9, L)                    # arm axis -> channels
with autocast(cuda, bfloat16):                             # e2b_fast.py:243
    for (wk, bk, k, relu) in enc_plan:                     # ONE grouped conv per encoder layer
        h = conv1d(h, P[wk].reshape(A*Cout, Cin, k), P[bk].reshape(-1), padding=k//2, groups=A)
        if relu: h = F.relu(h)
    z  = h.view(S, A, C, L).permute(1, 0, 3, 2)            # (A, S, L, C)
    zp = cat([z[:,:,:1], z, z[:,:,-1:]], 2)                # (A, S, L+2, C)
    X  = self._layer1(zp, Ws[0], bs[0], eps_dec, q, S, L, N, perturb, jitter)   # (A, S*N, H)
    X  = _mlp(X, Ws[1:], bs[1:], relus[1:])                # layers 2..5
return X.reshape(A, S, N).float()                          # fp32 LEAVES the autocast region
```

`_mlp_eager` (`e2b_fast.py:133-149`) branches on A: **at A == 1 it takes a 2-D `F.linear(Y, W[0],
b[0])` path**, explicitly because Inductor's `mm` templating beats its `bmm` templating. The
`baddbmm` path is **dead code in production**. `_base_index` and `_phase_coord`
(`e2b_fast.py:115-130`) cache `q`, `floor(q)` and `c_p` per `(L, R, device)` / `(R, device)` — these
caches are keyed partly by `str(device)` and a multi-device rewrite must keep them per-device.

`build_stacks` (`e2b_fast.py:419-433`) groups arms by `(cls_name, kind.startswith("det"))` in ARMS
insertion order and chunks by `GROUP_MAX`. For the seven arms at `GROUP_MAX = 1` the resulting stack
order is **measured to be identical to the ARMS dict order**:

```
0 det │ 1 es_marg │ 2 es_marg_l0.1 │ 3 es_split_l0.1 │ 4 es_erb_l0.1 │ 5 es_dec_l0.1 │ 6 es_dec_erb_l0.1
```

so the concatenated `(7, 4)` terms index coincides with ARMS order. Mixed det/es stacks are asserted
against (`e2b_fast.py:201`; `:200` computes the `is_det` flag the assert reads).

---

## 3. The objectives

### 3.1 The energy-score estimator

```
ES_d(y; y1, y2) = 0.5·(d(y,y1) + d(y,y2)) − 0.5·d(y1,y2)          # per-sample, shape (B,)
```

**Exactly two draws, coefficients exactly (0.5, 0.5, −0.5).** There is no n-draw generalisation
anywhere in the repo (`c1_model.py:167`, `e1_model.py:257`, `e2b_fast.py:358,372`). Both draws go
through ONE forward pass of `S = 2B` sequences built by `x.repeat(2,1)`, split `y1 = yh[:, :B]`,
`y2 = yh[:, B:]` (`e2b_fast.py:353-355`). **The two draws share the same low-rate conditioning** and
differ only through `eps` and the anchor jitter (both drawn at S = 2B, so draw 1 and draw 2 get
independent jitter).

ES is **linear in `d`**, so `ES_{a·d1 + b·d2} = a·ES_{d1} + b·ES_{d2}`. This is why λ may sit inside
`d` (as the research note writes it,
`notes/2026-09-16-score-geometry-decoder-noise-readouts.md:§1`) or outside (as `e2b_fast.py:382`
computes it), and why `ArmStack`'s `0.5·ES_logmag + 0.5·ES_erb` is **exactly**, not approximately,
the reference's `ES` of the combined distance `0.5·(d_logmag + d_erb)` (`e1_model.py:251-253` vs
`e2b_fast.py:372,381`).

### 3.2 The distances

All STFTs are `torch.stft(y, n, h, window=hann_window(n), return_complex=True)` — `center=True`,
`pad_mode='reflect'`, one-sided, unnormalised, **periodic** Hann (`c1_model.py:113-123`). The window
is cached per `(n, device)` in `_WIN`; `MultiScaleSTFTLoss` allocates a fresh one each call
(`build_notebook.py:1155`) — same values. At T = 48 000 the shapes are **verified** as
(1025, 94), (513, 188), (257, 376).

| distance | formula (per-sample, returns (B,)) | source |
|---|---|---|
| `d_wave(a,b)` | `mean over all non-batch axes of |a−b|` | `c1_model.py:133-134` |
| `d_logmag(a,b)` | `(1/3)·Σ_s mean_{f,t} |lm_s(a) − lm_s(b)|`, `lm_s = log(|STFT_s| + 1e-7)` | `c1_model.py:121-124,136-137` |
| `d_erb(a,b)` | `(1/3)·Σ_s mean_{band,t} |erb_s(a) − erb_s(b)|`, `erb_s = 0.5·log(W_s @ |STFT_s|² + 1e-8)` | `e1_model.py:200-214` |
| `d_ged(a,b)` | `(1/3)·Σ_s ‖lm_s(a) − lm_s(b)‖_F / sqrt(F_s·T_s)` | `e1_model.py:157-163` |
| `d_sliced(a,b)` | 64 random unit directions in R^F, `mean_{p,t} |θ·(Δlm)_{:,t}|` | `c1_model.py:139-153` |

`lowpass(y, R)` (`c1_model.py:126-131`): `Y = rfft(y)`, `k_cut = Y.shape[-1] // R`,
`Y[..., k_cut+1:] = 0`, `irfft(Y, n=T)`. At T = 48 000, R = 4: the rfft has **24 001** bins,
`k_cut = 6000`, bins 0…6000 are **kept** — DC to exactly 6000 Hz = `fs_lo/2`. Verified numerically.
An off-by-one in the retained bin **CHANGES MATH**.

**ERB bank** (`e1_model.py:166-197`): 32 triangular bands, centres equally spaced on the ERB-rate
scale `21.4·log10(1 + 4.37f/1000)` from `f_lo = 50` Hz to `fs/2 = 24000` Hz, built in float64 and cast
to float32, **every ROW normalised to sum 1** (`e1_model.py:195`) — so `W @ P` is a weighted **mean**
of power, not a sum. One bank per STFT scale, cached per
`(n_fft, fs, n_bands, f_lo, device)`. Shapes (32,1025), (32,513), (32,257). At the FULL preset the
uncovered-bin and empty-band repair loops (`e1_model.py:191-194`) **never fire**; bins below 50 Hz
(3 at n=2048, 2 at 1024, 1 at 512) carry zero weight in every band and are invisible to `d_erb`.

`d_sliced` **resamples its 64 directions on every call** (`th = sample_thetas(y.device)` at
`c1_model.py:176`, closed over at `:177` and consumed at `:178`; `sample_thetas` itself at
`c1_model.py:147-153`), so `es_slice`'s `d` is
not a fixed distance across steps. No arm in this run uses it, and `ArmStack` cannot express it — see
I19.

### 3.3 The seven arms, one formula each

λ multiplies the **spectral term only**, never the waveform term. Kinds and λ from
`e3_launch.py:7-15`; weight vectors from `e2b_fast.py:216-223`.

| # | arm | kind | λ | class | `w_lm` | `w_erb` | `w_l2` | `split` | objective |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `det` | `det` | 1e-2 | LISAS | — | — | — | F | `L1(y,ŷ) + λ·MSSTFT(y,ŷ)` |
| 2 | `es_marg` | `es_marg` | 1e-2 | LISAS | 1.0 | 0.0 | 0.0 | F | `mean_b[ ES_{d_wave} + λ·ES_{d_logmag} ]` |
| 3 | `es_marg_l0.1` | `es_marg` | 1e-1 | LISAS | 1.0 | 0.0 | 0.0 | F | as #2, λ = 1e-1 |
| 4 | `es_split_l0.1` | `es_split_marg` | 1e-1 | LISAS | 1.0 | 0.0 | 0.0 | **T** | `mean_b[ ES_{d_wave}(Π_lo y; Π_lo y1, Π_lo y2) + λ·ES_{d_logmag}(y; y1, y2) ]` |
| 5 | `es_erb_l0.1` | `es_marg_erb` | 1e-1 | LISAS | 0.5 | 0.5 | 0.0 | F | `mean_b[ ES_{d_wave} + λ·(0.5·ES_{d_logmag} + 0.5·ES_{d_erb}) ]` |
| 6 | `es_dec_l0.1` | `es_marg` | 1e-1 | **LISASD** | 1.0 | 0.0 | 0.0 | F | formula of #3, class LISASD |
| 7 | `es_dec_erb_l0.1` | `es_marg_erb` | 1e-1 | **LISASD** | 0.5 | 0.5 | 0.0 | F | formula of #5, class LISASD |

`w_l2` is **identically zero for all seven arms** — no arm uses `es_ged`, so the entire `d_ged` code
path at `e2b_fast.py:373-376` is dead code in this run. Keep it or drop it, but do not let a refactor
give any arm a non-zero `w_l2`.

**Arm 4, precisely: only the WAVEFORM term is low-banded.** Its spectral term is full-band
`d_logmag` (`w_lm = 1`, `e2b_fast.py:217,220`; reference `e1_model.py:238-246`). Low-banding the
spectral term too would be a different arm.

**Arm 1, `det`, in full** (`e2b_fast.py:337-352`; identical to `MultiScaleSTFTLoss`,
`build_notebook.py:1147-1161`):

```
ŷ      = forward(x, eps=None, perturb=True)                      # (A, B, T), eps identically 0
L1     = mean_{b,t} |ŷ − y|
MSSTFT = (1/3)·Σ_s [ ‖mag_s(y) − mag_s(ŷ)‖_F / (‖mag_s(y)‖_F + 1e-8)     # "sc"
                   + mean_{b,f,t} |log(mag_s(ŷ)+1e-7) − log(mag_s(y)+1e-7)| ]   # "lm"
loss   = L1 + 1e-2 · MSSTFT
```

**⚠ The `det` arm's spectral-convergence term is BATCH-COUPLED.** Both Frobenius norms are over the
whole `(B, F, T)` tensor — a single scalar per scale, not a per-sample quantity. The stacked path sums
dims (1,2,3) of an `(A,B,F,T)` tensor and divides by `tf.mag_norm[s] = mag_s(y).pow(2).sum().sqrt()`
(`e2b_fast.py:347`, `e2b_fast.py:414`). It is **not** the mean of per-sample ratios. Under any
data-parallel sharding, averaging per-shard `sc` values is a **different number**: the rewrite must
all-reduce `Σ‖Y−H‖²` and `Σ‖Y‖²` before the sqrt and the division, or the `det` arm's objective
silently changes. **Nothing in the repo does this today.** Every other term is a per-sample mean and
shards cleanly. This is the single highest-risk item in the whole specification.

**The sharded composition, as an ordered recipe** (this applies to multi-rank sharding **and** to
gradient accumulation / micro-batching, which are the same operation seen from a different angle):

```
(1)  each shard s computes, per arm and per scale:
         n_s = Σ_{b∈shard, f, t} (Y − H)²          # scalar, NO sqrt
         d_s = Σ_{b∈shard, f, t}  Y²               # scalar, NO sqrt
(2)  all-reduce n = Σ_s n_s  and  d = Σ_s d_s  as SUMS
(3)  sc  = sqrt(n) / (sqrt(d) + 1e-8)              # one scalar per arm per scale
(4)  broadcast sc; the backward flows through steps (1)-(3) as written
```

Three ways to get this wrong that the one-line instruction permits: adding the `1e-8` **inside** the
sum instead of to the already-square-rooted denominator; all-reducing **after** the sqrt; adding a
per-shard epsilon once per shard. All three change the number; none raises. The companion `lm` term
(`e2b_fast.py:348`) is a plain mean over `(1,2,3)` and must be weighted by shard size if the shards
are unequal. The six ES arms' terms are per-sample means over the batch (`.mean(1)`,
`e2b_fast.py:382-383`) and shard cleanly **only for equal shard sizes**.

**The ES branch verbatim** (`e2b_fast.py:353-383`):

```
dwf(a,b) = (a−b).abs().mean(-1)                                  # (A, B)
spread   = dwf(y1, y2)                                           # ALWAYS full band, even for arm 4
dw       = 0.5·(dwf(y,y1) + dwf(y,y2)) − 0.5·spread              # if need_full
dws      = 0.5·(dwf(y_lo,y1l) + dwf(y_lo,y2l)) − 0.5·dwf(y1l,y2l)  # if need_split
dw       = where(split, dws, dw)
spec = 0
for s, (n, h) in enumerate(SCALES):                              # ONE stft call covers BOTH draws
    S_ = |stft(yh.reshape(A·2B, T), n, h)|.view(A, 2B, F, T)
    Lm = log(S_ + 1e-7);  L1_, L2_ = Lm[:,:B], Lm[:,B:];  Ly = tf.lm[s]
    spec += w_lm  · (0.5·(dl(Ly,L1_)+dl(Ly,L2_)) − 0.5·dl(L1_,L2_)) / 3     # dl = mean over (-2,-1)
    spec += w_l2  · (0.5·(d2(Ly,L1_)+d2(Ly,L2_)) − 0.5·d2(L1_,L2_)) / 3     # dead for these arms
    E  = 0.5·log(W_s @ S_² + 1e-8);  E1,E2 = E[:,:B], E[:,B:];  Ey = tf.erb[s]
    spec += w_erb · (0.5·(de(Ey,E1)+de(Ey,E2)) − 0.5·de(E1,E2)) / 3         # de = mean over (-2,-1)
loss = (dw + lams.unsqueeze(1) · spec).mean(1)                   # (A,) : MEAN over the batch
return loss, dw.mean(1), spec.mean(1), spread.mean(1)            # -> history "wave","spec","spread"
```

`spread` is returned as `torch.zeros_like(l_w)` for the det arm (`e2b_fast.py:352`).

**Published cross-check** (`notes/analysis/dynamics_OV3_fast.md:§1`) confirming the λ placement:
`det` 0.005421 + 0.01·1.0821 = 0.016242 vs reported 0.016243; `es_marg_l0.1` 0.005458 + 0.1·0.5883 =
0.064288 vs 0.064285; `es_erb_l0.1` 0.006184 + 0.1·0.3391 = 0.040094 vs 0.040096.

**Load-bearing epsilons.** 1e-7 inside `log` for log-magnitude; **1e-8 inside `log` for ERB POWER**
(an amplitude floor of 1e-4, three decades above the log-magnitude floor); 1e-8 added to the
spectral-convergence denominator; 1e-6 added to the gradient norm in `clip_`. Changing any of them
**CHANGES MATH**. The **0.5** in `erb = 0.5·log(W@|S|² + 1e-8)` is what puts `d_erb` on the same
*slope* as `d_logmag` — removing it doubles the ERB term's effective weight. **CHANGES MATH.** (The
claim that the 0.5 also puts them on the same *scale* is an assertion by the original author,
`e1_model.py:201-203`; see §9.)

**Mean-vs-sum axes.** `d_logmag`/`d_erb` take the MEAN over (F,T) / (band,T), then SUM over scales and
divide by 3. `d_wave` takes the MEAN over time. `d_ged` takes an L2 norm over (F,T) divided by
`sqrt(F·T)`. The ES loss then takes the MEAN over the batch. Any switch from mean to sum on any axis
rescales the gradient and **CHANGES MATH**.

### 3.4 What TargetFeats shares across arms

`TargetFeats(y, R, need_lm, need_erb, need_mag, need_split)` (`e2b_fast.py:406-416`) is constructed
**once per training step** in `_fwd_bwd` (`e2b_fast.py:459`) and shared by every arm of every stack.
It runs in fp32 **outside** autocast. Need flags are the OR across all stacks
(`_needs`, `e2b_fast.py:436-437`), recomputed every step.

| field | value | condition | shape |
|---|---|---|---|
| `y_lo` | `lowpass(y, R)` | any arm splits | (B, T) |
| `mag[s]` | `|stft_s(y)|` | any det arm | (B, F_s, T_s) |
| `mag_norm[s]` | `mag[s].pow(2).sum().sqrt()` | any det arm | **scalar (GLOBAL over B,F,T)** |
| `lm[s]` | `log(mag[s] + 1e-7)` | any logmag arm OR any det arm | (B, F_s, T_s) |
| `erb[s]` | `0.5·log(W_s @ mag[s]² + 1e-8)` | any erb arm | (B, 32, T_s) |

For the seven arms: `need_lm = need_erb = need_mag = need_split = True`, so all five fields are built.
The y-side of every distance is therefore bit-identical across arms. Both draws of every arm go
through **one** `torch.stft` call of `A·2B` rows per scale (`e2b_fast.py:366-367`); the target STFT is
never recomputed per arm.

---

## 4. The training loop

Numbered, one line per operation, shapes at B = 64, A = 1. Source: `train_ov3_fast`
(`e2b_fast.py:570-733`) as invoked at `e8_launch_fast.py:39-42`.

### 4.1 Setup (once)

`arm3(spec)` (`e1_model.py:264-266`) is one line: `(kind, lam) or (kind, lam, cls_name) ->
(kind, lam, cls_name)`, padding a 2-tuple with the default `cls_name = "LISAS"`. It matters because
2-tuples appear elsewhere in the repo (`c1_model.py`'s `arm_loss` era) and a rewrite that reads ARMS
literally will meet them.

```
S1   names  = list(ARMS)                                             # dict insertion order
S2   specs  = {k: arm3(ARMS[k]) for k in names}                      # (kind, lam, cls_name)
S3   base["LISAS"]  = LISAS(CFG).to(DEVICE)                          # consumes the CPU default generator
S4   base["LISASD"] = copy_shared(base["LISAS"], LISASD(CFG).to(DEVICE))   # extra n_dec cols ZEROED
S5   models[k] = deepcopy(base[cls_of(k)])                           # export targets only
S6   models[k].tau = 0.0 if kind.startswith("det") else 1.0          # drives probe_metrics
S7   stacks = build_stacks(names, specs, base, DEVICE)               # 7 stacks, A=1 each, order of §2.6
S8   for each stack: P[k] = base.state_dict()[k].unsqueeze(0).repeat(A,…).clone()
S9   opts[i]   = Adam(stacks[i].params(), lr=1e-3, fused=True)       # betas (0.9,0.999), eps 1e-8, wd 0
S10  scheds[i] = MultiStepLR(opts[i], [int(f*steps) for f in (.2,.4,.5,.6,.7,.8)], gamma=0.5)
S11  hist[k]   = {16 keys, §4.5}
S12  vb   = val_batches(val_corpus, tag, 8, 64)                      # stream(f"{tag}/val"), GPU-resident
S13  rng  = stream(f"{tag}/batches")                                 # numpy Generator, blake2b(label) ^ 0
S14  naive = snr_db(probe, naive_upsample(probe, CFG))               # scalar, recorded into every dev point
S15  logf = open(ROOT/f"train_{tag}.log","a");  hist_path = ROOT/f"ov3_history_{tag}.json"
S16  pending = []            # (step, lr, (7,4) GPU tensor), drained by flush()
S17  dt_hist = []            # last <=50 wall deltas; NO cuda.synchronize anywhere
```

### 4.2 The step (`e2b_fast.py:640-658`)

**There is exactly ONE forward pass and ONE `sample_eps` call per arm per step.** `ArmStack.losses`
does the draw and the forward **itself** (`e2b_fast.py:353-354`) and takes no `yh` argument;
`_fwd_bwd` calls only `losses` (`e2b_fast.py:469`). T5a–T5d below are shown **indented, as the
internals of the T5 call**, not as separate operations — implementing them literally doubles both the
cost and the RNG consumption and contradicts I7's draw table.

```
T1   x, y = batch_pinned(corpus, rng, 64)          # x (64,12000) fp32, y (64,48000) fp32, SHARED
T2   for s in stacks: for p in s.params(): p.grad = None   # set_to_none, 7 x 18 tensors = 126
T3   need = (any need_lm, any need_erb, any is_det, any need_split)
T4   tf   = TargetFeats(y, R, *need)               # ONCE per step, fp32, outside autocast
T5   for s in stacks:                              # STRICT stack order; EACH consumes its own RNG slice
T5       loss, wave, spec, spread = s.losses(x, y, tf, perturb=True)      # each (A,) = (1,)
T5           |  # --- INSIDE losses(), e2b_fast.py:337-383 -------------------------------
T5a          |  if s.is_det:  yh = s.forward(x, None, None, perturb=True)     # (1, 64, 48000)
T5b          |  else:         e,d = s.sample_eps(128, 12000)  # eps_enc then eps_dec; ONE call
T5c          |                yh  = s.forward(x.repeat(2,1), e, d, perturb=True)  # (1,128,48000)
T5d          |                y1, y2 = yh[:, :64], yh[:, 64:]
T5e          |  # the jitter randn is drawn INSIDE forward -> _layer1 (e2b_fast.py:311)
T5f      loss.sum().backward()                     # sum over the stack's arms; arms are disjoint
T5g      s.clip_(1e-3)                             # PER-ARM norm, see below
T5h      terms.append(stack([loss,wave,spec,spread], 1).detach())         # (A, 4)
T6   for (s,o,sc) in zip(stacks,opts,scheds): o.step(); sc.step()   # scheduler steps EVERY step
T7   for (s,t) in zip(stacks,terms):
        s.ema = t[:,0].clone() if s.ema is None else s.ema*0.98 + t[:,0]*0.02      # GPU, every step
T8   now = time.time(); dt_hist.append(now - t_last); t_last = now; dt_hist = dt_hist[-50:]
```

`clip_` (`e2b_fast.py:386-395`) — exactly PyTorch's `clip_grad_norm_` coefficient formula, one norm
per arm over **all** of that arm's parameters jointly:

```
sq   = Σ_params  g.reshape(A,-1).pow(2).sum(1)                      # (A,)
coef = (clip / (sq.sqrt() + 1e-6)).clamp(max=1.0)                   # clip = 1e-3
for g in grads: g.mul_(coef.view(A, 1, 1, …))
```

### 4.3 Cadences

**The callback interface.** `train_ov3_fast(..., on_step=None, on_epoch=None, steps_per_epoch=None)`
(`e2b_fast.py:570-572`). The launcher supplies both callbacks from `make_dashboard(tag, arms)`
(`build_train_notebook.py:319-406`, called at `:489`; `e8_launch_fast.py:38`), which returns
`(on_step, on_epoch)`. Both take **one positional argument**, the `info` dict, and return nothing;
exceptions are caught and printed (`e2b_fast.py:665-674`). `info` has exactly these 16 keys
(`e2b_fast.py:623-627`):

```
tag, step (1-based = step+1), steps, frac (= step/steps),
epoch (= step/spe or None), epochs_total (= steps/spe or None),
t_elapsed, ms_per_step, eta_s, samples_per_s, audio_s_per_s,
batch, seg_s, lr, gpu (the gpu_stats() dict, e2b_fast.py:550-567),
arms  -> {arm: {loss_ema, loss, wave, spec, spread, val_loss, val_wave, val_spec,
                snr0, def0, snr1, def1}}      # the val/dev entries are the LAST recorded, or None
```

```
C1   every log_every = 25 steps  (step = 0, 25, 50, …):
        lr_now = scheds[0].get_last_lr()[0]          # STACK 0's lr, recorded for EVERY arm
        pending.append((step, lr_now, cat([t.reshape(-1,4) for t in terms], 0)))   # stays on GPU
        if on_step: info = _info(step, terms, lr_now)   # ONE .cpu().tolist() of a (7,5) tensor
                    on_step(info)                       # exceptions caught and printed
        if on_epoch and (step+1)//spe > epoch_mark: epoch_mark = int((step+1)//spe); on_epoch(info)
C2   every val_every = 500 steps, and on the last step:
        vals = val_loss_fast(stacks, vb, seed=1234)     # ONE host transfer, see below
        emas = cat([s.ema for s in stacks]).cpu().tolist()
        hist[k]["val_step"/"val_loss"/"val_wave"/"val_spec"/"train_loss_ema"].append(...)
        print one line; logf.write + flush
C3   every ckpt_every = 500 steps, and on the last step:
        flush()                                          # ONE .cpu().tolist() over the 20 pending rows
        for k in names:
            m  = stacks[si].export(a, models[k])         # no_grad copy_ of the stacked params
            pm = probe_metrics(m, probe, CFG, naive)     # tau (0.0,) det / (0.0,1.0) stochastic
            hist[k]["dev_step"/"snr_naive"/"snr0"/"def0"/"snr1"/"def1"].append(...)
            save_ckpt({...§4.5...}, CKPT/tag/f"{k}.pt")
        print one line; json.dump(hist, hist_path);  plot_curves(hist, tag)
```

At 104,950 steps: **4,198** training-grid rows per arm, **210** validation points (209 multiples of
500 plus the final step, since 104 950 mod 500 = 450) and **210** checkpoint events.

`flush()` (`e2b_fast.py:629-638`) writes **only** `wave` (r[1]), `spec` (r[2]), `spread` (r[3]) and
`lr`. **The per-step loss r[0] is never stored in the history JSON** — only its EMA at validation
points, as `train_loss_ema`.

**Host syncs.** There is NO `.item()`, no `.cpu()` and no `torch.cuda.synchronize()` in the common
path. Syncs are exactly:

- one `(7,5)` transfer every `log_every = 25` steps in `_info` (`e2b_fast.py:612-613`) — **only if
  `on_step` is supplied, or the epoch boundary fires** (`e2b_fast.py:662-663` guards the `_info` call);
- one `(7,3)` transfer in `val_loss_fast` plus one EMA transfer every 500 steps;
- one `(20,7,4)` transfer in `flush()` every 500 steps.

**This inventory is conditional on the callbacks.** The launcher always supplies them
(`build_train_notebook.py:489`), so the reference run had the 25-step sync. A rewrite that "saturates
the GPU" by dropping `on_step`/`on_epoch` removes that sync entirely, which changes the measured
ms/step (the timing deltas below are launch-time deltas *between* the callback syncs) and therefore
invalidates the cost model of §7.5 against which the budget is set. **Keep the callbacks, or re-measure
and say so in the run log.** Step timing uses `time.time()` deltas taken **after** the optimiser and EMA
but **before** the val/ckpt blocks, so a val+ckpt step's cost is charged to the FOLLOWING step; with
no explicit synchronize, the deltas measure launch time between the 25-step callback syncs rather than
kernel time.

### 4.4 Validation objective (`val_loss_fast`, `e2b_fast.py:528-547`)

```
@torch.no_grad()
gen = torch.Generator(device=DEVICE).manual_seed(1234)      # ONE generator, created fresh per call
for (x, y) in vb:                                           # 8 FIXED batches, 64 x 1 s
    tf = TargetFeats(y, R, *need)
    for i, s in enumerate(stacks):                          # SAME stack order as training
        eps = None if s.is_det else s.sample_eps(2B, L, gen)
        loss, wave, spec, _ = s.losses(x, y, tf, eps=eps, perturb=False)   # NO anchor jitter
        acc[i] += stack([loss, wave, spec], 1)
return {arm: tuple(v)} from cat(acc,0).div_(8).cpu().tolist()
```

`spread` is computed and **discarded**. The private generator means validation never perturbs the
training RNG stream. **Autocast bf16 is still in force** (the forward always enters `_autocast()`,
`e2b_fast.py:243`), so validation is bf16 in the stacked path and fp32 in the sequential reference.
`perturb=False` also puts `_layer1` on the gather-free branch (`e2b_fast.py:316-321`).

**KNOWN DISAGREEMENT with the reference.** `e2_trainer.val_loss` (`e2_trainer.py:26-30`) uses
`torch.random.fork_rng` + `manual_seed(1234)` **per arm**, so every arm sees the SAME validation
noise. `val_loss_fast` seeds ONE generator and consumes it across all batches **and** all stacks in
stack order, so each arm gets a different slice — and LISASD stacks consume extra draws that shift the
stream for everything after them. **The 50-epoch run uses the fast path.** Changing this is a
deliberate scientific decision, not a rewrite detail (§9).

**⚠ The `det` arm's `val_loss` inherits §3.3's batch coupling.** `val_loss_fast` rebuilds
`TargetFeats` **per validation batch** (`e2b_fast.py:536`), so the det branch divides by
`tf.mag_norm[s] = mag_s(y).pow(2).sum().sqrt()` — a scalar **global over that one batch of 64**
(`e2b_fast.py:347,414`) — and then averages the 8 batch values (`:542`). The det arm's `val_loss` is
therefore **a mean of 8 batch-global Frobenius ratios at B = 64**, not a per-utterance quantity and
not the ratio over the 512 validation segments. `n_val_batches = 8` and the validation batch size 64
are **part of the det arm's validation objective, not free parameters**: changing either, or sharding
validation across ranks, silently changes `val_loss` for the det arm **and for no other arm**. The
`s.losses(...)` line in the pseudocode above hides this.

Validation numbers are **each arm's own objective**. Read them along time within one arm; never
compare them across arms (`notes/analysis/dynamics_OV3_fast.md:§0`).

### 4.5 History and checkpoint contracts

**History: exactly 16 keys per arm**, created identically in both trainers (`e2b_fast.py:588-590`):

```
step, wave, spec, spread, lr,
val_step, val_loss, val_wave, val_spec, train_loss_ema,
dev_step, snr0, def0, snr1, def1, snr_naive
```

`step` is **0-based** on the `log_every` grid (0, 25, 50, …); `val_step` and `dev_step` are
**1-based** (`step+1` = 500, 1000, …) (`e2b_fast.py:661` vs `:683,:698`). `snr1`/`def1` are Python
`None` (JSON null, **not NaN**) for deterministic arms (`e2b_fast.py:700-701`) so the JSON
strict-parses; `spread` is exactly 0.0 for `det`. `audit/dynamics_ov3.py:51-52,83-86` reads all 16 and
maps null → NaN. **The keys are the contract; the constants in that script are not** — see §9.

**Dev probe** (`probe_metrics`, `c1_model.py:185-192`): for each tau in `(0.0,)` (det) or
`(0.0, 1.0)` (stochastic) it calls `reconstruct(m, probe, cfg, tau=tau, seed=0)` and returns
`(snr_db, mean band-energy ratio in [6000, 24000] Hz on the EVAL basis n_fft=1024, hop=256)`.
`reconstruct` decodes in chunks of `1<<15 = 32768` output samples, each ending in a `.cpu().numpy()`
transfer, and runs **13 times per checkpoint** (1 for det + 2 for each of the six stochastic arms).
It is deterministic given the weights (`torch.Generator(seed=0)`).

#### 4.5.1 The probe readout, verbatim — `snr0`, `def0`, `snr1`, `def1`, `snr_naive`

`def0` / `def1` is **the document's headline scientific number** (−19.09 dB for `det`, §0). Four of the
sixteen history keys are underivable without the following, so it is given in full.

**The analysis STFT is NOT `torch.stft`.** It is the repo's own numpy **float64** `stft()`
(`build_notebook.py:300-307`) with `_hann` (`:295-297`). No centering, no reflect padding, explicit
frame indexing:

```python
def _hann(n):                                    # PERIODIC Hann, float64
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)

def stft(x, n_fft, hop):                         # -> (n_frames, n_bins) complex128
    x = np.asarray(x, dtype=np.float64)
    length, pad = len(x), n_fft
    xp = np.pad(x, (pad, pad + n_fft))           # n_fft zeros in front, pad+n_fft at the end
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(xp[idx] * _hann(n_fft), axis=-1), pad, length
```

**Substituting `torch.stft` for this analysis STFT CHANGES MATH for the reported deficits** — it
centres, reflect-pads and (in the training path) runs fp32; the padding rule alone shifts every frame
and adds/removes frames at both ends.

```python
def snr_db(y, y_hat):                            # build_notebook.py:643-647
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    n = min(len(y), len(y_hat))                  # BOTH truncated to the shorter
    e = y[:n] - y_hat[:n]
    return 10.0 * np.log10(np.sum(y[:n] ** 2) / max(np.sum(e ** 2), 1e-20))

def third_octave_edges(fs, f_lo, f_hi):          # build_notebook.py:658-662
    f = [f_lo]
    while f[-1] < f_hi:
        f.append(f[-1] * 2 ** (1 / 3))           # repeated x 2^(1/3), inclusive of the overshoot edge
    return np.array(f)

def band_energy_ratio(y, y_hat, fs, n_fft, hop, f_lo, f_hi, frame_mask=None):   # :665-679
    Sy, Sh = np.abs(stft(y, n_fft, hop)[0]), np.abs(stft(y_hat, n_fft, hop)[0])
    freqs  = np.fft.rfftfreq(n_fft, 1.0 / fs)
    edges  = third_octave_edges(fs, f_lo, min(f_hi, fs / 2 - 1))                # = 23999.0 at fs=48k
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (freqs >= a) & (freqs < b)                                        # half-open [a, b)
        if sel.sum() == 0: continue                                             # empty bands SKIPPED
        num, den = np.sum(Sh[:, sel] ** 2), np.sum(Sy[:, sel] ** 2)             # over ALL FRAMES
        out.append((np.sqrt(a * b), 10.0 * np.log10((num + 1e-20) / (den + 1e-20))))
    return np.array(out)                          # (n_bands, 2): centre freq, dB

def naive_upsample(y, cfg):                      # build_notebook.py:1229-1232 -> snr_naive
    y = np.asarray(y, np.float64)
    return sps.resample_poly(decimate(y, cfg.upsample), cfg.upsample, 1)[:len(y)]
```

`probe_metrics` calls it as `band_energy_ratio(probe, yh, fs_hi, eval_n_fft=1024, eval_hop=256,
f_lo=fs_lo/2=6000, f_hi=fs_hi/2=24000)` (`c1_model.py:190`) and reduces with
`float(np.mean(b[:, 1]))` (`:191`) — **the UNWEIGHTED mean over third-octave bands of the per-band dB
ratios**, each band's ratio being a **global all-frame energy** ratio, not a per-frame mean of ratios.
`snr_naive = snr_db(probe, naive_upsample(probe, CFG))` is computed once at setup (S14,
`e2b_fast.py:596`) and recorded into every dev point.

**Two epsilons of 1e-20** (numerator and denominator inside the log) and one `max(..., 1e-20)` in
`snr_db`. Changing any of them **CHANGES MATH** for the reported readouts.

#### 4.5.2 Probe path contracts: length, chunking, and noise pairing

**Length and chunking.** `reconstruct` (`c1_model.py:66-78`) re-decimates the **raw** probe in
float64 (`:72`) — the same arithmetic as the training corpus, not a reuse of `corpus.X` — encodes
once, and decodes in chunks of `chunk = 1 << 15 = 32768`. It produces
`n_out = R · len(decimate(y, R))` samples and then **zero-pads and truncates to `len(y)`** (`:78`).

**`chunk` must remain a multiple of `R`.** Every chunk boundary `j0`, `j1` is then a multiple of 4,
which is exactly the guard `j0 % R == 0 and (j1 - j0) % R == 0` that keeps `_decode_subpixel` on the
**gather-free** branch (`e1_model.py:115`). A rewrite that changes `chunk` to a non-multiple of 4
silently moves the probe onto the jittered/gather branch — different kernel, different rounding, and
the probe stops being comparable with the reference run's `def0`/`def1`. Added as **I25**.

**Probe noise IS paired across arms; validation noise is NOT.** `reconstruct(..., seed=0)` builds a
**fresh** `torch.Generator(device).manual_seed(0)` **per call** (`c1_model.py:38`, `e1_model.py:36-38`),
so every stochastic arm's τ = 1 probe uses the **same** `eps_enc` draw — the exact opposite convention
to `val_loss_fast`'s single shared generator (§4.4's KNOWN DISAGREEMENT). Because `LISASD.sample_eps`
draws `eps_enc` **first** from that freshly seeded generator and `eps_dec` second
(`e1_model.py:37-38`), the LISASD arms receive the **same `eps_enc`** as the LISAS arms; only their
additional `eps_dec` is new. **Verified numerically 2026-09-19:** `randn(1,8,L, gen=seed 0)` is
bit-identical between the two classes. So `def1` is a genuinely paired cross-arm readout, and a
rewrite that replaces the per-call generator with a shared or per-arm one **breaks that pairing**.

#### 4.5.3 Artefact layout

Three distinct directories, all created at boot (`build_notebook.py:135-149`); `ROOT` is the Drive
mount on Colab and `./lisa_rtm_cache` locally:

| artefact | path | mode | writer |
|---|---|---|---|
| training log | `ROOT/train_{tag}.log` | **APPEND** (`open(..., "a")`, `e2b_fast.py:597`) — a restart CONCATENATES | trainer |
| launch plan header | `ROOT/train_{tag}.log` | **TRUNCATE** (`write_text`, `build_train_notebook.py:487`; `e8_launch_fast.py:36`) | launcher, before the trainer opens it |
| history JSON | `ROOT/ov3_history_{tag}.json` | overwritten every `ckpt_every` (`e2b_fast.py:598,713`) **and once more by the launcher** (`build_train_notebook.py:493`) | both |
| per-arm checkpoint | `CKPT/{tag}/{arm}.pt` where `CKPT = ROOT/checkpoints` (`build_notebook.py:148`); parent created at `e2b_fast.py:593-594` | overwritten every `ckpt_every`, staged via `/tmp/{stem}_{pid}.pt` (`build_notebook.py:1245`) | trainer |
| curves figure | `FIGS/ov3_curves_{tag}.png` where `FIGS = ROOT/figures` (`build_notebook.py:149`) | overwritten every `ckpt_every` (`e2_trainer.py:73`) | `plot_curves` |

`audit/dynamics_ov3.py:20` reads `lisa_rtm_cache/results/ov3_history_OV3_fast.json` — a `results/`
subdirectory the trainer never writes to. **The 50-epoch run must either write into `ROOT/results/`
or the history must be copied there before the audit runs; state which in the run log** (§9.15).

**Checkpoint dict** (`e2b_fast.py:703-705`), consumed by `load_arm` (`c1_model.py:278-284`), which
reads `ck['cls']`, `ck['n_noise']` and `ck['arm'][0]`:

```
{"model": m.state_dict(), "step": step+1, "history": hist[k],       # the LIVE dict, full history to date
 "arm": (kind, lam, cls_name), "n_noise": 8, "n_dec": 0|4,
 "cls": "LISAS"|"LISASD", "batch": 64, "seg": 48000, "tag": tag}
```

Written by `save_ckpt` (`build_notebook.py:1241`): torch.save to /tmp then `shutil.copyfile` with 4
retries (3/6/9/12 s backoff); on total failure it warns and keeps the /tmp copy. One file per arm at
`CKPT/tag/{k}.pt`, **overwritten** every 500 steps.

**There is no optimiser-state, scheduler-state or RNG-state checkpointing.** An interrupted 50-epoch
run cannot be resumed without changing the trajectory. At ~27 GPU-h of work this is a real risk; see
§9 for the ADDITION the rewrite probably needs and the test that it changes nothing.

### 4.6 The epoch boundary

A 50-**epoch** run makes this central, and the loop has no epoch structure at all. Three facts:

**(a) "Epoch" is a pure step-count convention; there is no epoch in the data.** Batches are drawn
i.i.d. **with replacement** from random R-aligned offsets into the concatenated corpus
(`c1_model.py:102-104`, `e2b_fast.py:444-446`). Nothing resets, reshuffles, re-seeds, validates or
checkpoints at a boundary, and **there is no coverage guarantee over the corpus** — at 50 nominal
epochs the expected fraction of the corpus never sampled is not zero, and some segments are drawn many
times. `steps_per_epoch = corpus.hours · 3600 / (BATCH · SEG / fs_hi)` (`e8_launch_fast.py:25`,
`build_train_notebook.py:476`) is an *audio-seconds* bookkeeping identity, nothing more.

**(b) The only epoch-triggered action is the `on_epoch` callback, and it fires LATE.** The crossing
test lives **inside** `if step % log_every == 0` (`e2b_fast.py:659,662,669`), so `on_epoch` can only
fire on the 25-step grid — **up to `log_every − 1 = 24` steps after** the true crossing. The mark is
`epoch_mark = int((step + 1) // spe)` (`:670`) with `spe` a **float** (`:608`), so the comparison
`(step + 1) // spe > epoch_mark` is float floor division. The `info` passed to `on_epoch` is the
**same dict** already built for `on_step` at that step (`:663`), not a fresh one.

**(c) `steps_per_epoch` is an OPTIONAL keyword and silently disables everything if omitted.**
`spe = float(steps_per_epoch) if steps_per_epoch else None` (`:608`). With `spe is None`:
`info["epoch"]` and `info["epochs_total"]` are `None` (`:624`), the `on_epoch` branch is never entered
(`:662,669`), and the dashboard's per-epoch table stays empty. No warning is printed. The launcher
passes it (`build_train_notebook.py:492`); a rewrite that drops the argument loses all epoch reporting
without any other symptom.

Nothing in (a)–(c) affects the gradients. It affects **what "50 epochs" is allowed to mean**: it is a
step count derived once at launch (§8), and after that the loop does not know epochs exist.

### 4.7 Teardown (`e2b_fast.py:720-733`)

After the step loop, in order:

```
E1   flush()                          # a SECOND flush; drains any pending rows the last ckpt did not
E2   if steps and (on_step or on_epoch):
E2a      info = _info(steps - 1, terms, scheds[0].get_last_lr()[0])
         #  ^ `terms` is the tensor list from the LAST TRAINING STEP, not a fresh forward,
         #    and lr is read AFTER that step's sched.step() (see §6.1 I6)
E2b      on_step(info)                                       # fires unconditionally if supplied
E2c      on_epoch(info)  if (spe and steps // spe > epoch_mark)   # conditional final epoch mark
E3   logf.close()                     # the append-mode train_{tag}.log
E4   for k in names: stacks[si].export(a, models[k])   # RE-EXPORT every arm's stacked params
E5   return (models, hist)
```

Then the **launcher** writes the history JSON a second time (`build_train_notebook.py:493`;
`e8_launch_fast.py:43`). A rewrite that stops at the final checkpoint produces a different artefact
set (no final callback, a stale `models` dict whose weights are those of the last checkpoint export
rather than the final step) and a different return value. E4 is what makes `models[k]` hold the
step-`steps` weights rather than the step-`steps` **checkpoint** weights — at `ckpt_every = 500` and
`steps = 104,950` the last checkpoint is on the final step anyway, but the two are not the same code
path and a rewrite must not assume they coincide.

---

## 5. The data pipeline

Six stages, one shared source of randomness.

**(1) Acquisition** (`overnight/cell1_corpus.py`, inlined verbatim as notebook §5 by
`build_train_notebook.py:178`). The corpus is **not** the DataShare zip — it is the Hugging Face
dataset `sanchit-gandhi/vctk`, 27 parquet shards (~11 GB), via
`snapshot_download(..., allow_patterns=["data/*.parquet"], max_workers=8)` (`cell1_corpus.py:9,14`).
A row survives if `"mic1"` appears in the `file` column **or** in `audio["path"]`
(`cell1_corpus.py:51-52`). Roles (`cell1_corpus.py:21-27`): `test` = {p236, p237, p238};
`skip` = any id not matching `p<digits>` (i.e. VCTK's `s5`); `paper_test` = numeric id ≥ 350;
`train` = everything else. Measured: **97 train speakers, 39,639 utterances, 37.3 h**
(`notes/2026-09-04-overnight-lambda-frontier.md:19`).
`assert not (set(train_spk) & set(test_spk))` at `cell1_corpus.py:84`.

**(2) Decoding / normalisation** (`load_bytes`, `cell1_corpus.py:29-39`): decode FLAC to **float64**;
mono by channel mean; `resample_poly` only if `fs != 48000` (never fires — the Hub files are 48 kHz);
**drop anything shorter than 1.0 s = 48,000 samples**; peak-normalise to **0.95** per utterance; cast
to float32; drop if `max|x| == 0`. Each role's list is then sorted by `(speaker_id, file)`
(`cell1_corpus.py:62-63`) — this sort is what makes the whole pipeline deterministic.

**(3) Decimation** (`decimate`, `build_notebook.py:1034-1036`) is literally
`scipy.signal.resample_poly(x, 1, 4)`. Expanded (scipy 1.17.1, `_signaltools.py:4024-4034,4073`):
`max_rate = 4`, `f_c = 1/4` of Nyquist, `half_len = 40`, `h = firwin(81, 0.25, window=("kaiser", 5.0))`,
`upfirdn(h, x, 1, 4, mode="constant", cval=0)`. **Measured in this repo's venv (2026-09-19):**
81 taps, `sum(h) = 1.0` exactly, **−6.02 dB at 6.0 kHz**, **−55.41 dB at 7.2 kHz**,
**−70.47 dB at 8.4 kHz**, passband ripple **+0.0178 / −0.0063 dB** over 0–4.8 kHz. Symmetric,
linear-phase, zero group delay after scipy's trim — so `x_lo[k]` is time-aligned with `y[4k]`, which
is exactly what makes LISA's `i = floor(j/R)` coordinate arithmetic correct. The signal is
**zero-padded** outside its support, so the first and last ~10 output samples of every utterance carry
a deterministic edge transient.

**(4) HostCorpus** (`c1_model.py:81-99`). Utterances shorter than `seg_hi` are dropped and the rest
truncated to a multiple of R (`c1_model.py:87-88`), so **every `off[i]` is a multiple of 4** and
`hi0 // R` is exact. Decimation is done in **float64** across a 32-thread pool (`c1_model.py:89-90`),
as is `reconstruct`'s (`c1_model.py:72`) — training input and inference input come from the same
arithmetic. `Y` and `X` are two flat float32 concatenations. At 37.3 h,
`sum(lens) = 6,445,440,000` samples: `Y.nbytes = 25.78 GB`, `X.nbytes = 6.45 GB`, **total 32.23 GB**
(derived; this is the docstring's "a 37 h corpus is 32 GB", `c1_model.py:83`). Peak during
construction ≈ **64.5 GB** (`train_utts` 25.78 + `xs` 6.45 + `Y` 25.78 + `X` 6.45; `ys` is a list of
views and costs nothing), settling to ~32.3 GB once the launcher trims `train_utts` to 200 utterances
(`e3_launch.py:36-38`).

**(5) The batch RNG contract.** One `np.random.default_rng` seeded by
`(int.from_bytes(blake2b(label, 8).digest(), "big") ^ SEED) % 2**63`, `SEED = 0`
(`build_notebook.py:157-162`), with `label = f"{tag}/batches"` (`e2b_fast.py:592`). **Per step it
makes exactly two calls, in this order:**

```
idx    = rng.integers(corpus.n, size=B)                      # (1)  c1_model.py:102
n_pos  = (corpus.lens[idx] - corpus.seg_hi) // R + 1         #      no RNG
starts = (rng.random(B) * n_pos).astype(int64) * R           # (2)  c1_model.py:104   R-ALIGNED
hi0    = corpus.off[idx] + starts                            #      hi0 % R == 0  (INVARIANT)
y = corpus.Y[hi0[:,None]        + _ar_hi]                    # (B, 48000) float32
x = corpus.X[(hi0//R)[:,None]   + _ar_lo]                    # (B, 12000) float32
```

`batch_pinned` (`e2b_fast.py:440-451`) reproduces those six lines verbatim and inserts only
`.pin_memory()` before `.to(DEVICE, non_blocking=True)`; if `PIN` is off or there is no CUDA it
delegates straight to `corpus.batch`. Its "identical RNG consumption" claim is therefore **proved by
inspection**, not merely asserted: same two calls, same arguments, same order ⇒ byte-identical `x`
and `y`.

The pre-run benchmark `time_ov3_fast` draws from `stream("timing")` (`e2b_fast.py:740`), a different
stream, so it does not perturb the training batch sequence — but it **does** advance the global torch
generators (§6.2, CHANGES-TRAJECTORY-1).

**(6) Splits.** `test_utts` = first 40 per held-out speaker in `(speaker, file)` order = **120**
(`cell1_corpus.py:69`); `EVAL12 = test_utts[:12]` (`e4_eval.py:253`);
`val_corpus = HostCorpus(test_utts[12:52])` = 40 utterances, commented as 28 of p236 + 12 of p237
(`e3_launch.py:35`); `probe = test_utts[0]`, resolved to **p236_002** in the measured run, whose own
naive sinc upsample scores **14.89 dB** — the probe's SNR ceiling
(`notes/analysis/dynamics_OV3_fast.md:§0`). `val_batches` (`e2_trainer.py:10-12`) draws 8 batches ONCE
from `stream(f"{tag}/val")` using the **non-pinned** `HostCorpus.batch`, and they stay resident on the
GPU for the whole run: `8 · 64 · (48000+12000) · 4 = 122.88 MB`.

**Per-step host work** (derived): a numpy fancy-index gather building `(64,48000)` and `(64,12000)`
int64 index arrays (30.72 MB of temporaries) from the 32 GB `Y`/`X`, a **fresh** `.pin_memory()`
page-locking copy of both tensors, and a 15.36 MB H2D transfer.

**What may be prefetched, and what may not.** Replacing the per-step `pin_memory()` with a persistent
pinned double-buffer + `copy_`, or running the numpy fancy-index **gather** and the **H2D copy** ahead
on a worker thread, is **byte-identical and touches no RNG**. The **two `rng` calls may not move**, and
"in step order" is not a sufficient condition:

- the calls must stay **interleaved within each step** — `rng.integers(corpus.n, size=B)` then
  `rng.random(B)`, **once each, per step**. A depth-K prefetcher written the obvious way, as
  `rng.integers(n, size=(K,B))` followed by `rng.random((K,B))`, consumes the same number of PCG64
  outputs **in a different order** and yields a completely different training set while remaining
  single-threaded and "in step order";
- the generator is **host-side numpy `np.random.default_rng` (PCG64)** seeded by
  `stream(f"{tag}/batches")` (`build_notebook.py:157-162`). Substituting a torch or CUDA RNG for batch
  sampling is forbidden regardless of how it is seeded;
- `rng.random(B)` returns **float64**, and `(rng.random(B) * n_pos).astype(np.int64)` must stay
  float64 end to end — in fp32 the multiply-and-truncate lands on a different integer for a small
  fraction of draws;
- `idx`, `starts`, `off`, `hi0` and both aranges are **int64** (I23).

**`steps_per_epoch = corpus.hours · 3600 / (BATCH · SEG / fs_hi)`** (`e8_launch_fast.py:25`), which at
BATCH = 64, SEG = 48000 reduces to `corpus.hours · 56.25`. **This must be recomputed at run time from
the corpus actually built, not hardcoded** — see §8.

---

## 6. The numerical contract

### 6.1 Invariants a rewrite MUST preserve

**I0 — FIXTURES ARE CAPTURED FROM THE UNMODIFIED CODE, FIRST. This clause is load-bearing for I1,
I3, I6, I10, I17 and I18, and without it those invariants are tautologies.** Several tests below
compare against "a checked-in fixture". **There is no fixtures directory in this repository**
(`ls /home/user/lisa-rtm` → `audit notes overnight overnight2 overnight3 demo sampler web`; `FIXTURES`
at `build_notebook.py:147` lives under the Drive cache `ROOT`, not in git), and the corpus manifest was
never committed (§9.28). As written, a rewrite satisfies every one of those tests by generating the
fixture **from its own rewritten code**. That is the failure mode most likely to be reached, because
it requires no bad intent at all.

**Therefore: before any rewrite lands**, run the **unmodified** `overnight3/e2b_fast.py` at a named
git SHA, commit the fixtures under a named path in this repository, and record that SHA in the run
log. The fixture set is exactly:

| fixture | contents |
|---|---|
| `batches` | the first **1,000** steps of `(idx, starts)` from `stream(f"{tag}/batches")` at B = 64, as int64 arrays |
| `corpus` | `corpus.n`; hashes of `corpus.lens` and `corpus.off`; `corpus.hours` to **six significant figures**; hashes of `corpus.X` and `corpus.Y`; `val_corpus.n`; `len(test_utts)` |
| `init` | `state_dict` hashes, per tensor, for the run's `LISAS` and `LISASD` (after `copy_shared`) |
| `stacks` | the `[s.names for s in stacks]` list |
| `schedule` | the realised milestone integers and the LR at `t = M−1` and `t = M` for all six |
| `randn` | a **20-step** `(order, shape, dtype)` trace of every `torch.randn` call, per arm |

A test that compares against a fixture generated by the code under test is **not a test**; say so in
the review.

**Flag requirements for the tests.** §6.2 mandates `use_deterministic_algorithms(False)` +
`cudnn.benchmark = True` for the run itself, under which run-to-run bitwise reproducibility of the
gradients is **not available and must not be claimed**. Several tests below need it anyway. Each test
is annotated:

- **[det]** — must be run with `DETERMINISTIC = True`, as a pre-flight only: **I4**, **I5**'s
  gradient comparison, **I22**, the §9.13 resume test. Under the run's own flags they will fail, and
  the person who runs them, sees the failure and concludes the invariant is broken is the failure
  mode this annotation exists to prevent (the non-determinism is in the `scatter_add` behind the
  layer-1 gather backward, `e2b_fast.py:313-314`, the `perturb=True` path).
- **[any]** — flag-independent, because no gradient atomics run: **I1, I2, I3, I10, I12, I17, I18,
  I23, I24, I25** (RNG/data/shape tests, no backward), and **I13** — the latter specifically because
  `perturb=False` takes the gather-free branch (`e2b_fast.py:316-321`) and `val_loss_fast` is
  `no_grad`, so there are no atomics at all.
- **[run]** — must hold under the run's own flags, as a runtime assertion: **I24**.

`DETERMINISTIC = True` is a **pre-flight-only** setting. It must be **off** for the ~27 GPU-h run
(3.9× slower, §6.2).

**I1 — Identical batches.** One numpy Generator, `stream(f"{tag}/batches")`, drawn ONCE per step
**outside** the arm loop (`e2b_fast.py:641`); every arm consumes bit-identical `(x, y)`.
*Test [any]:* assert `torch.equal` on `x` and `y` across arms at N random steps; replay 1,000 steps of
the stream and compare the recorded `(idx, starts)` arrays bit-for-bit against the **I0** fixture, and
**additionally compare the gathered `x` and `y` tensors themselves** — comparing only `(idx, starts)`
would not catch a wrapped index (I23). Assert that exactly two `rng` calls occur per step, in the
order `integers` then `random` (call counter + argument recorder), and that neither is batched across
steps (§5).
*Multi-process:* every rank must build the same stream label and consume it at the same rate, or one
rank draws and broadcasts. Per-rank independent generators silently destroy the paired design.
*Caveat:* numpy's bounded-integer path uses rejection sampling, so the number of underlying draws can
in principle depend on `B` and `corpus.n`. Probed at bounds 39,639 / 40,000 / 70,000 / 200,000 with
B = 64: no rejection occurred and the state matched — the dependence is latent, not observed. Do not
rely on that.

**I2 — The run tag is part of the numerics, and `e3_launch.py` is a trap.** `OV3_TAG` seeds BOTH the
batch stream and the fixed validation set (`stream(f"{tag}/val")`). Changing it changes the data.

**Only `e3_launch.py:7-15` (the `ARMS` literal) is in scope.** The rest of that file is a
**different, SEQUENTIAL launcher**: `:19` sets `BATCH, SEG = 32, 48000` and `:20` sets
`OV3_TAG = "OV3_es"` into the same globals, and `:42-55` calls `time_ov3` / `train_ov3`. Exec'ing it
therefore silently changes the tag (hence the entire batch **and** validation sequence, violating I2)
and the batch size (violating I17). The executed launcher inlines the ARMS dict literally instead
(`build_train_notebook.py:434-442`) and does not exec `e3_launch.py` at all.

*Test [any]:* before `train_ov3_fast` is called, assert explicitly

```
assert tag == "OV3_fast"            # or the chosen tag, stated in the run log
assert BATCH == 64
assert SEG == 48000 and corpus.seg_hi == 48000
```

and assert the first 100 `(idx, starts)` pairs for the chosen tag match the **I0** fixture.
*Decision required before launch:* keep `"OV3_fast"` (same batch sequence, longer) or change it.

**I3 — One init per class, shared across arms.** `base = LISAS(CFG)`;
`base_LISASD = copy_shared(base, LISASD(CFG))`; every arm's parameters are that one `state_dict`
repeated along the arm axis (`e2b_fast.py:510-512`, `:205`).
The initialisation **distribution** is `kaiming_uniform_(weight, a=sqrt(5))` + `bias ~ U(±1/sqrt(fan_in))`
in `nn.Sequential` construction order, 18 tensors per model, and `LISASD` consumes an **extra
14,688-draw** `nn.Linear(101,144)` that `copy_shared` then zeroes and overwrites — see **§1.1**, which
also fixes the boot-to-`_make_models` construction order and the (A)/(B) choice.
*Test [any]:* before step 0, assert every pair of same-class arms has bit-identical parameters; assert
`max|LISAS(x, eps=0) − LISASD(x, eps=0)| < 1e-6` (`smoke_ov3_local.py:67`); assert the 4 decoder-noise
columns of `dec.net.0.weight` are exactly 0; assert the per-tensor `state_dict` hashes match the
**I0** `init` fixture; and assert the CPU generator's draw count between the start of `_make_models`
and its return equals `87,777 + 88,353 + 14,688 = 190,818`.
*Multi-rank:* construct on CPU under `torch.manual_seed(0)` then `.to(device)`, or broadcast from rank
0 and hash-verify.

**I4 — Per-arm parameter and gradient independence.** Losses are summed and backwarded once
(`e2b_fast.py:470`); arms share no parameter.
*Test [det]:* multiply one arm's loss by 0 for one step; assert every OTHER arm's parameter delta is
bit-identical to the unmodified run. **This test requires `DETERMINISTIC = True`** — under the run's
own flags the layer-1 gather's `scatter_add` backward is non-deterministic and the assertion cannot
hold; see the flag note above I1.

**I5 — Per-arm gradient clipping at 1e-3.** `coef = clamp(clip/(‖g‖₂ + 1e-6), max=1.0)`, the norm
taken over ALL of that arm's parameters jointly, applied AFTER backward and BEFORE `opt.step()`
(`e2b_fast.py:386-395`). A global norm across the arm axis would couple the arms.
*Test [det]:* compare `ArmStack.clip_` against `torch.nn.utils.clip_grad_norm_` on the exported module
with the same gradients; require agreement to 1e-5 relative in fp32. The norm runs over all **18**
parameter tensors of that arm (§2.6).
*Under DDP:* gradients must be all-reduced BEFORE the clip is computed, or the coefficient differs per
rank.

**I6 — One Adam and one MultiStepLR per arm, identical hyperparameters, with these STEP SEMANTICS.**
lr 1e-3, betas/eps/wd at torch defaults, milestones `int(f·steps)`, gamma 0.5, one optimiser and one
scheduler per stack (`e2b_fast.py:519-525`), stepped once per **training step**, **optimiser before
scheduler**: `o.step(); sc.step()` (`e2b_fast.py:652-653`). One Adam over stacked tensors IS one Adam
per arm because Adam is elementwise. **`fused=True` is a kernel choice, not a math choice**, but fused
vs foreach differ in fp32 reduction order (~1e-7 relative) — flag any switch.

**The milestone integers alone do NOT pin the schedule.** `MultiStepLR.__init__` calls `step()` once,
leaving `last_epoch = 0`, and the loop steps the scheduler after the optimiser. Therefore:

> **Milestone `M` ⇒ the halved LR is in force from 0-based training step `M`, inclusive.**
> Step `M` is the FIRST step trained at the halved rate.
> **`hist["lr"]` records the POST-`sched.step()` value** (`e2b_fast.py:660`, read after `:653`),
> i.e. the LR for step `t+1`, not the LR used at step `t` — an off-by-one at every milestone
> boundary, which is what `audit/dynamics_ov3.py:151`'s `lr[st == M][0]` reads.

Without this, scheduler-before-optimiser and per-*epoch* stepping both reproduce the published
milestone integers while shifting the actual schedule.

*Test [any]:* assert every arm's scheduler reports the same `get_last_lr()` at every step; assert the
realised milestone list equals `[20990, 41980, 52475, 62970, 73465, 83960]` at steps = 104,950; and —
the operative test — for each of the six milestones `M`, read `opt.param_groups[0]["lr"]`
**immediately BEFORE `opt.step()`** at `t ∈ {M−1, M}` and assert it equals
`lr0 · 0.5^|{m ∈ milestones : m <= t}|`. That gives `lr0·0.5^(i)` at `t = M_i − 1` and
`lr0·0.5^(i+1)` at `t = M_i`.

**I7 — Noise draw order and shapes.** Per arm per step, in this order: `eps_enc
randn(A, S, 8, L)`; then, **only for LISASD**, `eps_dec randn(A, S, L·R, 4)`; then the anchor jitter
`randn(A, S, N)·0.5` drawn **inside decoder layer 1**, i.e. AFTER the encoder forward. det arms draw
jitter only, at S = B. Across arms the order is stack order (§2.6), which for these seven arms equals
dict order. **Arms share BATCHES, not NOISE** — each arm takes its own slice of the shared global CUDA
generator. Do not "fix" this by sharing `eps`.
*Test [any]:* monkeypatch `torch.randn`, run one step, assert the recorded
`(order, shape, dtype)` list equals (dtype `torch.float32` throughout, I11):

```
det              : (1, 64, 48000)
es_marg          : (1,128,8,12000) ; (1,128,48000)
es_marg_l0.1     : (1,128,8,12000) ; (1,128,48000)
es_split_l0.1    : (1,128,8,12000) ; (1,128,48000)
es_erb_l0.1      : (1,128,8,12000) ; (1,128,48000)
es_dec_l0.1      : (1,128,8,12000) ; (1,128,48000,4) ; (1,128,48000)
es_dec_erb_l0.1  : (1,128,8,12000) ; (1,128,48000,4) ; (1,128,48000)
```

*Derived total:* `3,072,000 + 4·18,432,000 + 2·43,008,000 = 162,816,000` randn elements per step;
over 104,950 steps, **1.709e13 draws** from the global per-device Philox generator (seeded at boot by
`seed_everything`'s `torch.manual_seed(0)` / `torch.cuda.manual_seed_all(0)` and **never re-seeded**,
§1.1).

**THE NOISE CONTRACT IS ORDER + COUNT + SHAPE + DTYPE. IT IS NOT VALUES.** ATen's CUDA distribution
kernels derive their launch grid from the device's `multiProcessorCount` (`calc_execution_policy`), and
both the Philox counter offset consumed per call and the element → substream mapping follow from that
grid. **The A100 the reference run used and the sm_120 RTX PRO 6000 the 50-epoch run targets have
different SM counts, so the same seed and the same draw sequence produce DIFFERENT `eps` and `jitter`
values on the target card.** Value-level equality of `eps`/`jitter` is claimed **only within one
(GPU model, torch build, CUDA build)**. Consequences:

- any value-level `eps` fixture recorded on one card **will fail** on the other; the person holding
  that failure must not "fix" it by loosening an invariant that is actually sound;
- §8 option (b) ("reproduce the interleaved global stream by pre-drawing per-arm offsets") is
  **struck** — it chases something unattainable across machines. Option (a), a per-arm
  `torch.Generator` with a fixed key, is the **only** offered path and the only portable one;
- the pre-flight artefact (§9.28) must record **torch, CUDA, cuDNN, numpy and scipy versions and the
  device name**, and a one-step A/B recording how far `eps` values move between the reference machine
  and the target, so the size of the effect is on the record rather than a surprise.

**Dispatch discipline.** While the global generator is shared, **the arm loop is dispatched from ONE
Python thread in stack order.** Multiple CUDA **streams** from one host thread are permitted
(`FAST["STREAMS"]`, `e2b_fast.py:455-466,463-476`); multiple host **threads** dispatching arms are
**not** — the Philox offset is advanced on the HOST at dispatch time, so multi-threaded dispatch makes
each arm's slice of the generator depend on thread scheduling: not reproducible run to run, and the
"identical RNG consumption" pairing across arms is gone. **If §8 option (a) is adopted, this
restriction lifts** — a further argument for (a).
*Test [any]:* record the per-arm draw order over 20 steps and assert it is constant.

**I8 — The two ES draws must not share noise or jitter.** Both are drawn at S = 2B and split
`y1 = yh[:, :B]`, `y2 = yh[:, B:]` (`e2b_fast.py:311,354-357`). Sharing the jitter between halves
would shrink `spread` and bias the estimator.
*Test [any]:* assert the jitter tensor's first and second B-blocks differ; assert `E[spread]` over a
few hundred steps matches the sequential trainer's to within Monte-Carlo error.

**I9 — Conditioning is shared between the two draws, in ONE forward at S = 2B, with ONE STFT call per
scale covering both draws.** `x.repeat(2,1)`; the draws differ only through `eps` and jitter
(`e2b_fast.py:354`); the spectral terms take one `torch.stft` of `A·2B` rows per scale
(`e2b_fast.py:366-367`). **This is an invariant, not a description.** Halving the ~8.6–8.9 GB per-arm
activation peak by running draw 1 and draw 2 as two separate B-sequence forwards is the most obvious
memory win on a 96 GB card, and it changes three things: the RNG stream (two `randn(A,B,8,L)` calls
are **not** one `randn(A,2B,8,L)` call — different Philox offsets and a different element mapping),
the I9 conditioning identity (currently free via `x.repeat(2,1)`, then needing an explicit assertion),
and the STFT batching.
**If the forward MUST be split for memory:** `e, d` must still be drawn **once** at the 2B shape via
`sample_eps(2*B, L)` and then sliced, so the RNG stream is preserved; and the split is
**CHANGES-TRAJECTORY** (kernel reduction order, STFT batching) and must be recorded.
*Test [any]:* assert `torch.equal(x_in[:B], x_in[B:])` inside forward; assert the `sample_eps` call
count is exactly **1 per ES arm per step** and that it yields 1 `randn` for LISAS and 2 for LISASD
(I7's table); assert the per-scale `torch.stft` call count is exactly 1 per ES arm per scale.

**I10 — Stack order.** `build_stacks` groups by `(cls, is_det)` in ARMS insertion order and chunks by
`GROUP_MAX` (`e2b_fast.py:419-433`). Reordering ARMS, or changing `GROUP_MAX`, changes which slice of
the global RNG stream each arm receives.
*Test [any]:* assert `[s.names for s in stacks]` equals the **I0** `stacks` fixture (whose expected
contents are the list printed in §2.6).

**I11 — The autocast boundary.** bf16 autocast covers the encoder convs and the decoder MLP **only**;
`forward` returns `.float()` before anything else touches it (`e2b_fast.py:243,274`). `lowpass`
(rfft/irfft), every `torch.stft`, the ERB matmul and every distance run in fp32. Parameters,
gradients and Adam state are fp32. **There is no GradScaler anywhere** (bf16 needs none; grep
confirms).
**The NOISE and JITTER dtypes are part of this invariant.** `eps_enc` (`e2b_fast.py:329`), `eps_dec`
(`:330`) and the anchor `jitter` (`:311`, and `:259`, `:303`) are drawn in **fp32** and stay fp32 until
`_layer1`'s explicit `.to(dt)` (`:322-323`) and `coord.to(dt)` (`:315`). **Drawing any of them in
bf16 or fp16 CHANGES MATH:**

- bf16 `randn` has an 8-bit mantissa, i.e. a visibly **quantised noise law** — that changes the
  conditional distribution the energy score is scoring, which is the entire experiment. It looks free
  (`eps_dec` is 98 MB and `eps_enc` 49 MB per arm per step, and both are immediately cast down);
- worse for the jitter: `torch.floor(q + jit)` at `:312` with `q` reaching 11,999.75, where bf16's
  ulp is **64**. A bf16 jitter destroys anchor selection outright.

**The anchor arithmetic is pinned at fp32 in BOTH directions.** `q`, `floor(q)` and `c_p`
(`_base_index` / `_phase_coord`, `e2b_fast.py:115-130`) and `floor(q + jit)` are fp32. **Raising them
to fp64 also changes which cell is selected** for a nonzero fraction of draws and is therefore
CHANGES MATH, not an improvement.

*Test [any]:* assert `yh.dtype is torch.float32` at the loss boundary; put a dtype-asserting hook on
the STFT / ERB / lowpass call sites — that catches a rewrite that widens the autocast region; put the
**same class of hook on the three `randn` call sites** (`:259`, `:303`, `:311`, `:329-330`) asserting
`torch.float32`, and on `_base_index`/`_phase_coord`'s outputs.
Widening autocast over the losses **CHANGES MATH**.

**I12 — TargetFeats once per step, shared.** (`e2b_fast.py:459,406-416`.) The y-side of every distance
is bit-identical across arms.
*Test [any]:* assert `TargetFeats` is constructed exactly once per step (call counter); assert
`mag_norm[s]` equals `torch.norm(Y,'fro')` from `MultiScaleSTFTLoss`.
Recomputing it per arm is numerically fine, but the need-flags are the OR across all stacks and the
1e-7 / 1e-8 floors and the **GLOBAL** `mag_norm` must be kept.

**I13 — Validation is fixed and jitter-free.** 8 batches from `stream(f"{tag}/val")` drawn once before
step 0 and never redrawn; eps from a generator seeded 1234 created fresh per call; `perturb=False`;
`no_grad` (`e2b_fast.py:528-539,591`).
*Test [any]:* run val twice on the same weights and assert bit-identical results; assert the
validation batch tensors are identical across val calls; assert `n_val_batches == 8` and the
validation batch size `== 64`, **which are part of the `det` arm's objective** (§4.4) and not free
parameters. This test is **flag-independent** — specifically because `perturb=False` takes the
gather-free branch (`e2b_fast.py:316-321`, `expand`'s backward is a sum-reduction) and the whole call
is `no_grad`, so no atomics run. That is why I13 is sound under the run's own flags while I4 is not.

**I14 — The probe is deterministic.** `probe_metrics → reconstruct(..., seed=0)` builds a
`torch.Generator(seed=0)`, so probe SNR/deficit depend only on the weights
(`c1_model.py:66-78,185-192`).
*Test [any]:* `smoke_ov3_local.py:119` — two reconstructions at the same seed differ by exactly 0.0.

**I15 — The per-arm loss recipe.** The weight vectors `(w_lm, w_erb, w_l2)` and the `split` mask
(`e2b_fast.py:216-223`) must keep reproducing `arm_loss3` for all seven kinds, with λ = 1e-2 for `det`
and `es_marg` and 1e-1 for the other five.
*Test [any]:* `gate_subpixel.py` check 2 — currently passes with worst |diff| **1.19e-07** (es_split_l0.1;
all six others exactly 0.00e+00) against a 1e-5 tolerance. **EXTEND IT to also run at GROUP_MAX = 1**,
the production setting, which it does not cover today (the gate sets `GROUP_MAX = None`,
`gate_subpixel.py:40`).

**I16 — Sub-pixel exactness is fp32-only and ~4e-7 relative.** Do not toggle `SUBPIXEL` mid-run; do
not compare it across AMP settings.
*Test [any]:* keep `gate_subpixel.py` as a pre-flight (it is fp32, AMP off, compile off, CPU, so no
atomics run and the flag is irrelevant). Add a bf16 row as a **recorded tolerance**, not a pass/fail —
and see §6.3's note that the CPU-bf16 proxy's *relative* figure is draw-dependent and must be
re-measured rather than quoted.

**I17 — Segment length and batch.** batch 64 × 1 s at 48 kHz (SEG = 48000, L = 12000, N = 48000)
(`e8_launch_fast.py:12-13`). Changing either changes the batch stream itself, the gradient-noise scale
and the audio-seconds per epoch, so 50 epochs would no longer mean the same step count.
*Test [any]:* assert `corpus.seg_hi == 48000`, `BATCH == 64`, and that the printed `steps_per_epoch`
equals `corpus.hours·56.25`. See also the hard pre-flight corpus gate in §8.

**I18 — The corpus itself.** Hub `sanchit-gandhi/vctk` mic1, the role split, the 1.0 s drop, peak
0.95, the `(speaker_id, file)` sort, the float64 `resample_poly(x,1,4)`, and the R-multiple truncation
(`c1_model.py:84-99`; `cell1_corpus.py`; `build_notebook.py:1034-1036`).
*Test [any]:* hash `corpus.lens`, `corpus.off` and a checksum of `corpus.X`/`corpus.Y` against the
**I0** `corpus` fixture. **Decimating each 1 s slice independently instead of the whole utterance at
build time is NOT the same signal** — it adds a fresh zero-padded edge transient at both ends of every
slice. **CHANGES MATH.**

**I19 — Validate the kind string; do not fall through.** `ArmStack` **silently degrades** an unknown
kind (and `es_slice`) to the waveform-only `es_wave` objective — `w_lm = w_erb = w_l2 = 0`,
`split = False` (`e2b_fast.py:216-220`) — whereas `arm_loss3` raises `ValueError`
(`e1_model.py:254-255`). A rewrite must validate against the supported set and **raise**.

**The supported set is exactly these ten** (`e1_model.py:218-219,227-255`):

```
det   det_split   es_wave   es_marg   es_split_marg
es_ged   es_erb   es_marg_erb   es_split_marg_erb   es_slice
```

`ArmStack` can express **nine** of them. **`es_slice` is the exception**: `arm_loss3` delegates it to
`arm_loss` (`e1_model.py:227-228`), whose `d_sliced` resamples 64 random directions per call
(§3.2) — there is no `w_*` weight for it. A rewrite must **reject `es_slice` with an exception**, not
silently degrade it to `es_wave`. None of the seven arms in this run uses it.
*Test [any]:* feed each of the ten strings plus one garbage string to the stack builder; assert the
nine construct, and that `es_slice` and the garbage string each raise.

**I20 — `perturb=True` in training, `perturb=False` in validation and `reconstruct()`.** Including for
the det arm (`e2b_fast.py:469,338`).

**I21 — Module-level caches are device-keyed.** `_WIN`, `_ERB`, `_IDX_CACHE`, `_PHASE_CACHE` are keyed
partly by `str(device)`. A multi-process rewrite must keep them per-device; a single-process
multi-device rewrite must not let one device's cached window or ERB bank be used on another.

**I22 — Gradient checkpointing, if used at all, may ONLY wrap decoder layers 2–5.** The author's
standing note says `preserve_rng_state=False` is "safe here only because the MLP block draws no
randomness, jitter and noise being sampled outside it" (`e2b_fast.py:43-45`). **That statement is
scoped to `_mlp_eager` alone.** The anchor jitter is drawn **inside decoder layer 1**, at
`e2b_fast.py:311` (`jit = torch.randn(A, S, N, device=dev) * 0.5`), inside `ArmStack._layer1` — and
§7.5's own memory accounting attributes the four saved ReLU outputs to decoder layers **1**–4. A
throughput engineer raising occupancy will therefore wrap `_layer1 + _mlp`, or the whole `forward`,
in `checkpoint(..., use_reentrant=False, preserve_rng_state=False)` — exactly as that sentence appears
to bless. **Recompute then draws a DIFFERENT jitter, so the backward is taken through a different
computational graph than the forward: the gradients are wrong, no exception is raised, and the loss
still decreases.** Nothing else in §6 catches it.

> **The ONLY permitted checkpoint boundary is `_mlp_eager` over `Ws[1:]` / `bs[1:]` (decoder layers
> 2–5)** — strictly downstream of the jitter draw at `e2b_fast.py:311` and of the `F.relu(X)` at
> `e2b_fast.py:325`. Any wider boundary — any function that can reach `e2b_fast.py:259`, `:303`,
> `:311`, or `sample_eps` at `:327-331` — requires `preserve_rng_state=True` **and** a check that the
> CUDA generator offset after the step is unchanged.

*Test [det]:* run one step with and without checkpointing from the same seed under
`DETERMINISTIC = True`; assert per-parameter gradients agree to **1e-5 relative**; and assert the
jitter tensor captured on the forward is elementwise equal to the one used on recompute.

**I23 — Every index is int64, on host and on device.** §5 computes `sum(lens) = 6,445,440,000`
samples, which exceeds int32 (2,147,483,647) by **3×**. The host path is safe because `off` is a numpy
int64 cumsum and `_ar_hi = np.arange(seg_hi)` is int64 (`c1_model.py:96-97`, `e2b_fast.py:444-449`).
A GPU-resident-corpus rewrite that halves index-tensor traffic by keeping `off`/`hi0` in **int32**
wraps to **negative** values — and torch advanced indexing **accepts negative indices by wrapping from
the end**, so the batch is silently drawn from the wrong utterances, with no exception, no NaN, and no
signature in the loss curves. **This is the cheapest way to destroy the run while appearing to satisfy
every other invariant**, and §7.5's "GPU-resident fp32 corpus ⇒ bit-for-bit identical batches, NO MATH
CHANGE" row is true **only** under this invariant.
*Test [any]:* assert `off.dtype == starts.dtype == hi0.dtype == int64` and that both aranges are
int64; for the first 1,000 steps assert
`hi0.min() >= 0 and (hi0.max() + seg_hi) <= Y.numel()` and the low-rate analogue. I1's fixture
comparison of the gathered `x`/`y` tensors is the backstop.

**I24 — The noise must CHANGE BETWEEN STEPS.** Every noise test above is *within* one step: I7 checks
`(order, shape)` for one step; I8 checks that the two B-blocks differ inside one step. **Nothing
asserts the draws differ across steps**, and several standard throughput moves freeze them: hand-rolled
CUDA-graph capture or `torch.cuda.make_graphed_callables` without registering the generator state;
`torch.compile(mode='max-autotune')` cudagraph trees (`e2b_fast.py:56-62` discusses this only as a
*stale-gradient* hazard); or a preallocated `eps` buffer filled once to avoid re-allocating 98 MB of
`eps_dec` per arm per step. With frozen `eps` and jitter the energy score degenerates — every step
scores the same two fixed draws — and `spread`, the run's headline diagnostic, **shrinks**, which reads
exactly like the intended convergence story in
`notes/2026-09-16-score-geometry-decoder-noise-readouts.md` rather than like a bug.
*Test [run] — a PERMANENT RUNTIME ASSERTION for the first 100 steps of the real run, not a pre-flight
gate*, because the freeze can be introduced by a compile mode that only engages after warm-up: record
a cheap hash of `eps_enc`, `eps_dec` and `jitter` at steps `k` and `k+1` and assert they differ; and
log `torch.cuda.get_rng_state()`'s offset delta per step, asserting it is a **fixed nonzero constant**
across steps.

**I25 — The probe decode chunk is a multiple of R.** `reconstruct(..., chunk=1<<15)`
(`c1_model.py:66,76`). Every chunk boundary being a multiple of 4 is what keeps `_decode_subpixel` on
the gather-free branch (`e1_model.py:115`). A non-multiple silently moves the probe onto the
jittered/gather branch. See §4.5.2.
*Test [any]:* `assert chunk % CFG.upsample == 0`; assert the gather-free branch is taken on every
probe chunk (branch counter).

**I26 — `lowpass` stays the exact brick-wall rfft projection.** See §6.2.

**I27 — No gradient accumulation or micro-batching without the §3.3 recipe.** See §6.2.

### 6.2 Knobs known to CHANGE MATH

| knob | effect | source |
|---|---|---|
| **`FAST["JITTER_PER_CELL"] = True`** | one anchor draw per INPUT cell (`randn(A,S,L)`) instead of per output sample (`randn(A,S,N)`); the R outputs of a cell then share one anchor and differ only by `c_p − 2m`. Also changes per-step RNG consumption by a factor of R. Must stay **False** inside the seven-arm comparison; run it as an eighth paired arm. | `e2b_fast.py:32-33,302-309`; `e1_model.py:67,102-106` |
| **bf16 autocast (`FAST["AMP"]`)** | not a formula change, but it changes every encoder/decoder value by ~4e-3 relative per op, and it is what makes SUBPIXEL and the gather path differ by ~1e-2 rather than ~4e-7. **The 16k reference run was bf16; the 50-epoch run must stay bf16 to be comparable.** Turning it on or off mid-comparison, or for some arms and not others, invalidates it. | `e2b_fast.py:111-112,243` |
| **TF32 + `set_float32_matmul_precision("high")`** | reduces fp32 matmul inputs to a 10-bit mantissa. The only significant fp32 matmul in the loss path is the **ERB contraction** `matmul(W(32,F), \|S\|²)`, so TF32 perturbs the ERB band energies (~1e-3 relative) of `es_erb_l0.1`, `es_dec_erb_l0.1` and the ERB target features while leaving the other five arms untouched. **TF32 is an ASYMMETRIC knob across the seven arms.** | `e2b_fast.py:102-105,378-379,416` |
| **`GROUP_MAX ≠ 1`** | changes the decoder MLP kernel from 2-D `F.linear` to `baddbmm` (different reduction order) AND changes which slice of the RNG stream each arm receives (arms in one stack share one `sample_eps` call, a contiguous draw at leading dim A). Not a formula change, but not reproducible against the GROUP_MAX = 1 run. | `e2b_fast.py:133-149,327-331` |
| **`CUDA_GRAPHS = True`** | gradients become static buffers never set to `None` between replays (`:494-504` vs `:644-645`); capture warm-up runs three extra fwd/bwd passes that consume RNG. **EXPERIMENTAL, untested on the day it was written.** Leave False. | `e2b_fast.py:22-23,480-504` |
| **changing the batch size** | changes the batch stream itself and the per-step gradient noise; batch 64 must be kept if the 50-epoch run is to be read against OV3_fast step-for-step. | I17 |
| **storing the corpus as int16 / fp16** | changes the data. A GPU-resident **fp32** corpus is fine (I18 / §7), **provided the indices stay int64** (I23). | **Derived; no repo source proposes it.** The repo is fp32 end-to-end: `overnight/cell2_trainer.py:33-34` (`torch.from_numpy(np.concatenate(ys)).to(DEVICE)`) on the GPU side and `c1_model.py:87,94-95` on the host side. `grep -rn 'int16\|float16\|fp16\|half()'` over `overnight/cell2_trainer.py`, `overnight2/c1_model.py` and `overnight3/*.py` returns only `e2b_fast.py:112` (the bf16 autocast). **This row is a prohibition this document ADDS.** |
| **gradient accumulation / micro-batching** (either the batch axis B **or** the draw axis 2B) | **CHANGES MATH for `det` unconditionally**, CHANGES TRAJECTORY for the six ES arms. It breaks four things at once, silently: (i) `det`'s spectral convergence is batch-global (`tf.mag_norm[s] = S_.pow(2).sum().sqrt()`, a single scalar over (B,F,T), numerator `.sum((1,2,3)).sqrt()`) — summing per-microbatch `sc` is a different number; (ii) `sample_eps(2*B, L)` becomes two half-size calls, a different RNG stream; (iii) `spread = dwf(y1, y2)` needs both draws live simultaneously, so a naive split mis-pairs or drops it; (iv) `clip_` must run **once** over the accumulated gradient, not per microbatch, or the per-arm coefficient is wrong. **If used at all:** `eps` is drawn once at the full `(A, 2B, 8, L)` / `(A, 2B, L·R, 4)` shape and **sliced**; `clip_` runs once after the last microbatch; and `det`'s `sc` is recomposed by the ordered recipe in §3.3, not by averaging. *Test:* one accumulated step vs one whole-batch step from the same init and the same `eps` must agree on every arm's post-step parameters to **1e-5 relative**. | `e2b_fast.py:347,353,357,386-395,414`; §3.3 |
| **reimplementing `lowpass`** | `lowpass` is a **brick-wall rfft projection with an in-place zeroing of an autograd tensor** (`c1_model.py:126-131`) and runs three times per step for arm 4 on `(A,B,48000)` tensors — an obvious target for a "faster equivalent". **It must remain exactly** `Y = rfft(y); Y[..., k_cut+1:] = 0; irfft(Y, n=y.shape[-1])`, including `n=y.shape[-1]` on the inverse. The **only** permitted rewrite is the functional `irfft(Y[..., :k_cut+1], n=T)`. A time-domain FIR or a `resample_poly` round trip is **not** equivalent and its adjoint is **not** the same projection, so arm 4's (`es_split_l0.1`) waveform gradient changes silently. Note `lowpass` is a projection, so its adjoint must equal itself. | `c1_model.py:126-131`; I26 |
| **any GPU / torch reimplementation of `decimate`** | different filter, different padding, or fp32 arithmetic all move the input signal and therefore the whole task. If a GPU decimator is used for speed it must be validated against scipy to ~1e-7 relative in float32 on a held-out batch before it is trusted. | §5 stage 3 |
| **per-shard `sc` for the det arm** | see §3.3 — the det arm's spectral convergence is batch-global. | `e2b_fast.py:347,414` |

**CHANGES-TRAJECTORY-1 (not a formula change, but it moves every number).** `e8_launch_fast.py:26`
runs `time_ov3_fast(corpus, ARMS, BATCH)` **before** `train_ov3_fast`. That call builds its own LISAS
and LISASD (advancing the CPU default generator) and runs `n + 3 = 23` steps of noise draws
(advancing the CUDA default generator). Dropping it, adding it, or re-parameterising it changes the
initial weights and the entire noise stream of the real run. A rewrite must either keep it verbatim
or re-seed immediately before `_make_models` **and say so in the run log**.

**NOT a math change: determinism OFF.** `use_deterministic_algorithms(False)` +
`cudnn.benchmark = True` changes only the float reduction ORDER inside the `scatter_add` behind
`torch.gather`'s backward, and cuDNN algorithm selection. The measured A/B on the sequential trainer,
7 arms, batch 32 × 1 s, A100-80GB: flag ON = **2261 ms/step**, flag OFF = **587 ms/step**, a **3.9×**
difference; the cause is ATen replacing the fused `fastAtomicAdd` with
`_scatter_via_index_put → index_put_with_sort_kernel`, materialising ~147M int64 keys (~1.2 GB) per
arm-draw and radix-sorting them (`e2b_fast.py:75-89`). Seeds, data order, noise draws, jitter draws,
init and arm pairing are untouched. The price is run-to-run bitwise reproducibility of the gradients,
which is **not available and must not be claimed**. *This is an assertion by the original author with
no test in the repo* — §9 says how to test it.

**Where the training kernel flags actually come from.** Not `e7_fastsettings.py` — that file is
**never inlined into the notebook** (`grep e7 build_train_notebook.py` returns nothing) and is the
batch-size sweep, not production. On the executed path, determinism-off, `cudnn.deterministic=False`,
`cudnn.benchmark=True`, `allow_tf32` on both backends and `torch.set_float32_matmul_precision("high")`
are **all set at IMPORT time of `e2b_fast.py`** (`:94-105`), driven by `FAST["DETERMINISTIC"]=False`
and `FAST["TF32"]=_CUDA` (`:63-65`). `FAST` is populated from `globals()` at `:66-68`, **before** those
lines run, so **any override must be set in a cell ABOVE the `e2b_fast.py` exec** — setting it
afterwards changes the dict but not the flags. A rewrite that correctly drops `e7_fastsettings.py`
loses nothing.

**Flag order is load-bearing.** The determinism block must run BEFORE any `torch.compile`:
`torch.__init__` mirrors `use_deterministic_algorithms()` into
`torch._inductor.config.deterministic`, which permanently disables Inductor autotuning for anything
compiled while it was on (`e2b_fast.py:70-73,94-101`; the compile call is at `:167`).

### 6.3 Tolerances — what "equivalent" means

| quantity | tolerance | basis |
|---|---|---|
| batch tensors `x`, `y` across arms and across a rewrite | **bit-for-bit** | I1; `batch_pinned` is `HostCorpus.batch` + `pin_memory` |
| RNG draw order and shapes | **exact** (order, count, shape) | I7 |
| initial parameters, all same-class arms | **bit-for-bit** | I3 |
| LISAS vs LISASD at eps = 0, step 0 | **< 1e-6 absolute** | `smoke_ov3_local.py:67` |
| sub-pixel vs gather path, **fp32, AMP off, compile off** | **< 1e-5 relative** (gate); **measured 1.96e-07 … 4.24e-07** | `gate_subpixel.py:51-73` |
| sub-pixel vs gather path, **bf16 autocast** | **not interchangeable**; record, do not gate. The CPU-bf16 proxy measures rel **~4e-03 … 2e-02**, and **the exact value is draw-dependent and does not reproduce across RNG placements** — see the note below. A pessimistic bound in any case, since CUDA tensor cores accumulate in fp32. | `overnight3/gate_subpixel.py` check 1 with the module-global `_autocast` rebound to `torch.autocast("cpu", torch.bfloat16)`; procedure below |
| stacked losses vs `arm_loss3`, all seven arms, fp32 | **< 1e-5 absolute** (gate); **measured worst 1.19e-07** (es_split_l0.1), six arms exactly 0 | `gate_subpixel.py:99-106` |
| `ArmStack.clip_` vs `clip_grad_norm_`, fp32 | **< 1e-5 relative** (reduction order only) | I5 |
| determinism OFF vs ON, gradients, fp32 | asserted **~1e-7 relative**; **UNTESTED** — measure and report the divergence, do not assert the bound | `e2b_fast.py:86-89` |
| fused vs foreach Adam | ~1e-7 relative (reduction order) | I6 |
| a batch sharded across ranks, per-shard means of equal-size shards | ~1e-7 relative in fp32 (reduction reorder) — the same tolerance the repo already accepts for non-deterministic `scatter_add` | `e2b_fast.py:88-89` |
| **a GPU-side gate at FULL dims** | reduction lengths are 24× longer than the CPU gate's (L = 12000 vs 512). **Relax the fp32 tolerance to 1e-5 relative and RECORD the number; do not tighten the bound.** | §9 |

**The CPU-bf16 proxy row is NOT a stable number, and must not be quoted as one.** Procedure for the
measurement above: `gate_subpixel.py`'s check-1 harness (SMOKE preset, `DEVICE = cpu`, B = 2,
L = 512, `TF32` off, `AMP` left False, `COMPILE` off, `GROUP_MAX = None`), with the **module-global
`_autocast` rebound** to `lambda: torch.autocast(device_type="cpu", dtype=torch.bfloat16)` after
`e2b_fast.py` is exec'd — i.e. `FAST["AMP"]` itself is *not* flipped, because it gates on `_CUDA`.
Re-run 2026-09-19 in this repo's venv, `torch.manual_seed(0)` immediately before the harness:

| case | max\|diff\| | relative |
|---|---|---|
| LISAS, `perturb=False` | 2.4414e-04 | 4.016e-03 |
| LISAS, `perturb=True`, identical jitter | 2.4414e-04 | 4.016e-03 |
| LISASD, `perturb=False` | 2.1362e-04 | 1.833e-02 |
| LISASD, `perturb=True`, identical jitter | 2.1362e-04 | 1.833e-02 |

An earlier draft of this document recorded **7.0e-03 … 3.46e-02** and an independent reviewer
recorded **6.41e-03 … 1.111e-02** under a different seed placement. All three sets are the same order
of magnitude and none reproduces the others: the ratio's denominator is
`outs[False].abs().max()`, a single extreme order statistic of a random tensor, so the *relative*
number moves with the draw while `max|diff|` (~2.4e-04, one bf16 ulp at these magnitudes) is stable.
**Read this row as "the bf16 deviation is ~1e-2 relative, three to four orders of magnitude above the
fp32 sub-pixel deviation of ~4e-07". Do not treat any particular figure as reproducible, and
re-measure on the target card rather than quoting it** (§9.12).

**The error hierarchy, largest first:** bf16 autocast (~1e-3 relative, and it is the run's baseline,
not a deviation) ≫ TF32 on the ERB matmul (~1e-3 relative, **asymmetric across arms**) ≫ sub-pixel
layer-1 reassociation (~4e-7 fp32) ≈ non-deterministic `scatter_add` (~1e-7 fp32) ≈ fused-Adam
reduction order (~1e-7 fp32). The authors state that bf16 rounding dominates the last three; that
statement is consistent with the measurements above but has never been measured end-to-end on a
trajectory.

---

## 7. Cost model and roofline

All FLOP and byte numbers below are **derived** and recomputed here; the arithmetic is shown. MAC = 2
FLOP. Backward is assumed to be **2× forward** (dgrad + wgrad for every GEMM/conv) — a standard
**ASSUMPTION**, not a measurement.

### 7.1 Work per step

Sequences per step at B = 64: det S = B = 64; each of the six ES arms S = 2B = 128
(`e2b_fast.py:338,353-354`). Total **832 sequences**, **39,936,000 decoder output rows**.

| quantity | value | arithmetic |
|---|---|---|
| encoder MACs / input sample | 10,736 | `9·16·7 + 16·32·3 + 32·64·3 + 64·32·1 = 1008+1536+6144+2048` |
| encoder MACs / sequence | 1.2883e8 | `10,736 · 12,000` |
| dec L1, **gather** MACs / output | 13,968 | `97 · 144` |
| dec L1, **sub-pixel** MACs / output | **3,600** | `(3·32·144)/4 + 144 = 3456 + 144` — **3.88× cheaper** |
| dec L2–5 MACs / output | 62,352 | `3·144·144 + 144·1` |
| dec L2–5 MACs / sequence | 2.9929e9 | `62,352 · 48,000` |
| decoder-noise MACs / output (LISASD) | 576 | `4 · 144` |
| **fwd MACs / sequence** | LISAS 3.2945e9, LISASD 3.3222e9 | sum of the above |
| **fwd GFLOP / sequence** | LISAS 6.589, LISASD 6.644 | ×2 |

### 7.2 FLOP breakdown per step (fwd + bwd = 3× fwd)

| component | TFLOP/step | share |
|---|---|---|
| encoder | 0.643 | 3.90 % |
| decoder layer 1 (sub-pixel) | 0.863 | 5.23 % |
| decoder layers 2–5 | 14.941 | **90.61 %** |
| decoder-noise term (2 LISASD arms) | 0.042 | 0.26 % |
| **network total** | **16.489** | 100 % |
| STFT chain (896 sequences × 28.9 MFLOP fwd) | 0.026 fwd / ~0.078 fwd+bwd | 0.47 % |
| ERB matmul (2 ERB arms + target) | 0.006 | 0.04 % |

Per arm: det **1.265 TFLOP**, each ES LISAS arm **2.530**, each ES LISASD arm **2.551**.
STFT frames/bins at T = 48 000: (1025, 94), (513, 188), (257, 376) at 5·n·log₂n FLOP/frame =
112,640 / 51,200 / 23,040 ⇒ 10.59 + 9.63 + 8.66 = **28.9 MFLOP per sequence-second**.

*This contradicts the module header's "~0.7 TFLOP per arm-step" (`e2b_fast.py:5`).* **The comparison
depends on which layer-1 path is counted, so both are given.** An ES arm at batch 32 is 64 sequences
(2B); MACs per sequence are `enc 1.2883e8 + L1 + L2–5 2.9929e9`:

| layer-1 path | MACs / output | MACs / sequence | fwd GFLOP @ 64 seq | fwd+bwd TFLOP |
|---|---|---|---|---|
| **sub-pixel** (`SUBPIXEL=True`, what the run uses) | 3,600 | 3.2945e9 | **421.7** | **1.265** |
| **gather** (`SUBPIXEL=False`) | 13,968 | 3.7922e9 | **485.4** | **1.456** |

The 421.7 / 1.265 row is **exactly the "det 1.265 TFLOP" figure printed two lines above** — det at
batch 64 and an ES arm at batch 32 are both 64 sequences.

**The header's 0.7 TFLOP should be read against the GATHER count**, because `e2b_fast.py:3-5`
explicitly describes the *pre-sub-pixel sequential* trainer ("sequential arms, ~28 `.item()` syncs per
step, **three gathers**, fp32"). On that basis 0.7 is a forward-only estimate inflated
**485.4/336.6 = 1.44×** (or a fwd+bwd estimate low by 2.1×). Against the sub-pixel count the same
0.7 would be inflated **1.66×**, not 1.45×. Resolvable in one line with
`torch.utils.flop_counter.FlopCounterMode`; §9.4 repeats the same pair of numbers.

### 7.3 HBM traffic (derived, ±20 %)

Per decoder output row, bf16, **unfused ReLUs** (which is what the run gets — see §7.5):

| term | bytes |
|---|---|
| L1 gather rd+wr+idx+coord | 588 |
| four ReLUs rd+wr | 4 × 576 = 2,304 |
| L2, L3, L4 rd+wr | 3 × 576 = 1,728 |
| L5 (rd 288 + wr 4) | 292 |
| **forward subtotal** | **4,912** |
| four ReLU backwards (rd out, rd dY, wr dX) | 4 × 864 = 3,456 |
| L2–4 dgrad + wgrad | 3 × 1,152 = 3,456 |
| L5 backward | 584 |
| L1 scatter_add atomics + dX read | 864 |
| **backward subtotal** | **8,360** |
| **row total** | **13,272 B** |

`39,936,000 · 13,272 = 530 GB` decoder + ~18 GB encoder + ~9 GB `u` + ~22 GB STFT/loss + ~2 GB misc
⇒ **~580 GB per step**.

### 7.4 Arithmetic intensity (bf16; A100 ridge = 312e12 / 2.039e12 = **153 FLOP/byte**)

| stage | FLOP/byte |
|---|---|
| encoder conv1 (9→16, k=7) | 40 |
| encoder conv3 (32→64, k=3) | 64 |
| encoder conv4 (64→32, k=1) | 21 |
| **dec L1 sub-pixel conv at 12 kHz** | **79** |
| dec L1 gather + rank-1 at 48 kHz | 0.49 |
| dec L2–4 fwd, ReLU epilogue **fused** | 72.0 — *exactly the author's figure at `e2b_fast.py:137`* |
| dec L2–4 fwd, **unfused** | 36 |
| dec L2–4 fwd+bwd, unfused | 39.3 |
| dec L5 (144→1) | 0.99 |
| standalone ReLU | ≈ 0 |
| STFT + abs + log + L1 chain | 2.3 |
| **whole step** (16.489 TFLOP / 580 GB) | **28.4 — 5.4× below the ridge** |

Nothing in the step is compute-bound. **The author's "~72 FLOP/byte at bf16" is the forward-only,
fused-epilogue figure**; the comment does not say which regime it means. I verified the arithmetic;
the run does not achieve it, because the epilogues are not fused (§7.5).

### 7.5 Measured vs theoretical

| row | measured | source |
|---|---|---|
| sequential, batch 32 | 589 ms/step, 13.2 GB, 54.3 audio-s/compute-s | `e8_launch_fast.py:3` |
| **stacked GROUP_MAX=1, batch 32** | **467 ms/step, 7.0 GB, 68.5** | `e8_launch_fast.py:4` |
| **stacked GROUP_MAX=1, batch 64** | **925 ms/step, 13.4 GB, 69.2** | `e8_launch_fast.py:5` |
| sequential sweep after e7 flags | 326 / 589 / 1167 ms at batch 16 / 32 / 64 | `e7_fastsettings.py:7` |
| the OV3_fast run itself | 16,000 steps, batch 64, **891 → 966 ms/step** | `notes/2026-09-16-…:47-48` |
| `bias_addmm(3072000×144, 144×144)` | 1.73 ms ⇒ **73.6 TFLOP/s (23.6 % MFU), 1.02 TB/s (50 % of peak)** | `e2b_fast.py:60` |
| `mm(144×3072000, 3072000×144)` wgrad | 1.34 ms ⇒ **95.1 TFLOP/s (30.5 % MFU), 1.32 TB/s (65 %)** | `e2b_fast.py:61` |

**At the measured 925 ms:** MFU = 16.489e12 / 0.925 / 312e12 = **5.71 %**; bandwidth =
580e9 / 0.925 = **628 GB/s = 30.8 % of HBM peak**. The two ATen kernels reach 50–65 % of peak, so
roughly half the step runs in kernels far below the best ones: four unfused ReLUs, the high-rate
gather, its `scatter_add` atomic backward, and the STFT chain.

**Two-point cost law (derived; fits BOTH measured stacked rows exactly):**

```
t(ms) = 9.0 + 22.9367 · (decoder output rows / 1e6)
   batch 32: 9.0 + 22.9367 · 19.968 = 467 ms   (measured 467)
   batch 64: 9.0 + 22.9367 · 39.936 = 925 ms   (measured 925)
```

**22.94 ns per output row; the fixed term is only 9 ms = 1 % of the step.** The step is
**per-element bound, not launch bound**, at batch ≥ 32 — which is exactly why throughput is flat in
batch (68.5 → 69.2 audio-s/compute-s) and why a larger batch buys nothing. Per arm run alone, at
batch 64: ES = `9.0 + 22.9367·6.144` = **149.9 ≈ 150 ms**, det = `9.0 + 22.9367·3.072` =
**79.5 ≈ 79 ms**. The sequential trainer fits 28.95 ns/row with ~11 ms fixed, so the stacked path is
**1.26× better per row**. **The law is affine, so per-arm times do NOT add** — combining two arms on
one GPU costs `9.0 + 22.9367·(rows₁+rows₂)`, one fixed term, not two (see §8's 6-GPU row).

**Memory.** Activations saved for backward on the sub-pixel path are the four ReLU outputs of decoder
layers 1–4, each `S·N·144` bf16 and each aliasing the next Linear's saved input:
`4 · 128 · 48000 · 144 · 2 = 7.078 GB` for one ES arm at batch 64, 3.539 GB for det; plus `u` at
442 MB, the gather index 49 MB, coord 25 MB, `yh` 25 MB, `eps_enc` 49 MB, `eps_dec` 98 MB (LISASD) and
~0.6–1.2 GB of STFT graph ⇒ **~8.6–8.9 GB live for one ES arm**. Because `GROUP_MAX = 1` and backward
runs immediately after each stack's forward (`e2b_fast.py:462-472`), **only one arm's activations are
live at a time** — peak does NOT scale with the number of arms. Predicted ~9.6 GB; **measured
13.4 GB**. Measured peak is **affine** in batch, not proportional: the two measured points (32 → 7.0 GB,
64 → 13.4 GB) fit **0.6 GB + 0.20 GB per unit of batch** exactly. (0.21 GB/unit is `13.4/64`, a
proportionality that ignores the intercept.)

**Gradient checkpointing is deliberately NOT used**, with a standing condition from the author:
"Revisit ONLY if a re-measured roofline shows the step compute-bound AND a larger batch is shown to
raise audio-seconds per compute-second" (`e2b_fast.py:41-48`). If it is ever used, it must be
`use_reentrant=False, preserve_rng_state=False` **and it may wrap ONLY `_mlp_eager` over decoder
layers 2–5** — the author's "the MLP block draws no randomness" is true of `_mlp_eager` and **false of
`_layer1`, which draws the anchor jitter at `e2b_fast.py:311`**. See **I22**, which is the binding
statement; a wider boundary silently computes wrong gradients. It costs one extra forward.

**torch.compile is effectively a no-op for the shapes that matter.** It targets `_mlp_eager` only,
`dynamic=False`, mode `"default"`, `triton.cudagraphs = False` (fused Adam + cudagraph trees ⇒ stale
gradients), recompile limit raised to 32, row axes marked dynamic at call time
(`e2b_fast.py:55-62,153-182`). ATen already beat the best Triton template on both decisive shapes
(1.73 vs 1.81 ms; 1.34 vs **18.14** ms). Failures are swallowed twice — at build time
(`:168-169`) and at first call (`:179-182`, which permanently flips `_MLP["compiled"]` to False). The
header itself suspects Dynamo silently skipped it in the stacked runs. **A rewrite must assert or log
`_MLP["compiled"]` after warm-up**, since a silent fallback changes step time and the decoder GEMM
reduction order without changing any flag.

**Headroom ladder** (derived):

| move | result |
|---|---|
| hit 80 % of HBM peak on the SAME traffic | 580 GB / 1.63 TB/s = **356 ms → 2.6×** |
| fuse the 5-layer decoder MLP into one kernel (a 144-wide row is 288 B and fits in registers) | forward traffic/row 4,912 → ~292 B (17×); forward intensity → **451 FLOP/byte**, i.e. the decoder becomes **compute-bound**; 16.489 TFLOP / (0.4·312 TF/s) ≈ **132 ms → 7×** |
| GPU-resident fp32 corpus (`overnight/cell2_trainer.py:22-49`) | 32.2 GB fp32 fits in 96 GB; consumes the numpy RNG identically ⇒ **bit-for-bit identical batches, NO MATH CHANGE — but ONLY under I23 (all indices int64).** `Y` has >2^31 elements; int32 index tensors wrap negative and torch indexes from the end, silently drawing the wrong utterances. Deletes the host gather, the per-step `pin_memory` and the H2D |
| larger batch | **no** — throughput is flat (68.5 vs 69.2) and batch 64 is contractual (I17) |

---

## 8. What 50 epochs means

**Steps.** `steps_per_epoch = corpus.hours · 56.25` at BATCH = 64, SEG = 48000
(`e8_launch_fast.py:25`, `build_train_notebook.py:476`). The OV3_fast run recorded **2,099
steps/epoch** and 16,000 steps = 7.62 epochs (`notes/analysis/dynamics_OV3_fast.md:3`), implying
`corpus.hours = 2099·64/3600 = 37.316 h`.

| basis | steps for 50 epochs | `int(f·steps)` milestones |
|---|---|---|
| rounded 2,099 steps/epoch | **104,950** | 20990, 41980, 52475, 62970, 73465, 83960 |
| 37.3 h exactly ⇒ 2,098.125 | 104,906 | 20981, 41962, 52453, 62943, 73434, 83924 |

**RESOLVED: the binding step count is `STEPS = 104,950`.** Earlier drafts gave a normative
instruction ("compute `steps = round(50 · steps_per_epoch)`", which yields 104,906 at the reference
corpus) while publishing the milestone integers for 104,950. They cannot both bind. 104,950 is chosen
because it is the number the run was scoped, costed and budgeted against, and because it is the one
whose milestones are published, audited and quoted downstream. **The six binding milestones are
`[20990, 41980, 52475, 62970, 73465, 83960]`** and the whole of this section uses 104,950.

`corpus.hours` remains **DERIVED, NOT CONSTANT** — it is only known to ~37.31–37.33 h from the
printed integers and the exact value is never written to the repo. So it is used as a **gate on the
corpus**, not as the source of `steps`:

**Hard pre-flight gate, before the optimiser is constructed** (`e3_launch.py:33-40` and
`build_train_notebook.py:467-472` contain a live trap: they build `corpus = HostCorpus(train_utts,…)`
and only **then** trim `train_utts` to 200 utterances to free RAM, and both have an
`isinstance(globals().get("corpus"), HostCorpus)` **reuse** branch. A rewrite that reorders those two
statements, or reuses a kernel-resident 200-utterance corpus, builds a ~0.19 h corpus, computes ~11
steps/epoch, and runs "50 epochs" in a few minutes — cheap, plausible-looking, and scientifically
void. Because the step count is derived there is otherwise no constant left to check it against):

```
print(f"corpus.hours = {corpus.hours:.6g}")     # SIX significant figures, into the run log
assert corpus.seg_hi == 48000
assert corpus.n == FIXTURE["corpus"]["n"]                       # the I0 fixture
assert 37.2 < corpus.hours < 37.4                               # brackets the reference 37.316 h
assert 2090 < corpus.hours * 56.25 < 2110                       # steps_per_epoch sanity
assert abs(round(50 * corpus.hours * 56.25) - 104950) < 200     # 104,950 is right FOR THIS corpus
assert val_corpus.n == 40 and len(test_utts) == 120             # §9.17
assert STEPS == 104950 and BATCH == 64
```

Abort on any failure. **Injection:** `steps` cannot reach 104,950 through the budget path (it is
`max(2000, (…//1000)*1000)`, always a multiple of 1000 — §1.2), so the run must set
`STEPS_OVERRIDE = 104950` in a cell **above** the launcher, or the launcher must be rewritten to take
`steps` directly. Whichever is done, assert `steps == 104950` after the launcher computes it.

**Learning-rate schedule.** Milestones are **fractions**, so stretching the run automatically stretches
the decay; the SHAPE (halve at 20/40/50/60/70/80 %, final lr = lr₀/64 = **1.5625e-5**) is what is
preserved, not the absolute step numbers. At 104,950 steps `int(f·steps)` lands exactly on
**20990, 41980, 52475, 62970, 73465, 83960** (verified — no truncation loss).

**Cost, at A100-class throughput.**

| basis | GPU-hours | $ at $1.79/GPU-hr |
|---|---|---|
| 925 ms/step (the measured batch-64 row) | 104,950 · 0.925 / 3600 = **26.97** | **$48.27** |
| 891 ms (run start) | 25.98 | $46.50 |
| 966 ms (run end) | 28.16 | $50.41 |
| + val (210 × ~2.2–2.7 s) | +0.13–0.16 | +$0.23–0.29 |
| + ckpt (210 × ~5–15 s: 7 exports, 13 `reconstruct`, 7 `torch.save`, 1 json, 1 matplotlib redraw) | +0.29–0.88 | +$0.52–1.58 |
| **total** | **27.3 – 28.0** | **$48.9 – $50.1** |

**Budget: $101 / $1.79 = 56.42 GPU-hours.** On A100-class throughput the run uses about half of it —
a ~2.0× margin.

**Blackwell RTX PRO 6000 PCIe (sm_120, 96 GB) — ASSUMPTION, not measured.** Nothing in this repo has
run on sm_120. Two scalings bracket it:

| assumption | ms/step | GPU-h | $ | verdict |
|---|---|---|---|---|
| time scales with **memory bandwidth** (2039 → ~1792 GB/s, 0.879×) | ~1053 | 30.7 | **$55.0** | within budget |
| time scales with **bf16 tensor peak** (312 → ~126 TF/s, 0.40×) | ~2290 | 66.8 | **$119.5** | **OVER BUDGET** |

The measured 5.71 % MFU against 30.8 % bandwidth utilisation says the first. **MEASURE IT BEFORE
COMMITTING THE BUDGET: rent ONE GPU, run `time_ov3_fast(corpus, ARMS, 64, n=20)`, and read the number.**

**Fan-out plans** (derived from the cost law; billing is wall-clock × GPUs, and batch 64 is
contractual so there is no data parallel):

| plan | wall-clock | GPU-hours billed | $ |
|---|---|---|---|
| 1 GPU, all 7 arms | 27.0 h | 27.0 | **$48.3** (cheapest) |
| **7 GPUs, 1 arm each** | **4.37 h** (the 150 ms ES arm sets the pace) | 30.6 | **$54.8** (6.2× faster, +13 %) |
| 6 GPUs (det paired with an ES arm) | **6.43 h** | **38.5** | **$69.0** (worse on both axes) |
| an 8th GPU | buys nothing without data parallel, which **CHANGES MATH** | — | — |

The 6-GPU row is computed **from the cost law applied to the combined work**, not by adding the two
standalone per-arm times: `3.072e6 + 6.144e6 = 9.216e6` rows ⇒ `9.0 + 22.9367·9.216 = 220.4 ms` ⇒
`104,950 · 0.2204 / 3600 = 6.43 h` wall ⇒ 38.5 GPU-h ⇒ $69.0. (Adding 150 + 79 = 229 ms
double-counts the law's 9 ms fixed term and gives 6.68 h; that is a pessimistic bound, not the law's
answer.) The row's verdict — worse on both axes — survives either way.

**Fanning arms across GPUs is statistically safe on the batch axis and NOT on the noise axis.** Every
arm already draws its own `eps` and jitter — but from a *shared* global CUDA generator in stack order,
so arm k's noise depends on arms 0…k−1 being present. Per-rank generators cannot reproduce that.

- **(a) — THE ONLY OFFERED PATH.** Give each arm its OWN `torch.Generator` seeded by a fixed per-arm
  key. This alters the realised noise but not the estimator, makes arms independent of each other's
  presence (arguably better science), lifts I7's single-host-thread dispatch restriction, and is a
  **CHANGES-TRAJECTORY**, not a CHANGES-MATH, decision. Record it in the run log.
- ~~**(b)** reproduce the interleaved global stream by pre-drawing per-arm offsets.~~ **STRUCK.**
  Per I7, the realised `eps`/`jitter` **values** are not portable across GPU models: ATen's CUDA
  distribution kernels size their launch grid from `multiProcessorCount`, and both the Philox offset
  consumed per call and the element → substream mapping follow from it. The A100 and the sm_120
  target have different SM counts, so (b) is engineering effort spent chasing something unattainable.

**The fan-out decision covers the VALIDATION generator too, and must say so explicitly.**
`val_loss_fast` seeds ONE generator per call and consumes it across all 8 batches **and** all 7 stacks
in stack order (`e2b_fast.py:534-540`), so arm k's validation `eps` depends on which arms precede it —
and the LISASD stacks consume 4× extra at `:330`, shifting everything after them. **Under "7 GPUs,
1 arm each" EVERY arm becomes stack 0 and receives the FIRST slice**, which changes `val_loss`,
`val_wave` and `val_spec` for **six of the seven arms** relative to OV3_fast — independently of how
the *training* noise decision is made. **Recommendation:** give each arm its own validation generator
seeded by a fixed per-arm key. That also repairs §4.4's KNOWN DISAGREEMENT with `e2_trainer.val_loss`'s
per-arm `fork_rng` (`e2_trainer.py:26-30`). Declare it in the run log as a **CHANGES-TRAJECTORY**
decision affecting the `val_*` curves — do not leave it to whichever way the rewrite falls out.

**Output topology for a fan-out — the artefact contract.** §4.5.3 fixes single, **tag-keyed** output
paths that all seven processes would write: `ROOT/ov3_history_{tag}.json` (rewritten every
`ckpt_every`), `ROOT/train_{tag}.log` — which the launcher **TRUNCATES** with `write_text` at launch
(`build_train_notebook.py:487`, `e8_launch_fast.py:36`) — and `plot_curves`'s figure. Seven same-tag
processes **clobber the history JSON, which is the primary scientific artefact of the run**: last
writer wins and six arms' curves are lost. Only `CKPT/{tag}/{k}.pt` survives, because `save_ckpt`
stages via `/tmp/{stem}_{os.getpid()}.pt` (`build_notebook.py:1245`). Therefore:

- each process writes `ov3_history_{tag}_{arm}.json` and `train_{tag}_{arm}.log`;
- the launcher's `write_text` becomes per-arm, or append;
- plotting is **deferred to a single merge step** after all processes exit;
- **the merge must reassemble the exact 16-key-per-arm dict of §4.5**, so `audit/dynamics_ov3.py`
  still reads it unchanged.

**Host resource contract for a fan-out.** Seven independent `HostCorpus` copies would need
7 × 32.2 GB = **226 GB**, plus **~64.5 GB of peak during each construction** (§5 stage 4:
`train_utts` 25.78 + `xs` 6.45 + `Y` 25.78 + `X` 6.45; the earlier "~72 GB" here was not derived and
is withdrawn). Build it once in shared memory, mmap one copy of `Y`/`X`, or put the fp32 corpus on
each GPU (32.2 GB in 96 GB — which also deletes the host gather, the per-step `pin_memory` and the
H2D, with no math change, **subject to I23**). Seven distinct output roots are needed as well as one
shared corpus.

---

## 9. Open questions and unverified assertions

Where the six specialist readings disagreed, or where the repo asserts without testing, this section
says so rather than papering over it.

**Contradictions inside the repo, resolved here.**

1. **"bit-for-bit" vs "~1e-7".** `e2b_fast.py:31` and `:288` say the jittered sub-pixel path "stays
   bit-for-bit the gather path's"; `e1_model.py:93-94` says only "up to the summation order of layer 1
   (~1e-7 relative in fp32)"; the gate measures **1.96e-07 … 4.24e-07**. **This document binds to
   `e1_model.py`.** Only the *coordinate arithmetic* is bit-for-bit; the layer-1 *sum* is reassociated.
2. **The memory model in `e2b_fast.py:37-42`** predicts `N·(97 + 4·144)` bytes per output row and
   "~8.3 GB fp32 / 4.2 GB bf16 per arm" at batch 32, and claims it reproduces measured
   8.1 / 15.7 / 30.8 GB at 1 / 2 / 7 arms per stack. **The 97-wide term only exists on the GATHER
   path**, and 8.1 GB matches the fp32 figure — so those measurements predate `AMP=True` and/or
   `SUBPIXEL=True`. The later `e8_launch_fast.py:5` measures 13.4 GB for the **whole batch-64 step**,
   less than the older note predicts for one arm. Trust `e8`.
3. **The `GROUP_MAX = 1` table.** `build_train_notebook.py:422-424` justifies the choice with
   16.6 / 16.4 audio-s per compute-s at one arm per stack (batch 16 / 32) — **4.2× slower** than the
   68.5 the same configuration measures in `e8_launch_fast.py:4` at batch 32. That table was taken
   under different flags, almost certainly `DETERMINISTIC=True` and/or `SUBPIXEL=False`. The
   `GROUP_MAX` choice is a pure-performance flag with no effect on the arithmetic (but see
   CHANGES-MATH-4 for the RNG-slice and kernel-reduction side effects) and **must be re-measured on
   sm_120**.
4. **"~0.7 TFLOP per arm-step"** (`e2b_fast.py:5`) — see §7.2. For an ES arm at batch 32: **1.456
   TFLOP fwd+bwd on the GATHER path** (which is what `e2b_fast.py:3-5` describes, "three gathers,
   fp32"), or **1.265 TFLOP on the sub-pixel path the run uses**. So 0.7 is a forward-only estimate
   inflated **1.44×** against gather, **1.66×** against sub-pixel. Resolve with `FlopCounterMode`.
5. **Decoder layers 2–5 share: 90.61 %** by the §7.2 arithmetic. *(Previously stated as a
   contradiction with "90.9 %". `grep -rn '90\.9' notes/ overnight3/ build_train_notebook.py audit/`
   matches only this spec file, so there is nothing in the repo to contradict. **Withdrawn as a
   contradiction; the 90.61 % derivation in §7.2 stands on its own.**)*

**Untested assertions by the original author.**

6. **"determinism OFF changes only the float reduction ORDER inside the gradient atomics, ~1e-7
   relative in fp32"** (`e2b_fast.py:86-89`). Never A/B'd numerically.
   *Test:* one step, fp32, DETERMINISTIC True vs False, same seeds, compare per-parameter gradient
   relative error; then 200 steps twice with the flag off, all other seeds pinned, and **report the
   max relative parameter divergence per arm** rather than asserting a bound.
7. **"the perturb=False gather-free path is immune to the determinism flag whichever way it is set"**
   (`e2b_fast.py:317-319`, `e1_model.py:116-117`). Asserted from the fact that `expand`'s backward is
   a sum-reduction; never A/B'd.
   *Test:* `time_ov3_fast` with perturb forced False under DETERMINISTIC True and False; compare
   bitwise output and ms/step.
8. **"the 0.5 in `erb_feats` puts `d_erb` on the same scale as `d_logmag`, so one λ is one spectral
   weight"** (`e1_model.py:201-203`). True for the *slope*, not for the *floor*: the 1e-8 power floor
   sits three amplitude decades above the 1e-7 magnitude floor, so quiet frames and the top bands
   saturate very differently.
   *Test (cheap):* log `d_logmag` and `d_erb` separately over a few hundred real batches and compare
   their distributions. If they differ by more than ~20 %, `es_marg_erb`'s "0.5/0.5" is not an
   equal-weight average of comparable quantities.
9. **"one Adam over the stacked tensors IS one Adam per arm (elementwise)"** (`e2b_fast.py:11-12`) is
   only exercised at A = 1 in the shipped configuration, so it is untested in the run.
   *Test:* at GROUP_MAX = 2, train two arms stacked and the same two separately from the same init on
   the same batches; compare parameters after N steps.
10. **Whether the stacked and sequential trainers realise IDENTICAL noise values, or only identical
    element counts.** The element-order argument (contiguous `randn` fill) is a reader's, not the
    repo's; no test compares a stacked step's `eps` against a sequential step's.
    *Test:* seed the global generator, run one step of each with the same batch, compare
    `eps_enc`/`eps_dec`/`jitter` elementwise.

**Gaps in the test surface.**

11. **The exactness gate does not cover the production kernel.** `gate_subpixel.py:40` sets
    `GROUP_MAX = None` (A > 1, `baddbmm`); the run uses `GROUP_MAX = 1` (A == 1, `F.linear`). Extend
    check 2 to run at both settings before launch.
12. **Nothing gates the fast path on a GPU at FULL dims.** The gate is CPU-only at SMOKE dims
    (fs 16 kHz, L = 512). Add a one-step GPU gate on the target card comparing SUBPIXEL against the
    gather path in fp32 with AMP off, and separately **recording** the AMP-on divergence alongside a
    bf16-vs-fp32 gather-path deviation, so the sub-pixel error can be read against the bf16 rounding
    floor. The gate has never been run in the precision the run actually uses.
13. **No resume test exists.** `save_ckpt` writes only `{model, step, history, arm, …}`
    (`e2b_fast.py:703-705`) — no optimiser state, no scheduler state, no RNG state. At ~27 GPU-h the
    rewrite probably needs to ADD them, and the addition must not change the uninterrupted
    trajectory. *Test:* checkpoint at step 50, restart, assert bit-identical parameters at step 100
    with determinism ON.
14. **`audit/dynamics_ov3.py` cannot read a 50-epoch history.** It hardcodes `STEPS = 16000`,
    `MILESTONES`, `STEPS_PER_EPOCH = 2099` and `GAP_STEPS = [4000, 8000, 12000, 16000]`; `at()`
    asserts an exact step match (`:55-58`), and `lr[st == M][0]` (`audit/dynamics_ov3.py:151`)
    additionally requires every milestone to be a multiple of `log_every = 25`. At 104,950 steps
    **FIVE of the six milestones (20990, 41980, 62970, 73465, 83960) are not multiples of 25; only
    52475 = 25·2099 is** (verified: `20990%25=15, 41980%25=5, 52475%25=0, 62970%25=20, 73465%25=15,
    83960%25=10`). The script will IndexError. The 16 history **keys** are the contract; the
    constants are not. **RECOMMENDED RESOLUTION: update `audit/dynamics_ov3.py`** — parameterise
    `STEPS`, `MILESTONES`, `STEPS_PER_EPOCH` and `GAP_STEPS` from the history itself, and read the LR
    at the nearest log-grid point at or after `M`. **Do NOT snap the milestones to the 25-grid**: that
    **CHANGES MATH** relative to `int(f·steps)` and moves the LR schedule to fit a plotting script.
15. **Path mismatch:** the audit reads `lisa_rtm_cache/results/ov3_history_OV3_fast.json`
    (`audit/dynamics_ov3.py:20`) while the trainer writes `ROOT/ov3_history_{tag}.json`
    (`e2b_fast.py:598`), i.e. `lisa_rtm_cache/ov3_history_{tag}.json` locally — the `results/`
    subdirectory is never created by the writer. See §4.5.3 for the full writer-side layout.
    **Decide and record: either the trainer writes into `ROOT/results/`, or the history is copied
    there before the audit runs.**
16. **Whether `torch.compile` was ever active** in the 16,000-step run is unknown
    (`e2b_fast.py:156-160`; failures are swallowed twice, at build time `:168-169` and at first call
    `:179-182`, the second permanently flipping `_MLP["compiled"]` to False). **On this axis the
    50-epoch run cannot be made "the same as the reference" in either direction**, so leaving the
    choice to the rewrite means the least-informed party picks it on sm_120 throughput alone — and
    compile changes the decoder GEMM reduction order.
    **DECISION, made here: the 50-epoch run runs with `COMPILE = False`.** Rationale: it removes a
    silent fallback, makes the decoder GEMM a known ATen kernel, and the repo's own autotune log
    already shows ATen winning both decisive shapes (1.73 vs 1.81 ms; 1.34 vs 18.14 ms,
    `e2b_fast.py:59-61`), so there is nothing to lose. If the target-card benchmark overturns that,
    `COMPILE = True` is permitted **only** with `_MLP["compiled"]`, the Dynamo counters and
    `TORCH_LOGS=recompiles` logged at warm-up and at the end. **Either way, the chosen value and the
    observed `_MLP["compiled"]` go into the run log's first artefact, next to the corpus manifest.**
17. **`overnight2/analysis_local.py:48`** selects its DataShare subset with `i % 40 < per`, silently assuming
    `test_FULL.npz` holds exactly 40 utterances per speaker. Nothing asserts that; the same assumption
    underlies the comment "28 of p236 + 12 of p237" at `e3_launch.py:35`.
    *Test:* assert `len(test_utts) == 120` and 40 per speaker before slicing.

**Scientific decisions this document does not make.**

18. **Sharding batch 64 across GPUs, or accumulating gradients.** Both need the `det` arm's
    batch-global spectral convergence recomposed by the **ordered recipe in §3.3** — all-reduce
    `Σ‖Y−H‖²` and `Σ‖Y‖²` as SUMS, *then* sqrt, *then* divide, with the `1e-8` added to the
    already-square-rooted denominator. **Nothing in the repo does this and no note flags it.**
    Highest-risk item in the specification. §6.2 now carries the micro-batching row and I27 the
    invariant; the one-line instruction "all-reduce before the sqrt" is **not** sufficient on its own
    — it admits at least three wrong implementations, enumerated in §3.3.
19. **Per-arm generators vs the interleaved global stream** for a fan-out (§8). **Resolved in §8:**
    option (a) only; option (b) is struck as unattainable across GPU models (I7). Flag (a) in the run
    notes as a deliberate trajectory-changing decision.
20. **Should the validation-noise convention be "fixed" to match the reference?** `val_loss_fast`
    does not pair validation noise across arms; `val_loss` does. Restoring per-arm seeding would make
    cross-arm val comparison honest, but changes the curves relative to OV3_fast — and
    `notes/analysis/dynamics_OV3_fast.md:§0` already forbids cross-arm comparison of `val_loss` on
    other grounds. Same decision applies to validation running under bf16 autocast in the fast path
    and fp32 in the reference. **This question is NOT independent of the fan-out** (§8): under
    "7 GPUs, 1 arm each" every arm becomes stack 0 and the convention changes for six of seven arms
    whether anyone decides it or not. §8 now requires it to be decided and declared explicitly.
21. **Should TF32 stay on?** It is the status quo and what the OV3_fast numbers were produced with,
    but it is **asymmetric across the seven arms** — it perturbs only the two ERB arms and the ERB
    target features. Forcing that one matmul to full fp32 is defensible and **CHANGES MATH** relative
    to the reference.
22. **Should the LR schedule stay at six halvings?** `notes/2026-09-16-…:181-185` argues the 16k run
    "ran out of step size, not gradient" and that three cuts would have done the work of six.
    Changing the milestone list is a scientific choice and needs the user's decision before launch.
23. **Is the training/inference sampler-law mismatch acceptable at 50 epochs?** The energy score is
    proper for the law that generated y1, y2 — which at training time **includes** the per-output-sample
    anchor jitter — while `reconstruct()` runs `perturb=False`. The properness argument in
    `notes/2026-09-16-…:§0` does not mention the jitter.
    *Test:* evaluate CRPS/PIT of a trained arm with `perturb=True` at inference and compare.
24. **Is 104,950 steps the right number at all?** The dynamics note reports all seven arms still
    learning at 16,000 steps but nearly stopped (val_loss slopes −0.05 to −0.52 % per 1,000 steps over
    the last 4,000). The cost model says 50 epochs is **affordable**; nothing says it will move the
    science, versus e.g. 40,000 steps plus a second variable.
25. **Should `CKPT_EVERY` be raised?** 210 matplotlib redraws, 210 × 7 `torch.save` to a network mount,
    210 history dumps and 210 × 13 full-utterance reconstructions is 0.3–0.9 GPU-h of pure overhead
    that was never separately timed in the 16k run. Raising it to 2000 recovers most of it, at the
    cost of coarser resume granularity and a coarser dev-probe curve. Staggering `val_every` and
    `ckpt_every` (they currently coincide at 500, so they serialise) changes no math but changes the
    history sampling grid.
26. **Where does the residual ~300–400 ms of the 925 ms step go?** My bottom-up budget accounts for
    ~520–620 ms (GEMMs 186, ReLUs 133, layer-1 gather+scatter 80–120, STFT/loss 40–70, encoder 25–35,
    misc 30–40, host batch 8–20). The repo contains **no per-kernel profile at batch 64**. One Nsight
    Systems trace or `torch.profiler` run of 20 steps would settle it and is the cheapest piece of
    evidence available. Related: the `scatter_add` atomic backward of the layer-1 gather is
    5.75e9 bf16 atomic adds per step into a 442 MB buffer per arm, with high intra-warp address
    conflict — it is the exact term the determinism flag inflates 3.9×, so it is known to be large,
    but its absolute size with determinism OFF has never been measured.
27. **Can a fused five-layer decoder MLP kernel keep the SAME per-layer accumulation order?** That is
    where the 7× headroom is. If the order must change, measure the deviation and check it against the
    bf16 rounding the run already carries.
28. **The corpus manifest is not in git.** `cell1_corpus.py:85-87` builds a MANIFEST and prints only
    its sizes; the JSON lives on Drive. The repo's own numbers (97 train + 3 ours + 8 paper_test +
    s5 skipped = 109) do not obviously reconcile with VCTK 0.92's 110 speakers + s5.
    **ACTION: the 50-epoch run should print and commit the manifest — speaker list, per-speaker
    utterance counts, `sum(lens)`, `corpus.hours` to six significant figures, and a hash of `Y` and
    `X` — as its first artefact**, so the corpus identity is checkable afterwards.
29. **Why Hub and DataShare measure 5 dB apart** on the same checkpoint
    (`notes/2026-09-04-…:77-81`, `notes/2026-09-08-…:74-78`) is asserted, never established. The
    DataShare route takes `[:40]` BEFORE the <1 s drop while the Hub route drops BEFORE taking `[:40]`
    (`cell1_corpus.py:56-58,69`), so their memberships can differ.
    *Test:* intersect the two 12-utterance sets by filename and compare sample-for-sample.

30. **Does the GPU model change the realised noise VALUES?** I7 now asserts that it does, from ATen's
    `calc_execution_policy` sizing its launch grid by `multiProcessorCount`. **This has not been
    measured on the two cards in question**, because no sm_120 device has been available.
    *Test (one step, cheap, do it on the rented GPU before committing the budget):* seed identically,
    draw `randn(1,128,8,12000)` on the A100 and on the RTX PRO 6000, and report the fraction of
    elements that differ and the `get_rng_state()` offset consumed by the call. Record the answer in
    the pre-flight artefact either way — if the values happen to agree, that is worth knowing too,
    but the contract stays ORDER + COUNT + SHAPE + DTYPE.

**Challenges raised against this document and REJECTED, with the reason.** Recorded so they are not
re-raised, and so a future reader can check the reasoning rather than the conclusion.

31. **"`§7.5`'s `det ≈ 79 ms` should be 80 ms."** *Rejected.* The cost law gives
    `9.0 + 22.9367·3.072 = 79.4615 ms`, which rounds to **79**, not 80. (The challenge computed
    "79.5" and rounded up from there.) The figure stands. The 6-GPU row in §8 was nevertheless wrong
    for a different reason — it added standalone per-arm times instead of applying the affine law to
    the combined work — and has been corrected to 220 ms / 6.43 h / $69.0.

32. **"`reconstruct(..., seed=0)` means LISASD arms do not share `eps_enc` with LISAS arms, because
    LISASD draws `eps_enc` then `eps_dec` from the one generator."** *Rejected.* `eps_enc` is the
    **first** draw from a **freshly seeded** generator in both classes, with identical shape
    `(B, n_noise, L)` (`c1_model.py:37-39` vs `e1_model.py:36-38`), so it is bit-identical across the
    two classes; only LISASD's additional `eps_dec` is new. **Verified numerically 2026-09-19:**
    `torch.equal` on the two draws returns True. The rest of that challenge — that probe noise IS
    paired across stochastic arms via the per-call `seed=0` generator, in direct opposition to
    `val_loss_fast`'s single shared generator, and that this belongs in the contract — was correct and
    is now §4.5.2.
