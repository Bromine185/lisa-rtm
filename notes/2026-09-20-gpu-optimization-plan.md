# GPU optimisation plan — seven-arm energy-score LISA at 50 epochs on Blackwell

**Date:** 2026-09-20.
**Target:** Hyperbolic `rtx-6000-pro`, PCIe, 96 GB, `us-east-3`, `machineType: virtual-machine`.
**Baseline:** 925 ms/step, seven arms in one loop, batch 64, 13.4 GB of 80 GB on an A100-80GB.
**Budget:** a hard ceiling of **$60**, covering training *and* evaluation. Account balance $101,
`maxOverdraftCents = 0`.
**Companion:** [`2026-09-19-algorithm-spec.md`](2026-09-19-algorithm-spec.md) is the correctness
contract. Where this plan and the spec disagree, **the spec wins** — it is the thing that keeps the
seven arms comparable.

**Provenance.** Twelve specialist probes (roofline, decoder fusion, the output-rate gather, precision,
activation memory, the loss side, the data pipeline, launch overhead, packing, multi-GPU, an sm_120
reality check, and fidelity gates) produced 105 candidates. The adversarial verification and synthesis
phases **did not run** — the session hit its rate limit after 14 of 67 agents. This document is
therefore a synthesis by the orchestrator over the twelve completed probes, the algorithm spec, and
live Hyperbolic API data. **The candidates in Tier 1 and below have had one pass of analysis, not
two.** Treat their predicted gains as the proposers' own estimates. Tier 0 items are the ones whose
reasoning I have checked against the spec's cost law myself.

---

## 0. The verdict in one page

**Three findings decide this run.**

**(1) The step is per-element bound, and that kills half the usual playbook.** Solving the two
measured rows, `467 = F + 32k` and `925 = F + 64k`, gives `k = 14.31 ms` per unit of batch and
`F = 9.0 ms`. So the *entire* batch-independent cost — every one of ~2,000 kernel launches, seven
fused-Adam steps, seven per-arm clips, the scheduler, all Python dispatch — is **at most 9 ms, under
1 % of the step**. That is not an estimate; it is forced by the two measurements. It rigorously rules
out CUDA graphs, CUDA streams, arm-stacking (`GROUP_MAX > 1`), larger batches, and host-sync removal
as speedups, and it retro-explains why the previous author measured every one of them as
no-effect-or-worse. The time is in per-output-row work, at 5.7 % MFU and ~31–44 % of HBM peak.

**(2) The two biggest wins are structural defects, not missing cleverness.**

- **The `u` gather reads 16× more memory than it needs to.** `e2b_fast.py:299` views `u` as
  `(A,S,L,H)` with stride 12000 on the `H` axis — verified numerically as strides
  `(1728000, 1728000, 1, 12000)`. Every output row therefore touches 144 separate 32-byte sectors
  where 9 would do: ~207 MB of sector traffic per sequence against 13.8 MB of actual data,
  ~172 GB/step forward and ~344 GB/step in the `scatter_add` backward. The same defect appears at
  `:249-250` and `:298`, where `zp` is built `(A,S,L+2,C)` and immediately permuted back to
  channel-first — a transpose round-trip through a strided read.
- **`H = 144` is pathological for cuBLAS.** 144 = 9 × 16, so every power-of-two tiling wastes.
  The header's own autotune log is the evidence: `addmm(3072000×144, 144×144)` takes 1.73 ms while
  moving only 1.77 GB of necessary bytes — 1.02 TB/s, exactly half of A100 peak. The consistent
  explanation is that cuBLAS splits `N = 144` into 2–3 column tiles and re-reads the whole `M × 144`
  activation once per tile. At 3× reads it is running at ~2.05 TB/s, i.e. *at peak, doing 3× the
  work*. **This inference is load-bearing and is settled in two minutes by `ncu`** (§5, PROFILE 3).

**(3) The headline optimisation may be blocked on this exact card.** The largest predicted win — a
hand-written fused Triton kernel for decoder layers 2–5, worth ~7× on paper — runs into
**pytorch/pytorch#176426, which is OPEN: "Triton kernels with 2 or more `tl.load()` calls segfault at
runtime on sm_120 (RTX PRO 6000)"**. It compiles clean and produces invalid code. A fused MLP kernel
is *nothing but* multiple `tl.load()` calls. Separately, **#178367** reports non-deterministic
segfaults on sm_120 under sustained work plus allocator churn. And the repo's own autotune log already
shows ATen beating the best Triton template on both decisive shapes (1.73 vs 1.81 ms; 1.34 vs
**18.14** ms, a 13× loss). **Conclusion: default `FAST["COMPILE"] = False` on sm_120, and treat every
Triton path as Tier 2 behind an explicit crash gate.**

**So the plan is:** fix the layout defects in plain PyTorch (no Triton, exact, large), put the corpus
on the GPU (exact), harden for an ephemeral instance, **then measure**, and only then decide whether
to fan out to eight GPUs and whether any Triton work is safe to attempt.

**Predicted end-to-end:** 925 ms → **440–620 ms** on an A100-equivalent from Tier 0 alone, with the
Blackwell factor still unmeasured and bracketed between 0.9× and 1.8×. The honest range for the full
50-epoch run is **$26–$71**, which straddles the ceiling. **That is why §5 comes before any commitment.**

---

## 1. Where the time goes

Per-step ranking at batch 64, A100, seven arms, from the roofline probe. Confidence is the probe's own.

| # | stage | ms | share | confidence |
|---|---|---|---|---|
| 1 | decoder layers 2–4 **backward** (`relu_bwd` + grad_in + grad_w) | 195 | 21 % | high |
| 2 | decoder layers 2–4 **forward** (`addmm` + standalone `relu`) | 127 | 14 % | high |
| 3 | layer-1 gather **backward** — `scatter_add`, bf16 atomics into a *strided* buffer | 86 | 9 % | medium |
| 4 | layer-1 HR epilogue (`coord*w_c`, `+g`, `+b`, `relu` as four separate pointwise kernels) | 61 | 7 % | high |
| 5 | encoder backward (skinny 9/16/32/64-channel convs) | 53 | 6 % | **low** |
| 6 | layer-1 HR gather of `u`, strided | 48 | 5 % | medium |
| 7–14 | layer-1 epilogue bwd 31; loss algebra bwd 27; encoder fwd 27; layer-1 relu bwd 23; spectral algebra fwd 18; STFT bwd 17; layer-5 bwd 15; STFT fwd 12 | 170 | 18 % | mixed |
| — | unmodelled residual | 148 | 16 % | — |

Grouped: **decoder layers 2–5 = 345 ms (37 %)**, **layer-1 HR block = 249 ms (27 %)**, encoder +
layout + indexing = 104 ms (11 %), losses = 74 ms (8 %), residual = 148 ms (16 %).

The byte model totals ~824 GB of nominal HBM traffic per step plus ~138 GB of cuBLAS re-reads and
~557 GB of L1/L2-only sector traffic, predicting 777 ms against 925 measured. The model independently
reproduces the measured 13.4 GB peak memory, which is the one free check available, and it does
(one live arm × 128 sequences × 4 saved HR bf16 tensors × 13.824 MB = 7.08 GB, plus backward
temporaries). That is why the byte model is trusted to ~20 % and the ranking above to about one rank.

---

## 2. The plan, in tiers

### Tier 0 — do unconditionally

Exact or equivalent, no Triton, no measurement needed first. These are the ones whose reasoning I have
checked against the cost law myself.

| id | what it changes | predicted gain | exactness | gate |
|---|---|---|---|---|
| **T0.1 `coalesce-u-gather`** | Make `u` contiguous in `(S,L,H)` before the HR gather; drop the `zp` transpose round-trip at `:249-250`/`:298`. Pure layout — `.contiguous()` and an index reorder, no new kernel. | **~110 ms (12 %)** on A100; the probe predicts **3.1×** on this stage on Blackwell's 3.2× L2 | equivalent (~1e-7, reduction order only) | G5 |
| **T0.2 `gather-to-index-select`** | Replace the element-wise `torch.gather` with a row `index_select` / `F.embedding` so the backward is `index_add` over rows, not per-element bf16 atomics | 25–35 ms; **forward exact** | forward EXACT, backward equivalent | G5, G2 |
| **T0.3 `gpu-resident-corpus`** | Y and X resident in GPU fp32 (32.2 GB of 96 GB); indices still drawn by the **same numpy Generator** on the host, only the gather moves to the device. Deletes the host gather, the per-step `pin_memory` and the H2D | 4–8 ms/step; more once the step is short | **exact, bit-for-bit** — *subject to spec I23, every index int64* | G3, G2 |
| **T0.4 `compile-off-gate`** | `FAST["COMPILE"] = False` by default on sm_120 | ~0 % throughput (ATen already wins both shapes); removes a **crash class** | exact | §8 |
| **T0.5 `foreach-clip` + `flat-param-buffer`** | `torch._foreach_norm` / `_foreach_mul_` for the per-arm clip (539 → ~14 launches); one flat param/grad buffer per arm (Adam 14 → 2 launches) | ≤ 9 ms total — bounded by `F`; take it for the launch-count hygiene, not the speed | equivalent (float reduction order in the clip norm) | G6 |
| **T0.6 `checkpointing-off-the-critical-path`** | Move `plot_curves` / `save_ckpt` / history JSON off the step path; cache the decimated probe input | 17–35 min of a 27 h run (1–2 %) | exact — changes only *when* artefacts are written | G11 |
| **T0.7 `resumable-checkpoints-and-watchdog`** | Optimiser + scheduler + RNG state in every checkpoint; auto-resume; watchdog | **not a speedup — it protects the spend.** Converts an instance loss from a total write-off into a ~10–15 min restart | exact | **G11** |
| **T0.8 `per-arm-output-topology`** | Per-arm `ov3_history_{tag}_{arm}.json` and logs; a single deferred merge that reassembles the exact 16-key-per-arm dict | prevents losing the **primary scientific artefact** under any fan-out | exact (output paths only) | G12 |

**T0.7 and T0.8 are not optional.** The instance is ephemeral — Hyperbolic's own docs say
*"Termination is permanent. All data on the instance will be lost."* A crash at step 80,000 without
T0.7 is a ~$37 write-off. Seven same-tag processes without T0.8 clobber the history JSON and six arms'
curves are simply lost.

### Tier 1 — do after the measurement in §5 confirms the bottleneck

| id | what it changes | predicted gain | exactness | the measurement that gates it |
|---|---|---|---|---|
| **T1.1 `fuse-layer1-hr-epilogue`** | Fuse gather + `coord*w_c` + bias + ReLU into one pass instead of four pointwise kernels | 115 ms → ~25 ms | exact for the fp32 path | PROFILE 1: pointwise kernels at shape `[1,128,48000,144]` ≥ 60 ms |
| **T1.2 `l2-row-blocking-recompute`** | Chunk the decoder MLP over rows so intermediates stay in L2 and recompute layers 3–5 in backward. **HBM traffic for `_mlp` 448.6 GB → 34.5 GB/step (13×)** | the single largest non-Triton win | equivalent | PROFILE 4: DRAM integral near 800–900 GB/step |
| **T1.3 `gemm-epilogue-autotune`** | `mode='max-autotune-no-cudagraphs'` so a `BLOCK_N ≥ 144` template reads the activation once | removes ~115 ms of standalone ReLU; more if the cuBLAS re-read is confirmed | equivalent | PROFILE 3 confirms re-read **and** the #176426 crash gate passes |
| **T1.4 `stft-fold-backward` + `fuse-spec-loss`** | Replace `torch.stft`'s `as_strided_backward` (int64 index, 4.8e8 atomics); fuse the spectrum→three-distances chain | ~15 GB → ~2.4 GB; ~37 GB → ~10 GB; ~700 → ~60 launches | equivalent | PROFILE 1: STFT + spectral algebra > 60 ms |
| **T1.5 `last-gemm-out-dtype-fp32`** | Final decoder GEMM writes fp32 directly | removes 36 % of output noise power (−1.9 dB) **for free** | equivalent-or-better | G1 |
| **T1.6 `fold-b1-into-subpixel-conv`** | Pass the layer-1 bias as the `conv1d` bias instead of a separate broadcast add | ~22 GB/step over seven arms | equivalent (≤ 1 ulp) | G5 |

### Tier 2 — only if budget and time allow, each behind its own gate

- **`triton-fused-mlp-128-16-split`** — the 7× prize, **blocked behind pytorch#176426**. Do not
  attempt without first running the crash gate in §5 (compile `_mlp_eager` at production shapes,
  run 5×, check for segfault *and* `allclose` against eager). If the gate fails, this tier is closed.
- **`group-max-after-fusion`** — revisit `GROUP_MAX > 1` *only* after the decoder is fused; the 9 ms
  fixed term is the only thing it can recover, so the ceiling is 1 %.
- **`mps-colocation` / `concurrency-decision-test`** — one 10-minute two-process test. Expect ≤ 3 %,
  for the same wave-count reason that sank CUDA streams on the A100.
- **`seeds-not-batch`** — if the optimisation lands well, spend the surplus on a **second seed**
  rather than a bigger batch. The repo's own notes record that the headline result has one seed and
  no confidence intervals; error bars are worth more than speed at that point.

### Tier 3 — rejected, with the reason (so nobody re-treads it)

| rejected | why |
|---|---|
| CUDA graphs (full-step capture) | bounded above by `F = 9 ms`; ≤ 1 % |
| CUDA streams, arm-stacking `GROUP_MAX > 1`, larger batch, host-sync removal | same `F = 9 ms` bound; all measured no-effect-or-worse already |
| **MPS with seven processes on one card at 2.5–4×** | **refuted.** The Blackwell probe predicted this from a "launch-bound" premise that the `F = 9 ms` bound falsifies. Seven processes cannot recover 1 % seven times over |
| MIG | partitions rather than adds; predicted −5 % to −15 % against running the same arms sequentially |
| FP8 / NVFP4 for the decoder | CHANGES MATH; sglang#21132 reports garbage FP8 output on this exact card; and at 5.7 % MFU there is no FLOP problem to solve |
| fp16 instead of bf16 | 0 % predicted (both are 2 bytes; the bound is bytes, not dtype) and it CHANGES MATH |
| bf16 spectrograms / log-magnitudes | saves ~3 GB (~0.2 %) and quantises the quantity the research measures |
| int16 corpus by requantising the *peak-normalised* float | CHANGES MATH for 12.9 GB we do not need — fp32 already fits in 96 GB. (The exact int16-from-original-PCM variant is fine but unnecessary for the same reason.) |
| conv-STFT (DFT as GEMM), cross-arm STFT batching | predicted net **loss** of ~9 ms plus a precision regression |

---

## 3. CHANGES MATH — what may NOT go inside the seven-arm comparison

These are not forbidden; they are forbidden **inside the seven**. Each may be run as a separate
eighth paired arm, declared in the run log.

1. **`JITTER_PER_CELL = True`** — one anchor draw per input cell instead of per output sample.
   Predicted 925 → ~790 ms (15 %). Already flagged CHANGES MATH in the source. Use it as a
   *stopwatch only* in §5, never in the run.
2. **Any larger global batch** (Plan C). Batch 64 is contractual (spec **I17**); "50 epochs" is
   *defined* against it, and a different global batch is a different optimisation trajectory.
3. **FP8 / NVFP4 anywhere in the decoder.**
4. **fp16 in place of bf16**, and **stochastic rounding**.
5. **Any reduced-precision spectrogram or log-magnitude.**
6. **`shared-anchors`** — sharing one anchor draw across arms. Worth ~3 ms and it breaks arm
   independence.
7. **Sharding or micro-batching the loss** without recomposing the spec's §3.3 recipe — the
   spectral-convergence term does **not** average over shards (spec **I27**). Gate **G8** detects it.

**And one that is CHANGES-TRAJECTORY rather than CHANGES-MATH, which must still be declared:**
fanning the arms across GPUs. Today every arm draws `eps` and jitter from a *shared* global CUDA
generator in stack order, so arm *k*'s noise depends on arms 0…*k*−1 being present. Per-rank
generators cannot reproduce that. Worse, it is unreproducible **in principle** across hardware: ATen's
CUDA distribution kernels size their launch grid from `multiProcessorCount`, and the A100 has 108 SMs
against this card's 188, so the realised noise values were never going to be portable. **Give each arm
its own `torch.Generator` seeded by a fixed per-arm key, for training *and* for validation, and record
it in the run log.** This also repairs a known disagreement between `val_loss_fast` and
`e2_trainer.val_loss`. It changes the `val_*` curves for six of the seven arms relative to OV3_fast —
say so in the note, do not let it fall out of the implementation silently.

---

## 4. Multi-GPU strategy

**The SKU constrains this more than the algorithm does.** Live from the API: there is no 7-GPU
option. Only `gpuCount: 1` at $1.79/hr and `gpuCount: 8` at $14.32/hr, priced exactly linearly, on
nodes of 48 vCPU / 180 GB RAM / 720 GB disk per GPU.

| plan | wall-clock (A100-equivalent) | GPU-hours | $ | verdict |
|---|---|---|---|---|
| **1 GPU, all seven arms in one loop** | 27.0 h | 27.0 | **$48.3** | cheapest; fits the ceiling in both Blackwell brackets |
| **8-GPU node, one arm per GPU** (7 used) | 4.37 h | 30.6 | **$62.6** | **6× faster, but over the $60 ceiling at baseline speed** |
| 6 GPUs (det paired with an ES arm) | 6.43 h | 38.5 | $69.0 | worse on both axes |
| Plan B: shard the global batch of 64 across 8 | 4.43 h | 35.5 | $63.5 | no faster than Plan A, and starves each kernel |
| Plan C: larger global batch | 4.21 h | 33.6 | $60.2 | **CHANGES MATH.** Rejected |

Two things make the fan-out cost *more* GPU-hours than the serial run: the cost law is **affine**, so
the 9 ms fixed term gets paid seven times instead of once; and the ES arm sets the pace while the det
arm's GPU idles at roughly half load.

**Recommendation: decide this after §5, not before.** If calibration says Blackwell is at the fast end
of the bracket, or if Tier 0 lands ≥ 30 %, the 8-GPU node comes in under $60 and buys a 4-hour run
instead of a 27-hour one. If calibration says Blackwell is at the slow end, **take the single GPU** —
it fits the ceiling in every bracket, and 27 hours of unattended training costs nothing but patience.

The eighth GPU, if the 8-node is taken, should run either a **second seed** of `es_erb_l0.1` (error
bars on the headline result) or the **`JITTER_PER_CELL` paired arm**. It is already paid for.

**Operational notes for the 8-node.** Staging burns $14.32/hr while using one card — parallelise it or
accept ~$5. One `HostCorpus` per process would need 7 × 32.2 GB = 226 GB (the node has 1440 GB, so it
fits, but build it once in `/dev/shm` and mmap it, or put it on each GPU). NCCL is irrelevant under
Plan A: zero communication. PCIe Gen5, no NVLink, which matters for nothing here.

---

## 5. The measurement protocol — spend $0.36 before committing $60

One GPU, ~12 minutes. Every number below decides something.

| # | measurement | command | decides |
|---|---|---|---|
| **M0** | **the wheel** | `nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv` | driver ≥ 580 → `torch==2.14.0+cu130`; 570/575 → `2.11.0+cu128`. **Release wheels ship SASS only, no PTX**, so `cu126` hard-fails with "no kernel image is available". Do not guess |
| **M1** | arch list sanity | `torch.cuda.get_device_capability()`, `get_arch_list()`, `get_device_properties(0)` | expect `(12,0)` and 188 SMs, ~128 MB L2. Do not be alarmed by `(12,2)` — pytorch#157549 quotes it; 2.14's cu130 cubins cover any 12.x |
| **M2** | **achievable bandwidth** | 2 GB device-to-device copy, 20×, CUDA events | **every millisecond prediction in §1 scales inversely with this.** Expect 1.25–1.35 TB/s of 1,597 nominal |
| **M3** | **isolated GEMM** | bf16 `(6_144_000,144) @ (144,144)`, timed | whether `N = 144` tile quantisation really costs ~1.8× |
| **M4** | **the step itself** | `time_ov3_fast(corpus, ARMS, 64, n=20)` | **the Blackwell factor.** This is the number the whole budget hinges on |
| **M5** | GPU busy % | `nsys profile -t cuda,nvtx`, then `sum(kernel durations)/wall` | predicted ≥ 95 %. **If below 85 %, the `F = 9 ms` reasoning is wrong** and launch work re-enters the ranking |
| **M6** | the ranked list | `torch.profiler`, `key_averages(group_by_input_shape=True)`, top 50 | if `gather` + `scatter_add_` < 60 ms combined, demote T0.1; if > 200 ms, promote it to rank 1 |
| **M7** | **the coalescing claim** | `ncu -k regex:'gather\|scatter' --section MemoryWorkloadAnalysis` → `l1tex__average_t_sectors_per_request…` | predicted ~32, ideal 4. **The single most load-bearing inference in §1** |
| **M8** | **the cuBLAS re-read** | `ncu --metrics dram__bytes_read.sum` on the one `addmm` | confirms or kills T1.3 |
| **M9** | **the Triton crash gate** | compile `_mlp_eager` at production shapes (6.1M rows, in=97 and in=101, width 144), run 5×, check segfault **and** `allclose` | opens or closes Tier 2 entirely |
| **M10** | fwd/bwd split | wrap `_fwd_bwd` in `no_grad()` | predicted 35/65. Says whether to spend effort on backward kernels |

**Decision rule after M4**, against the $60 ceiling and the 1.5×-of-prediction hard stop:

- ≤ 700 ms/step → Blackwell is at the fast end. 8-GPU fan-out is affordable. Proceed.
- 700–1,050 ms → as predicted. **Single GPU** unless Tier 0 measures ≥ 30 %, then re-evaluate.
- \> 1,400 ms → more than 1.5× the prediction. **Stop and report.** Something is wrong (wrong wheel,
  PTX-JIT, thermal cap, virtualised PCIe), and buying through it is the wrong move.

---

## 6. Fidelity gates — all must pass before the 50-epoch run

From the twelve-gate regime in the fidelity probe. **G0 first**: capture fixtures from the
*unmodified* code, or every later gate is self-referential.

| gate | what it proves | where it runs |
|---|---|---|
| **G0** | fixtures captured from unmodified code | CPU, free |
| **G1** | step-0 parity: losses, gradient vectors | CPU, 10 s |
| **G2** | **RNG consumption trace** — order, count, shape | CPU. *The only cheap detector for the most dangerous silent failure mode here* |
| **G3** | **batch identity** — same segments, same order | CPU. Protects the paired design itself |
| **G4** | arm symmetry — the optimisation hits all seven equally, so it cannot become a confounder | CPU |
| **G5** | GPU parity at **full** dimension on a real batch | GPU, ~1 min |
| **G6** | optimiser / clip / schedule semantics | CPU |
| **G7** | **the gradient-checkpoint boundary trap** — spec **I22**: `_layer1` draws the anchor jitter, so checkpointing may wrap **only** layers 2–5 | CPU |
| **G8** | shard / micro-batch recomposition (detects the non-averaging `sc` term) | CPU |
| **G9** | short end-to-end run inside a null-calibrated band | GPU, ~10 min |
| **G10** | permanent in-run assertions | during the run |
| **G11** | **resume equivalence** — a failed resume 20 h in wastes ~$36 | GPU |
| **G12** | the optimisation → gate coverage matrix | process |

Plus the spec's **hard pre-flight gate**: print `corpus.hours` to six significant figures and assert
`37.2 < corpus.hours < 37.4`, `corpus.n == FIXTURE.n`, `STEPS == 104950`, `BATCH == 64`. Without it,
the `e3_launch.py` trap (corpus built, *then* `train_utts` trimmed to 200) yields a 0.19 h corpus and
a "50-epoch" run that finishes in minutes — cheap, plausible-looking, and scientifically void.

---

## 7. Cost and schedule

All figures for 104,950 steps, seven arms, at $1.79/GPU/hr. The Blackwell factor is **unmeasured**;
the two columns bracket it. "Tier 0" assumes the mid-point of its predicted 30–50 % reduction.

| configuration | ms/step | fast bracket (0.7×) | slow bracket (1.13×) |
|---|---|---|---|
| baseline, 1 GPU | 925 | 18.9 h → **$33.8** | 30.5 h → **$54.6** |
| baseline, 8-GPU node (7 arms) | 150/arm | 3.06 h → **$43.8** | 4.94 h → **$70.8** ✗ |
| **+ Tier 0, 1 GPU** | ~555 | 11.3 h → **$20.3** | 18.3 h → **$32.7** |
| **+ Tier 0, 8-GPU node** | ~90/arm | 1.84 h → **$26.3** | 2.96 h → **$42.4** |
| + Tier 0 + Tier 1, 8-GPU node | ~60/arm | 1.22 h → **$17.5** | 1.97 h → **$28.2** |

Add to any row: staging (~$0.6 on 1 GPU, ~$5 on the 8-node), calibration $0.36, and the evaluation
pass (`e4_eval` + `e5_visqol`, estimated 1–2 GPU-h → $2–4).

**Does it fit $60?** **Yes in every configuration except one** — the 8-GPU node at baseline speed in
the slow bracket, at $70.8. Tier 0 alone moves every configuration comfortably inside the ceiling.
The recommended path (Tier 0, decide the fan-out after M4) lands between **$26 and $47 all-in**,
leaving real headroom for a second seed.

---

## 8. Risks and unknowns

1. **The Triton crash class (#176426, OPEN).** Closes Tier 2 if M9 fails. Mitigation: Tier 0 and most
   of Tier 1 need no Triton at all.
2. **#178367 — non-deterministic sm_120 segfaults** under sustained work plus allocator churn,
   partially mitigated by `PYTORCH_NO_CUDA_MEMORY_CACHING=1`. Our fixed-shape step is the best
   defence; T0.7 (resume) is the insurance. **Log `torch.cuda.memory_reserved()` every 500 steps for
   the first 5,000 — a still-growing curve is the warning sign.**
3. **Which roofline wall the step is on is genuinely unsettled.** The roofline probe says
   bandwidth/per-element and predicts Blackwell at 0.9–1.0× the A100. The Blackwell probe says
   latency/occupancy and predicts 1.3–1.8× *faster*. **I side with the roofline probe**, because the
   `F = 9 ms` bound is derived from measurements and falsifies the "launch-bound" premise the other
   argument rests on. But "per-element bound" is not the same as "DRAM-bound": if the binding
   resource is *sector/request throughput* (the 16× amplification), then Blackwell's 3.2× L2 and
   1.74× SMs do help, and the truth is between the two. **M2 + M7 settle it in four minutes.**
4. **Dense bf16 throughput of this card is inferred, not read** (~480 TF/s from
   120 TFLOP/s FP32 and 188 SMs at 2.49 GHz; NVIDIA's "1 PFLOPS" is the 2:4-sparse figure). It barely
   matters at 5.7 % MFU.
5. **Server vs Workstation Edition** — 1,597 vs 1,792 GB/s. The listing says PCIe + 96 GB, which
   points at Server. A 12 % swing in every §1 millisecond.
6. **The cuBLAS re-read inference** is the most load-bearing unverified claim in §1. M8 settles it.
7. **The encoder's 104 ms is a guess** — "30 % of roofline" for skinny 9/16/32/64-channel convs is not
   a measurement. It could plausibly be 30 ms or 150 ms.
8. **Storage volumes appear to be bare-metal-only**; our SKU is `virtual-machine`. Assume every rental
   re-stages the corpus, and make staging fast. Verify before relying on either answer.
9. **Unverified because the phases did not run:** Tier 1 and Tier 2 gains have had one analytical pass
   and no adversarial verification. The four-front refutation (does it help / does it exist / does it
   change the science / what does it cost) ran on only 2 of 105 candidates before the rate limit.
   Tier 0 I have checked myself against the cost law; below that, measure before you trust.
