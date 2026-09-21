# GPU optimisation plan — seven-arm energy-score LISA at 50 epochs on Blackwell

**Date:** 2026-09-20 (revised after adversarial verification).
**Target:** Hyperbolic `rtx-6000-pro`, PCIe, 96 GB, `us-east-3`, `machineType: virtual-machine`.
**Baseline:** 925 ms/step, seven arms in one loop, batch 64, 13.4 GB of 80 GB on an A100-80GB.
**Budget:** hard ceiling **$60**, covering training *and* evaluation. Balance $101, no overdraft.
**Companion:** [`2026-09-19-algorithm-spec.md`](2026-09-19-algorithm-spec.md) is the correctness contract.

**Provenance.** Twelve specialist probes produced 105 candidates. **44 were then adversarially
verified — 18 survived, 26 were refuted.** The escalation, synthesis and critique phases died on a
session rate limit; this revision was made by the orchestrator from the 44 verdicts. Verifiers were
instructed to refute, to redo the arithmetic, and to check hardware and library claims against
primary sources — and several of them *ran code in this repo's venv* rather than arguing. Where a
verifier measured something, it is marked MEASURED and it overrides the proposer.

**This revision is largely a record of the first draft being wrong.** Four of the five Tier 0 items
in the previous version are gone: two refuted outright, one demoted by a factor of five, one shown to
rest on a misreading. Read §0.1 before anything else.

---

## 0. The verdict in one page

### 0.1 The landmine — a silent, science-corrupting failure worth $55–75

This is the single most important finding, and it was not in the first draft.

Batches are drawn from `stream(f"{tag}/batches")` and validation from `val_batches(val_corpus, tag, …)`
(`e2b_fast.py:591-592`). `stream(label)` is **blake2b-hashed on the label — a NAMED stream, not a
positional one.** That is precisely why all seven arms see identical batches today, and why that
property survives being split across processes for free.

Now: any fan-out needs per-arm output paths (`ov3_history_{tag}_{arm}.json`, `CKPT/{tag}/{arm}.pt`),
and the obvious one-line way to get them is to pass `f"{tag}_{arm}"` as `tag`. **That is the same
string that reaches `stream()`.** Do it and every arm draws a different 105,000-step batch stream and
a different validation set.

Nothing crashes. No existing smoke test catches it — `overnight3/smoke_fast_local.py` runs all seven
arms in **one** process. The seven-arm paired comparison, which is the entire scientific product, is
invalidated, and the only symptom is that the per-arm loss curves stop sharing step-to-step noise,
which nobody thinks to look for.

**Mandatory:** `train_ov3_fast` takes a `batch_tag` / `val_tag` argument, held **fixed** across arms
while the output-path tag varies. Gate **G3** (batch identity) must run **across processes**, not
within one. Nothing else in this document matters if this is got wrong.

### 0.2 The cost law's intercept is degenerate — my previous headline was overstated

The first draft leaned on `F = 9.0 ms` (from `467 = F + 32k`, `925 = F + 64k`) to rule out whole
classes of optimisation "rigorously". A verifier reproduced the fit exactly (`a = 9.00`,
`b = 22.9367`) **and then showed it is not identified**: the per-Mrow rates at the two points differ
by only 0.97 %, and every plausible fixed cost in the step — the numpy gather, `pin_memory`, the
15.4 MB H2D, and `TargetFeats`' three STFTs plus ERB matmuls on `y` — scales with **B**, not with
arms. A two-point fit at two different B cannot separate a B-proportional host/target cost from the
per-row term. **The intercept is statistically indistinguishable from 0 or from 40 ms.**

What survives: even at 40 ms the intercept is ~4 % of the step, so claims of 2.5–4× from launch-bound
reasoning (MPS, seven processes on one card) remain **refuted** — a verifier independently killed
`mps-seven-processes` on exactly this ground. What does *not* survive is my calling it "under 1 %,
forced by measurement". It is a fit, not a measurement, and the fan-out cost depends on it directly
(§4). **M5 (GPU-busy %) and a three-point fit at B = 16/32/64 must settle it on the instance.**

### 0.3 The optimisation is worth less than claimed

Roughly **200 ms of 925 (≈22 %)**, not the 30–50 % the first draft projected — and the surviving
wins are concentrated in two places, not five. The single largest is the layer-1 high-rate epilogue
(~84 ms, 9.1 %), and it **CHANGES MATH as written** (MEASURED, see T1.1) with a free repair.

**So the plan is:** fix the landmine, pin the right wheel, calibrate for $0.36, take the two real
kernel wins, and **run on one GPU** — because the fan-out does not reliably fit $60 (§4).

---

## 1. Where the time goes

Unchanged from the roofline probe except where a verifier corrected it.

| stage | ms | share | note |
|---|---|---|---|
| decoder layers 2–4 backward | 195 | 21 % | |
| decoder layers 2–4 forward | 127 | 14 % | |
| layer-1 gather backward (`scatter_add`) | 86 | 9 % | **contested** — see T3 `gather-to-index-select` |
| layer-1 HR epilogue (4 separate pointwise kernels) | 61 | 7 % | the best target |
| encoder backward | 53 | 6 % | low confidence, a guess at 30 % of roofline |
| layer-1 HR gather of `u`, strided | 48 | 5 % | |
| remaining 8 stages | 170 | 18 % | |
| unmodelled residual | 148 | 16 % | |

**Corrected:** `_mlp` (decoder layers 2–5) is **33–35 % of wall time, not 45–50 %** — re-derived by a
verifier from the repo's own measured kernels (`bias_addmm` 1.73 ms, wgrad `mm` 1.34 ms at 3.072 M
rows): GEMMs 28.8 ms + L5 3.6 ms + three unfused ReLUs 17.7 ms ≈ 50 ms per ES arm against the cost
law's 149.9 ms for a whole ES arm.

---

## 2. The plan, in tiers

### Tier 0 — do unconditionally

| id | what | gain | exactness | score |
|---|---|---|---|---|
| **T0.1 batch-stream isolation** | `batch_tag`/`val_tag` fixed across arms while the path tag varies (§0.1) | prevents a $55–75 silent write-off | correctness | — |
| **T0.2 `torch-pin-cu130`** | `nvidia-smi` decides the wheel; driver ≥ 580 → `torch==2.14.0+cu130` (cp311 matches the venv). Release wheels are **SASS-only, no PTX**, so `cu126` (arch list stops at sm_90) hard-fails | 0 ms — a precondition. Avoids a dead instance | exact | **70** |
| **T0.3 `calibrate-first`** | ~12 min, $0.36, before committing the budget | 0 ms — option value | exact | **55** |
| **T0.4 `blackwell-rate-gate` + `measure-first-launch-bound`** | measure ns/row and GPU-busy % on one card; settle §0.2 | 0 ms — insurance | exact | 34 / 32 |
| **T0.5 `g3-batch-identity` + `g2-rng-consumption-trace`** | the two gates that survived verification, run **across processes** | 0 ms — insurance | exact | 34 / 30 |
| **T0.6 `foreach-clip`** | `_foreach_norm`/`_foreach_mul_` for the per-arm clip | 0.8–1.5 ms (0.1 %) — take it for hygiene, not speed | equivalent | 6 |

### Tier 1 — the actual speedups

| id | what | revised gain | exactness | score |
|---|---|---|---|---|
| **T1.1 `fuse-layer1-hr-epilogue`** | fuse gather + `coord*w_c` + bias + ReLU | **~84 ms (9.1 %)** | **CHANGES MATH as written — MEASURED.** `codegen_upcast_to_fp32` is True in torch 2.14.0, so the fused chain computes in fp32: mean rel 2.2e-3, **15.4 % of elements differ**. Repair is free: explicit `.to(bfloat16)` at all three rounding points, verified with `torch.equal`, **not** `allclose` | 32 |
| **T1.2 GEMM epilogue family** | `gemm-epilogue-autotune` / `max-autotune-no-cudagraphs` / `relu-epilogue-addmm` — one mechanism, count once | 46–92 ms, conditional on a Triton template winning the sweep on sm_120 | equivalent | 20–24 |
| **T1.3 `fold-b1-into-subpixel-conv`** | layer-1 bias as the `conv1d` bias | ~14.5 ms at 1792 GB/s, ~19 ms at 1597 | equivalent (≤ 1 ulp) | 28 |
| **T1.4 `fuse-spec-loss` + `stft-fold-backward`** | fuse the spectrum→distances chain; `F.fold` for the STFT backward | 12–22 ms + 4–9 ms | equivalent | 30 |
| **T1.5 `coalesce-u-gather`** | make `u` contiguous before the HR gather | **10–30 ms (1–3 %)**, not the 110 ms claimed — the backward half of the claim (59 % of the gain) was verified false | equivalent | 18 |
| **T1.6 `compile-layer1`** | extend `torch.compile` to `_layer1` | fidelity-driven; perf unaudited | equivalent | 62 |

**Realistic total: ~200 ms of 925 → ≈720 ms on an A100-equivalent.**

### Tier 3 — refuted (26). The ones that were in the previous draft:

| refuted | why — all verified, not argued |
|---|---|
| **`gpu-resident-corpus`**, `corpus-in-vram` | The host gather is **not on the critical path**: the step loop has no per-step sync, so on 24 of every 25 steps the host runs ahead and the ~8 ms of numpy hides behind queued GPU work. MEASURED: 7.39 ms fancy-index vs a contiguous-slice alternative recovering 82 % **for two lines and no VRAM** |
| **`gather-to-index-select`**, `embedding-deterministic-bwd` | MEASURED false premise: `u` is **not** contiguous. Strides `(17280,17280,1,120)`, `is_contiguous False`; `.reshape` silently **copies**. The "144-wide contiguous row" index_select needs *does not exist in memory*; materialising it costs ~11.5 GB/step. And "46 GB of index traffic" is L1/L2, not DRAM — `expand` gives stride 0, so it is a warp-wide broadcast |
| **`fused-mlp-layers-2-5`**, `triton-fused-mlp-128-16-split`, `fused-layer1-triton`, `fuse-l1-epilogue` | sm_120 shared memory is **128 KB/SM, 99 KB max per block** — the proposal's "228 KB/SM" is CC 10.0 (B200). The 125 KB weight set never fit. Plus `pytorch#176426` (open): Triton kernels with ≥2 `tl.load()` segfault on this card |
| **`l2-row-blocking-recompute`** | `_mlp` is 33–35 % of wall time, not 45–50 %; the time translation was wrong |
| **`plan-a-one-arm-per-gpu`** | the §0.1 landmine sits in its own step 5, and its cost model takes a mean where billing takes a max |
| **`last-gemm-out-dtype-fp32`** | "the implementation does not run, and the guard does not catch it" |
| `mps-seven-processes`, `hostpipe-prefetch`, `persistent-pinned-doublebuffer`, `pinned-double-buffer-prefetch`, `deterministic-batch-table`, `shared-mmap-corpus`, `prebuilt-corpus-artifact`, `seed-currency`, `concurrency-decision-test`, `one-pad-one-power`, `checkpoint-at-lr-boundary`, `fp16-instead-of-bf16`, `no-bf16-spectrogram`, `triton-mm-relu-epilogue` | various; the recurring pattern is optimising something that is not on the critical path, or pricing L1/L2 traffic as DRAM |

**`compile-off-gate` survived (18) but its reasoning was gutted**, and this matters because the
previous draft made it Tier 0. Both its pillars are misreadings of `e2b_fast.py`: (1) the "no benefit
from COMPILE" line is the repo **diagnosing a bug** — Dynamo hit the recompile limit and silently
skipped, so that datum is *eager measured against eager*, and the next two lines apply the fix;
(2) the autotune log showing ATen beating Triton is from `max-autotune`, which is **already off**.
**The post-fix compiled path has never been benchmarked on any card.** So: do not default COMPILE off
on a false premise — gate it on M9 (the `#176426` crash test) and measure it.

---

## 3. CHANGES MATH — not permitted inside the seven arms

1. **T1.1 as written** (see above) — repairable to exact at zero cost. Verify with `torch.equal`.
2. `JITTER_PER_CELL = True` — stopwatch only.
3. Any larger global batch. Batch 64 is contractual (spec **I17**).
4. FP8/NVFP4; fp16 for bf16; stochastic rounding; reduced-precision spectrograms.
5. `shared-anchors`.
6. Sharding the loss without recomposing the spec's §3.3 recipe (the `sc` term does not average).

**CHANGES-TRAJECTORY, must be declared:** per-arm noise generators. Arms currently draw from a shared
global CUDA generator in stack order, and Philox offsets derive from `multiProcessorCount` (A100 108
SMs vs 188 here), so the realised noise was never portable. Give each arm its own generator with a
fixed per-arm key, for training **and** validation, and record it.

---

## 4. Multi-GPU — the fan-out probably does not fit $60

There is no 7-GPU SKU: only `gpuCount` 1 at $1.79/hr and 8 at $14.32/hr, exactly linear. Billing is
**GPUs × wall-clock**, and wall-clock is the **max** over arms, so every idle second on the six fast
cards is paid for. A verifier re-derived the cost with that correction, and with the degenerate
intercept (§0.2) as the free parameter:

| intercept | 7 cards | ×8/7 for the 8-GPU SKU |
|---|---|---|
| 9 ms (best case, the fitted value) | $54.76 | **$62.59** |
| 25 ms | $60.61 | ~$69 |
| 40 ms | $66.09 | **$75.53** |
| single card, all seven arms | **$48.27** | — |

`TargetFeats` alone — recomputed in all seven processes — is plausibly 2–5 ms of that intercept.

**Recommendation: one GPU, 27 h, $48.27 at A100-equivalent speed.** It fits the ceiling in every
bracket; the fan-out fits in none of them once the 8-GPU SKU multiplier is applied. Revisit only if
calibration (M4) shows Blackwell at the fast end *and* Tier 1 lands.

---

## 5. Measurement protocol — $0.36 before committing $60

M0 wheel/driver · M1 arch list and `get_device_properties` · M2 achievable bandwidth ·
M3 isolated `(6.1M,144)@(144,144)` bf16 GEMM · **M4 the step itself** · **M5 GPU-busy %** ·
**M4b a three-point fit at B = 16/32/64 to identify the intercept (§0.2)** · M6 profiler ranking ·
M7 `ncu` sector-per-request on the gather · M8 `ncu` DRAM bytes on the one `addmm` ·
**M9 the `#176426` Triton crash gate** · M10 fwd/bwd split.

Decision rule after M4: ≤ 700 ms → fan-out affordable, reconsider §4. 700–1050 ms → single GPU.
\> 1400 ms → stop and report.

---

## 6. Fidelity gates — G1 as specified is refuted

**G1 (CPU step-0 parity) FAILED verification, and the verifier ran it.** Booting `e2b_fast.py` as
`gate_subpixel.py` does (SMOKE, CPU, all flags off, B=4, L=512) and comparing the certified-equivalent
SUBPIXEL change against a **null** (identical code, noise seed 4321 vs 1234) in float64:

- null ‖Δg‖/‖g‖ = 1.33e-2 … 9.24e-2 — **six of seven arms fall below the proposed 3e-2 pass threshold**
- null 1−cos = 8.91e-5 … 4.27e-3 — again six of seven below the proposed 1e-3
- null relative loss error on `es_marg` = 2.08e-7 — **below the proposed 1e-6 bound**

A reference with the entire noise stream scrambled passes. **A gate whose accept band contains its own
null is not a detector; a green light from it is worse than no gate.** G1 must be redesigned so its
thresholds sit below the measured null, or dropped in favour of G2/G3.

Surviving gates: **G2** (RNG consumption trace, 30), **G3** (batch identity, 34 — and it must run
across processes, per §0.1). Plus the spec's hard pre-flight gate on `corpus.hours` (the
`e3_launch.py` trap).

---

## 7. Cost

| configuration | A100-equiv ms/step | GPU-h | $ |
|---|---|---|---|
| baseline, 1 GPU | 925 | 27.0 | **$48.27** |
| + Tier 1, 1 GPU | ~720 | 21.0 | **$37.6** |
| baseline, 8-GPU SKU (7 arms) | 150/arm | — | **$62.59–$75.53** |

Plus calibration $0.36, staging ~$0.6 on one GPU, evaluation 1–2 GPU-h ($2–4).
**Recommended path: one GPU, Tier 0 + Tier 1 → ~$40–52 all-in, inside the $60 ceiling.**
Multiply every row by the unmeasured Blackwell factor (bracketed 0.9–1.8×), which is what M4 settles.

---

## 8. What is still unverified

1. **The intercept** (§0.2) — degenerate; M4b settles it. The fan-out decision hangs on it.
2. **35 of 105 candidates were never verified** — escalation, synthesis and critique all died on the
   session limit. Everything in Tier 1 has had one verify pass, not the planned three.
3. **Encoder time (104 ms)** is a guess at "30 % of roofline" for skinny convs.
4. **Server vs Workstation Edition** — 1597 vs 1792 GB/s, a 12 % swing.
5. **Whether COMPILE helps at all** — never benchmarked post-fix on any card (§ Tier 3 note).
6. **Storage volumes appear bare-metal-only**; assume every rental re-stages.
