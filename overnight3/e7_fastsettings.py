# ============================================================ OV3-7 training settings + re-benchmark
# The notebook's setup cell calls seed_everything(), which leaves torch.use_deterministic_algorithms(True,
# warn_only=True) and cudnn.benchmark=False in force.  The decoder gathers three latents per OUTPUT sample,
# so its backward is scatter_add_, and the deterministic implementation of that replaces the fused atomic
# kernel with a radix sort over ~147M keys per arm-draw.  overnight/cell2_trainer.py (the old, faster boot
# path) turned determinism OFF; the same sequential trainer measured 587 ms/step at batch 32 there and
# 2261 ms/step here.  Measured after this cell: 326 ms (batch 16), 589 ms (batch 32), 1167 ms (batch 64).
import torch, gc

torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("training settings: deterministic OFF, cudnn.benchmark ON, TF32 ON", flush=True)

RES7 = []


def _b(label, fn):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        dt = fn()
        mem = torch.cuda.max_memory_allocated() / 1e9
        RES7.append((label, dt * 1000, mem))
        print("%-44s %8.0f ms/step  %5.1f GB" % (label, dt * 1000, mem), flush=True)
    except Exception as e:
        RES7.append((label, float("nan"), float("nan")))
        print("%-44s FAILED %s" % (label, repr(e)[:100]), flush=True)
    gc.collect(); torch.cuda.empty_cache()


_keep = dict(FAST)
for B in (16, 32, 64):
    _b("sequential (e2)            batch %d" % B, lambda B=B: time_ov3(corpus, ARMS, B, n=8))
    FAST["GROUP_MAX"], FAST["STREAMS"] = 1, False
    _b("stacked GROUP_MAX=1        batch %d" % B, lambda B=B: time_ov3_fast(corpus, ARMS, B, n=8))
FAST.update(_keep)

print(flush=True)
print("| configuration | ms/step | peak GB | audio-s per compute-s |", flush=True)
print("|---|---|---|---|", flush=True)
for label, ms, mem in RES7:
    B = int(label.split("batch")[1])
    print("| %s | %s | %s | %s |" % (label, "OOM" if ms != ms else "%.0f" % ms,
                                     "-" if mem != mem else "%.1f" % mem,
                                     "-" if ms != ms else "%.1f" % (B / (ms / 1000))), flush=True)
ok = [r for r in RES7 if r[1] == r[1]]
if ok:
    best = max(ok, key=lambda r: int(r[0].split("batch")[1]) / r[1])
    B = int(best[0].split("batch")[1])
    print("\nBEST: %s  %.0f ms/step  %.1f audio-s per compute-s  (%.1f GB)"
          % (best[0].strip(), best[1], B / (best[1] / 1000), best[2]), flush=True)
print("SETTINGS BENCH DONE", flush=True)
