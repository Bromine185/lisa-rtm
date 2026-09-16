# ============================================================ OV3-6 step-time benchmark: sequential vs stacked
# Measures ms/step and peak memory for the seven arms under each configuration, so the launch picks a
# setting by measurement instead of by assumption.  Requires the training notebook's kernel (corpus, ARMS,
# time_ov3, time_ov3_fast, FAST).  Measured on an A100-80GB, 7 arms, 1 s segments, determinism OFF:
#   sequential batch 32  589 ms   13.2 GB   54.3 audio-s per compute-s
#   stacked GROUP_MAX=1 batch 32  467 ms    7.0 GB   68.5     <- best
#   stacked GROUP_MAX=2 / 7       slower at 2-4x the memory (grouped GEMMs tile-quantise badly)
import torch, gc, math

BENCH_BATCHES = globals().get("BENCH_BATCHES", (16, 32))
RES = []


def _bench(label, fn):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        dt = fn()
        mem = torch.cuda.max_memory_allocated() / 1e9
        RES.append((label, dt * 1000, mem))
        print("%-46s %8.0f ms/step  %5.1f GB" % (label, dt * 1000, mem), flush=True)
    except Exception as e:
        RES.append((label, float("nan"), float("nan")))
        print("%-46s FAILED  %s" % (label, repr(e)[:100]), flush=True)
    gc.collect(); torch.cuda.empty_cache()


_keep = dict(FAST)
for B in BENCH_BATCHES:
    _bench("sequential (e2)            batch %d" % B, lambda B=B: time_ov3(corpus, ARMS, B, n=8))
    for gm in (1, 2, 7):
        for st in (False, True):
            FAST["GROUP_MAX"], FAST["STREAMS"] = gm, st
            _bench("stacked GROUP_MAX=%d streams=%-5s batch %d" % (gm, st, B),
                   lambda B=B: time_ov3_fast(corpus, ARMS, B, n=8))
FAST.update(_keep)

ok = [r for r in RES if r[1] == r[1]]
print(flush=True)
print("| configuration | ms/step | peak GB | audio-s per compute-s |", flush=True)
print("|---|---|---|---|", flush=True)
for label, ms, mem in RES:
    B = int(label.split("batch")[1])
    thr = "-" if ms != ms else "%.1f" % (B * 1.0 / (ms / 1000))
    print("| %s | %s | %s | %s |" % (label, "OOM" if ms != ms else "%.0f" % ms, "-" if mem != mem else "%.1f" % mem, thr), flush=True)
if ok:
    best = max(ok, key=lambda r: int(r[0].split("batch")[1]) / r[1])
    B = int(best[0].split("batch")[1])
    print("\nBEST: %s  %.0f ms/step  %.1f audio-s per compute-s  (%.1f GB)"
          % (best[0], best[1], B / (best[1] / 1000), best[2]), flush=True)
print("BENCH DONE", flush=True)
