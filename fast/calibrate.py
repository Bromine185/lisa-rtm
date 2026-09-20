"""Calibration: ~12 minutes on ONE rented GPU, before committing the run.

    python fast/calibrate.py [--json out.json] [--rows 6144000]

Standalone on purpose -- no corpus, no repo model code, no staging.  It runs on a bare VM the minute
it boots, so a wrong wheel or a slow card is discovered for $0.36 instead of after 30 GPU-hours.

WHAT IT DECIDES.  Nothing in this repo has ever run on sm_120.  Two published predictions bracket
the Blackwell factor and they disagree by 2.5x:

  - roofline probe: the step is per-element/bandwidth bound, Blackwell is 0.9-1.0x the A100 (its
    1597 GB/s against the A100's 2039 is 0.78x, and the L2 and SM advantages only pay on the
    request-bound gather) -> ~$59 for the eight-arm run
  - Blackwell probe: the step is latency/occupancy bound, clock x SM count predicts 1.3-1.8x FASTER
    -> ~$35

At the pessimistic end (time x2.29) the run costs ~$136 against an account that CANNOT OVERDRAFT and
stops dead at $101.  So this is a feasibility test, not a cost optimisation.

MEASUREMENTS (plan section 5): M0 wheel/driver, M1 arch list and device properties, M2 achievable
bandwidth, M3 isolated GEMM plus the N=144 tile-quantisation check, M9 the Triton crash gate, and a
decoder proxy that times the dominant 37% of the real step without needing any data.
"""
import argparse
import json
import platform
import subprocess
import sys
import time

REF = {  # A100-80GB SXM reference points this repo's baseline was measured on
    "bw_peak_gbs": 2039.0,
    "bf16_tflops": 312.0,
    "step_ms_8arm": 995.0,        # cost law at 43.008 Mrows, the eight-arm set
    "steps": 104_950,
}
CARD = {  # RTX PRO 6000 Blackwell Server Edition, from NVIDIA's CUDA GPU table + datasheet
    "bw_peak_gbs": 1597.0,
    "bf16_tflops_dense": 480.0,   # "1 PFLOPS" in marketing is the 2:4-sparse figure
    "sm": 188,
    "cc": (12, 0),
}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"<failed: {e}>"


def m0_driver():
    """The single line that decides the wheel.  Release PyTorch wheels ship SASS only -- no PTX, no
    JIT fallback -- and cu126's arch list stops at sm_90, so torch==2.14.0+cu126 hard-fails on this
    card with 'no kernel image is available'.  CUDA 13.0 needs driver >= R580; cu128 tops out at
    torch 2.11.0, cu129 at 2.13.0."""
    out = sh("nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader")
    r = {"raw": out}
    if out and "<failed" not in out and "," in out:
        parts = [p.strip() for p in out.split(",")]
        r["name"], r["compute_cap"], r["driver"], r["memory"] = (parts + [""] * 4)[:4]
        try:
            major = int(str(r["driver"]).split(".")[0])
            r["driver_major"] = major
            if major >= 580:
                r["wheel"] = "torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130"
            elif major >= 575:
                r["wheel"] = "torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129"
            elif major >= 570:
                r["wheel"] = "torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128"
            else:
                r["wheel"] = f"NONE -- driver {major} is too old for any sm_120 wheel"
        except Exception:
            r["wheel"] = "unknown (could not parse driver version)"
    return r


def m1_device(torch):
    p = torch.cuda.get_device_properties(0)
    r = {"capability": list(torch.cuda.get_device_capability()),
         "arch_list": torch.cuda.get_arch_list(),
         "name": p.name, "sm": p.multi_processor_count,
         "total_gb": round(p.total_memory / 1e9, 1),
         "torch": torch.__version__, "cuda": torch.version.cuda}
    r["l2_mb"] = round(getattr(p, "L2_cache_size", 0) / 1e6, 1)
    cc = f"sm_{r['capability'][0]}{r['capability'][1]}"
    r["cubins_present"] = cc in r["arch_list"]
    # Do not be alarmed by (12,2): pytorch#157549 quotes that for this card, and 2.14's cu130 cubins
    # cover any 12.x device.  The thing that matters is that SOME sm_12x cubin is present.
    r["any_sm12x"] = any(a.startswith("sm_12") for a in r["arch_list"])
    return r


def _time(torch, fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters / 1000.0


def m2_bandwidth(torch):
    """Every millisecond prediction in the plan scales inversely with this number."""
    n = 1 << 29                                   # 512 Mi elements bf16 = 1.07 GB
    a = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    t = _time(torch, lambda: b.copy_(a))
    gbs = 2 * a.nbytes / t / 1e9                  # one read + one write
    del a, b
    torch.cuda.empty_cache()
    return {"copy_gbs": round(gbs, 1), "pct_of_peak": round(100 * gbs / CARD["bw_peak_gbs"], 1),
            "vs_a100_peak": round(gbs / REF["bw_peak_gbs"], 3)}


def m3_gemm(torch, rows):
    """The decoder's shape, and the tile-quantisation question.

    H = 144 = 9 x 16, so every power-of-two tiling wastes.  The repo's own autotune log has
    addmm(3072000x144, 144x144) at 1.73 ms while moving only 1.77 GB of necessary bytes -- 1.02 TB/s,
    exactly half of A100 peak -- which is consistent with cuBLAS splitting N=144 into 2-3 column
    tiles and re-reading the whole activation once per tile.  Timing 128 and 256 alongside 144
    settles it directly: if 144 is much worse per FLOP than its neighbours, the re-read is real.
    """
    out = {}
    for nwide in (128, 144, 256):
        try:
            a = torch.randn(rows, nwide, dtype=torch.bfloat16, device="cuda")
            w = torch.randn(nwide, nwide, dtype=torch.bfloat16, device="cuda")
            t = _time(torch, lambda: a @ w, warmup=3, iters=10)
            tf = 2 * rows * nwide * nwide / t / 1e12
            out[f"N{nwide}"] = {"ms": round(t * 1e3, 3), "tflops": round(tf, 1),
                                "pct_dense": round(100 * tf / CARD["bf16_tflops_dense"], 1)}
            del a, w
            torch.cuda.empty_cache()
        except RuntimeError as e:
            out[f"N{nwide}"] = {"error": repr(e)[:120]}
    if all("tflops" in v for v in out.values()):
        best = max(v["tflops"] for v in out.values())
        out["n144_penalty_vs_best"] = round(best / out["N144"]["tflops"], 3)
    return out


def m4_decoder_proxy(torch, rows):
    """Times the dominant 37% of the real step with no data and no repo code.

    The decoder is 4 GEMMs of (rows x 144) @ (144 x 144) with ReLU between, run at the OUTPUT rate.
    One ES arm at batch 64 is 2 x 64 x 48000 = 6.144M rows.  Forward+backward of exactly that, under
    bf16 autocast, is the single best standalone proxy for the Blackwell factor -- and unlike the
    full step it needs no corpus, so it runs before staging.
    """
    import torch.nn as nn
    net = nn.Sequential(nn.Linear(97, 144), nn.ReLU(), nn.Linear(144, 144), nn.ReLU(),
                        nn.Linear(144, 144), nn.ReLU(), nn.Linear(144, 144), nn.ReLU(),
                        nn.Linear(144, 1)).cuda()
    x = torch.randn(rows, 97, device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = net(x)
        y.float().sum().backward()
        for p in net.parameters():
            p.grad = None

    t = _time(torch, step, warmup=3, iters=8)
    flops = rows * (97 * 144 + 3 * 144 * 144 + 144) * 2 * 3      # fwd + bwd ~= 3x fwd
    peak = torch.cuda.max_memory_allocated() / 1e9
    del net, x
    torch.cuda.empty_cache()
    return {"rows": rows, "ms": round(t * 1e3, 1), "tflops": round(flops / t / 1e12, 1),
            "peak_gb": round(peak, 1),
            "mfu_pct": round(100 * flops / t / 1e12 / CARD["bf16_tflops_dense"], 2)}


def m9_triton_gate(torch, rows):
    """pytorch#176426 (OPEN): Triton kernels with >= 2 tl.load() calls segfault at runtime on
    sm_120 -- they compile clean and emit invalid code.  Every fused-kernel optimisation in the plan
    is nothing but tl.load calls, so this gate opens or closes that whole tier.

    Checks the realistic vehicle (Inductor via torch.compile on the decoder MLP) rather than a
    hand-written kernel, because that is what the trainer would actually use.  A crash here takes the
    process down, so the JSON is written incrementally by the caller -- see main().
    """
    import torch.nn as nn
    r = {"issue": "pytorch#176426", "rows": rows}
    try:
        net = nn.Sequential(nn.Linear(97, 144), nn.ReLU(), nn.Linear(144, 144), nn.ReLU(),
                            nn.Linear(144, 144)).cuda()
        x = torch.randn(rows // 8, 97, device="cuda")     # /8: this is a correctness gate, not a timing one
        with torch.no_grad():
            ref = net(x).float()
        c = torch.compile(net, dynamic=False)
        outs = []
        for _ in range(5):
            with torch.no_grad():
                outs.append(c(x).float())
        r["ran"] = True
        r["max_abs_diff"] = float(max((o - ref).abs().max().item() for o in outs))
        r["deterministic_across_calls"] = bool(all(torch.equal(outs[0], o) for o in outs[1:]))
        r["allclose"] = bool(r["max_abs_diff"] < 1e-2)
        r["verdict"] = "PASS -- compile is usable" if r["allclose"] else "FAIL -- compiled output differs"
        del net, x, c
        torch.cuda.empty_cache()
    except Exception as e:
        r["ran"] = False
        r["verdict"] = f"FAIL -- {type(e).__name__}: {repr(e)[:200]}"
    return r


def verdict(res):
    """Translate the decoder proxy into a Blackwell factor and a go/no-go against the account wall."""
    v = {}
    prox = res.get("m4_decoder_proxy", {})
    if "ms" not in prox:
        return {"status": "INCOMPLETE -- no proxy timing"}
    # The A100 reference for this exact proxy is the cost law's ES-arm figure minus its non-decoder
    # share: decoder layers 2-5 are 37% of the step (plan section 1), so 0.37 * 149.9 ms = 55.5 ms.
    a100_proxy_ms = 55.5
    factor = prox["ms"] / a100_proxy_ms
    est_step = REF["step_ms_8arm"] * factor
    hours_1gpu = REF["steps"] * est_step / 1000 / 3600
    v["blackwell_factor"] = round(factor, 3)
    v["est_step_ms_8arm"] = round(est_step, 1)
    v["est_1gpu_hours"] = round(hours_1gpu, 2)
    v["est_1gpu_usd"] = round(hours_1gpu * 1.79, 2)
    # 8 parallel rentals: 2 det arms at ~half the ES time, 6 ES arms, +0.3 h staging each
    es_h = REF["steps"] * (est_step * 6.144 / 43.008) / 1000 / 3600
    v["est_parallel_hours"] = round(es_h, 2)
    v["est_parallel_usd"] = round((2 * (es_h / 2 + 0.3) + 6 * (es_h + 0.3)) * 1.79, 2)
    cheapest = min(v["est_1gpu_usd"], v["est_parallel_usd"])
    v["cheapest_usd"] = cheapest
    # Both stops can apply at once, and they have different fixes -- "1.5x off prediction" means
    # something is broken (wrong wheel, PTX-JIT, thermal cap, virtualised PCIe) while "over the
    # account wall" means the plan is unaffordable even if nothing is broken.  Report every reason
    # that fires, so a reader does not act on the first and miss the second.
    stops = []
    if factor > 1.5:
        stops.append(f"{factor:.2f}x the A100 prediction (>1.5x) -- diagnose, do not buy through it")
    if cheapest > 101:
        stops.append(f"cheapest plan ${cheapest} exceeds the $101 account wall, which cannot overdraft")
    v["stop_reasons"] = stops
    if stops:
        v["status"] = "STOP -- " + "; ".join(stops)
    elif cheapest > 60:
        v["status"] = f"ASK -- ${cheapest} is over the $60 ceiling but under the account wall"
    else:
        v["status"] = f"GO -- ${cheapest}"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="calibration.json")
    ap.add_argument("--rows", type=int, default=6_144_000, help="one ES arm at batch 64")
    ap.add_argument("--wheel-only", action="store_true",
                    help="print the pip install line and exit; imports nothing, so it runs BEFORE torch exists")
    a = ap.parse_args()

    if a.wheel_only:
        d = m0_driver()
        print(d.get("wheel", "unknown"))
        return 0 if d.get("wheel", "").startswith("torch==") else 1

    res = {"host": platform.node(), "python": sys.version.split()[0], "m0_driver": m0_driver()}
    print("M0 driver/wheel:", json.dumps(res["m0_driver"], indent=1))

    def dump():
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)

    dump()
    try:
        import torch
    except ImportError as e:
        res["error"] = f"torch not installed: {e}"
        dump()
        print("torch missing -- install the wheel M0 names, then re-run")
        return 1
    if not torch.cuda.is_available():
        res["error"] = "no CUDA device"
        dump()
        print("no CUDA -- M1..M9 skipped")
        return 1

    rows = a.rows
    for name, fn in (("m1_device", lambda: m1_device(torch)),
                     ("m2_bandwidth", lambda: m2_bandwidth(torch)),
                     ("m3_gemm", lambda: m3_gemm(torch, rows // 2)),
                     ("m4_decoder_proxy", lambda: m4_decoder_proxy(torch, rows)),
                     ("m9_triton_gate", lambda: m9_triton_gate(torch, rows))):
        t0 = time.perf_counter()
        try:
            res[name] = fn()
        except Exception as e:                       # never lose earlier measurements to a later crash
            res[name] = {"error": f"{type(e).__name__}: {repr(e)[:200]}"}
        res[name + "_secs"] = round(time.perf_counter() - t0, 1)
        dump()                                       # incremental: a segfault in M9 keeps M0-M4
        print(f"{name}: {json.dumps(res[name])[:300]}")

    res["verdict"] = verdict(res)
    dump()
    print("\nVERDICT:", json.dumps(res["verdict"], indent=1))
    print(f"\nwritten to {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
