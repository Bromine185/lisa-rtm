"""Time each released model on an otherwise idle machine. Once sota/rtf.json exists the demo quotes it;
until then demo/tools/add_sota.py and make_results_sota.py fall back to the batch runs' _run.json wall clocks.

    lisa_rtm_cache/sota/sotaenv/bin/python sota/bench_rtf.py [--set core] [--n 8] [--repeats 3]

The batch runs' _run.json wall clocks are not a latency benchmark: they include file I/O, and some of them
ran beside the scorer. This loads each model once, warms it up on one utterance, then times `run()` (the
same function sota/run_models.py calls, so the same inference path) over the first n utterances of the
set, `repeats` times, and reports the median of the per-repeat RTF = compute seconds / audio seconds.
One process PER MODEL (the repos collide on module names such as utils.py, so run it once with
--models <m> for each model; results accumulate in rtf.json), nothing else running (the script refuses to start if another
run_models.py or score.py is alive). Devices as in the batch runs: AudioSR on MPS, the rest on CPU.

Writes sota/rtf.json: {model: {rtf, rtf_all, device, n_utts, audio_s, repeats, torch, machine, threads}}.
"""
import argparse, json, os, pathlib, platform, statistics, subprocess, sys, time

import numpy as np
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sota"))
import run_models as RM                                            # noqa: E402

DEVICE = {"flowhigh": "cpu", "apbwe": "cpu", "nuwave2": "cpu", "audiosr": "mps"}


def busy():
    out = subprocess.run(["pgrep", "-fl", "run_models.py|score.py|run_ours.py"], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if str(os.getpid()) not in l and "bench_rtf" not in l]


def machine():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return platform.processor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="core")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--models", nargs="*", default=list(DEVICE))
    ap.add_argument("--force", action="store_true", help="run even if other jobs are alive")
    a = ap.parse_args()
    b = busy()
    if b and not a.force:
        raise SystemExit("other jobs are running; RTF would be contended:\n  " + "\n  ".join(b))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    ws = RM.SOTA / "work" / a.set
    utts = json.loads((ws / "utts.json").read_text())["utts"][: a.n]
    xs = []
    for u in utts:
        x, _ = sf.read(str(ws / "lo12k" / f"{u['utt']}.wav"), dtype="float64")
        xs.append((u, x, int(round(u["seconds"] * RM.FS))))
    audio_s = sum(n for _, _, n in xs) / RM.FS
    out_p = REPO / "sota" / "rtf.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    import torch
    for m in a.models:
        cwd = os.getcwd()
        builder = {"flowhigh": RM.flowhigh, "apbwe": RM.apbwe, "nuwave2": RM.nuwave2, "audiosr": RM.audiosr}[m]
        run, _ = builder(DEVICE[m])
        os.chdir(cwd)
        u0, x0, n0 = xs[0]
        RM.RUN_CTX["truth"] = np.load(ws / "truth" / f"{u0['utt']}.npy"); RM.RUN_CTX["raw_peak"] = u0.get("raw_peak", 0.95)
        run(x0, n0, 0)                                               # warm-up: allocator, kernels, MPS graphs
        rtfs = []
        for r in range(a.repeats):
            t = 0.0
            for u, x, n in xs:
                RM.RUN_CTX["truth"] = np.load(ws / "truth" / f"{u['utt']}.npy"); RM.RUN_CTX["raw_peak"] = u.get("raw_peak", 0.95)
                if DEVICE[m] == "mps":
                    torch.mps.synchronize()
                t0 = time.perf_counter(); run(x, n, r)
                if DEVICE[m] == "mps":
                    torch.mps.synchronize()
                t += time.perf_counter() - t0
            rtfs.append(t / audio_s)
            print(f"  {m} repeat {r + 1}/{a.repeats}: RTF {rtfs[-1]:.4f}", flush=True)
        res[m] = {"rtf": round(statistics.median(rtfs), 4), "rtf_all": [round(v, 4) for v in rtfs], "device": DEVICE[m],
                  "n_utts": len(xs), "audio_s": round(audio_s, 2), "repeats": a.repeats, "torch": torch.__version__,
                  "machine": machine(), "threads": torch.get_num_threads(), "set": a.set,
                  "method": "median over repeats of (sum of run() wall time) / (audio seconds); one warm-up; idle machine"}
        print(f"{m}: RTF {res[m]['rtf']} on {DEVICE[m]}", flush=True)
        del run
        import gc; gc.collect()
        out_p.write_text(json.dumps(res, indent=1))
    print("->", out_p)


if __name__ == "__main__":
    main()
