"""Inference latency of every arm, measured on this machine, for the datasheet.

    python fast/bench_latency.py --src ~/lisa-results             # CPU, and MPS when present
    python fast/bench_latency.py --src ~/lisa-results --no-mps

WHAT IS MEASURED.  Batch 1, fp32, eager PyTorch.  "One pass" is a 12 kHz input tensor already on the
device to a 48 kHz output tensor on the host: sample_eps (samplers only), encode once, decode in
reconstruct()'s 32768-sample chunks.  It leaves out reconstruct()'s decimation of the 48 kHz truth,
which exists only because the evaluation starts from wideband audio, and the numpy round trip.  Three
lengths: a 20 ms frame (what a streaming build would issue), 1 s, and EVAL12's first utterance whole.
"Shipped" then times the evaluation's own pipeline on that utterance, through the same functions
e4_eval scores: one output or one draw + baseband passthrough; logmean16 + passthrough (16 draws, mean
of log|STFT|, phase of draw 0); mean16 + passthrough.  Median and p90 of repeated runs after warm-up.

WHY ARM BY ARM WHEN THE ARCHITECTURE IS SHARED.  Kind and lambda cannot change inference cost, and the
table shows that they do not.  Class can: LISASD draws four Gaussians per OUTPUT sample and multiplies
them in at the decoder's first layer, which is 1% of the MACs and measurably more of the wall time.
And the readout changes it 16x.  Those are the two facts a reader needs next to the quality columns.

WHAT IT IS NOT.  Not a deployment number: no export, no fused kernels, no int8, and the passthrough
step is scipy resample_poly plus two whole-utterance FFT brick-wall splits, which a deployed pipeline
would replace with a short filter.  Its cost is recorded separately so the model's share is visible.
The algorithmic lookahead (encoder receptive field plus the decoder's right-hand neighbour) is
computed from CFG, not measured.

Output: <src>/ov3/latency_<tag>.json, read by fast/datasheet.py.
"""
import argparse
import json
import os
import pathlib
import platform
import statistics as st
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fast.run_contract import RUN_TAG
from fast.train_arm import boot

REPO = pathlib.Path(__file__).resolve().parents[1]
EVAL_CELLS = (("def sample_batch(utts, cfg, rng):", "CKPT_PATH = "),)
ORDER = ["det_paper", "det", "es_marg", "es_dec_l0.01", "es_erb_l0.001", "es_erb_l0.01",
         "es_erb_l0.1", "es_dec_erb_l0.1"]
CHUNK = 1 << 15


def machine():
    m = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0],
         "cpu_count": os.cpu_count()}
    try:
        m["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=5).stdout.strip() or platform.processor()
    except Exception:
        m["cpu"] = platform.processor()
    return m


def timeit(fn, n, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return st.median(ts), sorted(ts)[int(0.9 * (n - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/lisa-results")
    ap.add_argument("--tag", default=RUN_TAG)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="CPU threads; the default is what fast/run_e4_eval.py uses")
    ap.add_argument("--no-mps", action="store_true")
    ap.add_argument("--utt", type=int, default=0, help="index into test_utts; 0 is EVAL12's first utterance")
    a = ap.parse_args()
    src = pathlib.Path(a.src).expanduser().resolve()

    cwd = pathlib.Path.cwd()
    os.chdir(src)
    try:
        G = boot(device="cpu", root=src, cells=EVAL_CELLS)
    finally:
        os.chdir(cwd)
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    CFG = G["CFG"]
    from fast.vctk_local import test_utts
    utts, names = test_utts(per_speaker=40, cache=src / "vctk_test", verbose=False)
    G["test_utts"], G["test_names"] = utts, names
    G["OV3_TAG"], G["OV3_M"], G["OV3_RUN_EVAL"] = a.tag, 16, False
    exec(compile((REPO / "overnight3/e4_eval.py").read_text(), "<e4_eval.py>", "exec"), G)
    models, arms = G["load_ov3"](a.tag)
    decimate, reconstruct, passthrough = G["decimate"], G["reconstruct"], G["passthrough"]
    logmag_ensemble_readout = G["logmag_ensemble_readout"]
    order = [k for k in ORDER if k in models] + sorted(set(models) - set(ORDER))

    # ---- the architecture's cost, from CFG ---------------------------------------------------------
    C, Hd, R = CFG.enc_channels[-1], CFG.dec_hidden, CFG.upsample
    enc, cin = 0, 1 + G["N_NOISE"]
    for c, k in zip(CFG.enc_channels, CFG.enc_kernels):
        enc += cin * c * k
        cin = c
    macs = {"enc": enc * CFG.fs_lo,                                   # convs at the input rate
            "dec1": 3 * C * Hd * CFG.fs_lo + Hd * CFG.fs_hi,             # sub-pixel layer 1 (e1_model) + coord term
            "dec_rest": (Hd * Hd * (CFG.dec_layers - 2) + Hd) * CFG.fs_hi,
            "dec_noise": G["N_DEC"] * Hd * CFG.fs_hi}                    # LISASD only
    lookahead = sum(k // 2 for k in CFG.enc_kernels) + 1
    info = {k: {"params": sum(p.numel() for p in models[k].parameters()), "cls": type(models[k]).__name__,
                "tau": models[k].tau, "kind": arms[k][0], "lam": arms[k][1]} for k in order}
    print(f"MACs per audio second: LISAS {(macs['enc'] + macs['dec1'] + macs['dec_rest']) / 1e9:.2f} G, "
          f"LISASD +{macs['dec_noise'] / 1e6:.0f} M; decoder layers 2-5 are "
          f"{100 * macs['dec_rest'] / (macs['enc'] + macs['dec1'] + macs['dec_rest']):.0f}%.  "
          f"Lookahead {lookahead} input samples = {1e3 * lookahead / CFG.fs_lo:.2f} ms.", flush=True)

    # ---- one pass, three lengths, each device --------------------------------------------------------
    @torch.no_grad()
    def infer(m, x_lo, tau, seed):
        eps = None if tau == 0 else m.sample_eps(x_lo, tau, seed)
        z = m.encode(x_lo, eps)
        n_out = x_lo.shape[1] * m.R
        return torch.cat([m.decode(z, s, min(s + CHUNK, n_out)) for s in range(0, n_out, CHUNK)], 1).cpu()

    y0 = np.asarray(utts[a.utt], np.float64)
    x0 = decimate(y0, R).astype(np.float32)
    dur = len(y0) / CFG.fs_hi
    utt_key = f"utt_{dur:.2f}s"
    segs = {"20ms": x0[: CFG.fs_lo // 50], "1s": x0[: CFG.fs_lo], utt_key: x0}
    devices = ["cpu"] + (["mps"] if (not a.no_mps and torch.backends.mps.is_available()) else [])
    LAT = {k: {} for k in order}
    for dev in devices:
        for k in order:
            m = models[k].to(dev).eval()
            tau = m.tau
            for seg, x in segs.items():
                xt = torch.from_numpy(x)[None].to(dev)

                def fn():
                    out = infer(m, xt, tau, 0)
                    if dev == "mps":
                        torch.mps.synchronize()
                    return out
                n = 30 if seg == "20ms" else (20 if seg == "1s" else 10)
                med, p90 = timeit(fn, n)
                sec = len(x) / CFG.fs_lo
                LAT[k][f"{dev}/{seg}"] = {"median_ms": 1e3 * med, "p90_ms": 1e3 * p90, "audio_s": sec, "rtf": med / sec}
                print(f"  {dev:3s} {k:16s} {seg:10s} median {1e3 * med:8.2f} ms  p90 {1e3 * p90:8.2f}  RTF {med / sec:.4f}", flush=True)
            models[k] = m.to("cpu")
    torch.set_num_threads(a.threads)

    # ---- the shipped pipelines, on CPU, through the eval's own functions -----------------------------
    G["DEVICE"] = torch.device("cpu")
    pt_med, _ = timeit(lambda: passthrough(y0, y0), 5)
    print(f"  passthrough alone (naive upsample + brick-wall split): {1e3 * pt_med:.1f} ms", flush=True)
    for k in order:
        m = models[k]
        if m.tau == 0:
            med, _ = timeit(lambda: passthrough(y0, reconstruct(m, y0, CFG, tau=0.0, seed=0)), 5, warm=1)
            LAT[k]["cpu/shipped_one_pt"] = {"median_ms": 1e3 * med, "rtf": med / dur}
            print(f"  {k:16s} shipped one output + pt   {1e3 * med:8.1f} ms  RTF {med / dur:.3f}", flush=True)
            continue

        def one():
            return passthrough(y0, reconstruct(m, y0, CFG, tau=1.0, seed=0))

        def draws():
            return np.stack([reconstruct(m, y0, CFG, tau=1.0, seed=s) for s in range(16)])

        def lm16():
            return logmag_ensemble_readout(draws(), y0, CFG, passthrough=True)

        def mean16():
            return passthrough(y0, draws().mean(0))
        for lab, fn, n in (("shipped_one_pt", one, 5), ("shipped_logmean16_pt", lm16, 3), ("shipped_mean16_pt", mean16, 3)):
            med, _ = timeit(fn, n, warm=1)
            LAT[k][f"cpu/{lab}"] = {"median_ms": 1e3 * med, "rtf": med / dur}
            print(f"  {k:16s} {lab:22s} {1e3 * med:8.1f} ms  RTF {med / dur:.3f}", flush=True)

    out = src / "ov3" / f"latency_{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"machine": machine(), "torch": torch.__version__, "threads": a.threads, "devices": devices,
               "utt": {"index": a.utt, "name": os.path.basename(names[a.utt]), "seconds": dur, "key": utt_key},
               "chunk": CHUNK, "macs": macs, "lookahead_samples_12k": lookahead, "fs_lo": CFG.fs_lo,
               "passthrough_ms": 1e3 * pt_med, "info": info, "latency": LAT}, open(out, "w"), indent=1)
    print(f"\n-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
