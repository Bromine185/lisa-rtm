"""Inference latency of the stochastic LISA (same architecture as LISA + 8 noise input channels) on the
Mac CPU and MPS, for 1 s of audio and for a 20 ms streaming chunk.  Weights are random: latency does
not depend on them."""
import json, os, pathlib, time, sys
import numpy as np, torch
os.environ.setdefault("MPLBACKEND", "Agg")
REPO = pathlib.Path(__file__).resolve().parents[1]
NB = json.loads((REPO / "lisa_rtm.ipynb").read_text())
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
G = {"__name__": "__main__"}
def run(mark):
    exec(compile(next(s for s in CODE if mark in s), "<nb>", "exec"), G)
run("import importlib, subprocess, sys"); G["DEVICE"] = torch.device("cpu")
run("@dataclasses.dataclass(frozen=True)"); G["CFG"] = G["FULL"]
run("def _hann(n):"); run("class QuantileMap:"); run("def snr_db(y, y_hat):"); run("VCTK_URL = ")
run("class LISAEncoder(nn.Module):")
exec(open(REPO / "overnight/cell2_trainer.py").read().split("def train_paired")[0], G)
exec("def save_ckpt" + open(REPO / "overnight/cell2_trainer.py").read().split("def save_ckpt")[1], G)
exec(open(REPO / "overnight/cell3_eval.py").read(), G)
exec(open(REPO / "overnight2/c1_model.py").read(), G)
CFG = G["CFG"]
torch.set_num_threads(os.cpu_count())

def bench(fn, n_rep=30, warmup=5, sync=None):
    for _ in range(warmup):
        fn()
    if sync: sync()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter(); fn()
        if sync: sync()
        ts.append(time.perf_counter() - t0)
    return np.percentile(ts, 50) * 1e3, np.percentile(ts, 95) * 1e3

OUT = {}
devices = [("cpu", torch.device("cpu"))]
if torch.backends.mps.is_available():
    devices.append(("mps", torch.device("mps")))
for dname, dev in devices:
    m = G["LISAS"](CFG).to(dev).eval()
    for label, n in [("1 s", CFG.fs_hi), ("100 ms", CFG.fs_hi // 10), ("20 ms", CFG.fs_hi // 50)]:
        x_lo = torch.randn(1, n // CFG.upsample, device=dev)
        eps = m.sample_eps(x_lo, 1.0, 0)
        with torch.no_grad():
            sync = (lambda: torch.mps.synchronize()) if dname == "mps" else None
            p50, p95 = bench(lambda: m(x_lo, eps=eps), sync=sync)
        OUT[f"{dname}/{label}"] = (p50, p95)
        print(f"{dname:4s} {label:>7s} of audio: p50 {p50:8.2f} ms  p95 {p95:8.2f} ms   ({n/CFG.fs_hi*1000/p50:6.1f}x real time)", flush=True)
json.dump(OUT, open(REPO / "overnight2/cpu_latency.json", "w"), indent=1)
print("threads", torch.get_num_threads(), "| CPU:", os.popen("sysctl -n machdep.cpu.brand_string").read().strip())
