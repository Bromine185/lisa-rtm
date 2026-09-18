import json, os, pathlib, time, numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
REPO = pathlib.Path(__file__).resolve().parents[1]
DRIVE = pathlib.Path(os.environ.get("LISA_DRIVE", pathlib.Path.home() /
    "Library/CloudStorage/GoogleDrive-naghavnarna@gmail.com/My Drive/lisa_rtm"))  # fixtures; override off a Mac
NB = json.loads((REPO / "lisa_rtm.ipynb").read_text()); CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
G = {"__name__": "__main__"}
def run(mark): exec(compile(next(s for s in CODE if mark in s), "<nb>", "exec"), G)
run("import importlib, subprocess, sys"); import torch; G["DEVICE"] = torch.device("cpu")
run("@dataclasses.dataclass(frozen=True)"); G["CFG"] = G["FULL"]
run("def _hann(n):"); run("class QuantileMap:"); run("def snr_db(y, y_hat):"); run("VCTK_URL = "); run("class LISAEncoder(nn.Module):")
exec(open(REPO / "overnight/cell2_trainer.py").read().split("def train_paired")[0], G)
exec("def save_ckpt" + open(REPO / "overnight/cell2_trainer.py").read().split("def save_ckpt")[1], G)
exec(open(REPO / "overnight/cell3_eval.py").read(), G)
torch.use_deterministic_algorithms(False)
exec(open(REPO / "overnight2/c1_model.py").read(), G)
exec(open(REPO / "overnight2/c1b_wide.py").read(), G)
with np.load(DRIVE / "fixtures/test_FULL.npz", allow_pickle=True) as d: utts = list(d["utts"])[:4]
CFG = G["CFG"]; corpus = G["HostCorpus"](utts, CFG, seg_hi=12000); rng = G["stream"]("smoke")
x, y = corpus.batch(rng, 2); m = G["LISASW"](CFG); spec = G["MultiScaleSTFTLoss"](CFG.n_fft)
for kind, lam in [("det", 1e-2), ("es_marg", 1e-2)]:
    loss, terms = G["arm_loss"](kind, lam, m, x, y, spec); loss.backward(); print(kind, loss.item(), terms); m.zero_grad()
yy = utts[0][:48000]; r0 = G["reconstruct"](m, yy, CFG, tau=0.0); r1 = G["reconstruct"](m, yy, CFG, tau=1.0, seed=1)
print("reconstruct", len(r0), np.abs(r0 - r1).max())
SCR = pathlib.Path(os.environ.get("LISA_SCRATCH", REPO / "lisa_rtm_cache" / "smoke"))  # gitignored; override to taste
G["CKPT"] = SCR / "ckpt"; G["ROOT"] = SCR; G["CKPT"].mkdir(parents=True, exist_ok=True); G["MODEL_CLS"] = G["LISASW"]
models, hist = G["train_ov2"](corpus, {"wide_det": ("det", 1e-2), "wide_es_marg": ("es_marg", 1e-2)}, 2, 2, 1e-3, (0.5,), 0.5, 1e-3, 1, "SMOKE_WIDE", yy)
m2, ck = G["load_arm"](G["CKPT"] / "SMOKE_WIDE" / "wide_es_marg.pt"); print("reloaded", type(m2).__name__, ck["cls"], ck["step"], "tau", m2.tau)
# latency of the wide model on CPU, 1 s
import time; xl = torch.randn(1, 12000); eps = m2.sample_eps(xl, 1.0, 0)
with torch.no_grad():
    for _ in range(3): m2(xl, eps=eps)
    t0 = time.perf_counter(); [m2(xl, eps=eps) for _ in range(10)]; print(f"wide CPU latency 1 s audio: {(time.perf_counter()-t0)*100:.1f} ms")
print("SMOKE WIDE OK")
