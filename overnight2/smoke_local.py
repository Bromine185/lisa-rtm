"""CPU smoke test of overnight2/c1_model.py: shapes, losses, reconstruct, a 4-step paired train.
Uses the notebook's definition cells and a handful of utterances from the Drive test cache."""
import json, os, sys, pathlib, time
import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
REPO = pathlib.Path(__file__).resolve().parents[1]
DRIVE = pathlib.Path.home() / "Library/CloudStorage/GoogleDrive-naghavnarna@gmail.com/My Drive/lisa_rtm"
NB = json.loads((REPO / "lisa_rtm.ipynb").read_text())
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
G = {"__name__": "__main__"}
def run(mark):
    src = next(s for s in CODE if mark in s)
    exec(compile(src, f"<nb {mark[:20]}>", "exec"), G)
run("import importlib, subprocess, sys")
import torch
G["DEVICE"] = torch.device("cpu")
run("@dataclasses.dataclass(frozen=True)")
G["CFG"] = G["FULL"]
run("def _hann(n):"); run("class QuantileMap:"); run("def snr_db(y, y_hat):"); run("VCTK_URL = ")
run("class LISAEncoder(nn.Module):")
exec(open(REPO / "overnight/cell2_trainer.py").read().split("def train_paired")[0], G)   # GPUCorpus, naive_upsample, hb_deficit_of
exec("def save_ckpt" + open(REPO / "overnight/cell2_trainer.py").read().split("def save_ckpt")[1], G)
exec(open(REPO / "overnight/cell3_eval.py").read(), G)
torch.use_deterministic_algorithms(False)
exec(open(REPO / "overnight2/c1_model.py").read(), G)

with np.load(DRIVE / "fixtures/test_FULL.npz", allow_pickle=True) as d:
    utts = list(d["utts"])[:6]
CFG = G["CFG"]
corpus = G["GPUCorpus"](utts, CFG, seg_hi=12000)
rng = G["stream"]("smoke")
x, y = corpus.batch(rng, 2)
print("batch", x.shape, y.shape)
m = G["LISAS"](CFG)
print("params", sum(p.numel() for p in m.parameters()), "vs LISA", sum(p.numel() for p in G["LISA"](CFG).parameters()))
spec = G["MultiScaleSTFTLoss"](CFG.n_fft)
for kind, lam in [("det", 1e-2), ("det_split", 1e-2), ("es_marg", 1e-2), ("es_slice", 1e-2), ("es_wave", 0.0)]:
    t0 = time.time()
    loss, terms = G["arm_loss"](kind, lam, m, x, y, spec)
    loss.backward()
    print(f"{kind:10s} loss {loss.item():.5f} terms {terms}  [{time.time()-t0:.2f}s]")
    m.zero_grad()
# lowpass sanity: energy above fs_lo/2 must vanish
lp = G["lowpass"](y, CFG.upsample)
Y = torch.fft.rfft(lp); k = Y.shape[-1] // CFG.upsample
print("lowpass residual above cut:", float(Y[..., k + 1:].abs().max()), " below:", float(Y[..., :k].abs().max()))
# reconstruct at tau 0 / 1, determinism of seeds
yy = utts[0][:48000]
r0 = G["reconstruct"](m, yy, CFG, tau=0.0)
r1a = G["reconstruct"](m, yy, CFG, tau=1.0, seed=1)
r1b = G["reconstruct"](m, yy, CFG, tau=1.0, seed=1)
r1c = G["reconstruct"](m, yy, CFG, tau=1.0, seed=2)
print("reconstruct lens", len(r0), len(yy), "seed-repeat max diff", np.abs(r1a - r1b).max(), "seed-change diff", np.abs(r1a - r1c).max())
m.tau = 1.0
print("probe_metrics", G["probe_metrics"](m, yy, CFG, 0.0))
# 4-step paired training with checkpoints into the local cache
G["CKPT"] = pathlib.Path("/private/tmp/claude-501/-Users-raghavsharma-projects-lisa-rtm/447211de-99bc-4ad9-8fd5-0b6a6dea79f4/scratchpad/ckpt")
G["ROOT"] = G["CKPT"]
G["CKPT"].mkdir(parents=True, exist_ok=True)
arms = {"det": ("det", 1e-2), "es_slice": ("es_slice", 1e-2)}
models, hist = G["train_ov2"](corpus, arms, 4, 2, 1e-3, (0.5,), 0.5, 1e-3, 2, "SMOKE_OV2", yy)
m2, ck = G["load_arm"](G["CKPT"] / "SMOKE_OV2" / "es_slice.pt")
print("reloaded", ck["step"], ck["arm"], "tau", m2.tau)
print("SMOKE OK")
