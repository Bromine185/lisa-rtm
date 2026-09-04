"""Independent CPU evaluation of the XL4 arms on the ORIGINAL test_FULL.npz (Drive-synced).
Execs the notebook's own definition cells so metrics are identical to Colab's."""
import json, os, sys, pathlib, time, io
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
run("import importlib, subprocess, sys")      # setup
import torch
G["DEVICE"] = torch.device("cpu")
run("@dataclasses.dataclass(frozen=True)")     # config (SMOKE on CPU) ...
G["CFG"] = G["FULL"]                           # ... but we evaluate FULL-config checkpoints
run("def _hann(n):")                           # spectral primitives
run("def snr_db(y, y_hat):")                   # metrics
run("VCTK_URL = ")                             # data utils: decimate
run("class LISAEncoder(nn.Module):")           # model classes
exec(open(REPO / "overnight/cell2_trainer.py").read().split("def time_steps")[0], G)     # naive_upsample, hb_deficit_of, GPUCorpus
exec("class LISAFF" + open(REPO / "overnight/cell2_trainer.py").read().split("class LISAFF")[1].split("def make_arm")[0], G)
exec(open(REPO / "overnight/cell3_eval.py").read(), G)
sps, CFG, LISA, LISAFF = G["sps"], G["CFG"], G["LISA"], G["LISAFF"]
torch.use_deterministic_algorithms(False)

@torch.no_grad()
def reconstruct(model, y, cfg, chunk=1 << 15):
    model.eval()
    x_lo = torch.from_numpy(G["decimate"](y, cfg.upsample)).float()[None]
    n_out = x_lo.shape[1] * cfg.upsample
    out = [model(x_lo, j0=s, j1=min(s + chunk, n_out)).squeeze(0).numpy() for s in range(0, n_out, chunk)]
    y_hat = np.concatenate(out).astype(np.float64)
    return np.pad(y_hat, (0, max(0, len(y) - len(y_hat))))[:len(y)]
G["reconstruct"] = reconstruct

with np.load(DRIVE / "fixtures/test_FULL.npz", allow_pickle=True) as d:
    test_utts, test_spk = list(d["utts"]), list(d["speakers"])
print("test_FULL.npz:", len(test_utts), sorted(set(test_spk)), flush=True)
first12 = test_utts[:12]
spread36 = [u for i, u in enumerate(test_utts) if i % 40 < 12]   # 12 per speaker

arms = {"relu_l1e-3": "relu", "relu_l1e-2": "relu", "relu_l1e-1": "relu", "ff_l1e-3": "ff"}
OUT = {}
t0 = time.time()
for k, kind in arms.items():
    ck = torch.load(DRIVE / f"checkpoints/XL4_b64_1s/{k}.pt", map_location="cpu", weights_only=False)
    m = (LISAFF(CFG) if kind == "ff" else LISA(CFG))
    m.load_state_dict(ck["model"]); m.eval()
    OUT[k] = {"step": ck["step"],
              "first12": G["evaluate_model"](m, first12, 12, f"{k} first12", CFG),
              "spread36": G["evaluate_model"](m, spread36, 36, f"{k} spread36", CFG),
              "phase12": [float(v) for v in G["phase_swap"](m, first12, 12, f"{k} phase12", CFG)]}
    print(f"  [{time.time()-t0:.0f}s]", flush=True)
json.dump(OUT, open(REPO / "overnight/local_eval_XL4.json", "w"), indent=1)
print("LOCAL EVAL DONE")
