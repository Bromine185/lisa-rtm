"""CPU smoke of c3_eval.py and the notebook-ladder handoff (c4) on the tiny smoke checkpoints."""
import json, os, sys, pathlib, time, dataclasses
import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
REPO = pathlib.Path(__file__).resolve().parents[1]
DRIVE = pathlib.Path(os.environ.get("LISA_DRIVE", pathlib.Path.home() /
    "Library/CloudStorage/GoogleDrive-naghavnarna@gmail.com/My Drive/lisa_rtm"))  # fixtures; override off a Mac
SCR = pathlib.Path(os.environ.get("LISA_SCRATCH", REPO / "lisa_rtm_cache" / "smoke"))  # gitignored; override to taste
NB = json.loads((REPO / "lisa_rtm.ipynb").read_text())
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
G = {"__name__": "__main__"}
def run(mark):
    exec(compile(next(s for s in CODE if mark in s), f"<nb {mark[:20]}>", "exec"), G)
run("import importlib, subprocess, sys")
import torch
G["DEVICE"] = torch.device("cpu")
run("@dataclasses.dataclass(frozen=True)"); G["CFG"] = G["FULL"]
run("def _hann(n):"); run("class QuantileMap:"); run("def snr_db(y, y_hat):"); run("VCTK_URL = ")
run("class LISAEncoder(nn.Module):")
exec(open(REPO / "overnight/cell2_trainer.py").read().split("def train_paired")[0], G)
exec("def save_ckpt" + open(REPO / "overnight/cell2_trainer.py").read().split("def save_ckpt")[1], G)
exec(open(REPO / "overnight/cell3_eval.py").read(), G)
torch.use_deterministic_algorithms(False)
exec(open(REPO / "overnight2/c1_model.py").read(), G)

with np.load(DRIVE / "fixtures/test_FULL.npz", allow_pickle=True) as d:
    utts = [u[:48000 * 2] for u in list(d["utts"])[:14]]
G["test_utts"] = utts[:12]
G["paper_test_utts"] = utts[12:14] + utts[:2]
G["train_utts"] = utts[:4]; G["train_spk"] = ["p225"] * 4; G["test_spk"] = ["p236"] * 12
G["MANIFEST"] = {"train_speakers": ["p225"], "test_speakers": ["p236", "p237", "p238"]}
G["ROOT"] = SCR; G["CKPT"] = SCR / "ckpt"; G["FIGS"] = SCR / "figs"; G["FIGS"].mkdir(parents=True, exist_ok=True)
G["OV2_TAG"] = "SMOKE_OV2"; G["OV2"] = str(REPO)      # c4 reads Path(OV2)/lisa_rtm.ipynb
# shrink the eval for CPU
src = open(REPO / "overnight2/c3_eval.py").read()
src = src.replace("M_DRAWS = 16", "M_DRAWS = 3").replace("EVAL12 = [np.asarray(u, np.float64) for u in test_utts[:12]]",
                                                         "EVAL12 = [np.asarray(u, np.float64) for u in test_utts[:3]]")
src = src.replace("TAUS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]", "TAUS = [0.0, 1.0]")
src = src.replace('r["ours120"] = ensemble_eval(m, test_utts, 1, m.tau, f"{k} ours120 single")', 'r["ours120"] = ensemble_eval(m, test_utts[:2], 1, m.tau, f"{k} ours120 single")')
src = src.replace('r["paper100"] = ensemble_eval(m, PAPER100, 1, m.tau, f"{k} paper100 single")', 'r["paper100"] = ensemble_eval(m, PAPER100[:2], 1, m.tau, f"{k} paper100 single")')
src = src.replace('r["phase12"] = [float(v) for v in phase_swap(m, EVAL12, 12, f"{k} phase-swap")]', 'r["phase12"] = [float(v) for v in phase_swap(m, EVAL12, 2, f"{k} phase-swap")]')
src = src.replace("for ui in (0, 5, 10):", "for ui in (0,):")
src = src.replace("[ensemble_eval(models_ov2[k], EVAL12, 8, t,", "[ensemble_eval(models_ov2[k], EVAL12, 2, t,")
t0 = time.time()
exec(compile(src, "<c3>", "exec"), G)
print(f"c3 smoke ok [{time.time()-t0:.0f}s]")
# c4: ladder handoff (SMOKE-sized fit)
src4 = open(REPO / "overnight2/c4_ladder.py").read()
G["dataclasses"] = dataclasses
exec(compile(src4, "<c4>", "exec"), G)
print(f"c4 smoke ok [{time.time()-t0:.0f}s]  CRPS", G["CRPS"])
