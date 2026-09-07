# ============================================================ OV2-0 boot
# Load the notebook's DEFINITION cells (no data fetch, no training) plus the XL corpus/trainer/eval
# cells, all from Drive.  Idempotent: safe to re-run after a kernel restart.
import json, sys
from pathlib import Path

OV2 = Path("/content/drive/MyDrive/lisa_rtm/ov2")
REPO_NB = OV2 / "lisa_rtm.ipynb"
_nb = json.loads(REPO_NB.read_text())
_code = ["".join(c["source"]) for c in _nb["cells"] if c["cell_type"] == "code"]

def _run(mark, label):
    src = next(s for s in _code if mark in s)
    print(f"--- notebook cell: {label}", flush=True)
    exec(compile(src, f"<nb {label}>", "exec"), globals())

_run("import importlib, subprocess, sys", "setup")          # Drive mount, ROOT/CKPT/FIGS, stream(), seeds
_run("@dataclasses.dataclass(frozen=True)", "config")      # CFG = FULL on a GPU
_run("def _hann(n):", "spectral")
_run("class QuantileMap:", "transport")
_run("def snr_db(y, y_hat):", "metrics")
_run("VCTK_URL = ", "data-utils")                           # decimate(); nothing is fetched
_run("class LISAEncoder(nn.Module):", "model")             # LISA, MultiScaleSTFTLoss

XL = Path("/content/drive/MyDrive/lisa_rtm/ov2/xl")
for f in ("cell1_corpus.py", "cell2_trainer.py", "cell3_eval.py"):
    print(f"--- xl cell: {f}", flush=True)
    exec(compile((XL / f).read_text(), f"<xl {f}>", "exec"), globals())
print("BOOT DONE  corpus:", len(train_utts), "train utts", flush=True)
