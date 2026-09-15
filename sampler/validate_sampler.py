"""Execute every code cell of lisa_rtm_sampler.ipynb on CPU in the SMOKE preset: synthetic corpus, no network.

A smoke test of the code path, not a scientific result.  The cache goes to $LISA_RTM_ROOT, or to a
fresh temporary directory when that is unset, so nothing lands in the repository.

    ./venv/bin/python validate_sampler.py
"""
import json
import os
import pathlib
import sys
import tempfile
import time

os.environ.setdefault("MPLBACKEND", "Agg")
import warnings
warnings.filterwarnings("ignore", message="FigureCanvasAgg")
os.environ["LISA_RTM_PRESET"] = "SMOKE"
os.environ.setdefault("LISA_RTM_ROOT", tempfile.mkdtemp(prefix="lisa_rtm_smoke_"))

NB = json.loads((pathlib.Path(__file__).parent / "lisa_rtm_sampler.ipynb").read_text())
SRC = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
NAMES = ["setup", "config", "data", "model", "gates", "objective", "metrics", "inference",
         "training", "evaluation", "latency", "listen", "summary"]
assert len(SRC) == len(NAMES), f"expected {len(NAMES)} code cells, notebook has {len(SRC)}"

# Outside a notebook, display() has nothing to render into; neutralise it.
import IPython.display as _disp
_disp.display = lambda *a, **k: None

G = {"__name__": "__main__"}
t_all = time.time()
for i, (name, src) in enumerate(zip(NAMES, SRC)):
    print(f"\n{'=' * 70}\n[{i}] {name}\n{'=' * 70}", flush=True)
    t0 = time.time()
    try:
        exec(compile(src, f"<cell {i} {name}>", "exec"), G)
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"\nFAILED at cell [{i}] {name}")
        sys.exit(1)
    print(f"--- [{i}] {name} done in {time.time() - t0:.1f}s", flush=True)

print(f"\n{'=' * 70}\nALL {len(SRC)} CELLS EXECUTED in {time.time() - t_all:.0f}s  (cache: {os.environ['LISA_RTM_ROOT']})")
