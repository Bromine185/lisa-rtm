"""Boot the repo's model and metric code outside a notebook, on CPU, at the FULL config.

The definitions live in notebook cells and in `overnight2/c1_model.py` / `overnight3/e1_model.py`.
This execs exactly the cells the audit scripts need into one namespace, the same trick
`overnight3/smoke_ov3_local.py` uses, and returns it.

    from audit.boot import boot
    G = boot()            # G["CFG"], G["load_arm"], G["reconstruct"], G["band_energy_ratio"], ...
"""
import json, os, pathlib, tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]


def boot(preset="FULL", chdir=True):
    os.environ.setdefault("MPLBACKEND", "Agg")
    if chdir:
        os.chdir(tempfile.mkdtemp(prefix="audit_"))
    cells = ["".join(c["source"]) for c in json.loads((REPO / "lisa_rtm.ipynb").read_text())["cells"]
             if c["cell_type"] == "code"]
    G = {"__name__": "__main__"}
    run = lambda mark: exec(compile(next(s for s in cells if mark in s), f"<nb {mark[:18]}>", "exec"), G)
    run("import importlib, subprocess, sys")
    import torch
    G["DEVICE"] = torch.device("cpu")                       # after the setup cell, which may pick mps
    run("@dataclasses.dataclass(frozen=True)")
    G["CFG"] = G[preset]                                    # the cell picks SMOKE without CUDA
    run("def _hann(n):")                                    # stft / istft / logmag
    run("def snr_db(y, y_hat):")                            # snr_db, lsd_db, band_energy_ratio, crps
    run("VCTK_URL = ")                                      # decimate, load_utterance
    run("class LISAEncoder(nn.Module):")                    # LISA, LISADecoder, MultiScaleSTFTLoss
    exec(compile((REPO / "overnight2/c1_model.py").read_text(), "<c1>", "exec"), G)
    exec(compile((REPO / "overnight3/e1_model.py").read_text(), "<e1>", "exec"), G)
    return G
