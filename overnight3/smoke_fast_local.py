"""CPU smoke test of the stacked trainer + dashboard callbacks (overnight3/e2b_fast.py).

Boots the notebook's definition cells exactly as smoke_ov3_local.py does, builds a synthetic corpus at SMOKE
dims, and runs train_ov3_fast for a few steps on the seven OV3 arms (both classes, so two stacks) with the
dashboard's on_step / on_epoch wired in.  Checks the callback payload, the history keys, the per-arm logging
index, and that the checkpoints reload.  No network, no Drive, no GPU.

    venv/bin/python overnight3/smoke_fast_local.py
"""
import json, os, sys, pathlib, time, tempfile, math
import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
REPO = pathlib.Path(__file__).resolve().parents[1]
OV3 = REPO / "overnight3"
SCR = pathlib.Path(tempfile.mkdtemp(prefix="ov3_fast_"))
os.chdir(SCR)
T0 = time.time()

NB = json.loads((REPO / "lisa_rtm.ipynb").read_text())
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
G = {"__name__": "__main__"}
def run(mark):
    exec(compile(next(s for s in CODE if mark in s), f"<nb {mark[:20]}>", "exec"), G)
run("import importlib, subprocess, sys")
import torch
G["DEVICE"] = torch.device("cpu")
run("@dataclasses.dataclass(frozen=True)"); G["CFG"] = G["SMOKE"]
run("def _hann(n):"); run("class QuantileMap:"); run("def snr_db(y, y_hat):"); run("VCTK_URL = ")
run("class LISAEncoder(nn.Module):")
_xl = open(REPO / "overnight/cell2_trainer.py").read()
exec(_xl.split("def train_paired")[0], G)
exec("def save_ckpt" + _xl.split("def save_ckpt")[1], G)
exec(open(REPO / "overnight/cell3_eval.py").read(), G)
torch.use_deterministic_algorithms(False)
torch.set_num_threads(max(1, os.cpu_count() // 2))
G["ROOT"] = SCR; G["CKPT"] = SCR / "ckpt"; G["FIGS"] = SCR / "figs"
for d in (G["CKPT"], G["FIGS"]):
    d.mkdir(parents=True, exist_ok=True)
exec(compile(open(REPO / "overnight2/c1_model.py").read(), "<c1>", "exec"), G)
exec(compile(open(OV3 / "e1_model.py").read(), "<e1>", "exec"), G)
exec(compile(open(OV3 / "e2_trainer.py").read(), "<e2>", "exec"), G)
exec(compile(open(OV3 / "e2b_fast.py").read(), "<e2b>", "exec"), G)
CFG = G["CFG"]
print(f"boot ok: CFG {CFG.name} fs {CFG.fs_lo}->{CFG.fs_hi}  [{time.time()-T0:.0f}s]", flush=True)

# ---- synthetic corpus ------------------------------------------------------------------------------------
_v = open(REPO / "validate.py").read()
_gen = "def synthetic_corpus" + _v.split("def synthetic_corpus")[1].split("\nfailures = []")[0]
_ns = {"G": G}
exec(_gen, _ns)
_ns["synthetic_corpus"](None)
train_utts, test_utts = G["train_utts"], G["test_utts"]
SEG, BATCH = 4096, 4
corpus = G["HostCorpus"](train_utts, CFG, seg_hi=SEG)
val_corpus = G["HostCorpus"](test_utts, CFG, seg_hi=SEG)

# ---- the seven OV3 arms (both classes -> two stacks) -----------------------------------------------------
ARMS = {
    "det":             ("det",           1e-2, "LISAS"),
    "es_marg":         ("es_marg",       1e-2, "LISAS"),
    "es_marg_l0.1":    ("es_marg",       1e-1, "LISAS"),
    "es_split_l0.1":   ("es_split_marg", 1e-1, "LISAS"),
    "es_erb_l0.1":     ("es_marg_erb",   1e-1, "LISAS"),
    "es_dec_l0.1":     ("es_marg",       1e-1, "LISASD"),
    "es_dec_erb_l0.1": ("es_marg_erb",   1e-1, "LISASD"),
}
TAG = "SMOKE_FAST"
seen = {"step": [], "epoch": []}


def on_step(info):
    seen["step"].append(info)


def on_epoch(info):
    seen["epoch"].append(info)


STEPS, SPE = 12, 4
models, hist = G["train_ov3_fast"](corpus, val_corpus, ARMS, STEPS, BATCH, 1e-3, (0.5,), 0.5, 1e-3, 6, TAG,
                                   test_utts[0], val_every=6, n_val_batches=2, log_every=2,
                                   on_step=on_step, on_epoch=on_epoch, steps_per_epoch=SPE)
print(f"trained {STEPS} steps  [{time.time()-T0:.0f}s]", flush=True)

# ---- the callback payload --------------------------------------------------------------------------------
assert seen["step"], "on_step never fired"
assert seen["epoch"], "on_epoch never fired"
i = seen["step"][-1]
for k in ("step", "steps", "frac", "epoch", "epochs_total", "t_elapsed", "ms_per_step", "eta_s",
          "samples_per_s", "audio_s_per_s", "gpu", "arms", "lr", "batch", "seg_s"):
    assert k in i, f"info is missing {k}"
for k in ("alloc_gb", "reserved_gb", "peak_gb", "total_gb", "free_gb", "util_pct"):
    assert k in i["gpu"], f"info['gpu'] is missing {k}"
assert set(i["arms"]) == set(ARMS), (set(i["arms"]), set(ARMS))
for k, r in i["arms"].items():
    for kk in ("loss_ema", "wave", "spec", "spread", "val_loss", "val_wave", "val_spec", "snr0", "def0", "snr1", "def1"):
        assert kk in r, f"{k} is missing {kk}"
    assert math.isfinite(r["loss_ema"]) and math.isfinite(r["wave"]), (k, r)
    assert r["val_loss"] is not None and math.isfinite(r["val_loss"]), (k, r["val_loss"])
    assert r["snr0"] is not None, (k, "no probe metrics by the last callback")
    assert (r["snr1"] is None) == k.startswith("det"), (k, r["snr1"])
assert abs(i["epoch"] - i["step"] / SPE) < 1e-9 and abs(i["epochs_total"] - STEPS / SPE) < 1e-9
print("callback payload ok: %d on_step, %d on_epoch; last epoch %.2f/%.2f, %.0f ms/step, gpu total %.1f GB"
      % (len(seen["step"]), len(seen["epoch"]), i["epoch"], i["epochs_total"], i["ms_per_step"], i["gpu"]["total_gb"]),
      flush=True)

# ---- per-arm logging index: every arm must get its OWN terms (the flush() bug) -----------------------------
losses = {k: i["arms"][k]["loss_ema"] for k in ARMS}
assert len(set(round(v, 9) for v in losses.values())) > 1, f"all arms report the same loss: {losses}"
lm = {k: hist[k]["wave"][-1] for k in ARMS}
assert len(set(round(v, 9) for v in lm.values())) > 1, f"history wave identical across arms: {lm}"
print("per-arm terms distinct:", {k: round(v, 4) for k, v in losses.items()}, flush=True)

# ---- history / checkpoints -------------------------------------------------------------------------------
for k in ARMS:
    h = hist[k]
    for key in ("step", "wave", "spec", "lr", "val_step", "val_loss", "train_loss_ema", "dev_step", "snr0", "def0"):
        assert h[key] and all(v is None or math.isfinite(v) for v in h[key]), (k, key, h[key])
    assert h["val_step"] == [6, 12], (k, h["val_step"])
    assert h["dev_step"] == [6, 12], (k, h["dev_step"])
assert (SCR / f"ov3_history_{TAG}.json").exists() and (G["FIGS"] / f"ov3_curves_{TAG}.png").exists()
json.load(open(SCR / f"ov3_history_{TAG}.json"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
print("history JSON strict-parses; curves PNG written", flush=True)

for k in ("es_erb_l0.1", "es_dec_erb_l0.1", "det"):
    m2, ck = G["load_arm"](G["CKPT"] / TAG / f"{k}.pt")
    assert type(m2).__name__ == ARMS[k][2] == ck["cls"], (type(m2).__name__, ck["cls"])
    yy = np.asarray(test_utts[1], np.float64)
    r1 = G["reconstruct"](m2, yy, CFG, chunk=8192, tau=1.0, seed=1)
    assert len(r1) == len(yy) and np.isfinite(r1).all()
    print(f"reload {k:<17} {type(m2).__name__:<7} step {ck['step']} tau {m2.tau:g}: SNR {G['snr_db'](yy, r1):.2f}", flush=True)

# ---- the dashboard cell of the built notebook, driven by the real payloads ---------------------------------
nbp = REPO / "sampler" / "lisa_rtm_train.ipynb"
if nbp.exists():
    cells = [("".join(c["source"])) for c in json.loads(nbp.read_text())["cells"] if c["cell_type"] == "code"]
    dsrc = next(s for s in cells if "def make_dashboard" in s)
    DG = dict(G); DG["__name__"] = "__main__"
    exec(compile(dsrc, "<dash>", "exec"), DG)
    DG["_HAS_IPY"] = False
    ds, de = DG["make_dashboard"](TAG, ARMS, every=0.0)
    for info in seen["step"]:
        ds(info)
    for info in seen["epoch"]:
        de(info)
    print("dashboard rendered every real payload", flush=True)

print(f"\nscratch: {SCR}\nSMOKE FAST OK  [{time.time()-T0:.0f}s]")
