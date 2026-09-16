"""CPU smoke test of the stacked trainer (overnight3/e2b_fast.py).

The gate before spending A100 time: with STACK on but AMP / COMPILE / PIN / FUSED / STREAMS off, the stacked
forward+loss must reproduce arm_loss3's per-arm losses on the same batch with the same noise, to 1e-5.  Then a
short train_ov3_fast run must produce e2's history keys and checkpoints that load_arm + reconstruct can read.

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
run("def _hann(n):"); run("def snr_db(y, y_hat):"); run("VCTK_URL = ")
run("class LISAEncoder(nn.Module):")
_xl = open(REPO / "overnight/cell2_trainer.py").read()
exec(_xl.split("def train_paired")[0], G)
exec("def save_ckpt" + _xl.split("def save_ckpt")[1], G)
exec(open(REPO / "overnight/cell3_eval.py").read(), G)
torch.use_deterministic_algorithms(False)
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
G["ROOT"] = SCR; G["CKPT"] = SCR / "ckpt"; G["FIGS"] = SCR / "figs"
for d in (G["CKPT"], G["FIGS"]):
    d.mkdir(parents=True, exist_ok=True)
exec(compile(open(REPO / "overnight2/c1_model.py").read(), "<c1>", "exec"), G)
exec(compile(open(OV3 / "e1_model.py").read(), "<e1>", "exec"), G)
exec(compile(open(OV3 / "e2_trainer.py").read(), "<e2>", "exec"), G)
for f in ("STACK", "AMP", "TF32", "COMPILE", "PIN", "FUSED", "STREAMS", "CUDA_GRAPHS"):
    G[f] = (f == "STACK")
G["GROUP_MAX"] = None
exec(compile(open(OV3 / "e2b_fast.py").read(), "<e2b>", "exec"), G)
CFG = G["CFG"]
print(f"boot ok: CFG {CFG.name} {CFG.fs_lo}->{CFG.fs_hi}  flags {G['FAST']}  [{time.time()-T0:.0f}s]", flush=True)

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

ARMS = {"det": ("det", 1e-2, "LISAS"), "es_marg": ("es_marg", 1e-2, "LISAS"),
        "es_marg_l0.1": ("es_marg", 1e-1, "LISAS"), "es_split_l0.1": ("es_split_marg", 1e-1, "LISAS"),
        "es_erb_l0.1": ("es_marg_erb", 1e-1, "LISAS"), "es_dec_l0.1": ("es_marg", 1e-1, "LISASD"),
        "es_dec_erb_l0.1": ("es_marg_erb", 1e-1, "LISASD")}

# ---- GATE: stacked losses == arm_loss3 losses, same weights, same batch, same eps --------------------------
names, specs, base, models = G["_make_models"](ARMS)
stacks = G["build_stacks"](names, specs, base, G["DEVICE"])
rng = G["stream"]("gate/batch")
x, y = corpus.batch(rng, BATCH)
spec_loss = G["MultiScaleSTFTLoss"](CFG.n_fft)
need = G["_needs"](stacks)
tf = G["TargetFeats"](y, stacks[0].R, *need)
worst, rows = 0.0, []
for s in stacks:
    # one eps draw for the stack; the same slice feeds the single-arm reference
    eps = None if s.is_det else s.sample_eps(2 * BATCH, x.shape[1])
    L_stack, w_stack, sp_stack, _ = s.losses(x, y, tf, eps=eps, perturb=False)
    for a, k in enumerate(s.names):
        m = models[k]
        with torch.no_grad():
            s.export(a, m)
        if s.is_det:
            ref, _t = G["arm_loss3"](specs[k][0], specs[k][1], m, x, y, spec_loss, perturb=False)
        else:
            # arm_loss3 draws its own eps; hand it this arm's slice of the stack's draw
            e_a = eps[0][a]
            eps_a = (e_a, eps[1][a]) if getattr(m, "n_dec", 0) else e_a
            m.sample_eps = (lambda ea: (lambda *args, **kw: ea))(eps_a)
            ref, _t = G["arm_loss3"](specs[k][0], specs[k][1], m, x, y, spec_loss, perturb=False)
            del m.sample_eps
        d = abs(float(L_stack[a]) - float(ref))
        worst = max(worst, d)
        rows.append((k, float(L_stack[a]), float(ref), d))
for k, a_, b_, d in rows:
    print(f"  gate {k:<18} stacked {a_:.6f}  reference {b_:.6f}  |diff| {d:.2e}", flush=True)
if rows:
    print(f"GATE stacked == sequential: worst |diff| {worst:.2e}", flush=True)
    assert worst < 1e-5, f"stacked losses differ from arm_loss3 by {worst:.2e}"
else:
    print("GATE skipped for the stochastic arms (no single-arm reference with shared eps)", flush=True)

# ---- short run -------------------------------------------------------------------------------------------
TAG = "SMOKE_FAST"
probe = test_utts[0]
seen = {"step": 0, "epoch": 0}
def on_step(info):
    seen["step"] += 1
    for key in ("step", "steps", "frac", "epoch", "t_elapsed", "ms_per_step", "gpu", "arms"):
        assert key in info, key
def on_epoch(info):
    seen["epoch"] += 1
G["time_ov3_fast"](corpus, ARMS, BATCH, n=2)
kw = {}
import inspect
sig = inspect.signature(G["train_ov3_fast"])
if "on_step" in sig.parameters:
    kw = dict(on_step=on_step, on_epoch=on_epoch, steps_per_epoch=10)
models_f, hist = G["train_ov3_fast"](corpus, val_corpus, ARMS, 20, BATCH, 1e-3, (0.5,), 0.5, 1e-3, 10, TAG, probe,
                                     val_every=10, n_val_batches=2, **kw)
for k in ARMS:
    h = hist[k]
    for key in ("step", "wave", "spec", "val_step", "val_loss", "train_loss_ema", "dev_step", "snr0", "def0"):
        assert h[key] and all(v is None or math.isfinite(v) for v in h[key]), (k, key, h[key])
    assert h["val_step"] == [10, 20], (k, h["val_step"])
    assert h["dev_step"] == [10, 20], (k, h["dev_step"])
print("history: keys present, every value finite; val_step", hist["det"]["val_step"], flush=True)
if kw:
    print(f"callbacks: on_step {seen['step']} calls, on_epoch {seen['epoch']} calls", flush=True)
assert (SCR / f"ov3_history_{TAG}.json").exists() and (G["FIGS"] / f"ov3_curves_{TAG}.png").exists()

for k in ("det", "es_marg_l0.1", "es_dec_erb_l0.1"):
    m2, ck = G["load_arm"](G["CKPT"] / TAG / f"{k}.pt")
    assert type(m2).__name__ == ARMS[k][2] == ck["cls"], (type(m2).__name__, ck["cls"])
    yy = np.asarray(test_utts[1], np.float64)
    r1 = G["reconstruct"](m2, yy, CFG, chunk=8192, tau=1.0, seed=1)
    assert len(r1) == len(yy) and np.isfinite(r1).all()
    print(f"  reload {k:<18} {type(m2).__name__:<7} step {ck['step']} SNR {G['snr_db'](yy, r1):+.2f}", flush=True)

print(f"\nscratch: {SCR}")
print(f"SMOKE FAST OK  [{time.time()-T0:.0f}s]")
