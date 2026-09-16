"""Exactness gate for the stacked trainer's fast paths.  Run before any GPU run that uses them.

Three checks, all in fp32 with autocast and compile OFF, so only the arithmetic is compared:
  1. SUBPIXEL   the input-rate layer 1 must reproduce the output-rate gather path, for perturb=False and for
                perturb=True with an IDENTICAL jitter tensor, on LISAS (97-wide layer 1) and LISASD (101).
  2. arm_loss3  the stacked per-arm losses must reproduce the reference single-arm objective, all seven arms.
  3. JITTER_PER_CELL is changes-math by construction: it is only checked to run and to stay finite.

    venv/bin/python overnight3/gate_subpixel.py
"""
import json, os, sys, pathlib, tempfile, time
import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
REPO = pathlib.Path(__file__).resolve().parents[1]
OV3 = REPO / "overnight3"
SCR = pathlib.Path(tempfile.mkdtemp(prefix="ov3_gate_"))
os.chdir(SCR)

NB = json.loads((REPO / "lisa_rtm.ipynb").read_text())
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
G = {"__name__": "__main__"}
run = lambda mark: exec(compile(next(s for s in CODE if mark in s), "<nb>", "exec"), G)
run("import importlib, subprocess, sys")
import torch
G["DEVICE"] = torch.device("cpu")
run("@dataclasses.dataclass(frozen=True)"); G["CFG"] = G["SMOKE"]
run("def _hann(n):"); run("def snr_db(y, y_hat):"); run("VCTK_URL = "); run("class LISAEncoder(nn.Module):")
_xl = open(REPO / "overnight/cell2_trainer.py").read()
exec(_xl.split("def train_paired")[0], G)
exec("def save_ckpt" + _xl.split("def save_ckpt")[1], G)
exec(open(REPO / "overnight/cell3_eval.py").read(), G)
G["ROOT"], G["CKPT"], G["FIGS"] = SCR, SCR / "ckpt", SCR / "figs"
for d in (G["CKPT"], G["FIGS"]):
    d.mkdir(parents=True, exist_ok=True)
exec(compile(open(REPO / "overnight2/c1_model.py").read(), "<c1>", "exec"), G)
exec(compile(open(OV3 / "e1_model.py").read(), "<e1>", "exec"), G)
exec(compile(open(OV3 / "e2_trainer.py").read(), "<e2>", "exec"), G)
for f in ("AMP", "TF32", "COMPILE", "PIN", "FUSED", "STREAMS", "CUDA_GRAPHS", "JITTER_PER_CELL"):
    G[f] = False
G["STACK"], G["SUBPIXEL"], G["GROUP_MAX"] = True, True, None
exec(compile(open(OV3 / "e2b_fast.py").read(), "<e2b>", "exec"), G)
CFG, FAST = G["CFG"], G["FAST"]
torch.manual_seed(0)
print("gate: fp32, autocast off, compile off; flags", {k: FAST[k] for k in ("SUBPIXEL", "AMP", "COMPILE", "JITTER_PER_CELL")}, flush=True)

B, L = 2, 512                     # 512 input samples -> 2048 output samples (longer than n_fft padding)
x = torch.randn(B, L)
y = torch.randn(B, L * CFG.upsample)
fails = []

# ---- 1. SUBPIXEL layer 1 == the gather path -------------------------------------------------------------
print("\n1. sub-pixel layer 1 vs the output-rate gather", flush=True)
for cls_name in ("LISAS", "LISASD"):
    base = G[cls_name](CFG)
    specs = {"a": ("es_marg", 1e-2, cls_name)}
    st = G["ArmStack"](["a"], specs, base, cls_name, torch.device("cpu"))
    S = 2 * B
    xs = x.repeat(2, 1)
    eps_enc, eps_dec = st.sample_eps(S, L)
    N = L * CFG.upsample
    jit = torch.randn(st.A, S, N) * 0.5                       # the SAME anchors for both paths
    for tag, kw in (("perturb=False", dict(perturb=False)), ("perturb=True (fixed jitter)", dict(perturb=True, jitter=jit))):
        outs = {}
        for sp in (False, True):
            FAST["SUBPIXEL"] = sp
            with torch.no_grad():
                outs[sp] = st.forward(xs, eps_enc, eps_dec, **kw)
        d = (outs[False] - outs[True]).abs().max().item()
        rel = d / max(outs[False].abs().max().item(), 1e-12)
        ok = rel < 1e-5
        fails.append(None if ok else f"SUBPIXEL {cls_name} {tag}: rel {rel:.2e}")
        print(f"   {cls_name:<7} {tag:<28} max|diff| {d:.3e}  rel {rel:.3e}  {'OK' if ok else 'FAIL'}", flush=True)
FAST["SUBPIXEL"] = True

# ---- 2. stacked losses == arm_loss3, all seven arms -------------------------------------------------------
print("\n2. stacked losses vs arm_loss3 (SUBPIXEL on, perturb off)", flush=True)
ARMS = {"det": ("det", 1e-2, "LISAS"), "es_marg": ("es_marg", 1e-2, "LISAS"),
        "es_marg_l0.1": ("es_marg", 1e-1, "LISAS"), "es_split_l0.1": ("es_split_marg", 1e-1, "LISAS"),
        "es_erb_l0.1": ("es_marg_erb", 1e-1, "LISAS"), "es_dec_l0.1": ("es_marg", 1e-1, "LISASD"),
        "es_dec_erb_l0.1": ("es_marg_erb", 1e-1, "LISASD")}
names, specs, base, models = G["_make_models"](ARMS)
stacks = G["build_stacks"](names, specs, base, torch.device("cpu"))
spec_loss = G["MultiScaleSTFTLoss"](CFG.n_fft)
tf = G["TargetFeats"](y, stacks[0].R, *G["_needs"](stacks))
worst = 0.0
for st in stacks:
    eps = None if st.is_det else st.sample_eps(2 * B, L)
    with torch.no_grad():
        Ls, _, _, _ = st.losses(x, y, tf, eps=eps, perturb=False)
    for a, k in enumerate(st.names):
        m = models[k]
        with torch.no_grad():
            st.export(a, m)
        if not st.is_det:
            e_a = eps[0][a]
            eps_a = (e_a, eps[1][a]) if getattr(m, "n_dec", 0) else e_a
            m.sample_eps = (lambda ea: (lambda *A_, **K_: ea))(eps_a)
        with torch.no_grad():
            ref, _ = G["arm_loss3"](specs[k][0], specs[k][1], m, x, y, spec_loss, perturb=False)
        if not st.is_det:
            del m.sample_eps
        d = abs(float(Ls[a]) - float(ref))
        worst = max(worst, d)
        print(f"   {k:<18} stacked {float(Ls[a]):.6f}  reference {float(ref):.6f}  |diff| {d:.2e}", flush=True)
fails.append(None if worst < 1e-5 else f"arm_loss3 equivalence: worst {worst:.2e}")
print(f"   worst |diff| {worst:.2e}  {'OK' if worst < 1e-5 else 'FAIL'}", flush=True)

# ---- 3. JITTER_PER_CELL runs and is finite (changes-math by construction) ---------------------------------
print("\n3. JITTER_PER_CELL (changes-math; finiteness only)", flush=True)
FAST["JITTER_PER_CELL"] = True
st = stacks[0]
with torch.no_grad():
    o = st.forward(x.repeat(2, 1) if not st.is_det else x, *( (None, None) if st.is_det else st.sample_eps(2 * B, L)), perturb=True)
print(f"   output finite: {bool(torch.isfinite(o).all())}  shape {tuple(o.shape)}", flush=True)
fails.append(None if bool(torch.isfinite(o).all()) else "JITTER_PER_CELL produced non-finite output")
FAST["JITTER_PER_CELL"] = False

bad = [f for f in fails if f]
print("\n" + ("GATE FAILED: " + "; ".join(bad) if bad else "GATE PASSED: sub-pixel is exact, all seven arms match the reference"), flush=True)
sys.exit(1 if bad else 0)
