"""Convert fast/train_arm.py checkpoints into the shape overnight3/e4_eval.py can load.

    python fast/convert_ckpt.py --src ~/lisa-results --out ~/lisa-results/ckpt/OV50

WHY THIS FILE EXISTS.  train_arm.py's save() writes a RESUME checkpoint: stacked parameters plus
optimiser moments, scheduler position and both RNG states, so a killed VM can continue bit-identically.
e4_eval.py's load_arm() (overnight2/c1_model.py:278) wants an INFERENCE checkpoint: a plain module
state_dict plus enough metadata to rebuild the right class at the right temperature.  The two blobs
share no key but "step".

    written by train_arm.save()          wanted by load_arm()
    ------------------------------       -----------------------------------------
    params   stack.P, (1, *shape)        model    nn.Module state_dict
    arm      "es_erb_l0.1"   (a str)     arm      (kind, lam, cls_name)   (a tuple)
    opt/sched/cpu_rng/cuda_rng           cls      "LISAS" | "LISASD"
    hist                                 n_noise  int

TWO OF THE THREE MISMATCHES FAIL SILENTLY.  A converter is needed, not a try/except.

1.  load_arm sets `m.tau = 0.0 if ck["arm"][0].startswith("det") else 1.0`.  Given the TUPLE it
    expects, ck["arm"][0] is the kind, and "det".startswith("det") is True.  Given the STRING
    train_arm writes, ck["arm"][0] is the first CHARACTER: "det_paper"[0] == "d", and
    "d".startswith("det") is False.  Both deterministic arms would then be evaluated at tau = 1 --
    noise injected into a network that never saw any during training.  Nothing raises.  The numbers
    are plausible and wrong, and they are wrong in the direction that flatters the samplers, because
    the reference line gets handicapped.

2.  ck.get("cls", "LISAS") defaults the class.  es_dec_l0.01 and es_dec_erb_l0.1 are LISASD: four
    decoder-noise channels, so the decoder's first Linear is 101 wide, not 97.  Here the shape
    mismatch does raise inside load_state_dict -- but only because n_dec happens to change a shape.
    The class is metadata that must travel; do not rely on a shape to carry it.

3.  params vs model is the loud one: a KeyError on ck["model"].

THE CONVERSION ITSELF IS MECHANICAL, AND IT ALREADY EXISTS TWICE.
ArmStack.export(a, module) (e2b_fast.py:399) copies stacked parameters into a plain module, and
e2b_fast.py:703 already writes exactly the blob load_arm reads.  This file does those two things to
a checkpoint instead of to a live trainer, taking (kind, lam, cls) from fast/run_contract.py's ARMS
-- the same table train_arm.py trained from -- so the arm name in the file and the spec in the blob
cannot drift apart.

VERIFICATION.  A converter that loads the wrong weights still runs, so --verify (on by default) is
the point of the file, not a flag on it:

    params->module   every exported tensor is bitwise equal to ck["params"][key][0]
    keys             the stacked key set is exactly the module's, at (1, *shape)
    forward          stack and module agree on one batch at fixed noise (rel < 1e-5, the
                     overnight3/gate_subpixel.py threshold: both run the sub-pixel decoder, whose
                     layer-1 summation order differs from the gather path by ~1e-7 relative)
    round-trip       load_arm() on the written file returns the right class, the right tau and a
                     bitwise-equal state_dict
    history          ck["step"] == STEPS and the checkpoint's own val_wave tail matches
                     history_<RUN_TAG>_<arm>.json, which ties this checkpoint to that curve

What these do NOT check is that the weights reproduce val_wave when re-run on the validation
batches: that needs the staged corpus, which lived on the terminated instance.  See
fast/val_wave_check.py, which rebuilds the 512 validation segments from VCTK and does exactly that.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fast.run_contract import ARMS, BATCH, RUN_TAG, SEG_HI, STEPS
from fast.train_arm import boot

REPO = pathlib.Path(__file__).resolve().parents[1]


def _pkey(k):
    """e2b_fast.py:188.  nn.ParameterDict rejects dots in keys."""
    return k.replace(".", "__")


def build(G, arm, device):
    """The stack and the plain module for one arm, both from ARMS[arm] -- never from the checkpoint.

    Taking the spec from the run contract rather than from ck["arm"] is deliberate: ck["arm"] is the
    arm NAME, and a name that does not appear in ARMS is a checkpoint from a different run, which is
    a KeyError here rather than a guess."""
    kind, lam, cls = ARMS[arm]
    names, specs, base, models = G["_make_models"]({arm: (kind, lam, cls)})
    stack = G["build_stacks"](names, specs, base, device)[0]
    return stack, models[arm], (kind, lam, cls)


def load_params(stack, ck, torch):
    """ck["params"] -> stack.P, with the key set checked rather than assumed.

    copy_ into the existing Parameters (not assignment) so a shape disagreement raises here, where
    the message names the key, instead of at the first forward."""
    P, got = stack.P, ck["params"]
    want = {_pkey(k) for k in stack.keys}
    if set(got) != want:
        raise SystemExit(f"key mismatch: missing {sorted(want - set(got))}, extra {sorted(set(got) - want)}")
    with torch.no_grad():
        for k, v in got.items():
            if tuple(v.shape) != tuple(P[k].shape):
                raise SystemExit(f"{k}: checkpoint {tuple(v.shape)} != stack {tuple(P[k].shape)}")
            P[k].copy_(v)
    return stack


def import_module(stack, module, torch):
    """The inverse of ArmStack.export: a plain module's weights back into a one-arm stack.

    fast/val_wave_check.py needs this because the thing worth re-running is the CONVERTED file --
    the one e4_eval will actually load -- and the objective lives on ArmStack, not on the module.
    Going module -> stack rather than re-reading the training checkpoint keeps the conversion inside
    the loop being tested."""
    sd = module.state_dict()
    with torch.no_grad():
        for k in stack.keys:
            stack.P[_pkey(k)][0].copy_(sd[k])
    return stack


def blob_for(module, ck, spec, torch):
    """e2b_fast.py:703's dict, from a checkpoint instead of a live trainer.

    "history" carries train_arm's history shape (step/loss/wave/spec/spread/lr/val_*), not the OV3
    trainer's (which also has dev_step/snr0/def0).  Nothing in e4_eval reads it; it is kept so the
    curve travels with the weights."""
    kind, lam, cls = spec
    return {"model": module.state_dict(), "step": ck["step"], "history": ck["hist"],
            "arm": (kind, lam, cls), "n_noise": module.n_noise, "n_dec": getattr(module, "n_dec", 0),
            "cls": cls, "batch": BATCH, "seg": SEG_HI, "tag": ck.get("run_tag", RUN_TAG)}


# ---- verification -----------------------------------------------------------------------------------
def check_export(module, ck, torch):
    """Every exported tensor bitwise equal to the stacked one at arm 0."""
    bad = []
    for k, v in module.state_dict().items():
        src = ck["params"][_pkey(k)][0]
        if not torch.equal(v.cpu(), src.cpu()):
            d = (v.cpu() - src.cpu()).abs().max().item()
            bad.append(f"{k} (max|diff| {d:.3e})")
    return not bad, ("all %d tensors bitwise equal" % len(module.state_dict()) if not bad
                     else "differ: " + ", ".join(bad))


def check_forward(G, stack, module, torch, S=2, L=256, seed=0):
    """Stack and module on one batch at the SAME noise.

    Both paths run the sub-pixel decoder, so this is tight but not bitwise: the stacked layer 1 is a
    grouped conv over A arms and the module's is a plain one, and float addition is not associative.
    1e-5 relative is overnight3/gate_subpixel.py's threshold for the same comparison."""
    g = torch.Generator(device=stack.device).manual_seed(seed)
    x = torch.randn(S, L, device=stack.device, generator=g)
    e, d = stack.sample_eps(S, L, gen=g)
    with torch.no_grad():
        out_stack = stack.forward(x, e, d, perturb=False)[0]
        eps = (e[0], d[0]) if getattr(module, "n_dec", 0) else e[0]
        out_mod = module.decode(module.encode(x, eps))
    diff = (out_stack - out_mod).abs().max().item()
    rel = diff / max(out_stack.abs().max().item(), 1e-12)
    return rel < 1e-5, f"max|diff| {diff:.3e}  rel {rel:.3e}"


def check_roundtrip(G, path, module, spec, torch):
    """load_arm() on what we just wrote: right class, right tau, same weights.

    This is the check that would have caught both silent failures, so it runs on the written file --
    not on the in-memory module -- and compares tau against the spec rather than against itself."""
    kind, lam, cls = spec
    m, ck = G["load_arm"](path)
    want_tau = 0.0 if kind.startswith("det") else 1.0
    notes = []
    if type(m).__name__ != cls:
        notes.append(f"class {type(m).__name__} != {cls}")
    if float(m.tau) != want_tau:
        notes.append(f"tau {float(m.tau)} != {want_tau} for kind {kind!r}")
    if getattr(m, "n_dec", 0) != getattr(module, "n_dec", 0):
        notes.append(f"n_dec {getattr(m, 'n_dec', 0)} != {getattr(module, 'n_dec', 0)}")
    for k, v in module.state_dict().items():
        if not torch.equal(v.cpu(), m.state_dict()[k].cpu()):
            notes.append(f"{k} differs after round-trip")
    return not notes, ("%s tau=%g n_noise=%d n_dec=%d" % (cls, m.tau, m.n_noise, getattr(m, "n_dec", 0))
                       if not notes else "; ".join(notes))


def check_history(arm, ck, hist_dir):
    """The checkpoint's own history tail against history_<RUN_TAG>_<arm>.json.

    train_arm writes both from the same dict at the same step, so a disagreement means the two files
    came from different runs -- the mix-up a per-arm conversion would otherwise carry through."""
    p = pathlib.Path(hist_dir) / f"history_{RUN_TAG}_{arm}.json"
    if not p.exists():
        return None, f"no {p.name}"
    h = json.loads(p.read_text())[arm]
    c = ck["hist"]
    notes = []
    if ck["step"] != STEPS:
        notes.append(f"step {ck['step']} != {STEPS}")
    if h["val_step"][-1] != c["val_step"][-1] or h["val_step"][-1] != STEPS:
        notes.append(f"val_step {c['val_step'][-1]} vs {h['val_step'][-1]}, expected {STEPS}")
    n = min(len(h["val_wave"]), len(c["val_wave"]), 5)
    if h["val_wave"][-n:] != c["val_wave"][-n:]:
        notes.append(f"val_wave tail differs: ckpt {c['val_wave'][-1]!r} json {h['val_wave'][-1]!r}")
    return not notes, (f"step {ck['step']}, final val_wave {c['val_wave'][-1]:.9f}" if not notes
                       else "; ".join(notes))


def convert_one(G, arm, src, out, hist_dir, torch, verify=True):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    stack, module, spec = build(G, arm, torch.device("cpu"))
    load_params(stack, ck, torch)
    with torch.no_grad():
        stack.export(0, module)
    module.eval()
    dst = pathlib.Path(out) / f"{arm}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    G["save_ckpt"](blob_for(module, ck, spec, torch), dst)

    rows = []
    if verify:
        rows.append(("params->module", *check_export(module, ck, torch)))
        rows.append(("forward", *check_forward(G, stack, module, torch)))
        rows.append(("round-trip", *check_roundtrip(G, dst, module, spec, torch)))
        rows.append(("history", *check_history(arm, ck, hist_dir)))
    return dst, spec, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/lisa-results", help="directory holding <arm>.pt from train_arm.py")
    ap.add_argument("--out", default=None, help="default <src>/ckpt/<RUN_TAG>, which is what e4_eval globs")
    ap.add_argument("--history", default=None, help="directory holding history_<tag>_<arm>.json (default --src)")
    ap.add_argument("--arms", nargs="*", default=sorted(ARMS))
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    src = pathlib.Path(a.src).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser().resolve() if a.out else src / "ckpt" / RUN_TAG
    hist_dir = pathlib.Path(a.history).expanduser().resolve() if a.history else src
    out.mkdir(parents=True, exist_ok=True)

    # boot() exec's the notebook's setup cell, which builds ./lisa_rtm_cache in the CWD before
    # boot() overrides ROOT.  Run it from the results tree so the repo stays clean and the VCTK
    # cache lands next to the data.  root=src makes G["CKPT"] == src/"ckpt", so CKPT/RUN_TAG is the
    # default --out and e4_eval's load_ov3 glob finds it without further argument.
    cwd = pathlib.Path.cwd()
    import os
    os.chdir(src)
    try:
        G = boot(device="cpu", root=src)
    finally:
        os.chdir(cwd)
    import torch

    print(f"\nconverting {len(a.arms)} arm(s)  {src} -> {out}\n", flush=True)
    fails = []
    for arm in a.arms:
        p = src / f"{arm}.pt"
        if not p.exists():
            fails.append(f"{arm}: no {p}")
            print(f"{arm:<18} MISSING {p}", flush=True)
            continue
        dst, spec, rows = convert_one(G, arm, p, out, hist_dir, torch, verify=not a.no_verify)
        kind, lam, cls = spec
        print(f"{arm:<18} {kind:<12} lam={lam:<7g} {cls:<7} tau={0.0 if kind.startswith('det') else 1.0:g} -> {dst.name}", flush=True)
        for name, ok, detail in rows:
            mark = "  ok " if ok else (" SKIP" if ok is None else "FAIL ")
            print(f"    {mark} {name:<16} {detail}", flush=True)
            if ok is False:
                fails.append(f"{arm}/{name}: {detail}")

    print()
    if fails:
        print("CONVERSION FAILED:\n  " + "\n  ".join(fails), flush=True)
        return 1
    print(f"CONVERSION OK: {len(a.arms)} arms in {out}", flush=True)
    print(f"point e4_eval at it with CKPT={out.parent} and OV3_TAG={out.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
