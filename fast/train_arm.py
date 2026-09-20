"""Train ONE arm in one process, resumably, on a staged corpus.

    python fast/train_arm.py --arm es_erb_l0.1 --corpus /data/corpus --out /data/run
    python fast/train_arm.py --arm det --smoke            # CPU, synthetic, ~20 steps

DESIGN: REUSE THE MATH, OWN THE LOOP.

`ArmStack` in overnight3/e2b_fast.py is the numerical contract -- the energy-score estimator, the
three distances, the per-arm clip, the sub-pixel decoder. It is exec'd here verbatim and used
unchanged, because a rewrite of the objective would need a parity gate this run cannot afford and
the algorithm spec's 27 invariants all point at that file.

What is NOT reused is `train_ov3_fast`'s loop, for one reason: it cannot resume. `save_ckpt` writes
{model, step, history, arm} with no optimiser state, no scheduler state and no RNG state (algorithm
spec section 9, item 13). At ~4.4 h per arm on an ephemeral VM whose termination is permanent, a
crash at 80% is a total write-off. So the loop lives here and checkpoints everything needed to
continue bit-identically.

THE BATCH STREAM IS EXPLICIT HERE, NOT DERIVED FROM A TAG.

e2b_fast.py:592 does `rng = stream(f"{tag}/batches")`, and `stream` is blake2b over the label, so the
tag IS the RNG. Eight arms on eight VMs each want their own output paths, and the one-line way to get
them -- passing f"{tag}_{arm}" as tag -- silently hands every arm a different 105,000-step batch
stream. This file never interpolates an arm name into a stream label: it calls
`stream(BATCH_TAG)` with the fixed constant from fast/run_contract.py, and asserts the consistency
of RUN_TAG/BATCH_TAG/VAL_TAG at startup. Per-arm separation comes from the `arms` dict having one
entry and from each VM owning its own filesystem.

NOISE IS PER-ARM AND DECLARED. The reference draws eps and jitter from the global CUDA generator in
stack order, so arm k's noise depends on which arms precede it -- unreproducible once arms are split
across processes, and unreproducible across GPU models in any case because ATen sizes its
distribution kernels from multiProcessorCount (A100 108 SMs, this card 188). Each arm therefore
seeds the global torch RNG from its own name, and that state is checkpointed. This is
CHANGES-TRAJECTORY, not CHANGES-MATH: same estimator, same distances, different realised draws.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fast.run_contract import (ARMS, BATCH, BATCH_TAG, GRAD_CLIP, LR, LR_FRACS, LR_GAMMA, RUN_TAG,
                               SEED, SEG_HI, STEPS, VAL_TAG, preflight, stream)

REPO = pathlib.Path(__file__).resolve().parents[1]


def arm_seed(arm, seed=SEED):
    """A per-arm torch seed that does not depend on which other arms exist."""
    h = hashlib.blake2b(f"{RUN_TAG}/noise/{arm}".encode(), digest_size=8).digest()
    return (int.from_bytes(h, "big") ^ seed) % (2 ** 63)


def boot(smoke=False, device=None, root=None):
    """Exec the notebook definition cells in dependency order, exactly as
    overnight3/smoke_fast_local.py does. Returns the globals dict."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    import torch
    nb = json.loads((REPO / "lisa_rtm.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    G = {"__name__": "__main__"}

    def run(mark):
        exec(compile(next(s for s in code if mark in s), f"<nb {mark[:20]}>", "exec"), G)

    run("import importlib, subprocess, sys")
    G["DEVICE"] = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run("@dataclasses.dataclass(frozen=True)")
    G["CFG"] = G["SMOKE"] if smoke else G["FULL"]
    for m in ("def _hann(n):", "class QuantileMap:", "def snr_db(y, y_hat):", "VCTK_URL = ",
              "class LISAEncoder(nn.Module):"):
        run(m)
    xl = (REPO / "overnight/cell2_trainer.py").read_text()
    exec(xl.split("def train_paired")[0], G)
    exec("def save_ckpt" + xl.split("def save_ckpt")[1], G)
    exec((REPO / "overnight/cell3_eval.py").read_text(), G)
    root = pathlib.Path(root or ".").resolve()
    G["ROOT"], G["CKPT"], G["FIGS"] = root, root / "ckpt", root / "figs"
    for d in (G["CKPT"], G["FIGS"]):
        d.mkdir(parents=True, exist_ok=True)
    for f in ("overnight2/c1_model.py", "overnight3/e1_model.py", "overnight3/e2_trainer.py",
              "overnight3/e2b_fast.py"):
        exec(compile((REPO / f).read_text(), f"<{f}>", "exec"), G)
    return G


class StagedCorpus:
    """HostCorpus's interface over the memmaps fast/stage_corpus.py wrote.

    batch() reproduces overnight2/c1_model.py:101-107 exactly, including the RNG consumption order
    (one rng.integers, then one rng.random) -- that order is the contract that keeps arms paired.
    """

    def __init__(self, path, cfg, seg_hi=SEG_HI, device=None):
        import torch
        p = pathlib.Path(path)
        man = json.loads((p / "manifest.json").read_text())
        self.R, self.seg_hi, self.seg_lo = cfg.upsample, seg_hi, seg_hi // cfg.upsample
        self.lens = np.load(p / "lens.npy")
        self.off = np.load(p / "off.npy")
        self.n, self.hours = int(man["n"]), float(man["hours"])
        self.Y = np.memmap(p / "Y.f32", dtype=np.float32, mode="r")
        self.X = np.memmap(p / "X.f32", dtype=np.float32, mode="r")
        self._ar_hi = np.arange(seg_hi)
        self._ar_lo = np.arange(seg_hi // cfg.upsample)
        self.manifest = man
        self.device = device or torch.device("cpu")

    def batch(self, rng, B):
        import torch
        idx = rng.integers(self.n, size=B)
        n_pos = (self.lens[idx] - self.seg_hi) // self.R + 1
        starts = (rng.random(B) * n_pos).astype(np.int64) * self.R
        hi0 = self.off[idx] + starts
        y = np.ascontiguousarray(self.Y[hi0[:, None] + self._ar_hi])
        x = np.ascontiguousarray(self.X[(hi0 // self.R)[:, None] + self._ar_lo])
        return (torch.from_numpy(x).to(self.device, non_blocking=True),
                torch.from_numpy(y).to(self.device, non_blocking=True))


def save(path, G, stack, opt, sched, step, hist, arm, torch):
    """Everything needed to continue bit-identically: weights, optimiser moments, scheduler
    position, and BOTH RNG states. Written to a temp file and renamed, so a kill during the write
    cannot leave a truncated checkpoint that resume would happily load."""
    blob = {"step": step, "arm": arm, "run_tag": RUN_TAG,
            "params": {k: v.detach().cpu() for k, v in stack.P.items()},
            "opt": opt.state_dict(), "sched": sched.state_dict(), "hist": hist,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    tmp = pathlib.Path(str(path) + f".tmp{os.getpid()}")
    torch.save(blob, tmp)
    tmp.replace(path)


def load(path, stack, opt, sched, torch):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    with torch.no_grad():
        for k, v in blob["params"].items():
            stack.P[k].copy_(v.to(stack.P[k].device))
    opt.load_state_dict(blob["opt"])
    sched.load_state_dict(blob["sched"])
    torch.set_rng_state(blob["cpu_rng"].cpu() if hasattr(blob["cpu_rng"], "cpu") else blob["cpu_rng"])
    if blob.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(blob["cuda_rng"])
    return blob["step"], blob["hist"]


def train(G, arm, corpus, val_corpus, out, steps=STEPS, batch=BATCH, ckpt_every=500, val_every=500,
          log_every=25, n_val_batches=8, resume=True, device=None, stop_after=None):
    """`steps` is the SCHEDULE length and must always be the full run: MultiStepLR milestones are
    int(f * steps), so passing a shorter `steps` to stop early would silently decay the lr on a
    different schedule. To stop early -- for a budget cap, or to test resume -- pass `stop_after`,
    which ends the loop without touching the schedule."""
    import torch
    from torch.optim.lr_scheduler import MultiStepLR
    kind, lam, cls = ARMS[arm]
    dev = device or G["DEVICE"]

    # consistency of the stream labels -- the whole point of run_contract
    assert BATCH_TAG == f"{RUN_TAG}/batches" and VAL_TAG == f"{RUN_TAG}/val", "stream tags drifted"

    torch.manual_seed(arm_seed(arm))                     # per-arm, declared CHANGES-TRAJECTORY
    names, specs, base, _ = G["_make_models"]({arm: (kind, lam, cls)})
    stack = G["build_stacks"](names, specs, base, dev)[0]
    opt = torch.optim.Adam(stack.params(), lr=LR,
                           **({"fused": True} if dev.type == "cuda" else {}))
    sched = MultiStepLR(opt, [int(f * steps) for f in LR_FRACS], LR_GAMMA)

    ck = pathlib.Path(out) / f"{arm}.pt"
    start, hist = 0, {"step": [], "loss": [], "wave": [], "spec": [], "spread": [], "lr": [],
                      "val_step": [], "val_loss": [], "val_wave": [], "val_spec": []}
    if resume and ck.exists():
        start, hist = load(ck, stack, opt, sched, torch)
        print(f"resumed {arm} from step {start}", flush=True)

    # The batch stream is advanced to `start` on resume so the sequence continues rather than
    # restarting -- a resumed run that re-draws batch 0 is a different experiment.
    rng = stream(BATCH_TAG)
    for _ in range(start):
        rng.integers(corpus.n, size=batch)
        rng.random(batch)
    # ONE generator, n draws -- e2_trainer.py:11-12. Creating stream(VAL_TAG) inside the
    # comprehension would hand every draw the same fresh generator, i.e. the same batch n times.
    _vr = stream(VAL_TAG)
    vb = [val_corpus.batch(_vr, batch) for _ in range(n_val_batches)]

    t0, ema = time.time(), None
    last = steps if stop_after is None else min(steps, stop_after)
    for step in range(start + 1, last + 1):
        x, y = corpus.batch(rng, batch)
        for p in stack.params():
            p.grad = None
        terms = G["_fwd_bwd"]([stack], x, y, GRAD_CLIP)[0]
        opt.step()
        sched.step()
        if step % log_every == 0:
            l, w, s, sp = terms[0].tolist()
            ema = l if ema is None else 0.98 * ema + 0.02 * l
            hist["step"].append(step); hist["loss"].append(ema); hist["wave"].append(w)
            hist["spec"].append(s); hist["spread"].append(sp)
            hist["lr"].append(float(opt.param_groups[0]["lr"]))
        if step % val_every == 0 or step == last:
            v = G["val_loss_fast"]([stack], vb)[arm]
            hist["val_step"].append(step); hist["val_loss"].append(v[0])
            hist["val_wave"].append(v[1]); hist["val_spec"].append(v[2])
            el = time.time() - t0
            done = step - start
            print(f"  {arm} step {step}/{steps}  val {v[0]:.6f}  "
                  f"{1000 * el / max(1, done):.0f} ms/step  eta {(steps - step) * el / max(1, done) / 3600:.2f} h",
                  flush=True)
        if step % ckpt_every == 0 or step == last:
            save(ck, G, stack, opt, sched, step, hist, arm, torch)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--corpus")
    ap.add_argument("--out", default="run")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--smoke", action="store_true", help="CPU, synthetic corpus, tiny dims")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    G = boot(smoke=a.smoke, device="cpu" if a.smoke else None, root=out)
    import torch
    print(f"arm {a.arm} {ARMS[a.arm]}  device {G['DEVICE']}  torch {torch.__version__}", flush=True)

    if a.smoke:
        v = (REPO / "validate.py").read_text()
        gen = "def synthetic_corpus" + v.split("def synthetic_corpus")[1].split("\nfailures = []")[0]
        ns = {"G": G}
        exec(gen, ns)
        ns["synthetic_corpus"](None)
        seg = 4096
        corpus = G["HostCorpus"](G["train_utts"], G["CFG"], seg_hi=seg)
        val_corpus = G["HostCorpus"](G["test_utts"], G["CFG"], seg_hi=seg)
    else:
        corpus = StagedCorpus(a.corpus, G["CFG"], device=G["DEVICE"])
        val_corpus = corpus
        print("preflight:", json.dumps(preflight(corpus.n, corpus.hours, SEG_HI)), flush=True)
        print("g3 (must match every other instance):",
              json.dumps({k: corpus.manifest["g3"][k] for k in ("corpus", "batches")}), flush=True)

    h = train(G, a.arm, corpus, val_corpus, out, steps=a.steps, batch=a.batch,
              ckpt_every=a.ckpt_every, val_every=min(a.ckpt_every, max(1, a.steps // 4)),
              log_every=max(1, min(25, a.steps // 8)), n_val_batches=2 if a.smoke else 8,
              resume=not a.no_resume)
    (out / f"history_{RUN_TAG}_{a.arm}.json").write_text(json.dumps({a.arm: h}))
    print(f"DONE {a.arm}: {len(h['step'])} log points, final val {h['val_loss'][-1]:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
