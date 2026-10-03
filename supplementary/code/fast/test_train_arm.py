"""Gate G11 and friends: resume must not change the trajectory.

    venv/bin/python fast/test_train_arm.py

Resume is the whole reason the training loop was rewritten rather than reused -- train_ov3_fast
checkpoints no optimiser, scheduler or RNG state, so at ~4.4 h per arm on an instance whose
termination is permanent, a crash at 80% is a total write-off. A resume that silently perturbs the
trajectory is worse than no resume at all, so this compares an interrupted run against an
uninterrupted one parameter by parameter.

CPU, synthetic corpus, SMOKE dims: about a minute, no GPU.
"""
import pathlib
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fast.run_contract import ARMS, BATCH_TAG, RUN_TAG, VAL_TAG, stream
from fast.train_arm import arm_seed, boot, train

ok = 0


def check(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    assert cond, name
    ok += 1


TMP = pathlib.Path(tempfile.mkdtemp(prefix="fast_train_"))
G = boot(smoke=True, device="cpu", root=TMP)
v = (pathlib.Path(__file__).resolve().parents[1] / "validate.py").read_text()
gen = "def synthetic_corpus" + v.split("def synthetic_corpus")[1].split("\nfailures = []")[0]
ns = {"G": G}
exec(gen, ns)
ns["synthetic_corpus"](None)
SEG = 4096
corpus = G["HostCorpus"](G["train_utts"], G["CFG"], seg_hi=SEG)
val_corpus = G["HostCorpus"](G["test_utts"], G["CFG"], seg_hi=SEG)


def run(arm, steps, out, ckpt_every, resume, stop_after=None):
    d = TMP / out
    d.mkdir(parents=True, exist_ok=True)
    train(G, arm, corpus, val_corpus, d, steps=steps, batch=4, ckpt_every=ckpt_every,
          val_every=steps, log_every=4, n_val_batches=2, resume=resume, device=torch.device("cpu"),
          stop_after=stop_after)
    blob = torch.load(d / f"{arm}.pt", map_location="cpu", weights_only=False)
    return blob, {k: t.clone() for k, t in blob["params"].items()}


print("1. stream labels are consistent and never carry an arm name")
check("BATCH_TAG derives from RUN_TAG", BATCH_TAG == f"{RUN_TAG}/batches", BATCH_TAG)
check("VAL_TAG derives from RUN_TAG", VAL_TAG == f"{RUN_TAG}/val", VAL_TAG)
check("no arm name appears in either", not any(a in BATCH_TAG or a in VAL_TAG for a in ARMS))

print("\n2. per-arm noise seeds are distinct and independent of the other arms")
seeds = {a: arm_seed(a) for a in ARMS}
check("eight distinct seeds", len(set(seeds.values())) == len(ARMS))
check("stable across calls", arm_seed("det") == arm_seed("det"))

print("\n3. G11 -- resume reproduces an uninterrupted run, bit for bit")
ARM = "det"
full, p_full = run(ARM, 12, "full", ckpt_every=12, resume=False)
# steps stays 12 so the lr schedule is identical; stop_after ends the loop at 6
run(ARM, 12, "part", ckpt_every=6, resume=False, stop_after=6)
part, p_part = run(ARM, 12, "part", ckpt_every=12, resume=True)   # resume 6 -> 12
check("uninterrupted reached step 12", full["step"] == 12)
check("resumed run reached step 12", part["step"] == 12)
worst = max(float((p_full[k] - p_part[k]).abs().max()) for k in p_full)
check("every parameter is bit-identical", all(torch.equal(p_full[k], p_part[k]) for k in p_full),
      f"max abs diff {worst:.3e} over {len(p_full)} tensors")

print("\n4. the batch stream continues rather than restarting on resume")
r_cont = stream(BATCH_TAG)
for _ in range(6):                                            # advance as train() does
    r_cont.integers(corpus.n, size=4)
    r_cont.random(4)
nxt = r_cont.integers(corpus.n, size=4)
r_fresh = stream(BATCH_TAG)
check("step 7 differs from step 1", not np.array_equal(nxt, r_fresh.integers(corpus.n, size=4)),
      f"resumed draws {nxt}, a restart would draw the first batch again")

print("\n5. a truncated checkpoint cannot be half-written")
d = TMP / "full"
check("no .tmp files left behind", not list(d.glob("*.tmp*")))
check("checkpoint carries optimiser state", "opt" in full and full["opt"]["state"])
check("checkpoint carries scheduler state", "sched" in full and "last_epoch" in full["sched"])
check("checkpoint carries CPU RNG state", full["cpu_rng"] is not None)
check("checkpoint records its run tag", full["run_tag"] == RUN_TAG)

print("\n6. an energy-score arm and a decoder-noise arm both train")
for arm in ("es_erb_l0.01", "es_dec_erb_l0.1"):
    b, p = run(arm, 4, f"x_{arm}", ckpt_every=4, resume=False)
    finite = all(torch.isfinite(t).all() for t in p.values())
    check(f"{arm} ({ARMS[arm][2]}) trains, finite params", b["step"] == 4 and finite,
          f"{len(p)} tensors")
    check(f"  {arm} logged a spread term", len(b["hist"]["spread"]) > 0)

shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{ok} checks passed")
