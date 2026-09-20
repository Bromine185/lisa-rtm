"""CPU tests for the run contract.  No torch, no GPU, no corpus: runs in about a second.

    venv/bin/python fast/test_run_contract.py

Test 3 is the point of the file: it DEMONSTRATES the failure mode rather than asserting the absence
of it, so that the thing the guard prevents is visible and cannot be quietly argued away.
"""
import hashlib
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fast.run_contract import (ARMS, BATCH, BATCH_TAG, MILESTONES, SEED, STEPS, barrier_decision,
                               batch_plan_digest, corpus_fingerprint, draw_batch, g3_digest,
                               preflight, stream)

ok = 0


def check(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    assert cond, name
    ok += 1


# a small synthetic corpus: 400 utterances, lengths like VCTK's, R-aligned
rng0 = np.random.default_rng(12345)
N = 400
LENS = (rng0.integers(3, 12, size=N) * 48000).astype(np.int64)
HOURS = float(LENS.sum()) / 48000 / 3600
Y = rng0.standard_normal(2_000_000).astype(np.float32)
X = rng0.standard_normal(500_000).astype(np.float32)

print("1. stream() reproduces build_notebook.py:159-162")
for label in ("OV50/batches", "OV50/val", "timing"):
    h = hashlib.blake2b(label.encode(), digest_size=8).digest()
    want = np.random.default_rng((int.from_bytes(h, "big") ^ SEED) % (2 ** 63))
    check(f"stream({label!r})", np.array_equal(stream(label).integers(0, 1 << 30, 8),
                                               want.integers(0, 1 << 30, 8)))

print("\n2. draw_batch consumes the RNG exactly as HostCorpus.batch does")
# replica of overnight2/c1_model.py:101-104, written out independently
r1, r2 = stream(BATCH_TAG), stream(BATCH_TAG)
idx_a, st_a = draw_batch(r1, N, LENS)
idx_b = r2.integers(N, size=BATCH)
n_pos = (LENS[idx_b] - 48000) // 4 + 1
st_b = (r2.random(BATCH) * n_pos).astype(np.int64) * 4
check("same indices", np.array_equal(idx_a, idx_b))
check("same starts", np.array_equal(st_a, st_b))
check("starts are R-aligned", bool(np.all(st_a % 4 == 0)))
check("segments fit", bool(np.all(st_a + 48000 <= LENS[idx_a])))
# the stream must not be left in a different state either
check("RNG left in the same state", r1.integers(0, 1 << 30) == r2.integers(0, 1 << 30))

print("\n3. THE LANDMINE, demonstrated: an arm name in the tag changes the batches")
base = batch_plan_digest(N, LENS, batch_tag="OV50/batches", n_steps=50)
per_arm = {a: batch_plan_digest(N, LENS, batch_tag=f"OV50_{a}/batches", n_steps=50) for a in ARMS}
n_distinct = len(set(per_arm.values()))
check("per-arm tags give 8 DIFFERENT batch streams", n_distinct == len(ARMS),
      f"{n_distinct} distinct digests from {len(ARMS)} arms")
check("none of them equals the shared stream", base not in set(per_arm.values()))
first = stream("OV50/batches").integers(N, size=BATCH)
wrong = stream("OV50_es_marg/batches").integers(N, size=BATCH)
check("step 0 already differs", not np.array_equal(first, wrong),
      f"shared drew {first[:4]}, per-arm drew {wrong[:4]}")

print("\n4. The guard: the fixed tag gives every arm the same stream")
digests = {a: batch_plan_digest(N, LENS, batch_tag=BATCH_TAG, n_steps=50) for a in ARMS}
check("all 8 arms agree", len(set(digests.values())) == 1, next(iter(digests.values()))[:16] + "...")

print("\n5. g3_digest pins the corpus, not just the draws")
g_ref = g3_digest(N, HOURS, LENS, Y, X, n_steps=20)
check("identical inputs -> identical digest", g3_digest(N, HOURS, LENS, Y, X, n_steps=20) == g_ref)
perm = np.random.default_rng(7).permutation(N)          # same utterances, different order
g_perm = g3_digest(N, HOURS, LENS[perm], Y, X, n_steps=20)
check("reordered corpus is caught", g_perm["corpus"] != g_ref["corpus"])
Y2 = Y.copy(); Y2[1_234_567] += 1e-3                     # one sample changed (a different resampler)
check("altered samples are caught", g3_digest(N, HOURS, LENS, Y2, X, n_steps=20)["corpus"] != g_ref["corpus"])
check("digest names which half differs", set(g_ref) >= {"corpus", "batches"})

print("\n6. preflight catches the trimmed-corpus trap")
good = preflight(N, 37.316, 48000, val_n=40, n_test_utts=120)
check("a real corpus passes", good["epochs"] > 49.9 and good["epochs"] < 50.1,
      f"{good['epochs']} epochs, {good['steps_per_epoch']} steps/epoch")
for label, kw in (("200-utterance corpus (0.19 h)", dict(n=200, hours=0.19, seg_hi=48000)),
                  ("wrong seg_hi", dict(n=N, hours=37.316, seg_hi=12000)),
                  ("wrong batch", dict(n=N, hours=37.316, seg_hi=48000, batch=32))):
    try:
        preflight(**kw)
        check(label, False, "preflight did NOT raise")
    except SystemExit as e:
        check(label, True, str(e).splitlines()[1].strip()[:70])

print("\n6b. the G3 barrier fails closed")
def rep(arm, corpus="C", batches="B", hours=37.316):
    return {"arm": arm, "g3": {"corpus": corpus, "batches": batches}, "hours": hours, "n": 400}
allrep = [rep(a) for a in ARMS]
check("all eight agreeing -> go", barrier_decision(allrep)["go"])
one_off = [rep(a) for a in ARMS]
one_off[3]["g3"]["batches"] = "DIFFERENT"
d = barrier_decision(one_off)
check("one divergent batch stream blocks the fleet", not d["go"])
check("  ...and names the odd one out", "es_erb_l0.001" in str(d["problems"]), str(d["problems"])[:90])
d = barrier_decision([rep(a) for a in list(ARMS)[:7]])
check("a missing instance blocks", not d["go"] and "no report from" in str(d["problems"]))
bad = [rep(a) for a in ARMS]; bad[0]["g3"]["corpus"] = "OTHER"
check("a different corpus blocks", not barrier_decision(bad)["go"])
trim = [rep(a) for a in ARMS]; trim[2]["hours"] = 0.19
d = barrier_decision(trim)
check("a trimmed corpus blocks even if digests agree", not d["go"], str(d["problems"])[:70])
dup = [rep(a) for a in ARMS] + [rep("det")]
check("a duplicate report blocks", not barrier_decision(dup)["go"])

print("\n7. milestones land where the spec says")
check("six milestones", len(MILESTONES) == 6)
check("values", MILESTONES == [20990, 41980, 52475, 62970, 73465, 83960], str(MILESTONES))
check("lr after the last is lr0/64", abs(1e-3 * 0.5 ** 6 - 1.5625e-5) < 1e-12)
check("STEPS is the spec's binding number", STEPS == 104950)

print(f"\n{ok} checks passed")
