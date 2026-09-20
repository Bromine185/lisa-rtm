"""The run contract: the eight arms, and the guard that stops the batch streams diverging.

WHY THIS FILE EXISTS.  The batch sampler is seeded by a NAME, not a position:

    build_notebook.py:159    stream(label) -> default_rng(blake2b(label) ^ SEED)
    e2b_fast.py:592          rng = stream(f"{tag}/batches")
    e2_trainer.py:11         rng = stream(f"{tag}/val")

That is a good design -- it is why all arms see identical batches, and why the property survives
being split across processes for free.  It is also a trap.  The 50-epoch run puts one arm on each of
eight separate VMs, and each VM needs its own output paths (`ov3_history_{tag}_{arm}.json`,
`CKPT/{tag}/{arm}.pt`).  The one-line way to get them is to pass f"{tag}_{arm}" as `tag` -- and that
same string reaches stream().  Every arm then draws a DIFFERENT 105,000-step batch stream and a
DIFFERENT validation set.

Nothing crashes.  No existing test catches it: overnight3/smoke_fast_local.py runs every arm in ONE
process, so the tags agree there by construction.  Eight separate VMs share no filesystem.  The
paired comparison -- the entire scientific product of the run -- is destroyed, and the only symptom
is that the per-arm loss curves stop sharing step-to-step noise, which nobody thinks to look for.

So: BATCH_TAG and VAL_TAG are fixed constants here.  They are NOT derived from the output tag and
NOT parameterised by arm.  `g3_digest()` is the cross-machine check: every instance computes it
before training and publishes it; if the eight digests are not identical, nothing trains.

Pure numpy and hashlib -- no torch, no GPU -- so it runs in CI and on a laptop.
"""
import hashlib
import numpy as np

# ---- run constants (algorithm spec section 1; arm allocation note) ----------------------------------
SEED = 0                      # build_notebook.py:157
STEPS = 104_950               # 50 epochs at batch 64 on the 37.3 h corpus; spec section 8 binds this
BATCH = 64
SEG_HI = 48_000               # 1 s at 48 kHz
R = 4                         # 12 kHz -> 48 kHz
LR = 1e-3
LR_FRACS = (0.2, 0.4, 0.5, 0.6, 0.7, 0.8)     # == LISA's [10,20,25,30,35,40] / 50
LR_GAMMA = 0.5
GRAD_CLIP = 1e-3

# The six lr milestones as absolute steps.  int(f * STEPS), matching MultiStepLR construction in
# e2b_fast.py::_make_opts.  Five of the six are NOT multiples of log_every=25 -- see the note in
# audit/dynamics_ov3.py about why the analysis must not require an exact hit.
MILESTONES = [int(f * STEPS) for f in LR_FRACS]

# ---- the eight arms (notes/2026-09-20-arm-allocation.md) --------------------------------------------
# (kind, lambda, class).  Kinds are the strings ArmStack.__init__ switches on (e2b_fast.py:217-220);
# "es_marg_erb" is the 0.5/0.5 blend of log-magnitude and ERB-band geometries.
ARMS = {
    "det_paper":       ("det",         0.0,  "LISAS"),    # LISA's released config: loss: l1, lambda = 0
    "det":             ("det",         1e-2, "LISAS"),    # our deterministic baseline
    "es_marg":         ("es_marg",     1e-2, "LISAS"),    # reference sampler, no ERB term
    "es_erb_l0.001":   ("es_marg_erb", 1e-3, "LISAS"),
    "es_erb_l0.01":    ("es_marg_erb", 1e-2, "LISAS"),    # matched-lambda partner of es_marg
    "es_erb_l0.1":     ("es_marg_erb", 1e-1, "LISAS"),    # the OV3_fast winner
    "es_dec_l0.01":    ("es_marg",     1e-2, "LISASD"),   # decoder noise at matched lambda
    "es_dec_erb_l0.1": ("es_marg_erb", 1e-1, "LISASD"),   # decoder noise + ERB at high lambda
}

# ---- THE FIXED STREAM LABELS.  Do not interpolate an arm name into these. --------------------------
# The output-path tag varies per arm; these do not.  Every instance of the run uses these exact
# strings, so every arm draws the same segments in the same order.
RUN_TAG = "OV50"              # output paths only: ov3_history_{RUN_TAG}_{arm}.json, CKPT/{RUN_TAG}/{arm}.pt
BATCH_TAG = "OV50/batches"    # <- reaches stream(); identical on all eight machines
VAL_TAG = "OV50/val"          # <- reaches stream(); identical on all eight machines


def stream(label, seed=SEED):
    """Byte-for-byte build_notebook.py:159-162.  Reimplemented rather than imported because that
    file is a notebook generator that cannot be imported standalone."""
    h = hashlib.blake2b(label.encode(), digest_size=8).digest()
    return np.random.default_rng((int.from_bytes(h, "big") ^ seed) % (2 ** 63))


def draw_batch(rng, n, lens, seg_hi=SEG_HI, r=R, batch=BATCH):
    """Exactly HostCorpus.batch's RNG consumption (overnight2/c1_model.py:101-104): one
    rng.integers(n, size=B), then one rng.random(B), in that order.  Returns (idx, starts).

    The ORDER and the COUNT are the contract.  Any rewrite that draws these differently -- even if it
    produces statistically identical batches -- gives a different realised sequence and breaks the
    pairing between arms.  Gate G2 (RNG consumption trace) exists to catch that.
    """
    idx = rng.integers(n, size=batch)
    n_pos = (lens[idx] - seg_hi) // r + 1
    starts = (rng.random(batch) * n_pos).astype(np.int64) * r
    return idx, starts


def batch_plan_digest(n, lens, batch_tag=BATCH_TAG, n_steps=1000, seed=SEED, **kw):
    """Digest of the first n_steps drawn (idx, starts) pairs.

    This is the thing that differs when someone interpolates an arm name into the tag, and it is
    cheap: 1000 steps of numpy integer draws, well under a second.
    """
    rng = stream(batch_tag, seed)
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{n}|{seg_of(kw)}|{n_steps}".encode())
    for _ in range(n_steps):
        idx, starts = draw_batch(rng, n, lens, **kw)
        h.update(idx.astype(np.int64).tobytes())
        h.update(starts.astype(np.int64).tobytes())
    return h.hexdigest()


def seg_of(kw):
    return f"{kw.get('seg_hi', SEG_HI)}/{kw.get('r', R)}/{kw.get('batch', BATCH)}"


def corpus_fingerprint(n, hours, lens, y, x, chunk=1 << 26):
    """Identity of the corpus itself, not of the draws into it.

    Identical indices into DIFFERENT corpora are different data, so G3 has to pin the corpus as well
    as the index stream.  Each VM builds its own corpus from its own download; if an utterance is
    dropped, reordered, decimated with a different filter, or arrives corrupt, this digest moves.

    Hashes EVERY sample, not a sample of them.  An earlier version probed 64 windows of 4096 samples
    and was rejected by fast/test_run_contract.py: at 32.2 GB that covers ~13 % of the corpus, so a
    corrupted block has an ~87 % chance of passing.  Measured cost of the full hash on this repo's
    venv is 637 MB/s through a no-copy memoryview, i.e. ~51 s for the 32.2 GB corpus -- $0.025 of
    instance time per VM, $0.20 across eight, against a $59 run that the sparse version could let
    through silently.  Chunked so the whole array is never copied into a bytes object.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{int(n)}|{float(hours):.6f}".encode())
    h.update(np.asarray(lens, np.int64).tobytes())
    for arr, name in ((y, "Y"), (x, "X")):
        a = np.ascontiguousarray(arr)
        h.update(f"{name}|{a.shape}|{a.dtype}".encode())
        mv = memoryview(a).cast("B")          # raw bytes, no copy
        for s in range(0, len(mv), chunk):
            h.update(mv[s:s + chunk])
    return h.hexdigest()


def g3_digest(n, hours, lens, y, x, batch_tag=BATCH_TAG, n_steps=1000):
    """The cross-machine gate.  Every instance prints this before training; a launcher collects all
    eight and refuses to start the run unless they are identical.

    Returns a dict rather than one string so a mismatch says WHICH half differs: the corpus build or
    the index stream.  Those have different fixes.
    """
    return {
        "corpus": corpus_fingerprint(n, hours, lens, y, x),
        "batches": batch_plan_digest(n, lens, batch_tag=batch_tag, n_steps=n_steps),
        "batch_tag": batch_tag,
        "val_tag": VAL_TAG,
        "steps": STEPS,
        "batch": BATCH,
    }


def preflight(n, hours, seg_hi, val_n=None, n_test_utts=None, steps=STEPS, batch=BATCH):
    """The hard gate, run before the optimiser is constructed.  Raises on any failure.

    e3_launch.py:33-40 builds `corpus = HostCorpus(train_utts, ...)` and only THEN trims train_utts
    to 200 utterances to free RAM, and it has an `isinstance(globals().get("corpus"), HostCorpus)`
    reuse branch.  A rewrite that reorders those two statements, or reuses a kernel-resident
    200-utterance corpus, builds a ~0.19 h corpus, computes ~11 steps/epoch, and runs "50 epochs" in
    a few minutes -- cheap, plausible-looking, and scientifically void.  Because the step count is
    DERIVED from corpus.hours there is no constant left to check it against, which is why this gate
    brackets corpus.hours explicitly.
    """
    fails = []
    spe = hours * 56.25          # steps/epoch at BATCH=64, SEG=48000 (e8_launch_fast.py:25)
    if seg_hi != SEG_HI:
        fails.append(f"seg_hi {seg_hi} != {SEG_HI}")
    if not (37.2 < hours < 37.4):
        fails.append(f"corpus.hours {hours:.6g} outside (37.2, 37.4) -- the trimmed-corpus trap")
    if not (2090 < spe < 2110):
        fails.append(f"steps/epoch {spe:.2f} outside (2090, 2110)")
    if abs(round(50 * spe) - STEPS) > 200:
        fails.append(f"50 epochs is {round(50 * spe)} steps, not within 200 of {STEPS}")
    if steps != STEPS:
        fails.append(f"steps {steps} != {STEPS}")
    if batch != BATCH:
        fails.append(f"batch {batch} != {BATCH}")
    if val_n is not None and val_n != 40:
        fails.append(f"val_corpus.n {val_n} != 40")
    if n_test_utts is not None and n_test_utts != 120:
        fails.append(f"len(test_utts) {n_test_utts} != 120")
    if fails:
        raise SystemExit("PREFLIGHT FAILED:\n  " + "\n  ".join(fails))
    return {"corpus_hours": round(hours, 6), "steps_per_epoch": round(spe, 3),
            "epochs": round(steps / spe, 4), "n_utts": int(n), "milestones": MILESTONES}


if __name__ == "__main__":
    for k, (kind, lam, cls) in ARMS.items():
        print(f"{k:<18} {kind:<12} lam={lam:<7g} {cls}")
    print(f"\nSTEPS {STEPS}  BATCH {BATCH}  milestones {MILESTONES}")
    print(f"BATCH_TAG {BATCH_TAG!r}  VAL_TAG {VAL_TAG!r}  (fixed: never per-arm)")
