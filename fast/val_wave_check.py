"""Re-run each converted arm's own objective on the run's actual validation batches.

    python fast/val_wave_check.py --src ~/lisa-results

WHAT IT IS FOR.  fast/convert_ckpt.py proves the conversion is a faithful copy: every exported
tensor is bitwise equal to the stacked one, and load_arm() round-trips it.  What it cannot prove is
that those weights are the weights that produced the curve in history_OV50_<arm>.json -- a converter
that silently paired es_marg's weights with es_erb_l0.1's name would pass every check in that file.
This one puts the weights back through val_loss_fast on the same validation batches the trainer used
and compares the waveform term against the recorded one, for all eight arms at once.

REBUILDING THE VALIDATION BATCHES WITHOUT THE 32 GB CORPUS.  fast/train_arm.py sets
`val_corpus = corpus`, so the 8 x 64 validation segments are drawn out of the full 37.3 h training
corpus, which lived on the terminated instance.  It does not have to be rebuilt:

  * `stream(VAL_TAG)` draws `idx = rng.integers(n, size=64)` and then `starts` from `lens[idx]`.
    Only `n` and the LENGTHS are needed to reproduce the draws -- not the samples.
  * `off[idx] + starts` always lands a whole segment inside utterance `idx`, so the concatenation is
    never needed either: decoding those ~512 utterances and slicing each one gives the same audio.

So the expensive part is `lens` for all 39,639 -- and a length is a FLAC header, not a decode.  The
index below reads every shard once and calls sf.info on the bytes, and is then checked against the
run's own manifest.json: n == 39639 AND sum(lens) == 3600 * 48000 * manifest["hours"] exactly, to
the sample.  That integer is the gate.  If the utterance set, the sort order, the mic1 filter, the
1-second drop or the truncation to a multiple of R were off by one utterance, the sum would not land
on 6,449,303,392, and the batch indices would be indices into a different corpus.

WHY THE NUMBERS AGREE CLOSELY RATHER THAN EXACTLY.  Two of the recorded val_wave's inputs do not
exist off the training hardware, and neither is a bug:

  1. AMP.  FAST["AMP"] is _CUDA, so training ran the encoder and decoder under bf16 autocast
     (e2b_fast.py:112) with the distances in fp32.  On this laptop FAST["AMP"] is False and the
     whole forward is fp32.  bf16 carries 8 mantissa bits, so yhat differs in the third decimal
     place and a mean absolute error over it differs in the third or fourth.
  2. eps.  val_loss_fast draws the sampler's noise from torch.Generator(device=DEVICE) at a fixed
     seed.  A CUDA generator and a CPU generator at seed 1234 are different streams, and per
     fast/train_arm.py's header the CUDA stream is not even portable between GPU models.  The six
     stochastic arms therefore see a different (valid) eps, and their val_wave carries the energy
     score's Monte-Carlo noise over 512 segments.  The two deterministic arms take the
     `self.is_det` branch of ArmStack.losses, which never touches eps, so only (1) applies to them.

THE TEST IS THEREFORE A MATCHING, NOT AN EQUALITY.  Both val_wave and val_spec are recomputed for
all eight converted modules on the same batches and compared against all eight recorded pairs; an
arm has to be the closest match on BOTH terms.  Two terms are needed because one is not enough:
es_erb_l0.01 and es_marg were recorded 0.09% apart on val_wave, which is inside the reproduction
floor, and they are 35% apart on val_spec.  The match is taken among arms of the same CLASS, because
a LISAS checkpoint cannot be loaded as a LISASD or the reverse -- n_dec makes the decoder's first
Linear 101 wide instead of 97 and load_state_dict raises.  The pairs that remain hardest are
es_marg/es_dec_l0.01 and es_erb_l0.1/es_dec_erb_l0.1, and those are precisely the cross-class
matched-lambda partners that the shape check already separates.

The reproduction floor is not assumed: det and det_paper never draw eps, so their diagonal error is
fp32-against-bf16 and nothing else, and it is printed as the scale everything else is read against.
"""
import argparse
import io
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fast.run_contract import ARMS, BATCH, R, RUN_TAG, SEG_HI, VAL_TAG, stream
from fast.stage_corpus import HF_REPO, role
from fast.train_arm import boot

REPO = pathlib.Path(__file__).resolve().parents[1]
COLS = ["speaker_id", "file", "audio"]
N_VAL_BATCHES = 8              # fast/train_arm.py's train(..., n_val_batches=8)


def _mic1(file, aud):
    """c0b_boot_light.py:67-68.  Either field may carry the mic tag."""
    return ("mic1" in (file or "")) or ("mic1" in ((aud or {}).get("path") or ""))


def build_index(local, cache=None, verbose=True):
    """[(speaker, file, frames, shard, row_group, row)] for every train-role mic1 utterance.

    Row groups are read one at a time and the audio bytes dropped straight after sf.info, so peak
    memory is one row group (~20 MB) rather than the 11.7 GB of shards."""
    import pyarrow.parquet as pq
    import soundfile as sf
    if cache and pathlib.Path(cache).exists():
        d = json.loads(pathlib.Path(cache).read_text())
        if verbose:
            print(f"  index: {len(d['rows'])} rows cached in {cache}", flush=True)
        return [tuple(r) for r in d["rows"]]
    shards = sorted(pathlib.Path(local).glob("data/*.parquet"))
    rows, t0, odd = [], time.time(), {}
    for si, sh in enumerate(shards):
        pf = pq.ParquetFile(sh)
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=COLS)
            spks, files, auds = (t.column(c).to_pylist() for c in COLS)
            for i in range(len(files)):
                if role(spks[i]) != "train" or not _mic1(files[i], auds[i]):
                    continue
                info = sf.info(io.BytesIO(auds[i]["bytes"]))
                if info.samplerate != 48000 or info.channels != 1:
                    odd[(info.samplerate, info.channels)] = odd.get((info.samplerate, info.channels), 0) + 1
                rows.append((spks[i], files[i], int(info.frames), si, rg, i))
            del t, auds
        if verbose:
            print(f"  shard {si + 1:>2}/{len(shards)}  rows {len(rows):>6}  [{time.time() - t0:.0f}s]", flush=True)
    if odd:
        raise SystemExit(f"not all utterances are mono 48 kHz: {odd}; load_bytes would resample and "
                         f"sf.info's frame count would no longer be the length")
    rows.sort(key=lambda r: (r[0], r[1]))            # c0b_boot_light.py:85
    if cache:
        pathlib.Path(cache).write_text(json.dumps({"repo": HF_REPO, "rows": rows}))
    return rows


def corpus_from_index(rows, manifest, seg_hi=SEG_HI, r=R):
    """The surviving utterances and their lengths, checked against the run's own manifest.

    load_bytes drops anything under a second and assemble truncates to a multiple of R, so the
    length of utterance i is (frames // R) * R over frames >= seg_hi.  n and the SAMPLE TOTAL both
    have to match what the trainer recorded, and the total is the strict one: it is an integer that
    every utterance contributes to."""
    keep = [t for t in rows if t[2] >= seg_hi]
    lens = np.array([(t[2] // r) * r for t in keep], np.int64)
    want_n, want_s = int(manifest["n"]), int(round(manifest["hours"] * 3600 * 48000))
    got_s = int(lens.sum())
    fails = []
    if len(keep) != want_n:
        fails.append(f"n {len(keep)} != manifest {want_n}")
    if got_s != want_s:
        fails.append(f"sum(lens) {got_s} != manifest {want_s} (diff {got_s - want_s})")
    if fails:
        raise SystemExit("corpus index does not match the run:\n  " + "\n  ".join(fails))
    print(f"  corpus: n={len(keep)}  {got_s} samples  {got_s / 48000 / 3600:.11f} h  "
          f"== manifest ({manifest['hours']:.11f} h)", flush=True)
    return keep, lens


def draw_val(n, lens, n_batches=N_VAL_BATCHES, batch=BATCH, seg_hi=SEG_HI, r=R):
    """fast/train_arm.py:189-190's `vb`, as indices.

    ONE generator for all the batches -- e2_trainer.py:11-12.  The RNG consumption order inside a
    batch (one integers, then one random) is overnight2/c1_model.py:101-104 and is the contract that
    keeps the arms paired; getting it wrong here draws a different validation set and every number
    below moves together, which is exactly the failure that looks like success."""
    rng = stream(VAL_TAG)
    out = []
    for _ in range(n_batches):
        idx = rng.integers(n, size=batch)
        n_pos = (lens[idx] - seg_hi) // r + 1
        starts = (rng.random(batch) * n_pos).astype(np.int64) * r
        out.append((idx, starts))
    return out


def decode(local, keep, need, verbose=True):
    """{utterance index: (y_f32_truncated, x_f32_decimated)} for the indices in `need`.

    Grouped by (shard, row group) so each row group is read once, and both bands are built from the
    WHOLE utterance: resample_poly is not shift invariant, so decimating a slice is not the same as
    slicing the decimation, and X in the trainer was decimated per utterance before concatenation."""
    import pyarrow.parquet as pq
    from fast.stage_corpus import decimate, load_bytes
    shards = sorted(pathlib.Path(local).glob("data/*.parquet"))
    groups = {}
    for u in need:
        _, _, _, si, rg, row = keep[u]
        groups.setdefault((si, rg), []).append((row, u))
    out, t0 = {}, time.time()
    for gi, ((si, rg), items) in enumerate(sorted(groups.items())):
        t = pq.ParquetFile(shards[si]).read_row_group(rg, columns=["audio"])
        auds = t.column("audio").to_pylist()
        for row, u in items:
            y = load_bytes(auds[row]["bytes"])
            if y is None:
                raise SystemExit(f"utterance {u} ({keep[u][1]}) decoded to None; the index and the "
                                 f"corpus disagree about which utterances survive")
            y = y[: (len(y) // R) * R]
            if len(y) != (keep[u][2] // R) * R:
                raise SystemExit(f"utterance {u}: decoded {len(y)} samples, header said {keep[u][2]}")
            out[u] = (y, decimate(y.astype(np.float64), R).astype(np.float32))
        del t, auds
        if verbose and (gi + 1) % 50 == 0:
            print(f"    {gi + 1}/{len(groups)} row groups  [{time.time() - t0:.0f}s]", flush=True)
    if verbose:
        print(f"  decoded {len(out)} utterances from {len(groups)} row groups  [{time.time() - t0:.0f}s]", flush=True)
    return out


def make_vb(plan, cache, torch, seg_hi=SEG_HI, r=R, device=None, sub=None):
    """The (x, y) tensor pairs, exactly StagedCorpus.batch's slicing with off[] folded away.

    `sub` splits each 64-segment batch into equal chunks, for memory: the sub-pixel decoder
    materialises a (2B x 48000, 144) activation per MLP layer, which at 2B = 128 is 3.5 GB, and on a
    16 GB laptop the stochastic arms go to swap and lose an order of magnitude of speed.

    SPLITTING IS EXACT FOR THE STOCHASTIC ARMS AND NOT FOR THE DETERMINISTIC ONES, so
    val_wave_all() uses the split batches only for the former.  Every term of ArmStack.losses is a
    per-sample mean over the batch axis -- which averages correctly over equal chunks -- EXCEPT the
    spectral-convergence term in the `is_det` branch (e2b_fast.py:347):

        sc = (Y - H).pow(2).sum((1, 2, 3)).sqrt() / (tf.mag_norm[s] + 1e-8)

    That sums over the batch INSIDE the square root and divides by a norm taken over the whole
    batch, so it is a ratio of batch aggregates, and a mean of per-chunk ratios is not the same
    number.  Measured on `det`, splitting four ways moves val_spec by 0.2% and val_wave not at all.
    0.2% is small, but this file exists to say how far off a reproduction is, and an unexplained
    0.2% inside it would be indistinguishable from a real discrepancy."""
    vb = []
    for idx, starts in plan:
        ys = np.stack([cache[u][0][s: s + seg_hi] for u, s in zip(idx, starts)])
        xs = np.stack([cache[u][1][s // r: s // r + seg_hi // r] for u, s in zip(idx, starts)])
        n = len(ys) if not sub else sub
        if len(ys) % n:
            raise SystemExit(f"--sub-batch {n} does not divide the batch size {len(ys)}; the chunks "
                             f"must be equal or their means do not average to the batch mean")
        for i in range(0, len(ys), n):
            vb.append((torch.from_numpy(xs[i:i + n]).to(device), torch.from_numpy(ys[i:i + n]).to(device)))
    return vb


def val_wave_all(G, vb, arms, ckpt_dir, torch, vb_det=None):
    """{arm: (val_loss, val_wave, val_spec, cls)} from each arm's own stack on the shared batches.

    Deterministic arms get `vb_det` (the unsplit batches) because their spectral term is batch-
    coupled -- see make_vb.  They can afford it: `is_det` runs B sequences where a sampler runs 2B."""
    from fast.convert_ckpt import build, import_module
    out = {}
    for arm in arms:
        # load_arm, not torch.load: this re-runs the file e4_eval reads, through the code path
        # e4_eval reads it with, so a conversion that produced a loadable-but-wrong file fails here.
        module, ck = G["load_arm"](pathlib.Path(ckpt_dir) / f"{arm}.pt")
        stack, _, spec = build(G, arm, torch.device("cpu"))
        if tuple(ck["arm"]) != spec:
            raise SystemExit(f"{arm}: checkpoint says {tuple(ck['arm'])}, run_contract says {spec}")
        import_module(stack, module.cpu(), torch)
        t0 = time.time()
        v = G["val_loss_fast"]([stack], (vb_det or vb) if stack.is_det else vb)[arm]
        out[arm] = (v[0], v[1], v[2], spec[2])
        print(f"  {arm:<18} val_loss {v[0]:.6f}  val_wave {v[1]:.9f}  val_spec {v[2]:.6f}  [{time.time() - t0:.0f}s]", flush=True)
    return out


def report(got, recorded, arms):
    """Each arm's recomputed (val_wave, val_spec) against every arm's recorded pair.

    DISTANCE is the larger of the two relative errors, so an arm has to match on BOTH terms.
    val_wave alone does not separate the arms: es_erb_l0.01 and es_marg were recorded 0.09% apart on
    it, which is inside the fp32-vs-bf16 reproduction floor.  val_spec separates exactly those two
    (0.282 against 0.436), and the pairs val_spec cannot separate are separated by val_wave.

    CANDIDATES for an arm are the arms of the same class.  A LISAS checkpoint cannot be loaded as
    LISASD or the reverse: n_dec changes the decoder's first Linear from 97 to 101 inputs and
    load_state_dict raises.  Restricting the match to what could actually be confused is the honest
    comparison -- and it is not a loophole, because the two cross-class pairs here
    (es_marg/es_dec_l0.01, es_erb_l0.1/es_dec_erb_l0.1) are the matched-lambda partners, i.e. the
    closest pairs in the run by construction.

    PASSING means every arm's own recorded pair is its nearest candidate, in both directions."""
    A = list(arms)
    W = {a: got[a][1] for a in A}
    S = {a: got[a][2] for a in A}
    cls = {a: got[a][3] for a in A}
    dw = lambda a, b: abs(W[a] - recorded[b][0]) / recorded[b][0]
    ds = lambda a, b: abs(S[a] - recorded[b][1]) / recorded[b][1]
    D = np.array([[max(dw(a, b), ds(a, b)) for b in A] for a in A])
    w = max(len(a) for a in A)

    print(f"\n{'arm':<{w}} {'cls':<7} {'val_wave':>12} {'recorded':>12} {'d%':>7}   "
          f"{'val_spec':>10} {'recorded':>10} {'d%':>7}")
    for a in A:
        print(f"{a:<{w}} {cls[a]:<7} {W[a]:>12.9f} {recorded[a][0]:>12.9f} {100 * dw(a, a):>7.3f}   "
              f"{S[a]:>10.6f} {recorded[a][1]:>10.6f} {100 * ds(a, a):>7.3f}")

    det = [a for a in A if a.startswith("det")]
    if det:
        print(f"\n  reproduction floor, from the arms whose objective never draws eps "
              f"({', '.join(det)}): {max(max(dw(a, a), ds(a, a)) * 100 for a in det):.3f}% "
              f"-- this is fp32 here against bf16 autocast there, nothing else.")

    print(f"\nmax(relative error on val_wave, on val_spec), in %  "
          f"(rows: converted arm, cols: recorded curve; . = different class, cannot be confused)\n")
    print(" " * (w + 2) + "".join(f"{b[:8]:>9}" for b in A))
    for i, a in enumerate(A):
        cells = ""
        for j, b in enumerate(A):
            if cls[a] != cls[b]:
                cells += " " + f"{'.':>8}"
            else:
                cells += ("*" if i == j else " ") + f"{100 * D[i, j]:>8.3f}"
        print(f"  {a:<{w}}" + cells)

    bad, margins = [], []
    for i, a in enumerate(A):
        cand = [j for j, b in enumerate(A) if cls[b] == cls[a]]
        best = min(cand, key=lambda j: D[i, j])
        if A[best] != a:
            bad.append(f"{a} matches {A[best]}'s curve better than its own")
        others = [D[i, j] for j in cand if A[j] != a]
        if others:
            margins.append(min(others) / max(D[i, i], 1e-12))
    for j, b in enumerate(A):
        cand = [i for i, a in enumerate(A) if cls[a] == cls[b]]
        best = min(cand, key=lambda i: D[i, j])
        if A[best] != b:
            bad.append(f"{b}'s curve is matched better by {A[best]} than by {b}")
    return not bad, sorted(set(bad)), margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/lisa-results")
    ap.add_argument("--tag", default=RUN_TAG)
    ap.add_argument("--arms", nargs="*", default=sorted(ARMS))
    ap.add_argument("--local", default=None, help="parquet snapshot dir (default: download/reuse the HF cache)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sub-batch", type=int, default=16,
                    help="split each 64-segment validation batch into chunks of this size (exact; see make_vb)")
    a = ap.parse_args()

    src = pathlib.Path(a.src).expanduser().resolve()
    manifest = json.loads((src / "manifest.json").read_text())
    ckpt_dir = src / "ckpt" / a.tag

    local = a.local
    if local is None:
        from huggingface_hub import snapshot_download
        print(f"corpus shards ({HF_REPO}):", flush=True)
        t0 = time.time()
        local = snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=["data/*.parquet"], max_workers=8)
        print(f"  {local}  [{time.time() - t0:.0f}s]", flush=True)

    print("\nindex:", flush=True)
    rows = build_index(local, cache=src / "train_index.json")
    keep, lens = corpus_from_index(rows, manifest)

    print("\nvalidation batches:", flush=True)
    plan = draw_val(len(keep), lens)
    need = sorted({int(u) for idx, _ in plan for u in idx})
    print(f"  {len(plan)} x {BATCH} segments from {len(need)} distinct utterances", flush=True)
    cache = decode(local, keep, need)

    cwd = pathlib.Path.cwd()
    import os
    os.chdir(src)
    try:
        G = boot(device="cpu", root=src)
    finally:
        os.chdir(cwd)
    import torch
    torch.set_num_threads(a.threads)
    vb_det = make_vb(plan, cache, torch, device=torch.device("cpu"))
    vb = make_vb(plan, cache, torch, device=torch.device("cpu"), sub=a.sub_batch) if a.sub_batch else vb_det
    print(f"  vb: {len(vb_det)} x {tuple(vb_det[0][1].shape)} hi for the det arms; "
          f"{len(vb)} x {tuple(vb[0][1].shape)} hi for the samplers "
          f"(split {BATCH // a.sub_batch} ways for memory)" if a.sub_batch else
          f"  vb: {len(vb_det)} x {tuple(vb_det[0][1].shape)} hi", flush=True)

    print("\nrecomputed (fp32, CPU eps -- see this file's docstring):", flush=True)
    got = val_wave_all(G, vb, a.arms, ckpt_dir, torch, vb_det=vb_det)
    recorded = {}
    for arm in a.arms:
        h = json.loads((src / f"history_{a.tag}_{arm}.json").read_text())[arm]
        recorded[arm] = (h["val_wave"][-1], h["val_spec"][-1], h["val_loss"][-1])

    out = src / "ov3" / f"val_wave_{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({arm: {"cls": got[arm][3],
                                     "recomputed": {"val_loss": got[arm][0], "val_wave": got[arm][1],
                                                    "val_spec": got[arm][2]},
                                     "recorded": {"val_loss": recorded[arm][2], "val_wave": recorded[arm][0],
                                                  "val_spec": recorded[arm][1]}} for arm in a.arms}, indent=1))

    ok, bad, margins = report(got, recorded, a.arms)
    print(f"\n  results written to {out}")
    if not ok:
        print("\nVAL_WAVE CHECK FAILED:\n  " + "\n  ".join(bad), flush=True)
        return 1
    print(f"\nVAL_WAVE CHECK PASSED: all {len(a.arms)} arms match their own recorded curve best among "
          f"the arms they could be confused with, by a margin of {min(margins):.1f}x at worst.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
