"""Fetch just the utterances an evaluation needs from sanchit-gandhi/vctk, without the 11.7 GB.

    python fast/vctk_local.py --speakers p236 p237 p238 --per-speaker 40

WHY NOT snapshot_download.  overnight2/c0b_boot_light.py builds `test_utts` by downloading all 27
parquet shards (11.7 GB) and throwing away everything that is not a held-out speaker.  On the
instance that was free; on a laptop with 55 GB spare it is the largest single cost of the evaluation,
and EVAL12 is twelve utterances.

The shards carry per-row-group statistics on `speaker_id`, and they are written in speaker order, so
a held-out speaker occupies a contiguous run of ~11 row groups in ONE shard.  Reading the footers
(one range request each, ~1 s per shard) locates that run; reading those row groups costs ~20 MB
each.  p236's first 40 utterances cost about 230 MB instead of 11.7 GB.

THE CONSTRUCTION IS c0b_boot_light.py:84-90, NOT A REIMPLEMENTATION OF IT.  That file is what the
OV2/OV3 evaluations ran against, so the ordering has to be identical or EVAL12 is a different twelve
utterances and every number moves:

    keep mic1 only, by `file` or by `audio.path`
    load_bytes: mono, resample to 48 kHz (a no-op -- VCTK mic1 is already 48 kHz), drop < 1 s,
                peak-normalise to 0.95 in float64, cast to float32 ONCE at the end
    sort by (speaker_id, file) -- `file` is an absolute path with a constant prefix, so within a
                speaker this is basename order, p236_001 before p236_002
    take the first `per_speaker` of each speaker, speakers in sorted order

`load_bytes` is imported from fast/stage_corpus.py rather than copied, because that copy is the one
the training corpus was built with.

WHAT THIS IS NOT.  It cannot build the 39,639-utterance TRAINING corpus: every train utterance's
length is needed for the batch sampler, and a length needs the audio.  fast/val_wave_check.py solves
that a different way.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fast.stage_corpus import HF_REPO, OUR_TEST, load_bytes

GLOB = f"datasets/{HF_REPO}/data/*.parquet"
COLS = ["speaker_id", "file", "audio"]


def _fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem()


def shards(fs=None):
    return sorted((fs or _fs()).glob(GLOB))


def locate(fs, paths, speakers):
    """{speaker: [(shard_path, row_group), ...]} from the footers alone.

    A row group is kept when its [min, max] speaker_id range can contain the speaker; the boundary
    groups hold two speakers and are read by both, which costs one extra row group per speaker."""
    import pyarrow.parquet as pq
    want, out = set(speakers), {s: [] for s in speakers}
    for p in paths:
        with fs.open(p, "rb") as f:
            m = pq.ParquetFile(f).metadata
            col = [m.schema.column(i).path for i in range(m.num_columns)].index("speaker_id")
            for rg in range(m.num_row_groups):
                st = m.row_group(rg).column(col).statistics
                if st is None:
                    raise SystemExit(f"{p} rg{rg}: no speaker_id statistics; fall back to a full read")
                for s in want:
                    if st.min <= s <= st.max:
                        out[s].append((p, rg))
    return out


def read_utts(fs, plan, speaker, min_keep):
    """Row groups in order until `min_keep` valid mic1 utterances are in hand, then one more.

    The extra group is the ordering guard: `file` is only known to be sorted within what has been
    read, so stopping the instant the count is reached could leave a later row that sorts before the
    last kept one.  One group beyond (100 rows, ~50 utterances) is far more than the parquet's
    observed disorder, and read_utts asserts sortedness of what it got."""
    import pyarrow.parquet as pq
    keep, extra = [], 0
    for p, rg in plan:
        if len(keep) >= min_keep:
            extra += 1
            if extra > 1:
                break
        with fs.open(p, "rb") as f:
            t = pq.ParquetFile(f).read_row_group(rg, columns=COLS)
        spks, files, auds = (t.column(c).to_pylist() for c in COLS)
        for i in range(len(files)):
            if spks[i] != speaker:
                continue
            if not (("mic1" in (files[i] or "")) or ("mic1" in ((auds[i] or {}).get("path") or ""))):
                continue
            a = load_bytes(auds[i]["bytes"])
            if a is not None:
                keep.append((files[i], a))
        del t, auds
    names = [f for f, _ in keep]
    if names != sorted(names):
        keep.sort(key=lambda r: r[0])
        print(f"  {speaker}: parquet order was not file order; sorted {len(keep)} rows", flush=True)
    return keep


def test_utts(speakers=tuple(sorted(OUR_TEST)), per_speaker=40, cache=None, verbose=True):
    """c0b_boot_light.py:86-90's `test_utts`, from a partial read.

    Cached as one .npz per speaker plus a manifest, so a second evaluation costs nothing."""
    speakers = tuple(sorted(speakers))
    cache = pathlib.Path(cache).expanduser() if cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    out, names = [], []
    fs = paths = plan = None
    for s in speakers:
        npz = cache / f"{s}.npz" if cache else None
        if npz and npz.exists():
            z = np.load(npz)
            got = [(str(k), z[k]) for k in z.files]
            got.sort(key=lambda r: r[0])
            if len(got) >= per_speaker:
                if verbose:
                    print(f"  {s}: {len(got)} cached", flush=True)
                out += [a for _, a in got[:per_speaker]]
                names += [f for f, _ in got[:per_speaker]]
                continue
        if fs is None:
            t0 = time.time()
            fs = _fs()
            paths = shards(fs)
            plan = locate(fs, paths, speakers)
            if verbose:
                print(f"  located {len(paths)} shards, "
                      + " ".join(f"{k}:{len(v)}rg" for k, v in plan.items()) + f"  [{time.time()-t0:.0f}s]", flush=True)
        t0 = time.time()
        got = read_utts(fs, plan[s], s, per_speaker)
        if len(got) < per_speaker:
            raise SystemExit(f"{s}: only {len(got)} utterances, need {per_speaker}")
        if verbose:
            print(f"  {s}: {len(got)} read, keeping {per_speaker}  [{time.time()-t0:.0f}s]", flush=True)
        if npz:
            np.savez(npz, **{f: a for f, a in got})
        out += [a for _, a in got[:per_speaker]]
        names += [f for f, _ in got[:per_speaker]]
    return out, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", nargs="*", default=sorted(OUR_TEST))
    ap.add_argument("--per-speaker", type=int, default=40)
    ap.add_argument("--cache", default="~/lisa-results/vctk_test")
    a = ap.parse_args()
    utts, names = test_utts(a.speakers, a.per_speaker, a.cache)
    print(f"\n{len(utts)} utterances, {sum(len(u) for u in utts)/48000:.1f} s total")
    for i, (n, u) in enumerate(zip(names[:12], utts[:12])):
        print(f"  [{i:>2}] {pathlib.Path(n).name:<24} {len(u):>7} samples  {len(u)/48000:5.2f} s  peak {np.abs(u).max():.4f}")
    print(f"\nEVAL12 = the first 12 above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
