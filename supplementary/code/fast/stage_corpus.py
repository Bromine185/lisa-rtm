"""Stage the 37.3 h VCTK corpus onto an ephemeral VM, then prove it is the same corpus everywhere.

    python fast/stage_corpus.py --out /data/corpus            # the real thing, ~15-20 min
    python fast/stage_corpus.py --out /tmp/c --synthetic 400  # no network, for tests

WHY THIS IS NOT JUST A DOWNLOAD.  Eight VMs each build their own corpus from their own download,
with no shared filesystem between them (Hyperbolic storage volumes appear to be bare-metal only, and
"termination is permanent -- all data on the instance will be lost").  Identical batch INDICES into
DIFFERENT corpora are different data, so staging ends by emitting the g3_digest that all eight
instances must agree on before any of them starts training.

The assembly must therefore be deterministic across machines:
  - utterances sorted by (speaker, file) before concatenation, never by arrival order
  - scipy.signal.resample_poly for both the 48 kHz normalisation and the decimation -- same filter
    everywhere, unlike torchaudio's Kaldi resampler which LISA uses and which is looser in the
    transition band (audit note section 2)
  - peak normalisation to 0.95 in float64, cast to float32 once at the end

WHAT IT IS NOT.  overnight2/c0b_boot_light.py is the LIGHT boot: it keeps 1 train utterance in 200
(TRAIN_EVERY = 200) so the ladder fit runs in 2 GB of RAM. That yields ~0.19 h, not 37.3 h. Staging
for training keeps every train utterance -- and fast.run_contract.preflight() brackets corpus.hours
to (37.2, 37.4) precisely so that a subsampled corpus cannot be mistaken for the real one and run
"50 epochs" in a few minutes.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fast.run_contract import R, SEG_HI, g3_digest, preflight

HF_REPO = "sanchit-gandhi/vctk"
OUR_TEST = {"p236", "p237", "p238"}       # our held-out speakers
PAPER_TEST_MIN = 350                      # LISA's split: speaker id >= 350
PEAK = 0.95
FS_HI = 48_000


def spk_num(s):
    return int(s[1:]) if (s[:1] == "p" and s[1:].isdigit()) else -1


def role(spk):
    """overnight2/c0b_boot_light.py:40-44, verbatim."""
    if spk in OUR_TEST:
        return "test"
    n = spk_num(spk)
    return "skip" if n < 0 else ("paper_test" if n >= PAPER_TEST_MIN else "train")


def load_bytes(b, fs_hi=FS_HI, peak=PEAK, min_seconds=1.0):
    """overnight2/c0b_boot_light.py:46-56, verbatim. float64 throughout, one cast at the end."""
    import io
    import math
    import soundfile as sf
    import scipy.signal as sps
    x, fs = sf.read(io.BytesIO(b), dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if fs != fs_hi:
        g = math.gcd(int(fs), int(fs_hi))
        x = sps.resample_poly(x, fs_hi // g, fs // g)
    if len(x) < min_seconds * fs_hi:
        return None
    m = float(np.max(np.abs(x)))
    return (x * (peak / m)).astype(np.float32) if m > 0 else None


def decimate(x, r=R):
    """build_notebook.py:1034-1036. The anti-aliased downsample that makes the high band
    unrecoverable -- and the thing that must be byte-identical across all eight VMs."""
    import scipy.signal as sps
    return sps.resample_poly(x, 1, r)


def assemble(ys, seg_hi=SEG_HI, r=R, workers=8):
    """HostCorpus.__init__ (overnight2/c1_model.py:84-99), as arrays rather than an object.

    Drops utterances shorter than one segment, truncates each to a multiple of R so the low- and
    high-rate grids stay aligned, then concatenates. Returns everything the sampler and the digest
    need, and nothing else.
    """
    from concurrent.futures import ThreadPoolExecutor
    ys = [np.asarray(y, np.float32) for y in ys if len(y) >= seg_hi]
    ys = [y[: (len(y) // r) * r] for y in ys]
    with ThreadPoolExecutor(workers) as ex:
        xs = list(ex.map(lambda y: decimate(y.astype(np.float64), r).astype(np.float32), ys))
    lens = np.array([len(y) for y in ys], np.int64)
    off = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    return {"Y": np.concatenate(ys), "X": np.concatenate(xs), "lens": lens, "off": off,
            "n": len(ys), "hours": float(lens.sum()) / FS_HI / 3600}


def download(workers=8):
    """Every train utterance -- no TRAIN_EVERY subsample. Returns (train, test, paper) lists of
    float32 arrays, each sorted by (speaker, file) so the order is identical on every machine."""
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq
    t0 = time.time()
    local = snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=["data/*.parquet"],
                              max_workers=min(8, workers))
    shards = sorted(pathlib.Path(local).glob("data/*.parquet"))
    print(f"  {len(shards)} shards in {time.time() - t0:.0f}s", flush=True)
    out = {"train": [], "test": [], "paper_test": []}
    pool = ThreadPoolExecutor(workers)
    for si, sh in enumerate(shards):
        pf = pq.ParquetFile(sh)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["speaker_id", "file", "audio"])
            spks, files, auds = (t.column(c).to_pylist() for c in ("speaker_id", "file", "audio"))
            keep = [i for i in range(len(files))
                    if (("mic1" in (files[i] or "")) or ("mic1" in ((auds[i] or {}).get("path") or "")))
                    and role(spks[i]) != "skip"]
            arrs = list(pool.map(lambda i: load_bytes(auds[i]["bytes"]), keep))
            for i, a in zip(keep, arrs):
                if a is not None:
                    out[role(spks[i])].append((files[i], spks[i], a))
            del t, auds
        print(f"  shard {si + 1:>2}/{len(shards)}  train {len(out['train'])} test {len(out['test'])}"
              f" paper {len(out['paper_test'])}  [{time.time() - t0:.0f}s]", flush=True)
    pool.shutdown()
    for k in out:
        out[k].sort(key=lambda rec: (rec[1], rec[0]))          # (speaker, file): the canonical order
    return out


def synthetic(n, seconds=(3, 9), seed=0):
    """Harmonic-plus-fricative speech-ish audio, for testing the pipeline without the network.
    Same role the repo's own validators play: a smoke test of the code path, not a scientific corpus."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        t = np.arange(int(rng.uniform(*seconds) * FS_HI)) / FS_HI
        f0 = rng.uniform(90, 220)
        y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
        y += 0.05 * rng.standard_normal(t.size)
        y *= np.clip(np.sin(2 * np.pi * 3.1 * t) ** 2 + 0.15, 0, None)     # syllable-rate envelope
        out.append((f"s{i:04d}.wav", f"p{200 + i % 50:03d}", (PEAK * y / np.abs(y).max()).astype(np.float32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--synthetic", type=int, default=0, help="skip the network; build N fake utterances")
    ap.add_argument("--digest-steps", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    if man_path.exists() and not a.force:
        print(f"{man_path} exists; --force to rebuild")
        print(json.dumps(json.loads(man_path.read_text())["g3"], indent=1))
        return 0

    t0 = time.time()
    if a.synthetic:
        recs = {"train": synthetic(a.synthetic), "test": [], "paper_test": []}
        print(f"synthetic corpus: {a.synthetic} utterances")
    else:
        print(f"downloading {HF_REPO} ...", flush=True)
        recs = download(a.workers)

    c = assemble([r[2] for r in recs["train"]], workers=a.workers)
    print(f"assembled: {c['n']} utts, {c['hours']:.4f} h, "
          f"Y {c['Y'].nbytes / 1e9:.1f} GB + X {c['X'].nbytes / 1e9:.1f} GB  [{time.time() - t0:.0f}s]")

    for name in ("Y", "X"):
        p = out / f"{name}.f32"
        c[name].tofile(p)
        print(f"  wrote {p} ({p.stat().st_size / 1e9:.1f} GB)")
    np.save(out / "lens.npy", c["lens"])
    np.save(out / "off.npy", c["off"])

    t1 = time.time()
    g3 = g3_digest(c["n"], c["hours"], c["lens"], c["Y"], c["X"], n_steps=a.digest_steps)
    print(f"  g3 digest in {time.time() - t1:.0f}s")

    man = {"repo": HF_REPO if not a.synthetic else "synthetic", "n": c["n"], "hours": c["hours"],
           "seg_hi": SEG_HI, "R": R, "fs_hi": FS_HI, "peak": PEAK,
           "test_speakers": sorted({r[1] for r in recs["test"]}),
           "n_test_utts": len(recs["test"]), "n_paper_test": len(recs["paper_test"]),
           "bytes": {"Y": int(c["Y"].nbytes), "X": int(c["X"].nbytes)}, "g3": g3,
           "staged_secs": round(time.time() - t0, 1)}
    man_path.write_text(json.dumps(man, indent=1))

    print("\nG3 DIGEST -- all eight instances must print these two identical strings:")
    print(f"  corpus  {g3['corpus']}")
    print(f"  batches {g3['batches']}")

    if not a.synthetic:
        print("\npreflight:", json.dumps(preflight(c["n"], c["hours"], SEG_HI), indent=1))
    else:
        print(f"\n(synthetic: preflight skipped -- {c['hours']:.3f} h would fail the 37.2-37.4 bracket, "
              "which is exactly what it is for)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
