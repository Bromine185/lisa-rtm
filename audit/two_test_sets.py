"""The same speakers, two acquisition routes: which one is the truth?

The 4 September and 8 September notes report the same checkpoint measuring a high-band deficit
5-8 dB apart on two sets both labelled "p236-238, mic1":

    route          built by                             lambda=1e-2 deficit   naive SNR
    Hub            overnight/cell1_corpus.py            -10.03 dB (120 utts)  19.48 dB
    DataShare      build_notebook.py build_split ->     - 4.52 dB ( 36 utts)  19.20 dB
                   fixtures/test_FULL.npz

Until that is settled every number in this repo inherits it, so this script settles it WITHOUT a
model.  The tell is in the table above: `naive` is sinc upsampling, no network, no checkpoint, and
it already differs by 0.28 dB.  A model cannot explain a difference in a baseline it is not part
of.  So the discrepancy lives in the audio or in the selection, and both are checkable here.

What it does: fetch the same nominal utterances by BOTH routes, pair them by FLAC basename, and
report, per pair, every model-free quantity the deficit is built out of.

    venv/bin/python audit/two_test_sets.py                       # p236-238, 4 utts each
    venv/bin/python audit/two_test_sets.py --speakers p236 --per-speaker 12
    venv/bin/python audit/two_test_sets.py --max-shards 6        # stop scanning the Hub early

Needs `huggingface_hub` and `pyarrow` on top of the repo's venv (the Hub route's own dependencies,
from overnight/cell1_corpus.py) -- everything else is numpy/scipy, as in audit/snr_scale.py.

Hypotheses this discriminates, in the order the code makes them likely:

  H1  Different audio under the same name.  The DataShare route takes members matching
      `wav48_silence_trimmed/*_mic1.flac`; the Hub route takes any row whose file or audio path
      contains "mic1", with no equivalent constraint.  If `sanchit-gandhi/vctk` ships the untrimmed
      `wav48` -- or anything at a lower stored rate, which load_bytes() then resamples UP to 48 kHz
      -- the high band of the "truth" is not the same signal.  Tells: `samples`, `stored fs`,
      `quiet frames`, `HB fraction`, `bit-identical`.

  H2  Different utterances under the same label.  Both routes take the first 40 per speaker by
      name, but by different name formats: `sorted(zip member paths)` against `sorted((speaker_id,
      file))`.  Tell: the paired/unpaired counts and the `only in` lists.

  H3  Different subsets of the same audio.  Not a pipeline bug, but it is conflated with one in the
      notes: the 8 September Hub column is `test_utts[:12]`, and since both routes lay out 40
      utterances per speaker in speaker order, that is p236 ALONE -- while the DataShare column is
      `[u for i, u in enumerate(test_utts) if i % 40 < 12]`, 12 from each of the three speakers.
      Tell: the per-speaker summary rows.  If p236 alone accounts for the gap, the sets never
      disagreed and only the labels did.

Nothing here is a model claim.  See notes/2026-09-17-snr-ceiling-gating-and-scale.md section 2.1 for
why a metric averaged over frames is this sensitive to how much silence a set carries.
"""
import argparse
import math
import pathlib
import sys

import numpy as np
import scipy.signal as sps

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from audit.vctk_fixtures import fetch, load                       # noqa: E402

FS, R = 48000, 4                       # CFG.fs_hi, CFG.upsample
N_FFT, HOP = 1024, 256                 # CFG.eval_n_fft, CFG.eval_hop
F_CUT = FS / R / 2                     # 6 kHz: everything above this is what decimation destroys
HF_REPO = "sanchit-gandhi/vctk"
QUIET_DB = -40.0                       # a frame this far below the utterance's loudest is a "gap"


# ---- the notebook's spectral primitives, inlined (no torch, so this runs with no checkpoint) -------
def _hann(n):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def stft(x, n_fft=N_FFT, hop=HOP):
    x = np.asarray(x, dtype=np.float64)
    xp = np.pad(x, (n_fft, n_fft + n_fft))
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(xp[idx] * _hann(n_fft), axis=-1)


def third_octave_edges(f_lo, f_hi):
    f = [f_lo]
    while f[-1] < f_hi:
        f.append(f[-1] * 2 ** (1 / 3))
    return np.array(f)


def snr_db(y, y_hat):
    n = min(len(y), len(y_hat))
    e = y[:n] - y_hat[:n]
    return 10.0 * np.log10(np.sum(y[:n] ** 2) / max(np.sum(e ** 2), 1e-20))


def naive_upsample(y):
    """Polyphase sinc interpolation of the decimated signal: the model-free baseline."""
    y = np.asarray(y, np.float64)
    return sps.resample_poly(sps.resample_poly(y, 1, R), R, 1)[:len(y)]


# ---- the model-free description of one utterance --------------------------------------------------
def describe(y):
    y = np.asarray(y, np.float64)
    S = np.abs(stft(y))
    fr = S.sum(1)
    fr_db = 20 * np.log10(fr + 1e-12)
    quiet = float(np.mean(fr_db < fr_db.max() + QUIET_DB))

    freqs = np.fft.rfftfreq(N_FFT, 1.0 / FS)
    hb = freqs >= F_CUT
    P = S ** 2
    hb_frac = float(P[:, hb].sum() / max(P.sum(), 1e-20))

    edges = third_octave_edges(F_CUT, FS / 2 - 1)
    bands = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (freqs >= a) & (freqs < b)
        if sel.sum():
            bands.append((math.sqrt(a * b), float(P[:, sel].sum())))

    return {"samples": len(y), "seconds": len(y) / FS,
            "rms": float(np.sqrt(np.mean(y ** 2))), "peak": float(np.max(np.abs(y))),
            "quiet_frac": quiet, "hb_frac_db": 10 * np.log10(max(hb_frac, 1e-20)),
            "snr_naive": snr_db(y, naive_upsample(y)),
            "bands": bands}


# ---- route A: DataShare, over HTTP ranges, exactly as audit/ and build_notebook.py take it --------
def datashare(speakers, per_speaker):
    out = {}
    for p in fetch(tuple(speakers), per_speaker=per_speaker):
        out[p.name] = load(p, FS)          # mono, 48 kHz, peak 0.95 -- load_utterance's treatment
    return out


# ---- route B: the Hub parquet shards, exactly as overnight/cell1_corpus.py takes them -------------
def hub(speakers, per_speaker, max_shards):
    import io
    import soundfile as sf
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    def load_bytes(b, peak=0.95, min_seconds=1.0):
        """overnight/cell1_corpus.py load_bytes, verbatim, plus the stored rate it hides."""
        x, fs = sf.read(io.BytesIO(b), dtype="float64", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
        if fs != FS:
            g = math.gcd(int(fs), int(FS))
            x = sps.resample_poly(x, FS // g, fs // g)
        if len(x) < min_seconds * FS:
            return None, fs
        m = float(np.max(np.abs(x)))
        return (x * (peak / m) if m > 0 else None), fs

    shards = sorted(f for f in list_repo_files(HF_REPO, repo_type="dataset")
                    if f.startswith("data/") and f.endswith(".parquet"))
    want = {s: per_speaker for s in speakers}
    out, rates = {}, {}
    for si, sh in enumerate(shards[:max_shards]):
        if all(v <= 0 for v in want.values()):
            break
        local = hf_hub_download(HF_REPO, sh, repo_type="dataset")
        pf = pq.ParquetFile(local)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["speaker_id", "file", "audio"])
            spks, files, auds = (t.column(c).to_pylist() for c in ("speaker_id", "file", "audio"))
            rows = [i for i in range(len(files)) if spks[i] in want
                    and (("mic1" in (files[i] or "")) or ("mic1" in ((auds[i] or {}).get("path") or "")))]
            for i in sorted(rows, key=lambda i: (spks[i], files[i] or "")):
                name = pathlib.Path((auds[i] or {}).get("path") or files[i] or "").name
                if not name.endswith(".flac"):
                    name = pathlib.Path(name).stem + ".flac"
                if name in out:
                    continue
                a, fs = load_bytes(auds[i]["bytes"])
                if a is None:
                    continue
                out[name], rates[name] = a, fs
            del t, auds
        # keep only the first `per_speaker` per speaker, by name, as both routes do
        for s in speakers:
            keep = sorted(n for n in out if n.startswith(s + "_"))[:per_speaker]
            want[s] = per_speaker - len(keep)
        print(f"  shard {si + 1}/{min(len(shards), max_shards)}  have "
              + " ".join(f"{s}:{per_speaker - want[s]}" for s in speakers), flush=True)
    trimmed = {}
    for s in speakers:
        for n in sorted(n for n in out if n.startswith(s + "_"))[:per_speaker]:
            trimmed[n] = out[n]
    return trimmed, rates


# ---- report ---------------------------------------------------------------------------------------
def main(speakers, per_speaker, max_shards):
    print(f"speakers {speakers}  per_speaker {per_speaker}\n")
    print("route A: DataShare zip, wav48_silence_trimmed/*_mic1.flac")
    A = datashare(speakers, per_speaker)
    print(f"route B: Hub {HF_REPO}, rows whose file/path contains mic1")
    B, rates = hub(speakers, per_speaker, max_shards)

    both = sorted(set(A) & set(B))
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    print(f"\n=== H2: selection ===")
    print(f"  paired {len(both)}   only DataShare {len(only_a)}   only Hub {len(only_b)}")
    if only_a:
        print(f"  only in DataShare: {only_a[:6]}{' ...' if len(only_a) > 6 else ''}")
    if only_b:
        print(f"  only in Hub:       {only_b[:6]}{' ...' if len(only_b) > 6 else ''}")
    if not both:
        print("\n  The two routes share no utterance name.  That IS the discrepancy: H2 confirmed,\n"
              "  and no comparison below is possible until the naming is reconciled.")
        return

    print(f"\n=== H1: the same audio? ===")
    hdr = (f"  {'utterance':<24} {'route':<10} {'stored':>7} {'samples':>9} {'sec':>6} {'rms':>7} "
           f"{'quiet':>6} {'HB dB':>7} {'naiveSNR':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = {}
    for n in both:
        da, db = describe(A[n]), describe(B[n])
        rows[n] = (da, db)
        ident = np.array_equal(np.asarray(A[n], np.float64), np.asarray(B[n], np.float64))
        for tag, d, fs in (("DataShare", da, 48000), ("Hub", db, rates.get(n, 0))):
            print(f"  {n if tag == 'DataShare' else '':<24} {tag:<10} {fs:>7} {d['samples']:>9} "
                  f"{d['seconds']:>6.2f} {d['rms']:>7.4f} {d['quiet_frac']:>6.2f} "
                  f"{d['hb_frac_db']:>7.2f} {d['snr_naive']:>9.2f}")
        print(f"  {'':<24} {'delta':<10} {'':>7} {db['samples'] - da['samples']:>9} "
              f"{db['seconds'] - da['seconds']:>6.2f} {db['rms'] - da['rms']:>7.4f} "
              f"{db['quiet_frac'] - da['quiet_frac']:>6.2f} "
              f"{db['hb_frac_db'] - da['hb_frac_db']:>7.2f} "
              f"{db['snr_naive'] - da['snr_naive']:>9.2f}"
              f"   bit-identical: {ident}")

    print(f"\n=== H3: per speaker (the label the notes compare) ===")
    print(f"  {'speaker':<10} {'n':>3} {'quiet':>16} {'HB dB':>16} {'naive SNR':>16}")
    print(f"  {'':<10} {'':>3} {'DS':>7} {'Hub':>8} {'DS':>7} {'Hub':>8} {'DS':>7} {'Hub':>8}")
    for s in speakers:
        ns = [n for n in both if n.startswith(s + "_")]
        if not ns:
            continue
        m = lambda i, k: float(np.mean([rows[n][i][k] for n in ns]))
        print(f"  {s:<10} {len(ns):>3} {m(0,'quiet_frac'):>7.2f} {m(1,'quiet_frac'):>8.2f} "
              f"{m(0,'hb_frac_db'):>7.2f} {m(1,'hb_frac_db'):>8.2f} "
              f"{m(0,'snr_naive'):>7.2f} {m(1,'snr_naive'):>8.2f}")

    print(f"\n=== the third-octave spectrum, pooled over the paired utterances ===")
    print(f"  {'centre Hz':>10} {'DataShare dB':>13} {'Hub dB':>13} {'Hub - DS dB':>13}")
    nb = len(rows[both[0]][0]["bands"])
    for bi in range(nb):
        fc = rows[both[0]][0]["bands"][bi][0]
        ea = sum(rows[n][0]["bands"][bi][1] for n in both)
        eb = sum(rows[n][1]["bands"][bi][1] for n in both)
        da_db, db_db = 10 * np.log10(max(ea, 1e-20)), 10 * np.log10(max(eb, 1e-20))
        print(f"  {fc:>10.0f} {da_db:>13.2f} {db_db:>13.2f} {db_db - da_db:>13.2f}")

    ga = float(np.mean([rows[n][0]["snr_naive"] for n in both]))
    gb = float(np.mean([rows[n][1]["snr_naive"] for n in both]))
    print(f"\n  naive SNR, mean over {len(both)} paired utterances: "
          f"DataShare {ga:.2f} dB, Hub {gb:.2f} dB, delta {gb - ga:+.2f} dB")
    print("  The notes report 19.20 vs 19.48 (delta +0.28).  If this delta reproduces on paired\n"
          "  utterances, the routes carry different audio (H1) and every deficit inherits it.\n"
          "  If it vanishes here, the gap is selection (H2/H3) and the fix is to name the set.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--speakers", default="p236,p237,p238")
    ap.add_argument("--per-speaker", type=int, default=4)
    ap.add_argument("--max-shards", type=int, default=27)
    a = ap.parse_args()
    main([s for s in a.speakers.split(",") if s], a.per_speaker, a.max_shards)
