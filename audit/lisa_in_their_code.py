"""Section 9 of the audit note, run inside ml-postech/LISA's own code.

`audit/lisa_paper_protocol.py` already answers this question, but every number in it comes from a
REIMPLEMENTATION: our transcription of `utils.calc_snr`, of `utils.compute_log_distortion`, and of
torchaudio 0.6.0's `kaldi.resample_waveform`.  That is the last caveat on the repo's strongest claim
-- that 24.16 dB sits above the task's ceiling -- and this script removes it by calling the originals.

Nothing is trained and no checkpoint is loaded.  Per section 9, the "model" is replaced by a plain
sinc upsample of the input, and the three conditions are:

  (a) the full validation set, their degradation operator (torchaudio kaldi 48k -> 12k)
  (b) batch index 3 only -- what `eval_lisa.py`'s `if ii == 3:` guard actually scores
  (c) the full set with the input made by `scipy.signal.resample_poly(x, 1, 4)` instead

and section 9's reading of them:

  (a) ~20-21 dB and (b) ~24 dB  -> the headline is a sinc baseline on a lucky eight seconds
  (a) ~24 dB                    -> the ceiling is in their data pipeline, not the model
  (a) - (c)                     -> prices their looser transition band, the one legitimate
                                   asymmetry between their protocol and ours

Usage -- you need a clone of their repo and the torchaudio their code was written against:

    git clone https://github.com/ml-postech/LISA /tmp/LISA
    venv/bin/python audit/lisa_in_their_code.py --lisa /tmp/LISA

Only their `utils.py` is imported, not their training harness: the audio comes from this repo's own
fixture loader (`audit/vctk_fixtures.py`, speakers id >= 350 -- their validation split, per
`datasets/audio_dataset.py:59`) and the chunking from `wrappers.py:287`, chunk_len = 1 s.  That keeps
the input identical to `audit/lisa_paper_protocol.py`, so the two scripts' outputs are directly
comparable and any disagreement is a bug in our transcription, which is the whole point.

See notes/2026-09-17-lisa-reported-numbers-audit.md section 9.
"""
import argparse
import pathlib
import sys

import numpy as np
import scipy.signal as sps

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from audit.vctk_fixtures import fetch, load, PAPER_TEST              # noqa: E402

FS, R_IN, R = 48000, 12000, 4
PAPER_4X = 24.16
BATCH = 8                       # eval_lisa.py scores one batch of eight 1-second chunks


def their_utils(clone):
    """Import ml-postech/LISA's utils.py and return (calc_snr, compute_log_distortion)."""
    clone = pathlib.Path(clone).expanduser().resolve()
    if not clone.is_dir():
        sys.exit(f"--lisa {clone} is not a directory.  git clone https://github.com/ml-postech/LISA")
    sys.path.insert(0, str(clone))
    try:
        import utils                                                 # their file, not ours
    except Exception as e:                                           # noqa: BLE001
        sys.exit(f"could not import utils.py from {clone}: {e}")
    missing = [n for n in ("calc_snr", "compute_log_distortion") if not hasattr(utils, n)]
    if missing:
        sys.exit(f"{clone}/utils.py has no {missing}.  Found: "
                 f"{sorted(n for n in dir(utils) if not n.startswith('_'))}")
    print(f"their utils.py: {pathlib.Path(utils.__file__).resolve()}")
    return utils.calc_snr, utils.compute_log_distortion


def their_resampler():
    """torchaudio's kaldi resampler -- the only resampler in their codebase (wrappers.py:317)."""
    import torch
    import torchaudio
    from torchaudio.compliance import kaldi
    print(f"torchaudio {torchaudio.__version__} (their pin is 0.6.0)")

    def resample(x, orig, new):
        t = torch.as_tensor(np.asarray(x, np.float64), dtype=torch.float32)[None]
        return kaldi.resample_waveform(t, float(orig), float(new))[0].numpy().astype(np.float64)

    return resample, torch


def chunks(paths):
    """The 1-second chunks wrappers.py:287 cuts, over their held-out speakers."""
    for p in paths:
        y = load(p, FS)
        if len(y) < FS:
            continue
        for j in range(len(y) // FS + 1):
            off = int((len(y) - FS) * (j / max(len(y) // FS, 1)))
            c = y[off:off + FS]
            if len(c) == FS:
                yield c


def score(calc_snr, cld, torch, pred, gt):
    """Call their two metrics on one chunk.  They take torch tensors in their own eval loop."""
    tp = torch.as_tensor(pred, dtype=torch.float32)
    tg = torch.as_tensor(gt, dtype=torch.float32)
    out = []
    for fn in (calc_snr, cld):
        try:
            v = fn(tp, tg)
        except Exception:                                            # noqa: BLE001
            v = fn(pred, gt)                                         # numpy fallback
        out.append(float(v.item() if hasattr(v, "item") else v))
    return out


def main(clone, per_speaker):
    calc_snr, cld = their_utils(clone)
    resample, torch = their_resampler()

    print(f"\nvalidation split: {len(PAPER_TEST)} speakers id >= 350, {per_speaker} utterances each")
    rows = []
    for c in chunks(fetch(PAPER_TEST, per_speaker=per_speaker)):
        # their ground truth passes through Resample(48000, 48000) -- not identity (section 3.1)
        gt = resample(c, FS, FS)

        # (a)/(b): their degradation, then a plain sinc upsample standing in for `pred = model(batch)`
        lo_kaldi = resample(c, FS, R_IN)
        up_kaldi = sps.resample_poly(lo_kaldi, R, 1)

        # (c): the same, with scipy's tighter decimation instead of torchaudio's
        lo_poly = sps.resample_poly(c, 1, R)
        up_poly = sps.resample_poly(lo_poly, R, 1)

        n = min(len(gt), len(up_kaldi), len(up_poly))
        s_k, l_k = score(calc_snr, cld, torch, up_kaldi[:n], gt[:n])
        s_p, l_p = score(calc_snr, cld, torch, up_poly[:n], gt[:n])
        rows.append((s_k, l_k, s_p, l_p))

    a = np.array(rows)
    if len(a) < BATCH * 4:
        print(f"\nonly {len(a)} chunks -- raise --per-speaker for a meaningful (b)")
    snr_k, lsd_k, snr_p, lsd_p = (a[:, i] for i in range(4))
    nb = len(a) // BATCH
    batches_k = snr_k[:nb * BATCH].reshape(-1, BATCH).mean(1)
    b3 = float(batches_k[3]) if nb > 3 else float("nan")

    print(f"\n{len(a)} one-second chunks -> {nb} batches of {BATCH}, scored by THEIR utils.py\n")
    print(f"  (a) full validation set, their operator     SNR {snr_k.mean():7.2f} dB   "
          f"LSD {lsd_k.mean():6.3f}")
    print(f"  (b) batch index 3 only (the `ii == 3` guard) SNR {b3:7.2f} dB   "
          f"LSD {lsd_k[3*BATCH:4*BATCH].mean() if nb > 3 else float('nan'):6.3f}")
    print(f"  (c) full set, scipy resample_poly input     SNR {snr_p.mean():7.2f} dB   "
          f"LSD {lsd_p.mean():6.3f}")
    print(f"\n  LISA reports                               SNR {PAPER_4X:7.2f} dB")
    print(f"  (a) - paper   {snr_k.mean() - PAPER_4X:+7.2f} dB")
    print(f"  (b) - paper   {b3 - PAPER_4X:+7.2f} dB")
    print(f"  (a) - (c)     {snr_k.mean() - snr_p.mean():+7.2f} dB   "
          f"(the price of their looser transition band)")

    print(f"\n  batch-to-batch spread of (a): sd {batches_k.std():.2f} dB, "
          f"range {batches_k.min():.2f} to {batches_k.max():.2f}, "
          f"{100 * (batches_k >= PAPER_4X).mean():.1f} % of batches >= {PAPER_4X}")
    print("\n  Compare with `venv/bin/python audit/lisa_paper_protocol.py`, which computes (a) and (b)\n"
          "  from our transcription of these same functions.  Agreement removes the reimplementation\n"
          "  caveat from section 7; disagreement is a bug in ours and the note must be corrected.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lisa", required=True, help="path to a clone of github.com/ml-postech/LISA")
    ap.add_argument("--per-speaker", type=int, default=8)
    a = ap.parse_args()
    main(a.lisa, a.per_speaker)
