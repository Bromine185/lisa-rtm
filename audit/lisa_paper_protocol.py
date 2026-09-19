"""LISA's own evaluation protocol, re-run on real VCTK, with no model.

Reimplements torchaudio 0.6.0's `kaldi.resample_waveform` (the only resampler in ml-postech/LISA)
and LISA's `utils.calc_snr` / `utils.compute_log_distortion` from their source, then applies them
to the chunking `eval_lisa.py` uses.  Reports:

  1. the frequency response of Resample(48000, 48000), which their ground truth passes through
  2. the protocol's hard ceiling -- perfect low band, empty high band -- and where naive sits
  3. what `if ii == 3` does to the variance: the metric is one batch of eight 1-second chunks
  4. LSD in their definition against the conventional one, on the same signal
  5. how much of their LSD is pinned at its epsilon floor

Nothing is trained.  See notes/2026-09-17-lisa-reported-numbers-audit.md.

    venv/bin/python audit/lisa_paper_protocol.py
"""
import math, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from audit.vctk_fixtures import fetch, load, PAPER_TEST          # noqa: E402

FS, R_IN = 48000, 12000
LFW = 6                       # kaldi lowpass_filter_width default
PAPER_4X = 24.16


# ---- torchaudio 0.6.0 kaldi.resample_waveform, in numpy --------------------------------------------
def kaldi_weights(orig, new, lfw=LFW):
    cut = 0.99 * 0.5 * min(orig, new)
    base = math.gcd(int(orig), int(new))
    n_in, n_out = int(orig) // base, int(new) // base
    ww = lfw / (2.0 * cut)
    out_t = np.arange(n_out, dtype=np.float64) / new
    first = np.ceil((out_t - ww) * orig)
    W = int((np.floor((out_t + ww) * orig) - first + 1).max())
    dt = (first[:, None] + np.arange(W, dtype=np.float64)[None, :]) / orig - out_t[:, None]
    w = np.zeros_like(dt)
    ins = np.abs(dt) < ww
    w[ins] = 0.5 * (1 + np.cos(2 * math.pi * cut / lfw * dt[ins]))       # Hann over the full width
    z = dt == 0.0
    w[~z] *= np.sin(2 * math.pi * cut * dt[~z]) / (math.pi * dt[~z])     # sinc
    w[z] *= 2 * cut                                                      # its limit at t = 0
    return first.astype(int), w / orig, n_in, n_out


def n_out_samples(n, orig, new):
    orig, new = int(orig), int(new)
    tick = abs(orig * new) // math.gcd(orig, new)
    total = n * (tick // orig)
    if total <= 0:
        return 0
    last = total // (tick // new)
    return last if last * (tick // new) == total else last + 1


def kaldi_resample(x, orig, new, lfw=LFW):
    first, w, n_in, n_out = kaldi_weights(orig, new, lfw)
    N, W = n_out_samples(len(x), orig, new), w.shape[1]
    pad = W + max(0, -first.min()) + n_in * (N // max(n_out, 1) + 2)
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    n = np.arange(N)
    i, unit = n % n_out, n // n_out
    cols = (first[i] + unit * n_in + pad)[:, None] + np.arange(W)[None, :]
    return (xp[cols] * w[i]).sum(1)


# ---- LISA's metrics, transcribed from utils.py ------------------------------------------------------
def calc_snr(pred, gt):                                   # utils.py:162, one value per chunk
    return 20 * np.log10(np.linalg.norm(gt) / np.linalg.norm(gt - pred) + 1e-8)


def spectrogram(x, n_fft=2048):
    """torchaudio Spectrogram(n_fft=2048): Hann, hop = n_fft // 2, power = 2 -> |X|**2."""
    hop, w = n_fft // 2, np.hanning(n_fft + 1)[:-1]
    xp = np.pad(x, n_fft // 2, mode="reflect")
    fr = 1 + (len(xp) - n_fft) // hop
    S = np.stack([np.fft.rfft(xp[i * hop:i * hop + n_fft] * w) for i in range(fr)], axis=1)
    return np.abs(S) ** 2                                  # (freq, time)


def lsd_lisa(pred, gt):                                   # utils.py:176 -- logs an ALREADY squared spectrogram
    f = lambda x: np.log10(np.abs(spectrogram(x)) ** 2 + 1e-8)
    d = f(gt) - f(pred)
    return min(float(np.mean(np.sqrt(np.mean(d ** 2 + 1e-8, axis=1)), axis=0)), 10.0)


def lsd_standard(pred, gt):                               # log10|S|**2, RMS over frequency within a frame
    f = lambda x: np.log10(spectrogram(x) + 1e-10)
    d = f(gt) - f(pred)
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=0))))


def main():
    first, w, _, _ = kaldi_weights(FS, FS)
    H = np.fft.rfft(w[0], 8192)
    f = np.fft.rfftfreq(8192, 1 / FS)
    mag = 20 * np.log10(np.abs(H) + 1e-12)
    print(f"1. Resample(48000, 48000) is not identity: {len(w[0])} taps, cutoff 0.99*24000 = 23760 Hz")
    print("   it is applied to LISA's ground truth (wrappers.py:317). its response:")
    for q in (6000, 12000, 16000, 20000, 22000, 23760):
        print(f"     {q/1000:5.1f} kHz  {np.interp(q, f, mag):+6.2f} dB")

    rows, floors, mags = [], [], []
    for p in fetch(PAPER_TEST, per_speaker=8):
        y = load(p, FS)
        if len(y) < FS:
            continue
        for j in range(len(y) // FS + 1):                 # wrappers.py:287, chunk_len = 1 s
            off = int((len(y) - FS) * (j / max(len(y) // FS, 1)))
            c = y[off:off + FS]
            if len(c) < FS:
                continue
            gt = kaldi_resample(c, FS, FS)
            up = kaldi_resample(kaldi_resample(c, FS, R_IN), R_IN, FS)
            n = min(len(gt), len(up))
            gt, up = gt[:n], up[:n]
            G = np.fft.rfft(gt)
            fr = np.fft.rfftfreq(n, 1 / FS)
            lo = G.copy()
            lo[fr >= R_IN / 2] = 0
            rows.append((calc_snr(up, gt), calc_snr(np.fft.irfft(lo, n), gt),
                         lsd_lisa(up, gt), lsd_standard(up, gt),
                         float((np.abs(G[fr >= R_IN / 2]) ** 2).sum() / (np.abs(G) ** 2).sum())))
            if len(floors) < 16:
                P = spectrogram(gt)
                floors.append(float((P ** 2 < 1e-8).mean()))
                mags.append(np.sqrt(P).ravel())
    a = np.array(rows)
    snr, ceil, l_lisa, l_std, e_hi = (a[:, i] for i in range(5))
    print(f"\n{len(a)} one-second chunks from {len(PAPER_TEST)} of the paper's own held-out speakers\n")
    print("2. mean of per-chunk dB, exactly as utils.calc_snr + Averager do it")
    print(f"     energy at or above 6 kHz                      {100*e_hi.mean():7.3f} %")
    print(f"     naive sinc upsample of the 12 kHz input       {snr.mean():7.2f} dB")
    print(f"     perfect low band, empty high band (ceiling)   {ceil.mean():7.2f} dB")
    print(f"     LISA reports                                  {PAPER_4X:7.2f} dB  ({PAPER_4X-ceil.mean():+.2f})")

    cons = snr[:len(snr) // 8 * 8].reshape(-1, 8).mean(1)
    rng = np.random.default_rng(0)
    boot = np.array([snr[rng.choice(len(snr), 8, replace=False)].mean() for _ in range(20000)])
    print(f"\n3. eval_lisa.py scores batch index 3 only -- 8 chunks, 0.04 % of their validation set")
    print(f"     consecutive 8-chunk batches ({len(cons)})   sd {cons.std():5.2f} dB  "
          f"range {cons.min():.2f} to {cons.max():.2f}")
    print(f"     random 8-chunk batches (20000)      sd {boot.std():5.2f} dB")
    print(f"     P(a naive batch scores >= {PAPER_4X})   {100*(boot >= PAPER_4X).mean():.1f} % random, "
          f"{100*(cons >= PAPER_4X).mean():.1f} % consecutive")

    print(f"\n4. LSD of that same naive upsample")
    print(f"     LISA's compute_log_distortion   {l_lisa.mean():.3f}")
    print(f"     the conventional definition     {l_std.mean():.3f}")
    m = np.concatenate(mags)
    print(f"\n5. LISA logs |X|**4, so +1e-8 floors every bin with |X| < 1e-2 "
          f"(median |X| here is {np.median(m):.2e})")
    print(f"     share of time-frequency bins pinned at that floor: {100*np.mean(floors):.1f} %")


if __name__ == "__main__":
    main()
