"""How far apart are a model's draws? Measured in three domains on the saved draws, no inference.

    venv/bin/python sota/draw_agreement.py --set core --models flowhigh flowhigh_std1

For each utterance, draws s0..s{M-1} as saved by run_models.py (raw output, no gain alignment):
    wave_maxabs   max over samples and draws of |x_k - x_0|                    (waveform, absolute; peak is ~1)
    wave_rel      RMS over draws of (x_k - xbar), over RMS of xbar            (waveform, relative)
    hb_rel        the same on the FFT band above 6 kHz                          (high-band waveform, relative)
    hbmag_rel     the same on |STFT| above 6 kHz, 1024/256 Hann                 (high-band magnitude, relative)
    logmel_maxabs max |log-mel_k - log-mel_0|, natural log, 128 mels to 24 kHz  (log-mel, absolute)
Reported as the median and maximum over utterances. Writes <work>/<set>/draw_agreement_<set>.json.
"""
import argparse, json, os, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from score import WORK, FS, CUT, fft_split                                          # noqa: E402


def stft_mag(x, n=1024, hop=256):
    w = np.hanning(n + 1)[:-1]
    fr = np.lib.stride_tricks.sliding_window_view(np.pad(x, n // 2), n)[::hop] * w
    return np.abs(np.fft.rfft(fr, axis=1))


def mel_fb(n=1024, n_mels=128):
    hz2m = lambda f: 2595 * np.log10(1 + f / 700); m2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = m2hz(np.linspace(0, hz2m(FS / 2), n_mels + 2)); f = np.fft.rfftfreq(n, 1 / FS)
    fb = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0, None)
    return fb


def rel(D):
    xb = D.mean(0)
    return float(np.sqrt(np.mean((D - xb[None]) ** 2)) / np.sqrt(np.mean(xb ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="core")
    ap.add_argument("--models", nargs="+", default=["flowhigh", "flowhigh_std1"])
    a = ap.parse_args()
    ws = WORK / a.set; FB = mel_fb(); hbk = np.fft.rfftfreq(1024, 1 / FS) >= CUT
    utts = [u["utt"] for u in json.load(open(ws / "utts.json"))["utts"]]
    out = {"_meta": {"set": a.set, "n_utts": len(utts), "doc": __doc__.strip()}}
    for m in a.models:
        rows = []
        for utt in utts:
            fs = sorted((ws / "out" / m).glob(f"{utt}_s*.npy"), key=lambda q: int(q.stem.rsplit("_s", 1)[1]))
            D = np.stack([np.load(q).astype(np.float64) for q in fs]); n = min(len(d) for d in D); D = D[:, :n]
            H = np.stack([fft_split(d, False) for d in D])
            S = np.stack([stft_mag(d) for d in D])
            L = np.log(np.stack([s @ FB.T for s in S]) + 1e-5)
            rows.append({"utt": utt, "M": len(D),
                         "wave_maxabs": float(np.abs(D - D[0:1]).max()), "wave_rel": rel(D), "hb_rel": rel(H),
                         "hbmag_rel": rel(S[:, :, hbk]), "logmel_maxabs": float(np.abs(L - L[0:1]).max())})
        keys = [k for k in rows[0] if k not in ("utt", "M")]
        agg = {k: {"median": float(np.median([r[k] for r in rows])), "max": float(np.max([r[k] for r in rows]))}
               for k in keys}
        out[m] = {"M": rows[0]["M"], "agg": agg, "per_utt": rows}
        print(f"{m:15s} M={rows[0]['M']}  " + "  ".join(f"{k} {v['median']:.2e} (max {v['max']:.2e})"
                                                     for k, v in agg.items()))
    (ws / f"draw_agreement_{a.set}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
