"""The SNR ceiling of 48 kHz VCTK bandwidth extension, at every input rate the literature reports.

    venv/bin/python sota/ceiling.py [--set wide]

For input rates 8, 12, 16, 24 kHz, per utterance of the NU-Wave 2 / FLowHigh / AP-BWE test speakers
(p360-p364, p374, p376, s5; sota/fetch_wide.py), SNR = 10 log10(|y|^2 / |y - w|^2), averaged in dB over
utterances -- the cal_snr of NU-Wave 2's for_test.py and AP-BWE's cal_metrics.py, whole utterances, no chunking.

  empty      w = y brick-walled at fs_in / 2: the true low band, nothing above. The ceiling for any estimate
             whose high band is incoherent with the truth and whose level is chosen to minimise squared error
             (it is the MMSE answer when the missing band carries no recoverable phase).
  input      w = NU-Wave 2's test input: cheby1(8, 0.05 dB) sosfiltfilt, resample_poly down and back up. Their
             Table 1 "Input" row, recomputed on these utterances; below `empty` by the filter's transition band.
Also the high band's share of the energy, which fixes the ceiling: C ~ -10 log10(share).

Writes <work>/<set>/ceiling_<set>.json.
"""
import os
import argparse, json, pathlib, sys
import numpy as np
import scipy.signal as sps

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
WORK = pathlib.Path(os.environ.get("BWE_CACHE", str(pathlib.Path(__file__).resolve().parents[1] / "cache")) + "/sota/work")
FS = 48000


def brick(y, fc):
    Y = np.fft.rfft(y); f = np.fft.rfftfreq(len(y), 1 / FS); Y[f >= fc] = 0
    return np.fft.irfft(Y, n=len(y))


def nuwave2_input(y, fs_in):
    sos = sps.cheby1(8, 0.05, (fs_in / 2) / (FS / 2), btype="lowpass", output="sos")
    wl = sps.resample_poly(sps.sosfiltfilt(sos, y), fs_in, FS)
    return sps.resample_poly(wl, FS, fs_in)[: len(y)]


def snr(y, w):
    return float(10 * np.log10(np.sum(y ** 2) / max(np.sum((y - w) ** 2), 1e-30)))


ap = argparse.ArgumentParser(); ap.add_argument("--set", default="wide"); a = ap.parse_args()
ws = WORK / a.set
utts = json.load(open(ws / "utts.json"))["utts"]
res = {}
for fs_in in (8000, 12000, 16000, 24000):
    rows = []
    for u in utts:
        y = np.load(ws / "truth" / f"{u['utt']}.npy"); y = y / np.max(np.abs(y))       # their wav /= max
        e = brick(y, fs_in / 2); inp = nuwave2_input(y, fs_in)
        rows.append({"utt": u["utt"], "empty": snr(y, e), "input": snr(y, inp),
                     "share": float(np.sum((y - e) ** 2) / np.sum(y ** 2))})
    A = {k: np.array([r[k] for r in rows]) for k in ("empty", "input", "share")}
    spk = sorted({r["utt"].split("_")[0] for r in rows})
    res[str(fs_in)] = {
        "empty": float(A["empty"].mean()), "empty_se": float(A["empty"].std(ddof=1) / np.sqrt(len(rows))),
        "input": float(A["input"].mean()), "input_se": float(A["input"].std(ddof=1) / np.sqrt(len(rows))),
        "empty_global_db": float(-10 * np.log10(A["share"].mean())), "hb_share_pct": float(100 * A["share"].mean()),
        "empty_sd": float(A["empty"].std(ddof=1)),
        "by_speaker_empty": {s: float(np.mean([r["empty"] for r in rows if r["utt"].startswith(s + "_")])) for s in spk},
        "per_utt": rows}
    print(f"{fs_in//1000:>2} kHz -> 48: empty-band ceiling {res[str(fs_in)]['empty']:.2f} ± {res[str(fs_in)]['empty_se']:.2f} dB "
          f"(sd {res[str(fs_in)]['empty_sd']:.2f}); NU-Wave 2 input row {res[str(fs_in)]['input']:.2f}; "
          f"HB share {res[str(fs_in)]['hb_share_pct']:.2f} %", flush=True)
json.dump({"_meta": {"set": a.set, "n_utts": len(utts), "speakers": sorted({u['utt'].split('_')[0] for u in utts}),
                     "snr": "per-utterance 10log10(|y|^2/|y-w|^2), mean of dB, whole utterances (NU-Wave 2 / AP-BWE cal_snr)"},
           "by_rate": res}, open(ws / f"ceiling_{a.set}.json", "w"), indent=1)
