"""One set of test inputs, written once, that every model below reads.

    venv/bin/python sota/make_inputs.py --set core --per-speaker 4
    venv/bin/python sota/make_inputs.py --set wide --per-speaker 30
    venv/bin/python sota/make_inputs.py --set ourtest --src audit --speakers p236 p237 p238 --per-speaker 11
    venv/bin/python sota/make_inputs.py --set ourtest_poly --src audit --speakers p236 p237 p238 --per-speaker 11 --resampler poly

Truth: VCTK 0.92 mic1 silence-trimmed, the NU-Wave 2 / FLowHigh test speakers (p360 p361 p362 p363 p364
p374 p376 s5), from cache/audit_wide (sota/fetch_wide.py), loaded by audit.vctk_fixtures.load
(48 kHz, peak 0.95), trimmed to a multiple of 3840 samples so every model's hop divides it (NU-Wave 2 256,
FLowHigh 480, AP-BWE 80 x 4).

Input: the FLowHigh / NU-Wave 2 evaluation input -- order-8 Chebyshev I low-pass, 0.05 dB ripple, by
sosfiltfilt, then scipy resample_poly down to 12 kHz (fast/paper_split_eval.py::down_cheby); or, with
--resampler poly, the repo's own decimate() (resample_poly alone), what EVAL12 and the training corpus use.
Written as a float32 WAV so no model sees 16-bit quantisation the others do not. --src audit reads the
audit fixtures (cache/audit: p236-p238, our held-out speakers, and the first files of the paper split)
instead of audit_wide. raw_peak is the flac's stored peak, which AP-BWE needs (it is not level invariant).

Layout: <work>/<set>/truth/<utt>.npy, <work>/<set>/lo12k/<utt>.wav, <work>/<set>/utts.json.
`--per-speaker k` takes k utterances evenly spaced through what audit_wide holds for each speaker.
"""
import os
import argparse, json, pathlib, sys
import numpy as np
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from audit.vctk_fixtures import load                               # noqa: E402
from fast.paper_split_eval import down_cheby, down_poly            # noqa: E402

SRC = {"wide": pathlib.Path(os.environ.get("BWE_CACHE", str(pathlib.Path(__file__).resolve().parents[1] / "cache")) + "/audit_wide"),
       "audit": pathlib.Path(os.environ.get("BWE_CACHE", str(pathlib.Path(__file__).resolve().parents[1] / "cache")) + "/audit")}
WORK = pathlib.Path(os.environ.get("BWE_CACHE", str(pathlib.Path(__file__).resolve().parents[1] / "cache")) + "/sota/work")
SPEAKERS = ("p360", "p361", "p362", "p363", "p364", "p374", "p376", "s5")
FS, R, QUANT = 48000, 4, 3840

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True)
ap.add_argument("--per-speaker", type=int, default=4)
ap.add_argument("--src", default="wide", choices=list(SRC))
ap.add_argument("--speakers", nargs="*", default=list(SPEAKERS))
ap.add_argument("--resampler", default="cheby", choices=["cheby", "poly"])
ap.add_argument("--utts", nargs="*", default=None, help="explicit utterance ids (e.g. p236_007); overrides --per-speaker")
ap.add_argument("--no-quant", action="store_true", help="keep the full length (the demo's fixtures are untrimmed)")
a = ap.parse_args()
if a.no_quant:
    QUANT = 1
WIDE = SRC[a.src]
out = WORK / a.set
(out / "truth").mkdir(parents=True, exist_ok=True); (out / "lo12k").mkdir(exist_ok=True)
utts = []
for spk in a.speakers:
    have = sorted((WIDE / spk).glob("*_mic1.flac"))
    if not have:
        print(f"{spk}: nothing in {WIDE}", flush=True); continue
    if a.utts:
        pick = [p for p in have if p.stem.replace("_mic1", "") in a.utts]
    else:
        pick = [have[i] for i in np.unique(np.linspace(0, len(have) - 1, min(a.per_speaker, len(have))).round().astype(int))]
    for p in pick:
        utt = p.stem.replace("_mic1", "")
        raw, _ = sf.read(str(p)); y = load(p, FS); y = y[: (len(y) // QUANT) * QUANT]
        np.save(out / "truth" / f"{utt}.npy", y.astype(np.float64))
        x_lo = down_cheby(y, R, FS) if a.resampler == "cheby" else down_poly(y, R)
        sf.write(str(out / "lo12k" / f"{utt}.wav"), x_lo.astype(np.float32), FS // R, subtype="FLOAT")
        utts.append({"utt": utt, "seconds": len(y) / FS, "raw_peak": float(np.abs(raw).max())})
json.dump({"speakers": sorted({u["utt"].split("_")[0] for u in utts}), "utts": utts,
           "input": {"cheby": "cheby1(8, 0.05 dB) sosfiltfilt + resample_poly 48k->12k", "poly": "resample_poly 48k->12k (the repo's decimate)"}[a.resampler],
           "src": str(WIDE), "truth_peak": 0.95, "raw_peak": "peak |x| of the flac as stored, before load() normalised to 0.95"},
          open(out / "utts.json", "w"), indent=1)
print(f"{len(utts)} utterances, {sum(u['seconds'] for u in utts):.0f} s -> {out}", flush=True)
