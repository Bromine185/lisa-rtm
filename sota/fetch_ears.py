"""The EARS test set, cut into 4 s utterances by a fixed rule (notes/2026-11-27-ears-predictions.md).

    python sota/fetch_ears.py [--per-speaker 16] [--keep-zips]

EARS (Richter et al., Interspeech 2024; CC BY-NC 4.0): 48 kHz anechoic speech, 107 speakers. Its own test
split is p102-p107. For each test speaker this downloads the release zip, reads only the read-speech files in
regular style (`*_regular.wav`), and cuts them, in name order, into consecutive non-overlapping 4 s windows.
A window is kept when at least 80% of its 20 ms frames lie within 40 dB of the file's loudest frame. From the
kept windows, `per_speaker` evenly spaced ones are written (np.linspace, as sota/fetch_wide.py picks VCTK files).

Output: $BWE_CACHE/ears_seg/<spk>/<spk>_<k>.wav (float32, 48 kHz, raw level) and
$BWE_CACHE/ears_seg/manifest.json (source file and offset of every utterance, speaker metadata).
Then:  python sota/make_inputs.py --set ears --src ears --speakers p102 p103 p104 p105 p106 p107 --per-speaker 16
"""
import argparse, io, json, os, pathlib, urllib.request, zipfile
import numpy as np
import soundfile as sf

CACHE = pathlib.Path(os.environ.get("BWE_CACHE", str(pathlib.Path(__file__).resolve().parents[1] / "lisa_rtm_cache")))
RAW, OUT = CACHE / "ears_raw", CACHE / "ears_seg"
BASE = "https://github.com/facebookresearch/ears_dataset"
SPEAKERS = ("p102", "p103", "p104", "p105", "p106", "p107")        # EARS test split (Richter et al., Sec. 3)
FS, WIN, FRAME, FLOOR_DB, ACTIVE = 48000, 4.0, 0.02, 40.0, 0.8


def get(url, dst):
    if not dst.exists():
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
        tmp.rename(dst)
    return dst


def windows(x):
    """Consecutive 4 s windows of one file that pass the activity rule; returns start samples."""
    n, fl = int(WIN * FS), int(FRAME * FS)
    e = np.array([np.mean(x[i:i + fl] ** 2) for i in range(0, len(x) - fl + 1, fl)]) + 1e-20
    edb = 10 * np.log10(e); top = edb.max(); fpw = n // fl
    keep = []
    for s in range(0, len(x) - n + 1, n):
        f0 = s // fl
        if np.mean(edb[f0:f0 + fpw] >= top - FLOOR_DB) >= ACTIVE:
            keep.append(s)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-speaker", type=int, default=16)
    ap.add_argument("--keep-zips", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    stats = json.loads(get(f"https://raw.githubusercontent.com/facebookresearch/ears_dataset/main/speaker_statistics.json",
                           RAW / "speaker_statistics.json").read_text())
    man = {"rule": __doc__.strip(), "speakers": {}, "utts": []}
    for spk in SPEAKERS:
        z = get(f"{BASE}/releases/download/dataset/{spk}.zip", RAW / f"{spk}.zip")
        cand = []
        with zipfile.ZipFile(z) as zf:
            names = sorted(m for m in zf.namelist() if m.endswith("_regular.wav"))
            for m in names:
                x, sr = sf.read(io.BytesIO(zf.read(m)), dtype="float64")
                assert sr == FS, (m, sr)
                x = x.mean(1) if x.ndim > 1 else x
                cand += [(m, s, x[s:s + int(WIN * FS)]) for s in windows(x)]
        pick = np.unique(np.linspace(0, len(cand) - 1, min(a.per_speaker, len(cand))).round().astype(int))
        (OUT / spk).mkdir(exist_ok=True)
        for k, i in enumerate(pick):
            m, s, seg = cand[i]; utt = f"{spk}_{k:03d}"
            sf.write(str(OUT / spk / f"{utt}.wav"), seg.astype(np.float32), FS, subtype="FLOAT")
            man["utts"].append({"utt": utt, "src": m, "start_s": s / FS})
        man["speakers"][spk] = {**stats.get(spk, {}), "regular_files": len(names), "windows_kept": len(cand),
                                "picked": len(pick)}
        print(f"{spk}: {len(names)} regular files, {len(cand)} windows kept, {len(pick)} picked "
              f"({stats.get(spk, {}).get('gender')})", flush=True)
        if not a.keep_zips:
            z.unlink()
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"{len(man['utts'])} utterances -> {OUT}")


if __name__ == "__main__":
    main()
