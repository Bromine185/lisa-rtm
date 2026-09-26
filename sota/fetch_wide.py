"""A wider slice of the NU-Wave 2 / FLowHigh test speakers, spread through each speaker's script.

    venv/bin/python sota/fetch_wide.py [--per-speaker 30]

audit/vctk_fixtures.fetch() takes each speaker's FIRST files, and VCTK's first ~24 prompts are the same
sentences for every speaker. This takes `per_speaker` files evenly spaced through the sorted list instead,
into lisa_rtm_cache/audit_wide/, so the audit fixtures (and every number already computed on them) stay
untouched. Same HTTP-range reader, same mic1 / silence-trimmed members.
"""
import argparse, io, pathlib, sys, zipfile
import numpy as np
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from audit.vctk_fixtures import RangeFile                                    # noqa: E402

SPEAKERS = ("p360", "p361", "p362", "p363", "p364", "p374", "p376", "s5")    # NU-Wave 2 / FLowHigh test set
OUT = pathlib.Path("/Users/raghavsharma/projects/lisa-rtm/lisa_rtm_cache/audit_wide")

ap = argparse.ArgumentParser(); ap.add_argument("--per-speaker", type=int, default=30); a = ap.parse_args()
zf = zipfile.ZipFile(io.BufferedReader(RangeFile(), buffer_size=1 << 20))
names = {}
for n in zf.namelist():
    if n.endswith("_mic1.flac") and "wav48_silence_trimmed/" in n and n.split("/")[-2] in SPEAKERS:
        names.setdefault(n.split("/")[-2], []).append(n)
for spk in SPEAKERS:
    L = sorted(names.get(spk, []))
    pick = [L[i] for i in np.unique(np.linspace(0, len(L) - 1, a.per_speaker).round().astype(int))]
    (OUT / spk).mkdir(parents=True, exist_ok=True)
    for n in pick:
        p = OUT / spk / pathlib.Path(n).name
        if not p.exists():
            p.write_bytes(zf.read(n))
    print(spk, len(L), "in archive,", len(pick), "kept", flush=True)
