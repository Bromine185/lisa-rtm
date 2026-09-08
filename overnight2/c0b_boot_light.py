# ============================================================ OV2-0b light boot (evaluation-only runtimes)
# Same as c0_boot but the corpus cell is replaced by a filtered load: held-out speakers (p236-238), the
# paper's test split (id >= 350) and a 1-in-200 subsample of training utterances (for the ladder fit).
# Fits in ~2 GB of host RAM, so it runs on a T4 / standard-RAM runtime.
import json, sys, io, time, math
from pathlib import Path

OV2 = Path("/content/drive/MyDrive/lisa_rtm/ov2")
_nb = json.loads((OV2 / "lisa_rtm.ipynb").read_text())
_code = ["".join(c["source"]) for c in _nb["cells"] if c["cell_type"] == "code"]

def _run(mark, label):
    src = next(s for s in _code if mark in s)
    print(f"--- notebook cell: {label}", flush=True)
    exec(compile(src, f"<nb {label}>", "exec"), globals())

_run("import importlib, subprocess, sys", "setup")
_run("@dataclasses.dataclass(frozen=True)", "config")
_run("def _hann(n):", "spectral")
_run("class QuantileMap:", "transport")
_run("def snr_db(y, y_hat):", "metrics")
_run("VCTK_URL = ", "data-utils")
_run("class LISAEncoder(nn.Module):", "model")

# ---- filtered Hub corpus ------------------------------------------------------------------------
import numpy as np, soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download
import pyarrow.parquet as pq

HF_REPO, OUR_TEST, PAPER_TEST_MIN, TRAIN_EVERY = "sanchit-gandhi/vctk", {"p236", "p237", "p238"}, 350, 200
t0 = time.time()
local = snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=["data/*.parquet"], max_workers=8)
shards = sorted(Path(local).glob("data/*.parquet"))
print(f"{len(shards)} shards in {time.time()-t0:.0f}s", flush=True)

def spk_num(s):
    return int(s[1:]) if (s[:1] == "p" and s[1:].isdigit()) else -1

def role(spk):
    if spk in OUR_TEST:
        return "test"
    n = spk_num(spk)
    return "skip" if n < 0 else ("paper_test" if n >= PAPER_TEST_MIN else "train")

def load_bytes(b, fs_hi=CFG.fs_hi, peak=0.95, min_seconds=1.0):
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

corpus = {"train": [], "test": [], "paper_test": []}
pool = ThreadPoolExecutor(16)
n_train_seen = 0
for si, sh in enumerate(shards):
    pf = pq.ParquetFile(sh)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["speaker_id", "file", "audio"])
        spks, files, auds = (t.column(c).to_pylist() for c in ("speaker_id", "file", "audio"))
        keep = []
        for i in range(len(files)):
            if not (("mic1" in (files[i] or "")) or ("mic1" in ((auds[i] or {}).get("path") or ""))):
                continue
            r = role(spks[i])
            if r == "skip":
                continue
            if r == "train":
                n_train_seen += 1
                if n_train_seen % TRAIN_EVERY != 0:
                    continue
            keep.append(i)
        arrs = list(pool.map(lambda i: load_bytes(auds[i]["bytes"]), keep))
        for i, a in zip(keep, arrs):
            if a is not None:
                corpus[role(spks[i])].append((files[i], spks[i], a))
        del t, auds
    print(f"  shard {si+1:>2}/{len(shards)}  kept train {len(corpus['train'])} test {len(corpus['test'])} paper {len(corpus['paper_test'])}  [{time.time()-t0:.0f}s]", flush=True)
for k in corpus:
    corpus[k].sort(key=lambda r: (r[1], r[0]))
by_spk = {}
for f, s, a in corpus["test"]:
    by_spk.setdefault(s, []).append(a)
test_utts = [a for s in sorted(by_spk) for a in by_spk[s][:40]]
test_spk = [s for s in sorted(by_spk) for a in by_spk[s][:40]]
train_utts = [a for f, s, a in corpus["train"]]
train_spk = [s for f, s, a in corpus["train"]]
paper_test_utts = [a for f, s, a in corpus["paper_test"]]
paper_test_spk = [s for f, s, a in corpus["paper_test"]]
MANIFEST = {"repo": HF_REPO, "train_speakers": sorted(set(train_spk)), "test_speakers": sorted(set(test_spk)),
            "paper_test_speakers": sorted(set(paper_test_spk)), "n_train": len(train_utts), "light": True}
assert not (set(train_spk) & set(test_spk))
print(f"light corpus: train {len(train_utts)} (1 in {TRAIN_EVERY}), test {len(test_utts)}, paper test {len(paper_test_utts)}", flush=True)

XL = OV2 / "xl"
for f in ("cell2_trainer.py", "cell3_eval.py"):
    print(f"--- xl cell: {f}", flush=True)
    exec(compile((XL / f).read_text(), f"<xl {f}>", "exec"), globals())
OV2 = str(OV2).rstrip("/") + "/"           # later cells concatenate OV2 + "file.py"
print("LIGHT BOOT DONE", flush=True)
