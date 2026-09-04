# ============================================================ XL-1 corpus
# Full VCTK 0.92 (mic1) from the Hub -- 27 parquet shards, 48 kHz, no DataShare.
# Same loader semantics as load_utterance(): mono, 48 kHz, peak 0.95, drop < 1 s.
import io, time, json, math, numpy as np, soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download
import pyarrow.parquet as pq

HF_REPO  = "sanchit-gandhi/vctk"
OUR_TEST = {"p236", "p237", "p238"}          # continuity with test_FULL.npz
PAPER_TEST_MIN = 350                          # paper: train = speaker id < 350

t0 = time.time()
local = snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=["data/*.parquet"], max_workers=8)
shards = sorted(Path(local).glob("data/*.parquet"))
print(f"{len(shards)} shards downloaded in {time.time()-t0:.0f}s")

def spk_num(s):
    return int(s[1:]) if (s[:1] == "p" and s[1:].isdigit()) else -1

def role(spk):
    if spk in OUR_TEST:
        return "test"
    n = spk_num(spk)
    if n < 0:
        return "skip"
    return "paper_test" if n >= PAPER_TEST_MIN else "train"

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

corpus = {"train": [], "test": [], "paper_test": []}     # (file, spk, array)
pool = ThreadPoolExecutor(32)
n_rows = n_mic1 = 0
t0 = time.time()
for si, sh in enumerate(shards):
    pf = pq.ParquetFile(sh)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["speaker_id", "file", "audio"])
        spks, files, auds = (t.column(c).to_pylist() for c in ("speaker_id", "file", "audio"))
        n_rows += len(files)
        keep = [i for i in range(len(files))
                if ("mic1" in (files[i] or "")) or ("mic1" in ((auds[i] or {}).get("path") or ""))]
        keep = [i for i in keep if role(spks[i]) != "skip"]
        n_mic1 += len(keep)
        arrs = list(pool.map(lambda i: load_bytes(auds[i]["bytes"]), keep))
        for i, a in zip(keep, arrs):
            if a is not None:
                corpus[role(spks[i])].append((files[i], spks[i], a))
        del t, auds
    print(f"  shard {si+1:>2}/{len(shards)}  rows {n_rows:>6}  mic1 kept {n_mic1:>6}  [{time.time()-t0:.0f}s]", flush=True)

for k in corpus:
    corpus[k].sort(key=lambda r: (r[1], r[0]))

# our held-out trio: first 40 utterances per speaker, as in test_FULL.npz
by_spk = {}
for f, s, a in corpus["test"]:
    by_spk.setdefault(s, []).append(a)
test_utts = [a for s in sorted(by_spk) for a in by_spk[s][:40]]
test_spk  = [s for s in sorted(by_spk) for a in by_spk[s][:40]]

train_utts = [a for f, s, a in corpus["train"]]
train_spk  = [s for f, s, a in corpus["train"]]
paper_test_utts = [a for f, s, a in corpus["paper_test"]]
paper_test_spk  = [s for f, s, a in corpus["paper_test"]]

def hours(utts):
    return sum(len(u) for u in utts) / CFG.fs_hi / 3600

print()
print(f"train       {len(train_utts):>6} utts  {len(set(train_spk)):>3} speakers  {hours(train_utts):6.2f} h")
print(f"test (ours) {len(test_utts):>6} utts  {len(set(test_spk)):>3} speakers  {hours(test_utts):6.2f} h   {sorted(set(test_spk))}")
print(f"paper test  {len(paper_test_utts):>6} utts  {len(set(paper_test_spk)):>3} speakers  {hours(paper_test_utts):6.2f} h")
assert not (set(train_spk) & set(test_spk)) and not (set(train_spk) & set(paper_test_spk))
MANIFEST = {"repo": HF_REPO, "train_speakers": sorted(set(train_spk)), "test_speakers": sorted(set(test_spk)),
            "paper_test_speakers": sorted(set(paper_test_spk)), "n_train": len(train_utts), "train_hours": hours(train_utts)}
print("manifest:", json.dumps({k: (v if not isinstance(v, list) else f"{len(v)} items") for k, v in MANIFEST.items()}))
