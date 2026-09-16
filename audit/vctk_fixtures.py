"""A handful of VCTK 0.92 utterances, pulled out of the remote archive over HTTP ranges.

The archive is 11 GB; we need a few megabytes.  `zipfile` over a seekable range-reader reads the
central directory and then only the members asked for.  Files land in AUDIT_DATA (default
`lisa_rtm_cache/audit/`) and are reused on later runs.

    from audit.vctk_fixtures import fetch
    paths = fetch(("p360", "p361"), per_speaker=8)
"""
import io, math, os, pathlib, zipfile

URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
DATA = pathlib.Path(os.environ.get("AUDIT_DATA",
                                   pathlib.Path(__file__).resolve().parents[1] / "lisa_rtm_cache" / "audit"))

# LISA's validation split is speaker id >= 350 (datasets/audio_dataset.py:59).
PAPER_TEST = ("p351", "p360", "p361", "p362", "p363", "p364", "p374", "p376")
OUR_TEST = ("p236", "p237", "p238")


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests.

    The datashare handle 302s and answers HEAD with HTML, so the length comes from the
    Content-Range of a one-byte GET and every later request uses the resolved URL.
    """

    def __init__(self, url=URL):
        import requests
        self.s = requests.Session()
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=120)
        r.raise_for_status()
        self.url = r.url
        self.length = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0
        r.close()

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {0: off, 1: self.pos + off, 2: self.length + off}[whence]
        return self.pos

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.length - self.pos
        size = min(size, self.length - self.pos)
        if size <= 0:
            return b""
        r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + size - 1}"}, timeout=180)
        r.raise_for_status()
        self.pos += len(r.content)
        return r.content

    def readall(self):
        return self.read(-1)

    def readinto(self, b):                      # BufferedReader calls this, not read()
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def fetch(speakers=PAPER_TEST, per_speaker=8):
    """Returns sorted paths to `per_speaker` mic1 FLACs for each speaker, downloading what is missing."""
    DATA.mkdir(parents=True, exist_ok=True)
    have = {s: sorted((DATA / s).glob("*.flac"))[:per_speaker] for s in speakers}
    if all(len(v) >= per_speaker for v in have.values()):
        return [p for s in speakers for p in have[s]]
    zf = zipfile.ZipFile(io.BufferedReader(RangeFile(), buffer_size=1 << 20))
    names = {}
    for n in zf.namelist():
        if n.endswith("_mic1.flac") and "wav48_silence_trimmed/" in n:
            spk = n.split("/")[-2]
            if spk in speakers:
                names.setdefault(spk, []).append(n)
    out = []
    for spk in speakers:
        (DATA / spk).mkdir(exist_ok=True)
        for n in sorted(names.get(spk, []))[:per_speaker]:
            p = DATA / spk / pathlib.Path(n).name
            if not p.exists():
                p.write_bytes(zf.read(n))
            out.append(p)
    print(f"fetched {len(out)} utterances from {len(speakers)} speakers into {DATA}", flush=True)
    return out


def load(path, fs=48000, peak=0.95):
    """Mono, resampled to fs, peak-normalised -- the same treatment the corpus loader gives."""
    import numpy as np, scipy.signal as sps, soundfile as sf
    y, sr = sf.read(str(path), dtype="float64")
    if y.ndim > 1:
        y = y.mean(1)
    if sr != fs:
        g = math.gcd(int(sr), int(fs))
        y = sps.resample_poly(y, fs // g, sr // g)
    m = float(np.max(np.abs(y)))
    return y * (peak / m) if m > 0 else y
