"""Generate lisa_rtm_sampler.ipynb -- the sampler (LISAS under the energy score) as one clean notebook.

The notebook is the deliverable; this script exists so it can be regenerated deterministically
instead of hand-edited as JSON.  Run:  python3 build_sampler_notebook.py
"""
import json
import pathlib

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ============================================================ title
md(r"""
# LISA as a conditional sampler: the energy score ends regression to the mean

A deterministic network trained with a pointwise loss on a one-to-many problem converges to the conditional barycentre of its target. For 12 → 48 kHz bandwidth extension the band above 6 kHz has random phase given the input, so its barycentre has almost no energy and the output is muffled. The fix here is an objective, not an architecture: eight Gaussian channels are concatenated to the input waveform and the same 88k-parameter LISA is trained under the energy score, a strictly proper scoring rule, with two noise draws per step. One forward pass is then one calibrated sample from $p(\text{high band} \mid \text{low band})$ at zero extra inference cost. With the noise at zero the network is LISA exactly.

Sections: setup, config, data, model and gates, objective, metrics, inference, training, evaluation, latency, listen, summary. Presets: `SMOKE` (synthetic corpus, runs on a CPU in minutes), `QUICK` (two VCTK shards, well under a GPU hour), `FULL` (37 h of VCTK, the 8 September 2026 recipe, needs an A100-40GB).
""")

# ============================================================ 0 setup
md(r"""
## 0. Setup

Packages, device, a persistent cache (Drive on Colab), and seeded random streams. `stream(label)` gives an independent numpy generator per component so re-running one cell does not disturb another.
""")
code(r"""
import importlib, subprocess, sys

def ensure(pkg, import_name=None):
    try:
        importlib.import_module(import_name or pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

for _pkg in ("numpy", "scipy", "matplotlib", "soundfile"):
    ensure(_pkg)

import os, io, gc, json, math, time, copy, hashlib, shutil, tempfile, warnings, dataclasses
from pathlib import Path
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import soundfile as sf
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

IN_COLAB = "google.colab" in sys.modules
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("gpu     ", torch.cuda.get_device_name(0))
torch.use_deterministic_algorithms(False)          # deterministic scatter-add in the gather backward is the hot spot
print("python  ", sys.version.split()[0], "  torch", torch.__version__, "  numpy", np.__version__)
print("device  ", DEVICE, "  colab", IN_COLAB)

# ---- persistent cache: Drive on Colab, LISA_RTM_ROOT or ./lisa_rtm_cache elsewhere ------------------
if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        ROOT = Path("/content/drive/MyDrive/lisa_rtm")
    except Exception as e:
        warnings.warn(f"Drive not mounted ({e}); caching to /content, which a disconnect will lose.")
        ROOT = Path("/content/lisa_rtm")
else:
    ROOT = Path(os.environ.get("LISA_RTM_ROOT", "./lisa_rtm_cache")).resolve()
CKPT, FIGS, AUDIO = ROOT / "checkpoints", ROOT / "figures", ROOT / "audio"
for _d in (CKPT, FIGS, AUDIO):
    _d.mkdir(parents=True, exist_ok=True)
print("cache   ", ROOT)

# ---- seeding --------------------------------------------------------------------------------------
SEED = 0

def stream(label, seed=None):
    '''Independent seeded numpy Generator for a named component.'''
    h = hashlib.blake2b(label.encode(), digest_size=8).digest()
    return np.random.default_rng((int.from_bytes(h, "big") ^ (SEED if seed is None else seed)) % (2 ** 63))

def seed_everything(seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

seed_everything()
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "figure.facecolor": "white"})
print("seeded  ", SEED)
""")

# ============================================================ 1 config
md(r"""
## 1. Config

One frozen dataclass; `PRESET` is the one switch. The model dimensions are the paper's: encoder kernels (7, 3, 3, 1) with channels (16, 32, 64, 32), a 5-layer ReLU MLP decoder of width 144, ×4 upsampling. The optimisation recipe is the paper's. The spectral weight $\lambda = 10^{-2}$ is the value the 4 September frontier run found best; eight noise channels.
""")
code(r"""
@dataclasses.dataclass(frozen=True)
class Config:
    name: str
    fs_hi: int              # target rate; the input rate is fs_hi // upsample
    upsample: int
    source: str             # "synthetic" | "hub"
    shards: tuple           # Hub parquet shard indices (source == "hub")
    max_train_hours: float  # cap on the training corpus held in host RAM
    seg_samples: int        # training segment length at fs_hi
    batch_size: int
    steps: int
    ckpt_every: int
    n_fft: int              # largest STFT of the spectral term; scales n, n/2, n/4
    n_eval_utts: int
    m_draws: int            # ensemble size for evaluation
    m_sweep: int            # ensemble size for the tau sweep
    # model
    enc_channels: tuple = (16, 32, 64, 32)
    enc_kernels: tuple = (7, 3, 3, 1)
    dec_hidden: int = 144
    dec_layers: int = 5
    n_noise: int = 8
    # optimisation: the paper's recipe, lambda from the 4 Sep frontier run
    lr: float = 1e-3
    lr_milestones: tuple = (0.2, 0.4, 0.5, 0.6, 0.7, 0.8)
    lr_gamma: float = 0.5
    lambda_spec: float = 1e-2
    grad_clip: float = 1e-3
    # evaluation basis, deliberately different from the loss basis
    eval_n_fft: int = 1024
    eval_hop: int = 256

    @property
    def fs_lo(self):
        return self.fs_hi // self.upsample

    @property
    def eval_k_cut(self):
        '''First evaluation-STFT bin above the input Nyquist.'''
        return int(np.ceil((self.fs_lo / 2) * self.eval_n_fft / self.fs_hi))


SMOKE = Config(name="SMOKE", fs_hi=16000, upsample=4, source="synthetic", shards=(), max_train_hours=1.0,
               seg_samples=4096, batch_size=8, steps=300, ckpt_every=150, n_fft=512,
               n_eval_utts=4, m_draws=4, m_sweep=2)
QUICK = Config(name="QUICK", fs_hi=48000, upsample=4, source="hub", shards=(0, 1), max_train_hours=6.0,
               seg_samples=24000, batch_size=16, steps=6000, ckpt_every=1000, n_fft=2048,
               n_eval_utts=12, m_draws=16, m_sweep=8)
FULL = Config(name="FULL", fs_hi=48000, upsample=4, source="hub", shards=tuple(range(27)), max_train_hours=40.0,
              seg_samples=48000, batch_size=32, steps=38000, ckpt_every=1000, n_fft=2048,
              n_eval_utts=12, m_draws=16, m_sweep=8)

PRESET = "QUICK" if torch.cuda.is_available() else "SMOKE"     # <-- the one switch: "SMOKE" | "QUICK" | "FULL"
PRESET = os.environ.get("LISA_RTM_PRESET", PRESET)

CFG = {"SMOKE": SMOKE, "QUICK": QUICK, "FULL": FULL}[PRESET]
TAG = f"SAMPLER_{CFG.name}"
print(f"preset      {CFG.name}   checkpoints {CKPT / TAG}")
print(f"rates       {CFG.fs_lo} Hz -> {CFG.fs_hi} Hz ({CFG.upsample}x); the model fills {CFG.fs_lo // 2}-{CFG.fs_hi // 2} Hz")
print(f"training    {CFG.steps} steps of batch {CFG.batch_size} x {CFG.seg_samples / CFG.fs_hi:.3g} s, lambda {CFG.lambda_spec:g}, {CFG.n_noise} noise channels")
print(f"evaluation  n_fft {CFG.eval_n_fft} hop {CFG.eval_hop}, high band = bins {CFG.eval_k_cut}..{CFG.eval_n_fft // 2}, M = {CFG.m_draws} draws")
""")

# ============================================================ 2 data
md(r"""
## 2. Data

The input is the target decimated by $R = 4$ through an anti-aliasing filter, $x = D_R\,y$. Everything above $f_s / 2R$ is removed; that is the band the model must fill. The trivial baseline is polyphase (windowed-sinc) upsampling of $x$. It leaves the band empty, and it is the sinc interpolator every deterministic model here converges to.

`SMOKE` uses a synthetic harmonic-plus-fricative corpus. `QUICK` and `FULL` read VCTK 0.92 (mic1, 48 kHz) from the Hugging Face Hub, `sanchit-gandhi/vctk`, 27 parquet shards of about 430 MB each. Speakers p236, p237 and p238 are held out; the paper's own test speakers (id ≥ 350) are set aside. The corpus is concatenated in host RAM as float32 and a batch is a set of aligned slices, so the GPU holds only the model.
""")
code(r"""
def decimate(x, R):
    '''Anti-aliased downsample by R: the operator that makes the high band unrecoverable.'''
    return sps.resample_poly(np.asarray(x, np.float64), 1, R)

def naive_upsample(y, cfg):
    '''Polyphase interpolation of the decimated input: the trivial baseline, empty above fs_lo/2.'''
    y = np.asarray(y, np.float64)
    return sps.resample_poly(decimate(y, cfg.upsample), cfg.upsample, 1)[:len(y)]

# ---- synthetic corpus (SMOKE) ---------------------------------------------------------------------
def synthetic_utterance(label, cfg, seconds=1.6):
    '''Harmonic stack with a predictable low band plus fricative bursts: broadband and phase-random.'''
    rng, fs = stream(f"synth/{label}"), cfg.fs_hi
    n = int(seconds * fs)
    t = np.arange(n) / fs
    f0 = 110 + 40 * rng.standard_normal() + 15 * np.sin(2 * np.pi * 1.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / fs
    x = sum((1.0 / h ** 0.9) * np.sin(h * phase + rng.uniform(0, 2 * np.pi)) for h in range(1, 60))
    env = np.zeros(n)
    for _ in range(4):
        s = rng.integers(0, n - fs // 8)
        env[s:s + fs // 8] = np.hanning(fs // 8)
    x = x * (0.5 + 0.5 * np.sin(2 * np.pi * 2.3 * t)) + 3.0 * env * rng.standard_normal(n)
    x[:fs // 10] = 0.0
    x[-fs // 10:] = 0.0
    return (x / np.max(np.abs(x)) * 0.95).astype(np.float32)

# ---- VCTK 0.92 from the Hub (QUICK, FULL) ----------------------------------------------------------
HF_REPO, OUR_TEST, PAPER_TEST_MIN = "sanchit-gandhi/vctk", ("p236", "p237", "p238"), 350

def load_hub(cfg):
    '''Returns train_utts, train_spk, test_utts, test_spk.  mic1 only, 48 kHz, peak 0.95, utterances >= 1 s.'''
    ensure("huggingface_hub"); ensure("pyarrow")
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor

    def role(spk):
        if spk in OUR_TEST:
            return "test"
        n = int(spk[1:]) if (spk[:1] == "p" and spk[1:].isdigit()) else -1
        return "skip" if n < 0 else ("paper_test" if n >= PAPER_TEST_MIN else "train")

    def load_bytes(b, peak=0.95, min_seconds=1.0):
        x, fs = sf.read(io.BytesIO(b), dtype="float64", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
        if fs != cfg.fs_hi:
            g = math.gcd(int(fs), int(cfg.fs_hi))
            x = sps.resample_poly(x, cfg.fs_hi // g, fs // g)
        if len(x) < min_seconds * cfg.fs_hi:
            return None
        m = float(np.max(np.abs(x)))
        return (x * (peak / m)).astype(np.float32) if m > 0 else None

    corpus = {"train": [], "test": [], "paper_test": []}
    pool = ThreadPoolExecutor(32)

    def read_shard(i):
        pat = f"data/train-{i:05d}-of-00027-*.parquet"
        local = snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=[pat], max_workers=8)
        pf, t0, n = pq.ParquetFile(next(Path(local).glob(pat))), time.time(), 0
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["speaker_id", "file", "audio"])
            spks, files, auds = (t.column(c).to_pylist() for c in ("speaker_id", "file", "audio"))
            keep = [j for j in range(len(files))
                    if ("mic1" in (files[j] or "") or "mic1" in ((auds[j] or {}).get("path") or "")) and role(spks[j]) != "skip"]
            for j, a in zip(keep, pool.map(lambda j: load_bytes(auds[j]["bytes"]), keep)):
                if a is not None:
                    corpus[role(spks[j])].append((files[j], spks[j], a)); n += 1
            del t, auds
        print(f"  shard {i:>2}: {n} mic1 utterances kept, speakers so far "
              f"{sorted({s for k in corpus for _, s, _ in corpus[k]})} [{time.time() - t0:.0f}s]", flush=True)

    wanted = list(cfg.shards)
    for i in wanted:
        read_shard(i)
    nxt, extra = max(wanted) + 1, 0
    while not corpus["test"] and nxt < 27 and extra < 4:        # QUICK: read on until a held-out speaker appears
        print(f"  none of {OUR_TEST} in shards {wanted}; reading shard {nxt}", flush=True)
        read_shard(nxt); wanted.append(nxt); nxt += 1; extra += 1
    for k in corpus:
        corpus[k].sort(key=lambda r: (r[1], r[0]))
    if not corpus["test"]:                                        # fall back: hold out the last two train speakers
        spk = sorted({s for _, s, _ in corpus["train"]})[-2:]
        corpus["test"] = [r for r in corpus["train"] if r[1] in spk]
        corpus["train"] = [r for r in corpus["train"] if r[1] not in spk]
        print(f"  held-out trio absent from shards {wanted}; holding out {spk} instead", flush=True)
    by_spk = {}
    for _, s, a in corpus["test"]:
        by_spk.setdefault(s, []).append(a)
    test = [(a, s) for s in sorted(by_spk) for a in by_spk[s][:40]]     # first 40 per speaker, as in test_FULL.npz
    train, cap, tot = [], cfg.max_train_hours * 3600 * cfg.fs_hi, 0
    for _, s, a in corpus["train"]:
        if tot + len(a) > cap:
            break
        train.append((a, s)); tot += len(a)
    print(f"  shards {wanted}: train speakers {sorted({s for _, s in train})}, held out {sorted(by_spk)}, "
          f"paper-test utterances set aside {len(corpus['paper_test'])}")
    return [a for a, _ in train], [s for _, s in train], [a for a, _ in test], [s for _, s in test]


class HostCorpus:
    '''Corpus concatenated in host RAM (float32); a batch is B aligned (input, target) slices moved to the device.'''
    def __init__(self, utts, cfg, seg_hi):
        from concurrent.futures import ThreadPoolExecutor
        R = cfg.upsample
        ys = [np.asarray(u, np.float32) for u in utts if len(u) >= seg_hi]
        ys = [y[: (len(y) // R) * R] for y in ys]
        with ThreadPoolExecutor(32) as ex:
            xs = list(ex.map(lambda y: decimate(y, R).astype(np.float32), ys))
        self.n, self.R, self.seg_hi, self.seg_lo = len(ys), R, seg_hi, seg_hi // R
        self.lens = np.array([len(y) for y in ys])
        self.off = np.concatenate([[0], np.cumsum(self.lens)[:-1]])
        self.Y = np.concatenate(ys); del ys
        self.X = np.concatenate(xs); del xs
        self.hours = float(self.lens.sum()) / cfg.fs_hi / 3600
        self._ar_hi, self._ar_lo = np.arange(seg_hi), np.arange(seg_hi // R)
        print(f"HostCorpus: {self.n} utterances, {self.hours:.2f} h ({self.lens.sum() / cfg.fs_hi:.0f} s), "
              f"{(self.Y.nbytes + self.X.nbytes) / 1e9:.2f} GB in host RAM")

    def batch(self, rng, B):
        idx = rng.integers(self.n, size=B)
        n_pos = (self.lens[idx] - self.seg_hi) // self.R + 1
        starts = (rng.random(B) * n_pos).astype(np.int64) * self.R
        hi0 = self.off[idx] + starts
        y = self.Y[hi0[:, None] + self._ar_hi]
        x = self.X[(hi0 // self.R)[:, None] + self._ar_lo]
        return torch.from_numpy(x).to(DEVICE, non_blocking=True), torch.from_numpy(y).to(DEVICE, non_blocking=True)


if CFG.source == "synthetic":
    train_utts, train_spk = [synthetic_utterance(f"train/{i}", CFG) for i in range(8)], ["synth-train"] * 8
    test_utts, test_spk = [synthetic_utterance(f"test/{i}", CFG) for i in range(4)], ["synth-test"] * 4
else:
    train_utts, train_spk, test_utts, test_spk = load_hub(CFG)
assert not (set(train_spk) & set(test_spk)), "speaker overlap between train and held-out"
hours = lambda utts: sum(len(u) for u in utts) / CFG.fs_hi / 3600
print(f"train: {len(train_utts)} utterances, {len(set(train_spk))} speakers, {hours(train_utts):.2f} h   "
      f"held out: {len(test_utts)} utterances, speakers {sorted(set(test_spk))}, {hours(test_utts):.2f} h")
corpus = HostCorpus(train_utts, CFG, seg_hi=CFG.seg_samples)
del train_utts; gc.collect()                       # the corpus holds the data now; the list is 26 GB in FULL
""")

# ============================================================ 3 model
md(r"""
## 3. Model

LISA (Kim, Lee, Hong, Ok, ICASSP 2022). A 1-D convolutional encoder maps the low-rate waveform to one 32-d latent per input sample, $z_i = E(x)_i$, with a receptive field of 11 input samples. For an output time $t$ in input-sample units the decoder reads the anchor $i = \lfloor t \rfloor$, the relative coordinate $c = 2(t - i) - 1 \in [-1, 1]$ and the three latents around the anchor:

$$\hat y(t) = D\big([\,c,\; z_{i-1},\; z_i,\; z_{i+1}\,]\big), \qquad D = \text{a 5-layer ReLU MLP of width 144}.$$

LISAS is the same network with $n$ Gaussian channels concatenated to the input at the input rate,

$$z = E\big([\,x;\ \tau\varepsilon\,]\big), \qquad \varepsilon \sim \mathcal N(0, I),$$

so the only new weights are the extra input columns of the first convolution (+896 parameters for $n = 8$). At $\tau = 0$ the network is LISA. Training uses $\tau = 1$; $\tau$ is the noise temperature at inference.
""")
code(r"""
class LISAEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers, c_in = [], 1
        for i, (c, k) in enumerate(zip(cfg.enc_channels, cfg.enc_kernels)):
            layers.append(nn.Conv1d(c_in, c, k, padding=k // 2))
            if i < len(cfg.enc_channels) - 1:
                layers.append(nn.ReLU())
            c_in = c
        self.net, self.dim = nn.Sequential(*layers), c_in

    def forward(self, x):                     # (B, L_lo) -> (B, dim, L_lo)
        return self.net(x.unsqueeze(1))


class LISADecoder(nn.Module):
    def __init__(self, latent, cfg):
        super().__init__()
        layers, d = [], 1 + 3 * latent
        for _ in range(cfg.dec_layers - 1):
            layers += [nn.Linear(d, cfg.dec_hidden), nn.ReLU()]
            d = cfg.dec_hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, f):
        return self.net(f).squeeze(-1)


class LISA(nn.Module):
    '''The reference model of the paper, kept for the gate below.'''
    def __init__(self, cfg):
        super().__init__()
        self.enc = LISAEncoder(cfg)
        self.dec = LISADecoder(self.enc.dim, cfg)
        self.R = cfg.upsample

    def forward(self, x_lo, j0=0, j1=None, perturb=False):
        B, L_lo = x_lo.shape
        j1 = L_lo * self.R if j1 is None else j1
        z = self.enc(x_lo).transpose(1, 2)                      # (B, L_lo, C)
        q = (torch.arange(j0, j1, device=x_lo.device, dtype=torch.float32) / self.R).unsqueeze(0).expand(B, -1)
        anchor = q + torch.randn_like(q) * 0.5 if perturb else q
        idx = torch.floor(anchor).long().clamp(0, L_lo - 1)
        coord = (2.0 * (q - idx.float()) - 1.0).unsqueeze(-1)

        def take(ii):
            ii = ii.clamp(0, L_lo - 1).unsqueeze(-1).expand(-1, -1, z.shape[-1])
            return torch.gather(z, 1, ii)

        return self.dec(torch.cat([coord, take(idx - 1), take(idx), take(idx + 1)], -1))


class LISAS(nn.Module):
    '''LISA, stochastic: n_noise Gaussian channels enter the encoder with the waveform.  eps = 0 is LISA.'''
    def __init__(self, cfg, n_noise=None):
        super().__init__()
        self.n_noise, self.R = (cfg.n_noise if n_noise is None else n_noise), cfg.upsample
        layers, c_in = [], 1 + self.n_noise
        for i, (c, k) in enumerate(zip(cfg.enc_channels, cfg.enc_kernels)):
            layers.append(nn.Conv1d(c_in, c, k, padding=k // 2))
            if i < len(cfg.enc_channels) - 1:
                layers.append(nn.ReLU())
            c_in = c
        self.enc, self.dim = nn.Sequential(*layers), c_in
        self.dec = LISADecoder(self.dim, cfg)
        self.tau = 0.0          # noise temperature used by reconstruct() when none is given
        self.seed = 0

    def sample_eps(self, x_lo, tau=1.0, seed=None):
        B, L = x_lo.shape
        if seed is None:
            return tau * torch.randn(B, self.n_noise, L, device=x_lo.device)
        g = torch.Generator(device=x_lo.device).manual_seed(int(seed))
        return tau * torch.randn(B, self.n_noise, L, device=x_lo.device, generator=g)

    def encode(self, x_lo, eps=None):
        B, L = x_lo.shape
        if eps is None:
            eps = torch.zeros(B, self.n_noise, L, device=x_lo.device, dtype=x_lo.dtype)
        return self.enc(torch.cat([x_lo.unsqueeze(1), eps], 1)).transpose(1, 2)      # (B, L, C)

    def decode(self, z, j0=0, j1=None, perturb=False):
        B, L_lo, C = z.shape
        j1 = L_lo * self.R if j1 is None else j1
        q = (torch.arange(j0, j1, device=z.device, dtype=torch.float32) / self.R).unsqueeze(0).expand(B, -1)
        anchor = q + torch.randn_like(q) * 0.5 if perturb else q
        idx = torch.floor(anchor).long().clamp(0, L_lo - 1)
        coord = (2.0 * (q - idx.float()) - 1.0).unsqueeze(-1)

        def take(ii):
            ii = ii.clamp(0, L_lo - 1).unsqueeze(-1).expand(-1, -1, C)
            return torch.gather(z, 1, ii)

        return self.dec(torch.cat([coord, take(idx - 1), take(idx), take(idx + 1)], -1))

    def forward(self, x_lo, j0=0, j1=None, perturb=False, eps=None):
        return self.decode(self.encode(x_lo, eps), j0, j1, perturb)


n_lisa = sum(p.numel() for p in LISA(CFG).parameters())
n_lisas = sum(p.numel() for p in LISAS(CFG).parameters())
assert n_lisas - n_lisa == CFG.n_noise * CFG.enc_channels[0] * CFG.enc_kernels[0]
print(f"LISA {n_lisa:,} parameters (paper: ~89k)   LISAS {n_lisas:,}   difference {n_lisas - n_lisa} = "
      f"{CFG.n_noise} noise channels x {CFG.enc_channels[0]} filters x kernel {CFG.enc_kernels[0]}")
""")

md(r"""
### Gates

Two checks before any data is touched. (a) A LISAS whose noise columns are zero reproduces the reference LISA at $\varepsilon = 0$ to floating-point precision, and the noise path is live when $\varepsilon \neq 0$. (b) The two-draw energy-score estimator is unbiased for the CRPS. For a Gaussian forecast $\mathcal N(\mu, \sigma^2)$ and $z = (y - \mu) / \sigma$,

$$\mathrm{CRPS} = \sigma \Big[ z\,(2\Phi(z) - 1) + 2\varphi(z) - \tfrac{1}{\sqrt{\pi}} \Big],$$

and the expected score over $y \sim \mathcal N(0, 1)$ is smallest at the correct $\sigma = 1$. That is properness, observed numerically.
""")
code(r"""
from math import erf, sqrt, pi, exp

def lisas_from_lisa(lisa, cfg, n_noise):
    '''Copy a reference LISA into a LISAS; the noise-input columns of the first convolution are zero.'''
    m = LISAS(cfg, n_noise).to(DEVICE)
    sd = {}
    for k, v in lisa.state_dict().items():
        nk = k.replace("enc.net.", "enc.")
        if nk == "enc.0.weight":
            w = torch.zeros_like(m.state_dict()[nk]); w[:, :1] = v; v = w
        sd[nk] = v
    m.load_state_dict(sd)
    return m.eval()

torch.manual_seed(SEED)
_ref = LISA(CFG).to(DEVICE).eval()
_zero = lisas_from_lisa(_ref, CFG, CFG.n_noise)
_live = LISAS(CFG).to(DEVICE).eval()
_x = 0.1 * torch.randn(2, 600, device=DEVICE)
with torch.no_grad():
    d0 = (_ref(_x) - _zero(_x)).abs().max().item()
    d1 = (_live(_x) - _live(_x, eps=_live.sample_eps(_x, 1.0, 0))).abs().max().item()
assert d0 < 1e-6, d0
assert d1 > 1e-6, d1
print(f"GATE (a)  eps = 0 is LISA: max |LISA - LISAS| = {d0:.1e}    noise path live: max |f(x, 0) - f(x, eps)| = {d1:.2e}")

def crps_gauss(mu, sigma, y):
    z = (y - mu) / sigma
    Phi, phi = 0.5 * (1 + erf(z / sqrt(2))), exp(-0.5 * z * z) / sqrt(2 * pi)
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / sqrt(pi))

def es_two_draw(y, y1, y2):
    return 0.5 * (np.abs(y - y1) + np.abs(y - y2)) - 0.5 * np.abs(y1 - y2)

rng, N = stream("gate/crps"), 200_000
mu, sigma, y = 0.3, 1.0, 0.7
est = es_two_draw(y, mu + sigma * rng.standard_normal(N), mu + sigma * rng.standard_normal(N)).mean()
ref = crps_gauss(mu, sigma, y)
assert abs(est - ref) / ref < 0.01, (est, ref)
print(f"GATE (b)  two-draw estimator {est:.4f} vs closed-form CRPS {ref:.4f}: {100 * abs(est - ref) / ref:.2f} % off on {N} pairs")
scores = {}
for s in (0.5, 1.0, 2.0):
    yy = rng.standard_normal(N)
    scores[s] = es_two_draw(yy, s * rng.standard_normal(N), s * rng.standard_normal(N)).mean()
assert scores[1.0] < scores[0.5] and scores[1.0] < scores[2.0], scores
print("GATE (b)  properness, y ~ N(0,1), forecast N(0, sigma^2): expected score  "
      + "   ".join(f"sigma={s:g}: {v:.4f}" for s, v in scores.items()) + "   -> minimum at sigma = 1")
del _ref, _zero, _live, _x
""")

# ============================================================ 4 objective
md(r"""
## 4. Objective

The energy score of a forecast $P$ against an outcome $y$ is

$$\mathrm{ES}(P, y) = \mathbb{E}\, d(Y, y) - \tfrac{1}{2}\, \mathbb{E}\, d(Y, Y'), \qquad Y, Y' \sim P \text{ independent}.$$

It is strictly proper whenever $d$ is a metric of strong negative type, which holds for $d(a, b) = \|\phi(a) - \phi(b)\|_1$ and for the Euclidean norm. Its unique minimiser over $P$ is the conditional law of $y$ given the input. With two draws $\hat y_k = f(x, \varepsilon_k)$ per step the unbiased estimate is

$$\widehat{\mathrm{ES}} = \tfrac{1}{2}\big[d(y, \hat y_1) + d(y, \hat y_2)\big] - \tfrac{1}{2}\, d(\hat y_1, \hat y_2).$$

The first term asks for accuracy. The second rewards spread, exactly as much as the truth spreads and no more. The geometry $d$ decides what the score is proper for:

| arm | $d(a, b)$ | proper for |
|---|---|---|
| `es_wave` | $\mathrm{mean}_t \lvert a_t - b_t \rvert$ | the per-sample marginals of the waveform |
| `es_marg` | above $+\ \lambda \cdot \mathrm{mean}_{k,t} \big\lvert \log\lvert A_{kt}\rvert - \log\lvert B_{kt}\rvert \big\rvert$ over three STFT scales | plus the per-bin marginals of log-magnitude |
| `es_slice` | the log-magnitude term replaced by $\mathbb{E}_\theta \lvert \theta \cdot (A_t - B_t) \rvert$, $\theta$ uniform on the unit sphere of $\mathbb{R}^F$, 64 directions per step | the joint law of a whole frame, since $\mathbb{E}_\theta \lvert \theta \cdot v \rvert \propto \lVert v \rVert_2$ (Cramér–Wold) |

Two deterministic controls share the same batches: `det`, the paper's $L_1(\text{wave}) + \lambda \cdot \text{MS-STFT}$, and `det_split`, the same loss with the waveform term on the low-passed output so that no term ever asks for zero above $f_s / 2R$.
""")
code(r"""
SCALES = [(CFG.n_fft, CFG.n_fft // 4), (CFG.n_fft // 2, CFG.n_fft // 8), (CFG.n_fft // 4, CFG.n_fft // 16)]
_WIN = {}

def _win(n, device):
    k = (n, str(device))
    if k not in _WIN:
        _WIN[k] = torch.hann_window(n, device=device)
    return _WIN[k]

def logmag_feats(y):
    '''List over scales of (B, F, T) log-magnitudes.'''
    return [torch.log(torch.stft(y, n, h, window=_win(n, y.device), return_complex=True).abs() + 1e-7) for n, h in SCALES]

def lowpass(y, R):
    '''Brick-wall projection onto the input band [0, fs_lo / 2].'''
    Y = torch.fft.rfft(y)
    k_cut = Y.shape[-1] // R
    Y[..., k_cut + 1:] = 0
    return torch.fft.irfft(Y, n=y.shape[-1])

def d_wave(a, b):
    return (a - b).abs().mean(dim=tuple(range(1, a.ndim)))                          # (B,)

def d_logmag(fa, fb):
    return sum((A - B).abs().mean(dim=(1, 2)) for A, B in zip(fa, fb)) / len(fa)       # (B,)

def d_sliced(fa, fb, thetas):
    '''|theta . (A_t - B_t)| averaged over P random unit directions in R^F, frames t and scales.'''
    tot = 0.0
    for A, B, Th in zip(fa, fb, thetas):                       # A: (B, F, T), Th: (F, P)
        tot = tot + torch.einsum("bft,fp->bpt", A - B, Th).abs().mean(dim=(1, 2))
    return tot / len(fa)

def sample_thetas(device, P=64):
    out = []
    for n, _ in SCALES:
        th = torch.randn(n // 2 + 1, P, device=device)
        out.append(th / th.norm(dim=0, keepdim=True))
    return out


class MultiScaleSTFTLoss(nn.Module):
    '''The paper's spectral term: spectral convergence + log-magnitude L1 at three scales.'''
    def __init__(self, n_fft):
        super().__init__()
        self.scales = SCALES

    def forward(self, y, y_hat):
        total = 0.0
        for n, h in self.scales:
            w = _win(n, y.device)
            Y = torch.stft(y, n, h, window=w, return_complex=True).abs()
            H = torch.stft(y_hat, n, h, window=w, return_complex=True).abs()
            sc = torch.norm(Y - H, p="fro") / (torch.norm(Y, p="fro") + 1e-8)
            lm = F.l1_loss(torch.log(H + 1e-7), torch.log(Y + 1e-7))
            total = total + sc + lm
        return total / len(self.scales)


def arm_loss(kind, lam, m, x, y, spec_loss):
    '''Returns (loss, logged terms).  Energy-score arms run their two draws in one forward pass of batch 2B.'''
    if kind in ("det", "det_split"):
        y_hat = m(x, perturb=True)                                            # eps = 0
        l_w = F.l1_loss(lowpass(y_hat, m.R), lowpass(y, m.R)) if kind == "det_split" else F.l1_loss(y_hat, y)
        l_s = spec_loss(y, y_hat)
        return l_w + lam * l_s, {"wave": l_w.item(), "spec": l_s.item(), "spread": 0.0}
    B = x.shape[0]
    eps = m.sample_eps(x.repeat(2, 1), 1.0)
    yh = m(x.repeat(2, 1), perturb=True, eps=eps)
    y1, y2 = yh[:B], yh[B:]
    dw = 0.5 * (d_wave(y, y1) + d_wave(y, y2)) - 0.5 * d_wave(y1, y2)
    spread = d_wave(y1, y2).mean().item()
    if kind == "es_wave":
        return dw.mean(), {"wave": dw.mean().item(), "spec": 0.0, "spread": spread}
    fy, f1, f2 = logmag_feats(y), logmag_feats(y1), logmag_feats(y2)
    if kind == "es_marg":
        d = d_logmag
    elif kind == "es_slice":
        th = sample_thetas(y.device)
        d = lambda a, b: d_sliced(a, b, th)
    else:
        raise ValueError(kind)
    ds = 0.5 * (d(fy, f1) + d(fy, f2)) - 0.5 * d(f1, f2)
    return (dw + lam * ds).mean(), {"wave": dw.mean().item(), "spec": ds.mean().item(), "spread": spread}


# every objective on one batch, as a smoke check of the code path
_spec, _m = MultiScaleSTFTLoss(CFG.n_fft).to(DEVICE), LISAS(CFG).to(DEVICE)
_x, _y = corpus.batch(stream("objective/check"), 4)
for _kind in ("det", "det_split", "es_wave", "es_marg", "es_slice"):
    _loss, _terms = arm_loss(_kind, CFG.lambda_spec, _m, _x, _y, _spec)
    assert torch.isfinite(_loss)
    print(f"{_kind:<10} loss {_loss.item():.4f}   " + "  ".join(f"{k} {v:.4f}" for k, v in _terms.items()))
del _spec, _m, _x, _y
""")

# ============================================================ 5 metrics
md(r"""
## 5. Metrics

All on held-out utterances, in an evaluation STFT with $n_\mathrm{fft} = 1024$ and hop 256, a different basis from the loss on purpose. $Y$ and $P$ are the STFTs of target and prediction; the high band is the set of bins above the input Nyquist.

- SNR $= 10 \log_{10} \|y\|^2 / \|y - \hat y\|^2$. Phase-sensitive; the paper's Eq. (4).
- LSD $=$ mean over frames of the RMS over bins of $\log_{10}|Y|^2 - \log_{10}|P|^2$. HB-LSD restricts it to the high band.
- Band energy ratio, per third-octave band $b$: $10 \log_{10} \sum_b |P|^2 / \sum_b |Y|^2$. The deficit is its mean over the high band; negative is over-smoothed.
- CRPS on high-band log-magnitude, fair ensemble form: $\frac{1}{M}\sum_m |s_m - y| - \frac{1}{2M^2}\sum_{m, m'} |s_m - s_{m'}|$. For a deterministic model it is the MAE of its point forecast.
- Sliced CRPS: the same on 32 random unit projections across bins; it sees cross-bin structure.
- PIT: the rank of the truth among the $M$ draws. A flat histogram is calibrated.
- Coherent fraction and $\kappa$, per band: $\hat\rho(b) = \mathrm{Re}\langle Y_b, P_b \rangle / \langle Y_b, Y_b \rangle$ and $\kappa(b) = \mathrm{Re}\langle Y_b, P_b \rangle / \langle P_b, P_b \rangle$. For the exact conditional mean $\hat\rho$ equals the predictable fraction $\rho$ of target power and $\kappa = 1$. $\kappa \approx 0$ means the output energy is hallucinated.

Three identities turn the deficit into a measurement. Write $y = \mu(x) + e$ with $\mu = \mathbb{E}[y \mid x]$, and let $Y$ be a draw from a calibrated sampler, independent of $y$.

1. The energy ratio of the ensemble mean in band $b$ is $\rho(b)$, a predictability spectrum. The $M$-draw mean carries residual $(1 - \rho) / M$; the corrected estimate is $\hat\rho = (M r - 1) / (M - 1)$.
2. Any point predictor obeys $\mathrm{SNR} \le 10 \log_{10} \big( 1 / \sum_b (1 - \rho(b))\, p_b \big)$ with $p_b$ the target's power fraction in band $b$. This bounds every deterministic model on this data.
3. $\mathbb{E}\|y - Y\|^2 = 2\, \mathbb{E}\|y - \mu\|^2$, so $\mathrm{SNR}(\text{mean of } M) - \mathrm{SNR}(\text{one draw}) = 10 \log_{10} \big( 2 / (1 + 1/M) \big)$ for a calibrated sampler. A smaller gap means under-dispersed, a larger one over-dispersed.
""")
code(r"""
def _hann(n):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)

def stft(x, n_fft, hop):
    '''Real signal -> (n_frames, n_bins) complex.  Returns (S, pad, length).'''
    x = np.asarray(x, dtype=np.float64)
    length, pad = len(x), n_fft
    xp = np.pad(x, (pad, pad + n_fft))
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(xp[idx] * _hann(n_fft), axis=-1), pad, length

def logmag(S, eps=1e-8):
    return np.log(np.abs(S) + eps)

def snr_db(y, y_hat):
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    n = min(len(y), len(y_hat))
    e = y[:n] - y_hat[:n]
    return 10.0 * np.log10(np.sum(y[:n] ** 2) / max(np.sum(e ** 2), 1e-20))

def lsd_db(y, y_hat, n_fft, hop, k_from=0, eps=1e-10):
    '''Log-spectral distance on power spectra (log10), optionally restricted to bins k >= k_from.'''
    Sy = np.abs(stft(y, n_fft, hop)[0])[:, k_from:]
    Sh = np.abs(stft(y_hat, n_fft, hop)[0])[:, k_from:]
    d = np.log10(Sy ** 2 + eps) - np.log10(Sh ** 2 + eps)
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=1))))

def third_octave_edges(fs, f_lo, f_hi):
    f = [f_lo]
    while f[-1] < f_hi:
        f.append(f[-1] * 2 ** (1 / 3))
    return np.array(f)

def band_energy_ratio(y, y_hat, fs, n_fft, hop, f_lo, f_hi):
    '''Per third-octave band: (centre Hz, 10 log10 energy(pred) / energy(target)).  0 dB is correct energy.'''
    Sy, Sh = np.abs(stft(y, n_fft, hop)[0]), np.abs(stft(y_hat, n_fft, hop)[0])
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    edges = third_octave_edges(fs, f_lo, min(f_hi, fs / 2 - 1))
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (freqs >= a) & (freqs < b)
        if sel.sum() == 0:
            continue
        num, den = np.sum(Sh[:, sel] ** 2), np.sum(Sy[:, sel] ** 2)
        out.append((np.sqrt(a * b), 10.0 * np.log10((num + 1e-20) / (den + 1e-20))))
    return np.array(out)

def crps_ensemble(samples, truth):
    '''samples (M, ...) against truth (...): fair estimator averaged over all elements.'''
    term1 = np.abs(samples - truth[None]).mean(0)
    diff = np.abs(samples[:, None] - samples[None, :]).mean((0, 1))
    return float(np.mean(term1 - 0.5 * diff))

def pit_ranks(samples, truth):
    '''Rank of the truth among M samples, in 0..M.'''
    return (samples < truth[None]).sum(0).ravel()

def spread_skill(samples, truth, n_bins=8):
    '''Binned (mean ensemble spread, RMSE of the ensemble mean).  Unit slope is the target.'''
    spread, err = samples.std(0).ravel(), (samples.mean(0) - truth).ravel()
    order = np.argsort(spread)
    spread, err = spread[order], err[order]
    edges = np.linspace(0, len(spread), n_bins + 1).astype(int)
    return np.array([(spread[a:b].mean(), np.sqrt((err[a:b] ** 2).mean())) for a, b in zip(edges[:-1], edges[1:]) if b > a])

# high-band third-octaves starting exactly at the input Nyquist, for the coherence estimators
_FREQS = np.fft.rfftfreq(CFG.eval_n_fft, 1.0 / CFG.fs_hi)
_EDGES_HB = third_octave_edges(CFG.fs_hi, CFG.fs_lo / 2, CFG.fs_hi / 2 - 1)
HB_SEL = [(_FREQS >= a) & (_FREQS < b) for a, b in zip(_EDGES_HB[:-1], _EDGES_HB[1:]) if ((_FREQS >= a) & (_FREQS < b)).sum()]
HB_CENTRES = np.array([np.sqrt(a * b) for a, b in zip(_EDGES_HB[:-1], _EDGES_HB[1:]) if ((_FREQS >= a) & (_FREQS < b)).sum()])
BB_SEL = _FREQS < CFG.fs_lo / 2

def coherent(y, w):
    '''Per high-band third-octave: coherent fraction Re<Y,P>/<Y,Y>, kappa = Re<Y,P>/<P,P>, and the target's
    power fraction in the band.  Then the same two coherences over the whole baseband as an alignment check.'''
    Y, P = stft(y, CFG.eval_n_fft, CFG.eval_hop)[0], stft(w, CFG.eval_n_fft, CFG.eval_hop)[0]
    n = min(len(Y), len(P)); Y, P = Y[:n], P[:n]
    tot = np.sum(np.abs(Y) ** 2)
    yy = np.array([np.sum(np.abs(Y[:, s]) ** 2) for s in HB_SEL])
    pp = np.array([np.sum(np.abs(P[:, s]) ** 2) for s in HB_SEL])
    yp = np.array([np.sum((Y[:, s] * np.conj(P[:, s])).real) for s in HB_SEL])
    bb_yy, bb_pp = np.sum(np.abs(Y[:, BB_SEL]) ** 2), np.sum(np.abs(P[:, BB_SEL]) ** 2)
    bb_yp = np.sum((Y[:, BB_SEL] * np.conj(P[:, BB_SEL])).real)
    kappa = np.where(pp > 1e-6 * yy, yp / np.maximum(pp, 1e-20), np.nan)     # undefined for an empty band
    return (yp / np.maximum(yy, 1e-20), kappa, yy / max(tot, 1e-20),
            bb_yp / max(bb_yy, 1e-20), bb_yp / max(bb_pp, 1e-20))

# sanity: a signal against itself
_y = np.asarray(test_utts[0], np.float64)
_coh, _kap, _pf, _bc, _bk = coherent(_y, _y)
assert abs(snr_db(_y, _y)) > 100 and lsd_db(_y, _y, CFG.eval_n_fft, CFG.eval_hop) < 1e-9
assert np.allclose(_coh, 1) and np.allclose(_kap, 1) and abs(_bc - 1) < 1e-9
print(f"metrics defined; {len(HB_SEL)} high-band third-octaves from {CFG.fs_lo // 2} Hz, centres {np.round(HB_CENTRES).astype(int).tolist()} Hz; "
      f"target power above the cut on one held-out utterance: {100 * _pf.sum():.2f} %")
""")

# ============================================================ 6 inference
md(r"""
## 6. Inference

One noise draw per utterance, encoded once and decoded in chunks of query coordinates. `ensemble` stacks $M$ seeded draws. `passthrough` is the honest system output for bandwidth extension: the given baseband from the input, the model's band above it, brick-wall at $f_s / 2R$.
""")
code(r"""
@torch.no_grad()
def reconstruct(model, y, cfg, chunk=1 << 15, tau=None, seed=None):
    '''Full-utterance inference.  tau=None -> model.tau (0 for deterministic arms); seed=None -> model.seed.'''
    model.eval()
    tau = model.tau if tau is None else tau
    seed = model.seed if seed is None else seed
    x_lo = torch.from_numpy(decimate(np.asarray(y, np.float64), cfg.upsample)).float()[None].to(DEVICE)
    eps = None if tau == 0 else model.sample_eps(x_lo, tau, seed)
    z = model.encode(x_lo, eps)
    n_out = x_lo.shape[1] * cfg.upsample
    out = [model.decode(z, s, min(s + chunk, n_out)).squeeze(0).cpu().numpy() for s in range(0, n_out, chunk)]
    y_hat = np.concatenate(out).astype(np.float64)
    return np.pad(y_hat, (0, max(0, len(y) - len(y_hat))))[:len(y)]

def ensemble(model, y, cfg, M, tau=1.0):
    '''(M, T) draws with seeds 0..M-1.'''
    return np.stack([reconstruct(model, y, cfg, tau=tau, seed=s) for s in range(M)])

def passthrough(y_hat, y, cfg):
    '''Given baseband (naive upsampling of the input) below fs_lo/2, the model's output above it.'''
    n = min(len(y_hat), len(y))
    base, hat = np.fft.rfft(naive_upsample(y[:n], cfg)), np.fft.rfft(np.asarray(y_hat[:n], np.float64))
    hi = np.fft.rfftfreq(n, 1.0 / cfg.fs_hi) > cfg.fs_lo / 2
    base[hi] = hat[hi]
    return np.fft.irfft(base, n=n)

_m = LISAS(CFG).to(DEVICE); _m.tau = 1.0
_y = np.asarray(test_utts[0], np.float64)
_d = ensemble(_m, _y, CFG, 2)
assert _d.shape == (2, len(_y)) and np.abs(_d[0] - _d[1]).max() > 0 and np.allclose(reconstruct(_m, _y, CFG, seed=0), _d[0])
_p = passthrough(_d[0], _y, CFG)
print(f"reconstruct: {len(_y)} samples, two seeded draws differ by up to {np.abs(_d[0] - _d[1]).max():.3f}; "
      f"passthrough baseband SNR {snr_db(naive_upsample(_y, CFG), passthrough(naive_upsample(_y, CFG), _y, CFG)):.0f} dB (identity check)")
del _m, _y, _d, _p
""")

# ============================================================ 7 training
md(r"""
## 7. Training

One initialisation, one batch stream, one optimiser per arm, so every difference between arms is the objective. Adam at $10^{-3}$, halved at 20/40/50/60/70/80 % of the steps; gradient clipping at $10^{-3}$; decoder coordinates perturbed by $\mathcal N(0, 0.5^2)$ during training. All of that is the paper's recipe. Energy-score arms run their two draws in one forward pass of batch $2B$. A finished checkpoint for this tag and arm is loaded instead of retrained. One held-out utterance is the probe scored at every checkpoint.
""")
code(r"""
def save_ckpt(state, path, retries=4):
    '''Write locally, then copy: Colab's Drive mount drops files under long writes.'''
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.gettempdir()) / f"{path.name}.{os.getpid()}.tmp"
    torch.save(state, tmp)
    for i in range(retries):
        try:
            shutil.copyfile(tmp, path); tmp.unlink(); return
        except OSError as e:
            err = e; time.sleep(2 ** i)
    raise err

def load_arm(path, cfg):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m = LISAS(cfg, ck.get("n_noise", cfg.n_noise)).to(DEVICE)
    m.load_state_dict(ck["model"]); m.eval()
    m.tau = 0.0 if ck["arm"][0].startswith("det") else 1.0
    return m, ck

def probe_metrics(m, probe, cfg):
    '''SNR and high-band deficit of the probe at tau = 0 and, for stochastic arms, one draw at tau = 1.'''
    out = {}
    for tau in ((0.0,) if m.tau == 0 else (0.0, 1.0)):
        yh = reconstruct(m, probe, cfg, tau=tau, seed=0)
        b = band_energy_ratio(probe, yh, cfg.fs_hi, cfg.eval_n_fft, cfg.eval_hop, cfg.fs_lo / 2, cfg.fs_hi / 2)
        out[tau] = (snr_db(probe, yh), float(np.mean(b[:, 1])))
    return out

def time_step(corpus, arms, cfg, n=10):
    '''Seconds per training step for these arms at this batch, measured on throwaway models.'''
    models = {k: LISAS(cfg).to(DEVICE) for k in arms}
    opts = {k: torch.optim.Adam(models[k].parameters(), lr=cfg.lr) for k in arms}
    spec_loss, rng = MultiScaleSTFTLoss(cfg.n_fft).to(DEVICE), stream("timing")
    sync = torch.cuda.synchronize if DEVICE.type == "cuda" else (lambda: None)
    for i in range(n + 3):
        if i == 3:
            sync(); t0 = time.time()
        x, y = corpus.batch(rng, cfg.batch_size)
        for k, (kind, lam) in arms.items():
            loss, _ = arm_loss(kind, lam, models[k], x, y, spec_loss)
            opts[k].zero_grad(set_to_none=True); loss.backward(); opts[k].step()
    sync()
    dt = (time.time() - t0) / n
    mem = f", peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if DEVICE.type == "cuda" else ""
    print(f"{len(arms)} arm(s) at batch {cfg.batch_size} x {cfg.seg_samples}: {dt * 1000:.0f} ms/step{mem}", flush=True)
    del models, opts
    return dt

def train_arms(corpus, arms, cfg, tag, probe, log_every=25):
    '''arms: {name: (kind, lam)}.  One init, one batch stream, one optimiser per arm.  Returns (models, history).'''
    names, run_dir = list(arms), CKPT / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    base = LISAS(cfg).to(DEVICE)
    models = {k: copy.deepcopy(base) for k in names}
    keys = ("step", "wave", "spec", "spread", "lr", "dev_step", "snr0", "def0", "snr1", "def1", "snr_naive")
    hist = {k: {kk: [] for kk in keys} for k in names}
    todo = []
    for k in names:
        models[k].tau = 0.0 if arms[k][0].startswith("det") else 1.0
        p = run_dir / f"{k}.pt"
        ck = torch.load(p, map_location=DEVICE, weights_only=False) if p.exists() else None
        if ck is not None and tuple(ck["arm"]) == tuple(arms[k]) and ck["step"] >= cfg.steps:
            models[k].load_state_dict(ck["model"]); hist[k] = ck["history"]
            print(f"{k}: loaded {p} at step {ck['step']}; training skipped", flush=True)
        else:
            todo.append(k)
    if not todo:
        return models, hist
    dt = time_step(corpus, {k: arms[k] for k in todo}, cfg)
    epochs = cfg.steps * cfg.batch_size * cfg.seg_samples / (corpus.hours * 3600 * cfg.fs_hi)
    print(f"plan: {todo} x {cfg.steps} steps = {epochs:.1f} epochs of {corpus.hours:.2f} h ({corpus.hours * 3600:.0f} s); "
          f"est {cfg.steps * dt / 60:.1f} min at {dt * 1000:.0f} ms/step", flush=True)
    opts = {k: torch.optim.Adam(models[k].parameters(), lr=cfg.lr) for k in todo}
    scheds = {k: MultiStepLR(opts[k], [int(f * cfg.steps) for f in cfg.lr_milestones], cfg.lr_gamma) for k in todo}
    spec_loss = MultiScaleSTFTLoss(cfg.n_fft).to(DEVICE)
    rng = stream(f"{tag}/batches")
    probe = np.asarray(probe, np.float64)
    naive = snr_db(probe, naive_upsample(probe, cfg))
    logf, t0 = open(ROOT / f"train_{tag}.log", "a"), time.time()
    for step in range(cfg.steps):
        x, y = corpus.batch(rng, cfg.batch_size)
        for k in todo:
            kind, lam = arms[k]
            m = models[k]; m.train()
            loss, terms = arm_loss(kind, lam, m, x, y, spec_loss)
            opts[k].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.grad_clip)
            opts[k].step(); scheds[k].step()
            if step % log_every == 0:
                h = hist[k]
                h["step"].append(step); h["wave"].append(terms["wave"]); h["spec"].append(terms["spec"])
                h["spread"].append(terms["spread"]); h["lr"].append(scheds[k].get_last_lr()[0])
        if (step + 1) % cfg.ckpt_every == 0 or step + 1 == cfg.steps:
            line = f"step {step + 1:>6}/{cfg.steps} [{time.time() - t0:.0f}s]"
            for k in todo:
                m, h = models[k], hist[k]
                pm = probe_metrics(m, probe, cfg)
                s1, d1 = pm.get(1.0, (float("nan"), float("nan")))
                h["dev_step"].append(step + 1); h["snr_naive"].append(naive)
                h["snr0"].append(pm[0.0][0]); h["def0"].append(pm[0.0][1]); h["snr1"].append(s1); h["def1"].append(d1)
                save_ckpt({"model": m.state_dict(), "step": step + 1, "history": h, "arm": arms[k], "n_noise": m.n_noise,
                           "cls": type(m).__name__, "batch": cfg.batch_size, "seg": corpus.seg_hi, "tag": tag}, run_dir / f"{k}.pt")
                line += f" | {k}: w {h['wave'][-1]:.4f} s {h['spec'][-1]:.3f} SNR0 {pm[0.0][0]:5.2f} def0 {pm[0.0][1]:+6.2f}"
                if 1.0 in pm:
                    line += f" SNR1 {s1:5.2f} def1 {d1:+6.2f}"
            line += f"  (probe: one held-out utterance; naive {naive:.2f})"
            print(line, flush=True); logf.write(line + "\n"); logf.flush()
    logf.close()
    for m in models.values():
        m.eval()
    return models, hist


ARMS = {"det": ("det", CFG.lambda_spec), "es_marg": ("es_marg", CFG.lambda_spec)}
# all five arms of the 8 Sep run:
# ARMS = {"det": ("det", 1e-2), "det_split": ("det_split", 1e-2), "es_marg": ("es_marg", 1e-2),
#         "es_slice": ("es_slice", 1e-2), "es_wave": ("es_wave", 0.0)}
models, hist = train_arms(corpus, ARMS, CFG, TAG, test_utts[0])

fig, ax = plt.subplots(1, 4, figsize=(17, 3.4))
for k in ARMS:
    h = hist[k]
    ax[0].plot(h["step"], h["wave"], label=k); ax[1].plot(h["step"], h["spec"], label=k); ax[2].plot(h["step"], h["spread"], label=k)
    ax[3].plot(h["dev_step"], h["def0"], "o-", ms=3, label=f"{k} tau=0")
    if np.isfinite(h["def1"]).any():
        ax[3].plot(h["dev_step"], h["def1"], "s--", ms=3, label=f"{k} one draw tau=1")
for a, t in zip(ax, ("waveform term", "spectral term", "spread d(y1, y2)", "probe high-band deficit (dB)")):
    a.set_title(t, fontsize=9); a.set_xlabel("step"); a.legend(fontsize=7)
ax[3].axhline(0, color="k", lw=1)
plt.tight_layout(); plt.savefig(FIGS / f"train_{TAG}.png", dpi=130); plt.show()
""")

# ============================================================ 8 evaluation
md(r"""
## 8. Evaluation

Held-out utterances, $M$ draws per stochastic arm. The table reads left to right: one draw, the proper score, the coherence, then the ensemble mean and the calibration gap of identity 3. The $\tau$ sweep moves the sampler along the fidelity–calibration curve; $\tau = 1$ is the temperature the score trained. Every number names its set.
""")
code(r"""
EVAL = [np.asarray(u, np.float64) for u in test_utts[:CFG.n_eval_utts]]
EVAL_NAME = f"held-out {CFG.name} set: {len(EVAL)} utterances, speakers {sorted(set(test_spk[:CFG.n_eval_utts]))}"
K_HB = CFG.eval_n_fft // 2 + 1 - CFG.eval_k_cut
_th = stream("eval/theta").standard_normal((K_HB, 32))
THETA = _th / np.linalg.norm(_th, axis=0, keepdims=True)
CENTRES = band_energy_ratio(EVAL[0], EVAL[0], CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop, 200.0, CFG.fs_hi / 2)[:, 0]
HB_MASK = CENTRES >= CFG.fs_lo / 2
NAIVE_SNR = float(np.mean([snr_db(y, naive_upsample(y, CFG)) for y in EVAL]))
print(EVAL_NAME, f"| naive upsampling SNR {NAIVE_SNR:.2f} dB")

def lm_hb(w):
    return logmag(stft(w, CFG.eval_n_fft, CFG.eval_hop)[0][:, CFG.eval_k_cut:])

def wave_metrics(y, w):
    b = band_energy_ratio(y, w, CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop, 200.0, CFG.fs_hi / 2)
    coh, kappa, pfrac, bb_coh, bb_kappa = coherent(y, w)
    return dict(snr=snr_db(y, w), lsd=lsd_db(y, w, CFG.eval_n_fft, CFG.eval_hop),
                hb_lsd=lsd_db(y, w, CFG.eval_n_fft, CFG.eval_hop, CFG.eval_k_cut),
                deficit=float(b[HB_MASK, 1].mean()), baseband=float(b[~HB_MASK, 1].mean()), curve=b[:, 1],
                coh=coh, kappa=kappa, pfrac=pfrac, bb_coh=bb_coh, bb_kappa=bb_kappa)

def agg(rows):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if np.ndim(rows[0][k]) == 0}
        out.update({k: np.nanmean([r[k] for r in rows], 0).tolist() for k in rows[0] if np.ndim(rows[0][k]) > 0})
    return out

def fmt_kappa(v):
    return f"{v:+.3f}" if np.isfinite(v) else "-"    # undefined for an empty high band

def evaluate(label, draw, M, tau):
    '''draw(y, seed) -> waveform.  M = 1 for a deterministic condition (its CRPS is the MAE of the point forecast).'''
    single, mean_rows, crps, scrps, ranks, sk = [], [], [], [], [], []
    for y in EVAL:
        waves = np.stack([draw(y, s) for s in range(M)])
        truth = lm_hb(y)
        ens = np.stack([lm_hb(w)[:truth.shape[0]] for w in waves])
        crps.append(crps_ensemble(ens, truth)); scrps.append(crps_ensemble(ens @ THETA, truth @ THETA))
        single.append(wave_metrics(y, waves[0]))
        if M > 1:
            ranks.append(pit_ranks(ens, truth)); sk.append(spread_skill(ens, truth)); mean_rows.append(wave_metrics(y, waves.mean(0)))
    r = {"label": label, "M": M, "tau": tau, "set": EVAL_NAME, "crps": float(np.mean(crps)),
         "sliced_crps": float(np.mean(scrps)), "single": agg(single)}
    best = single
    if M > 1:
        r["mean"] = agg(mean_rows); best = mean_rows
        rk = np.concatenate(ranks)
        r["pit_hist"] = (np.histogram(rk, bins=M + 1, range=(-0.5, M + 0.5))[0] / len(rk)).tolist()
        r["spread_skill"] = np.mean(sk, 0).tolist()
        ratio = 10 ** (np.array(r["mean"]["curve"])[HB_MASK] / 10)
        r["rho_energy"] = np.clip((M * ratio - 1) / (M - 1), 0, 1).tolist()          # identity 1, bias-corrected
        r["snr_gap_db"] = r["mean"]["snr"] - r["single"]["snr"]                          # identity 3
        r["snr_gap_calibrated_db"] = 10 * math.log10(2 / (1 + 1 / M))
    rho = np.clip(np.mean([s["coh"] for s in best], 0), 0, 1)                            # predictability spectrum
    r["rho"] = rho.tolist()
    r["snr_bound_db"] = float(np.mean([10 * math.log10(1 / max(float(np.sum((1 - rho) * s["pfrac"])), 1e-12)) for s in best]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r["hb_kappa"] = float(np.nanmean(r["single"]["kappa"]))
    s = r["single"]
    line = (f"{label:<28} SNR {s['snr']:6.2f}  LSD {s['lsd']:.3f}  HB-LSD {s['hb_lsd']:.3f}  deficit {s['deficit']:+6.2f} dB  "
            f"| CRPS {r['crps']:.4f}  sCRPS {r['sliced_crps']:.4f}  | HB kappa {fmt_kappa(r['hb_kappa'])}  BB coh {s['bb_coh']:.3f}  bound {r['snr_bound_db']:.1f} dB")
    if M > 1:
        line += f"  | mean of {M}: SNR {r['mean']['snr']:6.2f} deficit {r['mean']['deficit']:+6.2f}  gap {r['snr_gap_db']:.2f} (calibrated {r['snr_gap_calibrated_db']:.2f})"
    print(line, flush=True)
    return r

RES = {"naive": evaluate("naive upsampling", lambda y, s: naive_upsample(y, CFG), 1, 0.0)}
for k, m in models.items():
    stoch = m.tau > 0
    RES[k] = evaluate(f"{k} one draw tau=1" if stoch else k, lambda y, s, m=m: reconstruct(m, y, CFG, tau=m.tau, seed=s),
                      CFG.m_draws if stoch else 1, m.tau)
    if stoch:
        RES[f"{k} tau=0"] = evaluate(f"{k} tau=0", lambda y, s, m=m: reconstruct(m, y, CFG, tau=0.0), 1, 0.0)

TAUS = (0.0, 0.5, 0.75, 1.0, 1.25)
SWEEP = {k: [evaluate(f"{k} tau={t:g} (M={CFG.m_sweep if t > 0 else 1})",
                      lambda y, s, m=m, t=t: reconstruct(m, y, CFG, tau=t, seed=s), CFG.m_sweep if t > 0 else 1, t) for t in TAUS]
         for k, m in models.items() if m.tau > 0}

def results_table(RES):
    rows = ["| condition | SNR | LSD | HB-LSD | deficit dB | CRPS | HB kappa | mean-of-M SNR | mean deficit | SNR gap (calibrated) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in RES.items():
        s, mm = r["single"], r.get("mean")
        tail = (f"{mm['snr']:.2f} | {mm['deficit']:+.2f} | {r['snr_gap_db']:.2f} ({r['snr_gap_calibrated_db']:.2f}) |" if mm else "- | - | - |")
        rows.append(f"| {r['label']} | {s['snr']:.2f} | {s['lsd']:.3f} | {s['hb_lsd']:.3f} | {s['deficit']:+.2f} | "
                    f"{r['crps']:.4f} | {fmt_kappa(r['hb_kappa'])} | " + tail)
    return "\n".join(rows)

print(f"\n{EVAL_NAME}\n" + results_table(RES))

# ---- figures ---------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].semilogx(CENTRES, np.maximum(RES["naive"]["single"]["curve"], -40), ":", color="grey", label="naive upsampling")
for k, m in models.items():
    r = RES[k]
    ax[0].semilogx(CENTRES, r["single"]["curve"], "o-", ms=3, label=f"{k} (one draw)" if m.tau else k)
    if "mean" in r:
        ax[0].semilogx(CENTRES, r["mean"]["curve"], "--", lw=1.2, label=f"{k} (mean of {r['M']})")
ax[0].axhline(0, color="k", lw=1); ax[0].axvline(CFG.fs_lo / 2, color="crimson", ls="--"); ax[0].axhspan(-2, 2, color="grey", alpha=0.15)
ax[0].set_ylim(-42, 8); ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("pred / target energy (dB)")
ax[0].set_title(f"Energy ratio, one draw vs ensemble mean ({EVAL_NAME.split(':')[0]})", fontsize=9); ax[0].legend(fontsize=7)
for k, rows in SWEEP.items():
    ax[1].plot([r["single"]["snr"] for r in rows], [r["crps"] for r in rows], "o-", ms=4, label=f"{k}: tau {TAUS[0]:g} -> {TAUS[-1]:g}")
    for r in rows:
        ax[1].annotate(f"{r['tau']:g}", (r["single"]["snr"], r["crps"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
for k, m in models.items():
    if m.tau == 0:
        ax[1].scatter([RES[k]["single"]["snr"]], [RES[k]["crps"]], marker="s", s=45, label=f"{k} (deterministic)")
ax[1].axvline(NAIVE_SNR, color="grey", ls=":", lw=1)
ax[1].set_xlabel("waveform SNR (dB); dotted = naive upsampling"); ax[1].set_ylabel("CRPS on high-band log-magnitude")
ax[1].set_title("Noise temperature: fidelity vs proper score", fontsize=9); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIGS / f"eval_{TAG}.png", dpi=130); plt.show()

stoch = [k for k, m in models.items() if m.tau > 0]
if stoch:
    fig, ax = plt.subplots(1, len(stoch) + 1, figsize=(4 * (len(stoch) + 1), 3.3))
    for a, k in zip(ax, stoch):
        h = RES[k]["pit_hist"]
        a.bar(range(len(h)), h, edgecolor="k", lw=0.5); a.axhline(1 / len(h), color="crimson", ls="--")
        a.set_title(f"PIT: {k}", fontsize=9); a.set_xlabel("rank of truth among draws")
        s = np.array(RES[k]["spread_skill"]); ax[-1].plot(s[:, 0], s[:, 1], "o-", ms=3, label=k)
    lim = ax[-1].get_xlim(); ax[-1].plot([0, lim[1]], [0, lim[1]], "k--", lw=1)
    ax[-1].set_title("spread-skill", fontsize=9); ax[-1].set_xlabel("ensemble spread"); ax[-1].set_ylabel("RMSE of ensemble mean"); ax[-1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIGS / f"calibration_{TAG}.png", dpi=130); plt.show()
""")

# ============================================================ 9 latency
md(r"""
## 9. Latency

One stochastic forward pass, noise sampling included, for one second of audio and for a 20 ms chunk. The algorithmic look-ahead is the encoder's right context plus the decoder's right neighbour: $\sum_k \lfloor k/2 \rfloor + 1 = 6$ input samples, 0.5 ms at 12 kHz.
""")
code(r"""
def bench(fn, n_rep=20, warmup=3):
    sync = torch.cuda.synchronize if DEVICE.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    return np.percentile(ts, 50) * 1e3, np.percentile(ts, 95) * 1e3

LAT = {}
_m = models[next(k for k, m in models.items() if m.tau > 0)] if any(m.tau > 0 for m in models.values()) else next(iter(models.values()))
_dev = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else f"cpu ({torch.get_num_threads()} threads)"
with torch.no_grad():
    for name, n in (("1 s", CFG.fs_hi), ("20 ms chunk", CFG.fs_hi // 50)):
        x_lo = torch.from_numpy(decimate(EVAL[0][:n], CFG.upsample)).float()[None].to(DEVICE)
        p50, p95 = bench(lambda: _m(x_lo, eps=_m.sample_eps(x_lo, 1.0)))
        LAT[name] = {"p50_ms": p50, "p95_ms": p95, "real_time_factor": 1000 * n / CFG.fs_hi / p50}
        print(f"{name:<12} on {_dev}: p50 {p50:.2f} ms  p95 {p95:.2f} ms  ({LAT[name]['real_time_factor']:.0f}x real time)")
lookahead = sum(k // 2 for k in CFG.enc_kernels) + 1
print(f"algorithmic look-ahead: {lookahead} input samples = {1000 * lookahead / CFG.fs_lo:.2f} ms at {CFG.fs_lo} Hz")
""")

# ============================================================ 10 listen
md(r"""
## 10. Listen

One held-out utterance: the truth, naive upsampling, the deterministic arm, two draws of the sampler, its ensemble mean, and the passthrough versions. Files go to the cache under `audio/`.
""")
code(r"""
try:
    from IPython.display import Audio, display
    SHOW = True
except Exception:
    SHOW = False

OUT_AUDIO = AUDIO / TAG
OUT_AUDIO.mkdir(parents=True, exist_ok=True)
y = EVAL[0]
clips = {"truth": y, "naive upsampling": naive_upsample(y, CFG)}
det_k = next((k for k, m in models.items() if m.tau == 0), None)
sto_k = next((k for k, m in models.items() if m.tau > 0), None)
if det_k:
    clips[det_k] = reconstruct(models[det_k], y, CFG, tau=0.0)
    clips[f"{det_k} passthrough"] = passthrough(clips[det_k], y, CFG)
if sto_k:
    draws = ensemble(models[sto_k], y, CFG, CFG.m_draws, 1.0)
    clips[f"{sto_k} draw 1"], clips[f"{sto_k} draw 2"], clips[f"{sto_k} mean of {CFG.m_draws}"] = draws[0], draws[1], draws.mean(0)
    clips[f"{sto_k} draw 1 passthrough"] = passthrough(draws[0], y, CFG)
print(f"first utterance of the {EVAL_NAME.split(':')[0]}, {len(y) / CFG.fs_hi:.1f} s")
for name, w in clips.items():
    w = np.clip(w, -1, 1)
    fn = OUT_AUDIO / (name.replace(" ", "_") + ".wav")
    sf.write(fn, w, CFG.fs_hi)
    print(f"  {name:<28} SNR {snr_db(y, w):6.2f} dB   {fn.name}")
    if SHOW:
        display(Audio(w, rate=CFG.fs_hi))
""")

# ============================================================ 11 summary
md(r"""
## 11. Summary

The table again, the three sentences that read it, and a JSON of everything measured.
""")
code(r"""
print(f"{EVAL_NAME}\n" + results_table(RES) + "\n")
if sto_k:
    S, D = RES[sto_k], (RES[det_k] if det_k else None)
    M = S["M"]
    print(f"1. {sto_k}: one draw has a high-band deficit of {S['single']['deficit']:+.1f} dB, the mean of {M} draws {S['mean']['deficit']:+.1f} dB. "
          f"A mean of independent draws loses 10 log10 M = {10 * math.log10(M):.1f} dB wherever the band is unpredictable; "
          f"the difference is the energy a pointwise objective removes by taking a conditional expectation.")
    kd = f"{fmt_kappa(D['hb_kappa'])} for {det_k}, " if D else ""
    print(f"2. High-band kappa is {kd}{fmt_kappa(S['hb_kappa'])} for one draw of {sto_k}. Kappa near 0 means the energy above the cut is uncorrelated with the truth: "
          f"the deterministic model's high band is one frozen sample, and the coherent (predictable) fraction bounds any point predictor at {S['snr_bound_db']:.1f} dB SNR "
          f"(naive upsampling {NAIVE_SNR:.2f} dB).")
    gap, cal = S["snr_gap_db"], S["snr_gap_calibrated_db"]
    verdict = "under-dispersed" if gap < cal - 0.3 else ("over-dispersed" if gap > cal + 0.3 else "calibrated")
    print(f"3. SNR(mean) - SNR(one draw) = {gap:.2f} dB against {cal:.2f} dB for a calibrated sampler at M = {M}: {verdict} in waveform energy on this set.")

summary = {"preset": CFG.name, "tag": TAG, "set": EVAL_NAME, "arms": ARMS, "naive_snr": NAIVE_SNR,
           "results": RES, "tau_sweep": SWEEP, "latency": LAT, "config": dataclasses.asdict(CFG)}
out = ROOT / f"results_{TAG}.json"
json.dump(summary, open(out, "w"), indent=1, default=float)
print(f"\nwritten {out}; checkpoints in {CKPT / TAG}; figures in {FIGS}; audio in {OUT_AUDIO}")
""")


nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path(__file__).parent / "lisa_rtm_sampler.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(CELLS)} cells)")
