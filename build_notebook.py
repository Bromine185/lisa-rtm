"""Generate lisa_rtm.ipynb.

The notebook is the deliverable; this script exists so it can be regenerated
deterministically instead of hand-edited as JSON.  Run:  python3 build_notebook.py
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
# Regression to the mean in LISA — a transport-map prototype

**LISA** ([arXiv:2111.00195](https://arxiv.org/abs/2111.00195), ICASSP 2022) is an audio
super-resolution model: a conv encoder produces local latent codes from a low-rate waveform, and
a 5-layer ReLU MLP decodes an amplitude at *any* continuous time coordinate.  It is ~89k
parameters and it beats models 800× its size.

It is also **deterministic, trained on a deterministic loss, on a one-to-many problem**.  Many
high-frequency contents are consistent with a given band-limited input, so the loss minimiser is
approximately the conditional mean, and the conditional mean of a random-phase high band has
suppressed energy.  The output is muffled.  That is regression to the mean.

This notebook measures that deficit and corrects it with a ladder of **optimal transport maps**
fitted to *low-order statistics* of the high-band log-magnitude distribution:

| rung | map | statistics used |
|---|---|---|
| `T0` | identity | none — the baseline |
| `T1` | per-bin quantile map $F_t^{-1}\circ F_s$ | full 1-D marginal per bin |
| `T2` | Bures–Wasserstein $m_t + A(\ell - m_s)$ | mean + covariance |
| `T3` | conditional (energy × voicing cells) | mean + covariance, *per frame type* |
| `T4` | $T_\lambda = (1-\lambda)\,\mathrm{Id} + \lambda T$ | the continuum between any rung and identity |
| `S`  | stochastic: noise through a conditional map | the only rung that produces *samples* |

**Read this before running.**  Three things are easy to get wrong here, and the notebook is built
around them:

1. **A deterministic map on a point prediction is still a point prediction.**  `T1`–`T4` make one
   output sharper; they cannot represent $p(y\mid x)$ or be calibrated.  Section 11 adds the
   stochastic rung, which is where the generative claim actually lives.
2. **A single global map matches the *pooled* marginal**, which is a mixture over frame types — so
   it inflates silence into hiss and under-corrects fricatives.  Section 10 conditions the map.
3. **LSD is an RMSE-type score on log-magnitudes, so it structurally rewards regression to the
   log-domain mean.**  A perfect sampler scores ~2× worse in expected squared error than the
   conditional-mean predictor.  This is very likely why LISA (deterministic, 89k) reports LSD 0.81
   against WSRGlow (a flow, 229M) at 1.01 *in LISA's own table*.  **Beating a generative model on
   LSD is evidence of the disease, not health.**  So band-energy ratio measures the disease, CRPS
   measures the cure, and LSD is reported but never trusted alone.

**Order of operations.**  Sections 2–5 are pure numpy and are validated against closed-form
optimal transport *before any data is downloaded or any model is trained* (§5 is a hard gate).
§9 is the second gate: if the trained model's high-band energy deficit is small, the premise is
wrong and the honest move is to stop and write that down.

**How to run.**  Runtime → Change runtime type → GPU, then Run all.  §1 picks `FULL` automatically
when CUDA is present and `SMOKE` otherwise; override it there if you want.

Rough budget for `FULL` on a T4: a few minutes to pull the VCTK subset (once — it is cached to
Drive), ~1–2 hours for 20k training steps, then a few minutes for the ladder and the figures.  The
training cell is resumable, so a disconnect costs you only the steps since the last checkpoint.
`SMOKE` runs the entire notebook on CPU in about two minutes and proves the code path without
proving anything scientific.
""")

# ============================================================ 0 setup
md(r"""
## 0 · Setup

Do **not** `pip install torch` on Colab — the preinstalled build is matched to the CUDA driver and
replacing it silently loses the GPU.  We install only what is missing.

The numpy side is fully deterministic: every component draws from its own generator, seeded by
hashing a label, so re-running one section does not perturb another's draws.  The torch side is
deterministic on CUDA but *not* on Apple MPS, where `index_put` and `scatter_reduce` have no
deterministic kernel — so local `SMOKE` runs vary slightly between invocations.  That affects the
model, never the transport maths or the tests.
""")

code(r"""
import importlib, subprocess, sys

def ensure(pkg, import_name=None):
    try:
        importlib.import_module(import_name or pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

for _pkg, _mod in [("numpy", "numpy"), ("scipy", "scipy"), ("matplotlib", "matplotlib"),
                   ("soundfile", "soundfile"), ("requests", "requests"), ("tqdm", "tqdm")]:
    ensure(_pkg, _mod)

import os, io, json, math, time, zipfile, hashlib, warnings, dataclasses, shutil, tempfile
from pathlib import Path

import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_COLAB = "google.colab" in sys.modules
print("python  ", sys.version.split()[0])
print("torch   ", torch.__version__)
print("numpy   ", np.__version__)
print("in colab", IN_COLAB)

# ---- device -----------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("gpu     ", torch.cuda.get_device_name(0))
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print("device  ", DEVICE)

# ---- persistent cache -------------------------------------------------------
# Colab disconnects.  Anything expensive (the VCTK subset, checkpoints) goes to Drive if we can
# mount it, so a reconnect costs seconds instead of an hour.
ROOT = None
if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        ROOT = Path("/content/drive/MyDrive/lisa_rtm")
    except Exception as e:
        warnings.warn(f"Drive not mounted ({e}); caching to /content, which a disconnect will lose.")
        ROOT = Path("/content/lisa_rtm")
else:
    ROOT = Path("./lisa_rtm_cache").resolve()

FIXTURES = ROOT / "fixtures"
CKPT     = ROOT / "checkpoints"
FIGS     = ROOT / "figures"
for _d in (FIXTURES, CKPT, FIGS):
    _d.mkdir(parents=True, exist_ok=True)
print("cache   ", ROOT)

# ---- seeding ----------------------------------------------------------------
# One global seed.  Per-component streams are derived by hashing a label, so re-running one
# section does not perturb another section's draws.
SEED = 0

def stream(label, seed=None):
    '''Independent seeded numpy Generator for a named component.'''
    h = hashlib.blake2b(label.encode(), digest_size=8).digest()
    return np.random.default_rng((int.from_bytes(h, "big") ^ (SEED if seed is None else seed)) % (2**63))

def seed_everything(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

seed_everything()
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.facecolor": "white"})
print("seeded  ", SEED)
""")

# ============================================================ 1 config
md(r"""
## 1 · Config

One switch drives everything below.  `SMOKE` and `FULL` are the *same code path* with different
numbers — the point of `SMOKE` is to prove correctness in minutes, not to produce a result.
""")

code(r"""
@dataclasses.dataclass(frozen=True)
class Config:
    name: str
    fs_hi: int            # target sample rate
    upsample: int         # R:  fs_lo = fs_hi // R
    train_speakers: tuple
    test_speakers: tuple
    utts_per_speaker: int
    seg_samples: int      # training segment length at fs_hi
    # model
    enc_channels: tuple
    enc_kernels: tuple
    dec_hidden: int
    dec_layers: int
    # training
    batch_size: int
    steps: int
    lr: float
    lr_milestones: tuple  # fractions of `steps` at which lr is multiplied by lr_gamma
    lr_gamma: float
    lambda_spec: float    # weight of the multi-scale STFT term.  Official LISA code: 1e-3
    grad_clip: float
    ckpt_every: int
    # analysis
    n_fft: int            # transport / operating basis
    hop: int
    eval_n_fft: int       # evaluation basis (deliberately different from the operating basis)
    eval_hop: int
    n_eval_utts: int
    n_quantiles: int
    lambdas: tuple

    @property
    def fs_lo(self): return self.fs_hi // self.upsample
    @property
    def k_cut(self):
        '''First STFT bin above the input Nyquist, in the transport basis.'''
        return int(np.ceil((self.fs_lo / 2) * self.n_fft / self.fs_hi))
    @property
    def eval_k_cut(self):
        return int(np.ceil((self.fs_lo / 2) * self.eval_n_fft / self.fs_hi))


# Two values below come from the paper / official code (ml-postech/LISA), not from taste:
#   * lambda_spec = 1e-3.  The paper never prints lambda; the released code uses spec_coeff=0.001 and
#     its shipped config is plain L1.  The paper's own ablation shows the spectral term moves LSD by
#     <= 0.01.  An earlier run here used 1.0, which made the phase-blind term 92% of the loss and
#     produced a model with correct magnitudes, random phase, and -5.8 dB SNR (worse than silence).
#   * lr halved at milestones (official code: epochs 10,20,25,30,35,40 of 50), grad clip 1e-3 (stated).
SMOKE = Config(
    name="SMOKE", fs_hi=16000, upsample=4,
    train_speakers=("p225", "p226"), test_speakers=("p236",),
    utts_per_speaker=6, seg_samples=4096,
    enc_channels=(16, 32, 64, 32), enc_kernels=(7, 3, 3, 1),
    dec_hidden=144, dec_layers=5,
    batch_size=8, steps=400, lr=1e-3, lr_milestones=(0.2, 0.4, 0.5, 0.6, 0.7, 0.8), lr_gamma=0.5,
    lambda_spec=1e-3, grad_clip=1e-3, ckpt_every=200,
    n_fft=512, hop=128, eval_n_fft=1024, eval_hop=256,
    n_eval_utts=4, n_quantiles=513, lambdas=(0.0, 0.25, 0.5, 0.75, 1.0),
)

FULL = Config(
    name="FULL", fs_hi=48000, upsample=4,
    train_speakers=("p225", "p226", "p227", "p228", "p229",
                    "p230", "p231", "p232", "p233", "p234"),
    test_speakers=("p236", "p237", "p238"),
    utts_per_speaker=40, seg_samples=12288,
    enc_channels=(16, 32, 64, 32), enc_kernels=(7, 3, 3, 1),
    dec_hidden=144, dec_layers=5,
    batch_size=16, steps=20000, lr=1e-3, lr_milestones=(0.2, 0.4, 0.5, 0.6, 0.7, 0.8), lr_gamma=0.5,
    lambda_spec=1e-3, grad_clip=1e-3, ckpt_every=1000,
    n_fft=2048, hop=512, eval_n_fft=1024, eval_hop=256,
    n_eval_utts=12, n_quantiles=1001,
    lambdas=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# ---------------------------------------------------------------------------
PRESET = "FULL" if torch.cuda.is_available() else "SMOKE"     # <-- the one switch
# ---------------------------------------------------------------------------

CFG = {"SMOKE": SMOKE, "FULL": FULL}[PRESET]
RUN = CKPT / f"{CFG.name}_lam{CFG.lambda_spec:g}"     # objective in the path: no silent resumes
RUN.mkdir(parents=True, exist_ok=True)

print(f"preset        {CFG.name}   run dir {RUN.name}")
print(f"rates         {CFG.fs_lo} Hz -> {CFG.fs_hi} Hz  ({CFG.upsample}x)")
print(f"speakers      {len(CFG.train_speakers)} train / {len(CFG.test_speakers)} held out")
print(f"transport     n_fft={CFG.n_fft} hop={CFG.hop}  high band = bins {CFG.k_cut}..{CFG.n_fft//2}")
print(f"evaluation    n_fft={CFG.eval_n_fft} hop={CFG.eval_hop} (different basis, on purpose)")
""")

# ============================================================ 2 spectral
md(r"""
## 2 · Spectral primitives

Hann window, hop = `n_fft // 4`, weighted overlap-add with division by $\sum_i w^2$.  That makes
`istft(stft(x)) == x` on interior samples to float64 precision, which §5 asserts.

Everything downstream obeys one rule: **a metric is never computed on a modified spectrogram.**
Modified magnitudes with unchanged phase are an *inconsistent* STFT — the pair
$(|S'|, \angle S)$ is generally not the STFT of any signal — so it can score well in the spectral
domain and sound bad after overlap-add.  We always resynthesise to a waveform first and re-analyse.
""")

code(r"""
def _hann(n):
    '''Periodic Hann, float64.'''
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def stft(x, n_fft, hop):
    '''Real signal -> (n_frames, n_bins) complex.  Returns (S, pad, length) for exact inversion.'''
    x = np.asarray(x, dtype=np.float64)
    length, pad = len(x), n_fft
    xp = np.pad(x, (pad, pad + n_fft))
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(xp[idx] * _hann(n_fft), axis=-1), pad, length


def istft(S, n_fft, hop, pad, length):
    '''Inverse of stft().  Weighted overlap-add; exact on interior samples.'''
    w = _hann(n_fft)
    frames = np.fft.irfft(S, n=n_fft, axis=-1) * w
    n_frames = frames.shape[0]
    total = (n_frames - 1) * hop + n_fft
    out = np.zeros(total)
    wsq = np.zeros(total)
    for i in range(n_frames):
        s = i * hop
        out[s:s + n_fft] += frames[i]
        wsq[s:s + n_fft] += w ** 2
    out /= np.maximum(wsq, 1e-12)
    return out[pad:pad + length]


def logmag(S, eps=1e-8):
    return np.log(np.abs(S) + eps)


def resynth(S_ref, new_logmag, hi_slice, n_fft, hop, pad, length, phase=None, eps=1e-8):
    '''Replace the high band's log-magnitude, keep (or replace) phase, return a waveform.'''
    S = S_ref.copy()
    mag = np.exp(new_logmag) - eps
    mag = np.maximum(mag, 0.0)
    ph = np.angle(S[:, hi_slice]) if phase is None else phase
    S[:, hi_slice] = mag * np.exp(1j * ph)
    return istft(S, n_fft, hop, pad, length)
""")

# ============================================================ 3 transport
md(r"""
## 3 · The transport ladder

All maps act on the **high-band log-magnitude** vector $\ell_n = \ln|S[k,n]|$ for $k \ge k_c$, and
are fitted on **train speakers only**.

**`T1` — per-bin quantile map.**  For each bin $k$,
$$T_1^{(k)} = \big(F_t^{(k)}\big)^{-1} \circ F_s^{(k)}.$$
In 1-D the monotone rearrangement is the *unique* optimal Monge map for any strictly convex cost,
so this rung is exact 1-D OT.  Applied per bin it matches every marginal but transports the joint
only under the comonotone coupling — cross-bin structure (formants, harmonic combs) is untouched.

**`T2` — Bures–Wasserstein.**  Model $\ell_s\sim N(m_s,\Sigma_s)$, $\ell_t\sim N(m_t,\Sigma_t)$.
The $W_2$-optimal map is affine and closed-form:
$$T_2(\ell) = m_t + A(\ell - m_s), \qquad
A = \Sigma_s^{-1/2}\big(\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2}\big)^{1/2}\Sigma_s^{-1/2},$$
with $A$ symmetric PSD and the identity $A\,\Sigma_s A = \Sigma_t$ — which §5 uses to test the
matrix-square-root path to $10^{-10}$ without Monte Carlo.  This is literally *"approximate the
transport using lower-order statistics"*: two moments and nothing else.  Diagonal $\Sigma$ collapses
it to per-bin $\sigma_t/\sigma_s$ scaling.

**`T3` — conditional.**  The pooled marginal is a mixture $p(\ell)=\int p(\ell\mid c)\,p(c)\,dc$
over frame types.  A global map matches the mixture by *reallocating mass across frame types*:
silence sits in the source's low quantiles and gets mapped onto the target's low quantiles, which
contain quiet-but-voiced frames — so silence is inflated into audible hiss.  We condition on
covariates computed **from the input only** (low-band frame energy, spectral flatness as a voicing
proxy) with *soft* assignment.  Hard switching between maps clicks between frames.

The soft combination happens in **parameter** space, not output space.  Averaging the *outputs* of
per-cell maps is the obvious implementation and it is unstable: it evaluates every cell's map on
every frame, so the map fitted on silence gets applied to a loud frame, extrapolates with a huge
slope, and a weight of $10^{-6}$ on a value of $10^{8}$ still detonates.  Blending the parameters
gives each frame a single affine map that cannot leave the convex hull of the fitted ones.

**`T4` — the continuum.**  $T_\lambda = (1-\lambda)\,\mathrm{Id} + \lambda T$.  If $T=\nabla\varphi$
with $\varphi$ convex, then $T_\lambda = \nabla\big((1-\lambda)\|\ell\|^2/2 + \lambda\varphi\big)$
is also a gradient of a convex function, hence itself the exact OT map onto its own pushforward.
The curve $\lambda \mapsto (T_\lambda)_\#\mu_s$ is **McCann displacement interpolation**, the
constant-speed geodesic in $W_2$: $W_2(\mu_s,\mu_\lambda) = \lambda\,W_2(\mu_s,\mu_t)$.  This is the
rigorous version of *"descending in the continuous case"*, and §5 checks the geodesic identity in
closed form.
""")

code(r"""
class QuantileMap:
    '''T1: per-bin 1-D optimal transport by monotone rearrangement.'''

    def __init__(self, q_src, q_tgt):
        self.q_src = np.asarray(q_src, dtype=np.float64)   # (K, Q) non-decreasing
        self.q_tgt = np.asarray(q_tgt, dtype=np.float64)

    @staticmethod
    def _monotone(q, jitter=1e-12):
        '''np.interp needs strictly increasing x; ties are broken by a monotone epsilon ramp.'''
        q = np.maximum.accumulate(q, axis=-1)
        return q + jitter * np.arange(q.shape[-1])

    @classmethod
    def fit(cls, L_src, L_tgt, n_q=1001):
        '''L_*: (K, N) high-band log-magnitudes pooled over frames.'''
        p = np.linspace(0.0, 1.0, n_q)
        return cls(cls._monotone(np.quantile(L_src, p, axis=1).T),
                   cls._monotone(np.quantile(L_tgt, p, axis=1).T))

    @classmethod
    def from_quantiles(cls, q_src, q_tgt):
        '''Build directly from (analytic) quantile tables — used by the exactness test.'''
        return cls(np.atleast_2d(q_src), np.atleast_2d(q_tgt))

    def __call__(self, L):
        L = np.atleast_2d(L)
        out = np.empty_like(L, dtype=np.float64)
        for k in range(L.shape[0]):
            xs, ys = self.q_src[k], self.q_tgt[k]
            out[k] = np.interp(L[k], xs, ys)
            # identity-slope extrapolation outside the fitted support: a frame far off the
            # training distribution is shifted, never clamped (clamping detonates the tails).
            lo = L[k] < xs[0]
            hi = L[k] > xs[-1]
            out[k][lo] = ys[0] + (L[k][lo] - xs[0])
            out[k][hi] = ys[-1] + (L[k][hi] - xs[-1])
        return out

    def inverse(self):
        return QuantileMap(self.q_tgt, self.q_src)


def target_range(L_tgt, pad=1.0, q=0.001):
    '''Per-bin observed dynamic range of the target, padded.

    The affine (Gaussian) maps need this.  Modelling log-magnitude as Gaussian is fine near the
    middle and wrong in the tails, and the map's slope sigma_t/sigma_s is routinely 2-4 because the
    over-smoothed source really does have less spread.  Multiplying a deviation by 4 in *log*
    space turns a +10 dB peak into +40 dB and a -60 dB dip into -240 dB.  Clamping to the range the
    target actually occupies is a regulariser, it is reported, and the alternative is nonsense.
    '''
    return (np.quantile(L_tgt, q, axis=1) - pad,
            np.quantile(L_tgt, 1 - q, axis=1) + pad)


def sqrtm_psd(C):
    '''Symmetric PSD square root via eigendecomposition.'''
    w, V = np.linalg.eigh((C + C.T) / 2.0)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def invsqrtm_psd(C, floor=1e-12):
    w, V = np.linalg.eigh((C + C.T) / 2.0)
    w = np.clip(w, floor, None)
    return (V / np.sqrt(w)) @ V.T


def shrink(C, rho):
    '''Ledoit-Wolf style shrinkage toward a scaled identity.'''
    K = C.shape[0]
    return (1.0 - rho) * C + rho * (np.trace(C) / K) * np.eye(K)


class BuresMap:
    '''T2: Bures-Wasserstein (Gaussian) optimal transport.  Full, diagonal, or block-diagonal.'''

    def __init__(self, m_src, m_tgt, A, blocks=None, clamp=None):
        self.m_src, self.m_tgt, self.A, self.blocks = m_src, m_tgt, A, blocks
        self.clamp = clamp

    @staticmethod
    def _A(C_s, C_t):
        Ch = sqrtm_psd(C_s)
        Ci = invsqrtm_psd(C_s)
        A = Ci @ sqrtm_psd(Ch @ C_t @ Ch) @ Ci
        return (A + A.T) / 2.0

    @classmethod
    def fit(cls, L_src, L_tgt, mode="diag", rho=0.1, n_blocks=12, ratio_clip=8.0):
        '''mode in {diag, full, block}.

        ratio_clip bounds sigma_t/sigma_s.  It binds where the model output is nearly silent, so
        sigma_s -> 0 and the unclipped ratio is enormous: the map would then amplify whatever
        numerical residue sits in that bin by a factor of thousands.  The fraction of clipped bins
        is reported, because this is a regulariser and it changes the result.
        '''
        m_s, m_t = L_src.mean(1), L_tgt.mean(1)
        K = L_src.shape[0]
        rng_t = target_range(L_tgt)
        if mode == "diag":
            raw = L_tgt.std(1) / np.maximum(L_src.std(1), 1e-12)
            cls.last_clipped = float(np.mean((raw > ratio_clip) | (raw < 1.0 / ratio_clip)))
            return cls(m_s, m_t, np.clip(raw, 1.0 / ratio_clip, ratio_clip), clamp=rng_t)
        C_s = shrink(np.cov(L_src), rho)
        C_t = shrink(np.cov(L_tgt), rho)
        if mode == "full":
            return cls(m_s, m_t, cls._A(C_s, C_t), clamp=rng_t)
        edges = np.unique(np.linspace(0, K, n_blocks + 1).astype(int))
        blocks = [slice(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        return cls(m_s, m_t, [cls._A(C_s[b, b], C_t[b, b]) for b in blocks], blocks, clamp=rng_t)

    @classmethod
    def from_moments(cls, m_s, C_s, m_t, C_t):
        return cls(m_s, m_t, cls._A(C_s, C_t))

    def __call__(self, L):
        d = L - self.m_src[:, None]
        if self.blocks is not None:
            out = np.empty_like(d)
            for A, b in zip(self.A, self.blocks):
                out[b] = A @ d[b]
        elif np.ndim(self.A) == 1:
            out = self.A[:, None] * d
        else:
            out = self.A @ d
        out = out + self.m_tgt[:, None]
        if self.clamp is not None:
            out = np.clip(out, self.clamp[0][:, None], self.clamp[1][:, None])
        return out


def bures_w2(m_s, C_s, m_t, C_t):
    '''Closed-form squared W2 between Gaussians (the geodesic test uses this).'''
    Ch = sqrtm_psd(C_s)
    cross = np.trace(sqrtm_psd(Ch @ C_t @ Ch))
    return float(np.sum((m_s - m_t) ** 2) + np.trace(C_s) + np.trace(C_t) - 2 * cross)


class Interpolated:
    '''T4: McCann displacement interpolation  T_lambda = (1-l) Id + l T.'''

    def __init__(self, T, lam):
        self.T, self.lam = T, lam

    def __call__(self, L):
        if self.lam == 0.0:
            return np.array(L, dtype=np.float64, copy=True)
        if self.lam == 1.0:
            return self.T(L)
        return (1.0 - self.lam) * L + self.lam * self.T(L)


class Identity:
    def __call__(self, L):
        return np.array(L, dtype=np.float64, copy=True)


def design(c, mu, sd):
    '''(N, 1+D) design matrix from standardised covariates.'''
    return np.vstack([np.ones(c.shape[1]), (c - mu[:, None]) / sd[:, None]]).T


class ConditionalMap:
    '''T3: a location-scale map whose location varies continuously with the frame covariates.

    Per bin, source and target means are affine in c and the scale is the ratio of *residual*
    spreads:
        T(l, c) = m_t(c) + a . (l - m_s(c)),   m(c) = B'[1, c],   a = sd(r_t) / sd(r_s).

    Discrete cells were tried first and abandoned, twice.  Averaging per-cell *outputs* detonates:
    every cell's map is evaluated on every frame, so the map fitted on silence extrapolates a loud
    frame with a huge slope, and a weight of 1e-6 on a value of 1e8 still ruins the frame (this
    drove SNR to -77 dB).  Averaging per-cell *parameters* is bounded but still biased: a frame
    whose energy falls between two cell centres gets a blended m_s that matches neither, and that
    offset is then multiplied by a (SNR -12 dB, +10 dB of excess energy).  The affine model has no
    cells to fall between, costs 3 coefficients per bin per side, and targets the failure mode
    directly -- the deficit is level-dependent, so the correction must be level-dependent too.
    '''

    def __init__(self, B_src, B_tgt, a, mu, sd, clamp=None):
        self.B_src, self.B_tgt, self.a = B_src, B_tgt, a
        self.mu, self.sd, self.clamp = mu, sd, clamp

    @classmethod
    def fit(cls, L_src, L_tgt, c_src, c_tgt, ratio_clip=4.0):
        '''c_*: (D, N) covariates, frame-aligned with L_*.'''
        mu, sd = c_src.mean(1), np.maximum(c_src.std(1), 1e-9)
        Xs, Xt = design(c_src, mu, sd), design(c_tgt, mu, sd)
        B_s = np.linalg.lstsq(Xs, L_src.T, rcond=None)[0]        # (1+D, K)
        B_t = np.linalg.lstsq(Xt, L_tgt.T, rcond=None)[0]
        r_s = L_src - (Xs @ B_s).T
        r_t = L_tgt - (Xt @ B_t).T
        a = np.clip(r_t.std(1) / np.maximum(r_s.std(1), 1e-12), 1.0 / ratio_clip, ratio_clip)
        return cls(B_s, B_t, a, mu, sd, clamp=target_range(L_tgt))

    def __call__(self, L, c):
        X = design(c, self.mu, self.sd)
        out = (X @ self.B_tgt).T + self.a[:, None] * (L - (X @ self.B_src).T)
        if self.clamp is not None:
            out = np.clip(out, self.clamp[0][:, None], self.clamp[1][:, None])
        return out


class ConditionalPower:
    '''Target high-band power as a function of the same covariates: the stochastic rung's sigma^2.

    Clamped to the observed range of log power for the same reason the affine maps are: a linear
    predictor evaluated on a silent frame whose covariates sit outside the fitted range is
    exponentiated, so a modest extrapolation in log space becomes an enormous power and the
    sampler fills the silences with noise.
    '''

    def __init__(self, B, mu, sd, clamp):
        self.B, self.mu, self.sd, self.clamp = B, mu, sd, clamp

    @classmethod
    def fit(cls, log_power, c, mu, sd):
        # pad=0: the clamp is a ceiling on power, and a 1-nat pad is +4.3 dB of headroom that
        # silent frames saturate against, which showed up as +4.8 dB of excess band energy.
        return cls(np.linalg.lstsq(design(c, mu, sd), log_power.T, rcond=None)[0], mu, sd,
                   target_range(log_power, pad=0.0))

    def __call__(self, c):
        p = design(c, self.mu, self.sd) @ self.B                 # (N, K)
        return np.exp(np.clip(p, self.clamp[0][None, :], self.clamp[1][None, :]))


class Conditioned:
    '''Freeze a ConditionalMap against one utterance's covariates so it has the T(L) signature.'''

    def __init__(self, cmap, c):
        self.cmap, self.c = cmap, c

    def __call__(self, L):
        return self.cmap(L, self.c)
""")

# ============================================================ 4 metrics
md(r"""
## 4 · Metrics

- **SNR** $=10\log_{10}\|y\|^2/\|y-\hat y\|^2$.  Phase-sensitive, and it *must* fall when we add the
  missing conditional variance: with $\hat y\approx E[y|x]$ the correction is roughly orthogonal to
  the residual, so $E\|T_\lambda(\hat y)-y\|^2 \approx E\|\hat y - y\|^2 + \lambda^2 E\|\Delta\|^2$.
  A correction that fixes the marginals at *zero* SNR cost is a bug, not a result.
- **LSD** with $X = \log_{10}|S|^2$:
  $\mathrm{LSD} = \frac{1}{N_f}\sum_n \sqrt{\frac{1}{K}\sum_k (X-\hat X)^2}$.  **HB-LSD** restricts
  $k \ge k_c$; full-band LSD is diluted by the (already correct) low band.
- **Band energy ratio** $\rho_b$ in dB per third-octave — the direct over-smoothing measure, and the
  one metric that is not confounded by the log-mean argument in the header.
- **CRPS** for the stochastic rung, $\mathrm{CRPS} \approx \frac1M\sum_m|\hat y_m - y|
  - \frac{1}{2M^2}\sum_{m,m'}|\hat y_m - \hat y_{m'}|$, plus **PIT rank histograms** (flat iff
  calibrated) and **spread–skill**.
""")

code(r"""
def snr_db(y, y_hat):
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    n = min(len(y), len(y_hat))
    e = y[:n] - y_hat[:n]
    return 10.0 * np.log10(np.sum(y[:n] ** 2) / max(np.sum(e ** 2), 1e-20))


def lsd_db(y, y_hat, n_fft, hop, k_from=0, eps=1e-10):
    '''Log-spectral distance on power spectra (log10), optionally restricted to k >= k_from.'''
    Sy = np.abs(stft(y, n_fft, hop)[0])[:, k_from:]
    Sh = np.abs(stft(y_hat, n_fft, hop)[0])[:, k_from:]
    d = np.log10(Sy ** 2 + eps) - np.log10(Sh ** 2 + eps)
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=1))))


def third_octave_edges(fs, f_lo, f_hi):
    f = [f_lo]
    while f[-1] < f_hi:
        f.append(f[-1] * 2 ** (1 / 3))
    return np.array(f)


def band_energy_ratio(y, y_hat, fs, n_fft, hop, f_lo, f_hi, frame_mask=None):
    '''rho_b in dB per third-octave band.  0 dB = correct energy, negative = over-smoothed.'''
    Sy, Sh = np.abs(stft(y, n_fft, hop)[0]), np.abs(stft(y_hat, n_fft, hop)[0])
    if frame_mask is not None:
        Sy, Sh = Sy[frame_mask], Sh[frame_mask]
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    edges = third_octave_edges(fs, f_lo, min(f_hi, fs / 2 - 1))
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (freqs >= a) & (freqs < b)
        if sel.sum() == 0:
            continue
        num, den = np.sum(Sh[:, sel] ** 2), np.sum(Sy[:, sel] ** 2)
        out.append((np.sqrt(a * b), 10.0 * np.log10((num + 1e-20) / (den + 1e-20))))
    return np.array(out)          # (n_bands, 2): centre freq, dB


def crps_ensemble(samples, truth):
    '''samples: (M, ...) ensemble;  truth: (...).  Fair estimator, averaged over all elements.'''
    M = samples.shape[0]
    term1 = np.abs(samples - truth[None]).mean(0)
    diff = np.abs(samples[:, None] - samples[None, :]).mean((0, 1))
    return float(np.mean(term1 - 0.5 * diff))


def pit_ranks(samples, truth):
    '''Rank of truth among M samples, in 0..M.  Flat histogram <=> calibrated.'''
    return (samples < truth[None]).sum(0).ravel()


def spread_skill(samples, truth, n_bins=8):
    '''Binned (mean ensemble spread, rmse of ensemble mean).  Unit slope is the target.'''
    spread = samples.std(0).ravel()
    err = (samples.mean(0) - truth).ravel()
    order = np.argsort(spread)
    spread, err = spread[order], err[order]
    edges = np.linspace(0, len(spread), n_bins + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            out.append((spread[a:b].mean(), np.sqrt((err[a:b] ** 2).mean())))
    return np.array(out)
""")

# ============================================================ 5 gate 1
md(r"""
## 5 · Gate 1 — closed-form validation

No model, no audio, no download.  The high band of the synthetic signal is
$Y[k,n] = \sigma_k\,\varepsilon[k,n]$ with $\varepsilon\sim \mathcal{CN}(0,1)$, so $|Y|$ is Rayleigh
with $F(r) = 1-e^{-r^2/\sigma^2}$ and quantile $\sigma\sqrt{-\ln(1-p)}$.  An over-smoothed "model"
is the same field scaled by $\beta<1$.  Scaling a Rayleigh shifts its log by a constant, so the
**exact** optimal transport is known in closed form:
$$T^\*(v) = v - \ln\beta.$$

That gives an exactness test the implementation either passes to machine precision or fails.  The
Bures rung gets the algebraic identity $A\Sigma_s A = \Sigma_t$, which pins the matrix square root
without any sampling.  If this section does not print `ALL PASS`, nothing below is worth reading.
""")

code(r"""
TESTS = []
def test(fn):
    TESTS.append(fn); return fn


@test
def t_stft_roundtrip():
    '''istft(stft(x)) == x on interior samples, float64.'''
    rng = stream("test/stft")
    for n_fft, hop in [(512, 128), (1024, 256), (2048, 512)]:
        x = rng.standard_normal(9000)
        S, pad, L = stft(x, n_fft, hop)
        err = np.max(np.abs(istft(S, n_fft, hop, pad, L) - x))
        assert err < 1e-12, f"n_fft={n_fft}: round-trip error {err:.2e}"


@test
def t_quantile_map_is_exact_shift():
    '''T1 built from analytic log-Rayleigh quantiles must reproduce v - ln(beta) exactly.'''
    beta, sigma = 0.37, 1.9
    p = np.linspace(1e-6, 1 - 1e-6, 4001)
    q_hi = np.log(sigma * np.sqrt(-np.log1p(-p)))          # ln|Y| quantiles, scale sigma
    q_lo = q_hi + np.log(beta)                             # the over-smoothed model
    T = QuantileMap.from_quantiles(q_lo, q_hi)
    v = np.linspace(q_lo[10], q_lo[-10], 500)[None, :]
    err = np.max(np.abs(T(v) - (v - np.log(beta))))
    assert err < 1e-12, f"analytic quantile map error {err:.2e}"


@test
def t_quantile_map_converges():
    '''With empirical CDFs the sup-error must shrink as O(n^-1/2).'''
    beta, sigma = 0.5, 1.0
    errs = []
    for n in (2000, 8000, 32000):
        rng = stream(f"test/qm/{n}")
        hi = np.log(np.abs(rng.standard_normal(n) + 1j * rng.standard_normal(n)) * sigma / np.sqrt(2))
        lo = np.log(np.abs(rng.standard_normal(n) + 1j * rng.standard_normal(n)) * sigma * beta / np.sqrt(2))
        T = QuantileMap.fit(lo[None], hi[None], n_q=501)
        v = np.linspace(np.quantile(lo, 0.05), np.quantile(lo, 0.95), 400)[None, :]
        errs.append(np.max(np.abs(T(v) - (v - np.log(beta)))))
    assert errs[-1] < errs[0], f"error did not shrink with n: {errs}"
    assert errs[-1] < 0.15, f"empirical map too inaccurate: {errs[-1]:.3f}"


@test
def t_quantile_inverse():
    rng = stream("test/qm-inv")
    L_s, L_t = rng.standard_normal((6, 4000)), 1.7 * rng.standard_normal((6, 4000)) + 0.4
    T = QuantileMap.fit(L_s, L_t, n_q=801)
    v = L_s[:, :500]
    err = np.max(np.abs(T.inverse()(T(v)) - v))
    assert err < 1e-9, f"T^-1(T(v)) error {err:.2e}"


@test
def t_bures_identity():
    '''A Sigma_s A = Sigma_t, and A symmetric.  Pins the matrix square root exactly.'''
    rng = stream("test/bures")
    for K in (5, 20, 60):
        Qs, Qt = rng.standard_normal((K, K)), rng.standard_normal((K, K))
        C_s = Qs @ Qs.T + K * np.eye(K)
        C_t = Qt @ Qt.T + K * np.eye(K)
        A = BuresMap._A(C_s, C_t)
        rel = np.linalg.norm(A @ C_s @ A - C_t, "fro") / np.linalg.norm(C_t, "fro")
        assert rel < 1e-10, f"K={K}: ||A Cs A - Ct|| / ||Ct|| = {rel:.2e}"
        assert np.max(np.abs(A - A.T)) < 1e-12


@test
def t_bures_diag_matches_full():
    '''On diagonal covariances the full map must collapse to per-bin sigma_t/sigma_s.'''
    rng = stream("test/bures-diag")
    s_s, s_t = rng.uniform(0.5, 2.0, 12), rng.uniform(0.5, 2.0, 12)
    A = BuresMap._A(np.diag(s_s ** 2), np.diag(s_t ** 2))
    assert np.max(np.abs(np.diag(A) - s_t / s_s)) < 1e-10
    assert np.max(np.abs(A - np.diag(np.diag(A)))) < 1e-10


@test
def t_bures_pushforward():
    '''Pushing samples through T2 must reproduce the target moments.'''
    rng = stream("test/bures-push")
    K, n = 8, 200000
    Q = rng.standard_normal((K, K)); C_s = Q @ Q.T + np.eye(K)
    Q = rng.standard_normal((K, K)); C_t = Q @ Q.T + np.eye(K)
    m_s, m_t = rng.standard_normal(K), rng.standard_normal(K)
    X = (np.linalg.cholesky(C_s) @ rng.standard_normal((K, n))) + m_s[:, None]
    Y = BuresMap.from_moments(m_s, C_s, m_t, C_t)(X)
    assert np.max(np.abs(Y.mean(1) - m_t)) < 5 / np.sqrt(n) * np.sqrt(np.trace(C_t))
    rel = np.linalg.norm(np.cov(Y) - C_t, "fro") / np.linalg.norm(C_t, "fro")
    assert rel < 0.02, f"pushforward covariance off by {rel:.3f}"


@test
def t_mccann_endpoints_and_geodesic():
    '''T_0 = Id and T_1 = T exactly; and W2(mu_s, mu_lambda) = lambda W2(mu_s, mu_t).'''
    rng = stream("test/mccann")
    K = 10
    Q = rng.standard_normal((K, K)); C_s = Q @ Q.T + np.eye(K)
    Q = rng.standard_normal((K, K)); C_t = Q @ Q.T + np.eye(K)
    m_s, m_t = np.zeros(K), rng.standard_normal(K)
    T = BuresMap.from_moments(m_s, C_s, m_t, C_t)
    L = rng.standard_normal((K, 50))
    assert np.array_equal(Interpolated(T, 0.0)(L), L)
    assert np.max(np.abs(Interpolated(T, 1.0)(L) - T(L))) == 0.0
    total = np.sqrt(bures_w2(m_s, C_s, m_t, C_t))
    for lam in (0.25, 0.5, 0.75):
        A = (1 - lam) * np.eye(K) + lam * T.A
        m_l, C_l = (1 - lam) * m_s + lam * m_t, A @ C_s @ A
        got = np.sqrt(bures_w2(m_s, C_s, m_l, C_l))
        assert abs(got - lam * total) < 1e-8 * max(total, 1.0), \
            f"lambda={lam}: geodesic {got:.6f} != {lam * total:.6f}"


@test
def t_resynth_identity():
    '''The full modify-and-resynthesise path with an unchanged magnitude is the identity.'''
    rng = stream("test/resynth")
    x = rng.standard_normal(8000)
    n_fft, hop, kc = 512, 128, 64
    S, pad, L = stft(x, n_fft, hop)
    hi = slice(kc, None)
    y = resynth(S, logmag(S[:, hi]), hi, n_fft, hop, pad, L)
    assert np.max(np.abs(y - x)) < 1e-9, f"resynth identity error {np.max(np.abs(y - x)):.2e}"


@test
def t_affine_maps_stay_in_range():
    '''The Gaussian rungs must never leave the target's observed dynamic range.

    Regression test for a real bug: fitted on an over-smoothed source, sigma_t/sigma_s is large,
    and an unclamped affine map in log space sent a loud frame to +38 dB of excess energy and SNR
    to -77 dB.
    '''
    rng = stream("test/clamp")
    K, n = 12, 5000
    L_t = rng.standard_normal((K, n)) * 2.0
    L_s = rng.standard_normal((K, n)) * 0.05           # over-smoothed: almost no spread
    lo, hi = target_range(L_t)
    wild = np.concatenate([L_s, L_s * 50, L_s - 20], axis=1)
    for T in (BuresMap.fit(L_s, L_t, mode="diag"), BuresMap.fit(L_s, L_t, mode="block", n_blocks=3)):
        out = T(wild)
        assert np.all(out >= lo[:, None] - 1e-9), "affine map fell below the target range"
        assert np.all(out <= hi[:, None] + 1e-9), "affine map exceeded the target range"
    c = np.stack([rng.standard_normal(n), rng.uniform(0, 1, n)])
    C = ConditionalMap.fit(L_s, L_t, c, c)
    out = C(L_s * 50, c)
    assert np.all(out >= lo[:, None] - 1e-9) and np.all(out <= hi[:, None] + 1e-9)


@test
def t_metrics_sanity():
    rng = stream("test/metrics")
    x = rng.standard_normal(6000)
    assert snr_db(x, x) > 100
    assert lsd_db(x, x, 512, 128) < 1e-9
    b = band_energy_ratio(x, x, 16000, 512, 128, 500, 7000)
    assert np.max(np.abs(b[:, 1])) < 1e-9
    truth = np.zeros(400)
    tight = rng.standard_normal((64, 400)) * 0.1
    loose = rng.standard_normal((64, 400)) * 3.0
    assert crps_ensemble(tight, truth) < crps_ensemble(loose, truth)


@test
def t_determinism():
    assert np.array_equal(stream("abc").standard_normal(5), stream("abc").standard_normal(5))
    assert not np.array_equal(stream("abc").standard_normal(5), stream("abd").standard_normal(5))


def run_tests():
    failed = []
    for fn in TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed.append(fn.__name__)
            print(f"  FAIL  {fn.__name__}: {e}")
    print()
    if failed:
        raise AssertionError(f"{len(failed)} test(s) failed: {failed}")
    print(f"ALL PASS ({len(TESTS)} tests)")

run_tests()
""")

# ============================================================ 6 data
md(r"""
## 6 · Data — a VCTK subset, fetched once

The paper trains on CSTR VCTK 0.92 at 48 kHz.  The archive is ~11 GB and we need ~13 speakers, so
the loader opens the remote zip over **HTTP range requests** and extracts only the members it wants
— the zip central directory lives at the end of the file, so `zipfile` can seek to it and pull
individual FLACs without downloading the rest.  If the server refuses ranges we fall back to a full
streaming download.

The trimmed subset (~50 MB) is written to the Drive cache, so this cell is slow exactly once.
Nothing downstream touches the network.
""")

code(r"""
VCTK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"


class HTTPRangeFile(io.RawIOBase):
    '''Minimal seekable read-only file over HTTP range requests.'''

    def __init__(self, url, session=None):
        import requests
        self.session = session or requests.Session()
        r = self.session.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        if r.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise OSError("server does not advertise byte ranges")
        self.url, self.length, self.pos = r.url, int(r.headers["Content-Length"]), 0

    def readable(self):  return True
    def seekable(self):  return True
    def tell(self):      return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset,
                    io.SEEK_CUR: self.pos + offset,
                    io.SEEK_END: self.length + offset}[whence]
        return self.pos

    def read(self, size=-1):
        if size < 0:
            size = self.length - self.pos
        size = min(size, self.length - self.pos)
        if size <= 0:
            return b""
        end = self.pos + size - 1
        r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=180)
        r.raise_for_status()
        self.pos += len(r.content)
        return r.content


def _members_for(zf, speakers, per_speaker):
    wanted = {}
    for name in zf.namelist():
        if not (name.endswith("_mic1.flac") and "wav48_silence_trimmed/" in name):
            continue
        spk = name.split("/")[-2]
        if spk in speakers:
            wanted.setdefault(spk, []).append(name)
    return {s: sorted(v)[:per_speaker] for s, v in wanted.items()}


def fetch_vctk(speakers, per_speaker, dest):
    '''Extract selected speakers' FLACs into dest/<spk>/.  Returns {speaker: [paths]}.'''
    from tqdm.auto import tqdm
    dest.mkdir(parents=True, exist_ok=True)
    have = {s: sorted((dest / s).glob("*.flac")) for s in speakers}
    if all(len(v) >= per_speaker for v in have.values()):
        return {s: v[:per_speaker] for s, v in have.items()}

    try:
        handle = io.BufferedReader(HTTPRangeFile(VCTK_URL), buffer_size=1 << 20)
        print("opening remote archive over HTTP ranges (no full download)")
    except Exception as e:
        print(f"range requests unavailable ({e}); downloading the full archive once")
        import requests
        local = ROOT / "VCTK-Corpus-0.92.zip"
        if not local.exists():
            with requests.get(VCTK_URL, stream=True, timeout=300) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                with open(local, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True) as bar:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk); bar.update(len(chunk))
        handle = open(local, "rb")

    out = {}
    with zipfile.ZipFile(handle) as zf:
        members = _members_for(zf, set(speakers), per_speaker)
        missing = set(speakers) - set(members)
        if missing:
            raise RuntimeError(f"speakers not found in archive: {sorted(missing)}")
        for spk, names in members.items():
            (dest / spk).mkdir(exist_ok=True)
            paths = []
            for name in tqdm(names, desc=spk, leave=False):
                p = dest / spk / Path(name).name
                if not p.exists():
                    p.write_bytes(zf.read(name))
                paths.append(p)
            out[spk] = paths
    return out


def load_utterance(path, fs_hi, peak=0.95, min_seconds=1.0):
    '''Read a FLAC, mono, resample to fs_hi, peak-normalise.  None if too short.'''
    x, fs = sf.read(str(path), dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if fs != fs_hi:
        g = math.gcd(int(fs), int(fs_hi))
        x = sps.resample_poly(x, fs_hi // g, fs // g)
    if len(x) < min_seconds * fs_hi:
        return None
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else None


def decimate(x, R):
    '''Anti-aliased downsample by R -- this is what makes the high band unrecoverable.'''
    return sps.resample_poly(x, 1, R)


""")

code(r"""
def build_split(speakers, cfg, tag):
    cache = FIXTURES / f"{tag}_{cfg.name}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        print(f"{tag}: {len(d['utts'])} utterances (cached)")
        return list(d["utts"]), list(d["speakers"])
    files = fetch_vctk(speakers, cfg.utts_per_speaker, FIXTURES / "vctk")
    utts, owners = [], []
    for spk in speakers:
        for p in files[spk]:
            y = load_utterance(p, cfg.fs_hi)
            if y is not None:
                utts.append(y.astype(np.float32)); owners.append(spk)
    np.savez_compressed(cache, utts=np.array(utts, dtype=object),
                        speakers=np.array(owners))
    print(f"{tag}: {len(utts)} utterances from {len(speakers)} speakers -> {cache.name}")
    return utts, owners


train_utts, train_spk = build_split(CFG.train_speakers, CFG, "train")
test_utts,  test_spk  = build_split(CFG.test_speakers,  CFG, "test")
print(f"total audio: train {sum(map(len, train_utts)) / CFG.fs_hi:.1f} s, "
      f"held-out {sum(map(len, test_utts)) / CFG.fs_hi:.1f} s")
""")

# ============================================================ 7 model
md(r"""
## 7 · LISA

Faithful to the paper.  The encoder is 4 layers of 1-D convolution with kernels $\{7,3,3,1\}$ and
channels $\{16,32,64,32\}$; the half-widths sum to $3+1+1+0 = 5 = k$, so each latent
$z_i = g_\phi(F(t_{i-k}),\dots,F(t_{i+k}))$ sees $2k+1 = 11$ input samples.  The decoder is a
5-layer ReLU MLP:
$$\hat F(t) = f_\theta\big(t - t_{i(t)};\; z(t)\big), \qquad
z(t) = (z_{i(t)-1},\, z_{i(t)},\, z_{i(t)+1}).$$

Because $R$ is an integer, the coordinate arithmetic is exact: output sample $j$ has
$t\cdot f_{lo} = j/R$, so $i = \lfloor j/R \rfloor$ and the relative coordinate is $(j \bmod R)/R$,
rescaled to $[-1,1)$.

**Perturbed prediction.**  During training the anchor index is $\tilde\imath(t) = i(t+\eta)$ with
$\eta\sim N(0,\delta^2)$ and $\delta$ = half an input sample, so $\delta f_{lo} = 0.5$.  The relative
coordinate is then measured from the *perturbed* anchor and can fall outside $[0,1)$ — which is the
point: each latent is forced to predict beyond its own cell.

**Loss.** $\mathcal{L} = \|x-\hat x\|_1 + \lambda\,\mathcal{L}_{spec}$, with $\mathcal{L}_{spec}$ the
multi-scale STFT loss (spectral convergence + log-magnitude L1) at three resolutions.  Note this
loss is *itself* a partial fix for over-smoothing — which is exactly why §9 measures the remaining
headroom before we build anything on top.
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
    def __init__(self, cfg):
        super().__init__()
        self.enc = LISAEncoder(cfg)
        self.dec = LISADecoder(self.enc.dim, cfg)
        self.R = cfg.upsample

    def forward(self, x_lo, j0=0, j1=None, perturb=False):
        B, L_lo = x_lo.shape
        j1 = L_lo * self.R if j1 is None else j1
        z = self.enc(x_lo).transpose(1, 2)                      # (B, L_lo, C)
        q = (torch.arange(j0, j1, device=x_lo.device, dtype=torch.float32) / self.R)
        q = q.unsqueeze(0).expand(B, -1)                        # t * fs_lo
        anchor = q + torch.randn_like(q) * 0.5 if perturb else q
        idx = torch.floor(anchor).long().clamp(0, L_lo - 1)
        coord = (2.0 * (q - idx.float()) - 1.0).unsqueeze(-1)

        def take(ii):
            ii = ii.clamp(0, L_lo - 1).unsqueeze(-1).expand(-1, -1, z.shape[-1])
            return torch.gather(z, 1, ii)

        return self.dec(torch.cat([coord, take(idx - 1), take(idx), take(idx + 1)], -1))


class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, n_fft):
        super().__init__()
        self.scales = [(n_fft, n_fft // 4), (n_fft // 2, n_fft // 8), (n_fft // 4, n_fft // 16)]

    def forward(self, y, y_hat):
        total = 0.0
        for n, h in self.scales:
            w = torch.hann_window(n, device=y.device)
            Y = torch.stft(y, n, h, window=w, return_complex=True).abs()
            H = torch.stft(y_hat, n, h, window=w, return_complex=True).abs()
            sc = torch.norm(Y - H, p="fro") / (torch.norm(Y, p="fro") + 1e-8)
            lm = F.l1_loss(torch.log(H + 1e-7), torch.log(Y + 1e-7))
            total = total + sc + lm
        return total / len(self.scales)


model = LISA(CFG).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"LISA: {n_params:,} parameters  (paper reports ~89k)")
print(f"  encoder {sum(p.numel() for p in model.enc.parameters()):,}"
      f"   decoder {sum(p.numel() for p in model.dec.parameters()):,}")
""")

# ============================================================ 8 training
md(r"""
## 8 · Training

Resumable, because Colab disconnects and a training cell you cannot resume is a training cell you
will run three times.  Checkpoints go to the Drive cache every `ckpt_every` steps; re-running this
cell picks up where it stopped.

The loop logs two things on a held-out utterance at every checkpoint, and both matter:

* the **high-band energy deficit** — §9's headroom gate can be faked by an undertrained model, so we
  need the deficit as a function of training step and read the asymptote, not the current value;
* the **waveform SNR**, against naive polyphase upsampling of the same input.  A band-energy ratio
  only says the *energy* per band is right.  A model can pass it with random phase, and one did:
  at `lambda_spec=1.0` the run reached −11 dB deficit and −5.8 dB SNR — worse than emitting
  silence.  SNR is the metric that cannot be fooled that way, and it is the one the paper reports
  (24.16 dB at 12 kHz → 48 kHz).

Resuming is guarded: a checkpoint trained under a different training config raises instead of
silently continuing.
""")

code(r"""
def sample_batch(utts, cfg, rng):
    ys = []
    for _ in range(cfg.batch_size):
        u = utts[rng.integers(len(utts))]
        if len(u) <= cfg.seg_samples:
            u = np.pad(u, (0, cfg.seg_samples - len(u)))
            ys.append(u)
        else:
            s = rng.integers(len(u) - cfg.seg_samples)
            ys.append(u[s:s + cfg.seg_samples])
    y = np.stack(ys).astype(np.float64)
    x = np.stack([decimate(s, cfg.upsample) for s in y])
    return (torch.from_numpy(x).float().to(DEVICE),
            torch.from_numpy(y).float().to(DEVICE))


@torch.no_grad()
def reconstruct(model, y, cfg, chunk=1 << 15):
    '''Full-utterance inference, chunked over query coordinates.'''
    model.eval()
    x_lo = torch.from_numpy(decimate(y, cfg.upsample)).float()[None].to(DEVICE)
    n_out = x_lo.shape[1] * cfg.upsample
    out = [model(x_lo, j0=s, j1=min(s + chunk, n_out)).squeeze(0).cpu().numpy()
           for s in range(0, n_out, chunk)]
    y_hat = np.concatenate(out).astype(np.float64)
    return np.pad(y_hat, (0, max(0, len(y) - len(y_hat))))[:len(y)]


def hb_deficit(model, y, cfg):
    '''Mean high-band energy ratio in dB.  Negative = over-smoothed.'''
    b = band_energy_ratio(y, reconstruct(model, y, cfg), cfg.fs_hi,
                          cfg.eval_n_fft, cfg.eval_hop, cfg.fs_lo / 2, cfg.fs_hi / 2)
    return float(np.mean(b[:, 1])) if len(b) else float("nan")


def naive_upsample(y, cfg):
    '''Polyphase (sinc-windowed) interpolation of the decimated input: the trivial baseline.'''
    y = np.asarray(y, np.float64)
    return sps.resample_poly(decimate(y, cfg.upsample), cfg.upsample, 1)[:len(y)]


def dev_snr(model, y, cfg):
    '''Waveform SNR of the reconstruction and of naive upsampling, both against y.'''
    y = np.asarray(y, np.float64)
    return snr_db(y, reconstruct(model, y, cfg)), snr_db(y, naive_upsample(y, cfg))


def save_ckpt(state, path, retries=4):
    '''Write locally, then copy to `path` with retries.  Colab's Drive mount intermittently claims a
    directory it wrote to seconds ago does not exist (seen at ~20 s write cadence on an A100).  A
    training run must not die of that; if Drive stays unreachable the file is kept in /tmp.'''
    tmp = Path(tempfile.gettempdir()) / f"{path.stem}_{os.getpid()}.pt"
    torch.save(state, tmp)
    err = None
    for attempt in range(retries):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tmp, path)
            return True
        except (OSError, RuntimeError) as e:
            err = e
            time.sleep(3 * (attempt + 1))
    warnings.warn(f"step {state['step']}: checkpoint not written to {path} ({err}); kept at {tmp}")
    return False


CKPT_PATH = RUN / "lisa.pt"
spec_loss = MultiScaleSTFTLoss(CFG.n_fft).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=CFG.lr)
sched = torch.optim.lr_scheduler.MultiStepLR(
    opt, milestones=[int(f * CFG.steps) for f in CFG.lr_milestones], gamma=CFG.lr_gamma)
history = {"step": [], "loss": [], "wave": [], "spec": [], "lr": [],
           "dev_step": [], "deficit": [], "snr": [], "snr_naive": []}
start_step = 0

# Fields that define the optimisation problem.  A checkpoint that disagrees on any of them is a
# different experiment and must not be resumed into this one.
TRAIN_KEYS = ("fs_hi", "upsample", "train_speakers", "utts_per_speaker", "seg_samples",
              "enc_channels", "enc_kernels", "dec_hidden", "dec_layers", "batch_size",
              "lr", "lr_milestones", "lr_gamma", "lambda_spec", "grad_clip")

def _norm(v):
    return tuple(v) if isinstance(v, list) else v

if CKPT_PATH.exists():
    ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    saved, now = ck.get("cfg", {}), dataclasses.asdict(CFG)
    diff = {k: (saved.get(k), now[k]) for k in TRAIN_KEYS if _norm(saved.get(k)) != _norm(now[k])}
    if diff:
        raise RuntimeError(f"{CKPT_PATH} was trained under a different config: {diff}\n"
                           f"Move or delete it, or change RUN, before training.")
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    if "sched" in ck:
        sched.load_state_dict(ck["sched"])
    history, start_step = ck["history"], ck["step"]
    print(f"resumed from step {start_step}")

rng = stream("train/batches")
probe = test_utts[0]
t0 = time.time()
model.train()
for step in range(start_step, CFG.steps):
    x_lo, y = sample_batch(train_utts, CFG, rng)
    y_hat = model(x_lo, perturb=True)
    l_wave = F.l1_loss(y_hat, y)
    l_spec = spec_loss(y, y_hat)
    loss = l_wave + CFG.lambda_spec * l_spec

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
    opt.step()
    sched.step()

    if step % 25 == 0:
        history["step"].append(step); history["loss"].append(loss.item())
        history["wave"].append(l_wave.item()); history["spec"].append(l_spec.item())
        history["lr"].append(sched.get_last_lr()[0])
    if (step + 1) % CFG.ckpt_every == 0 or step + 1 == CFG.steps:
        d = hb_deficit(model, probe, CFG)
        s, s0 = dev_snr(model, probe, CFG)
        model.train()
        history["dev_step"].append(step + 1); history["deficit"].append(d)
        history["snr"].append(s); history["snr_naive"].append(s0)
        save_ckpt({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                   "step": step + 1, "history": history, "cfg": dataclasses.asdict(CFG)}, CKPT_PATH)
        print(f"step {step+1:>6}/{CFG.steps}  loss {loss.item():.4f}  "
              f"(wave {l_wave.item():.4f}, spec {l_spec.item():.4f})  "
              f"HB deficit {d:+.2f} dB  SNR {s:5.2f} dB (naive {s0:5.2f})  "
              f"lr {sched.get_last_lr()[0]:.2e}  [{time.time()-t0:.0f}s]")

fig, ax = plt.subplots(1, 3, figsize=(15, 3.4))
ax[0].plot(history["step"], history["wave"], label="L1 waveform")
ax[0].plot(history["step"], history["spec"], label=f"multi-scale STFT (x{CFG.lambda_spec:g} in loss)")
ax[0].set_yscale("log"); ax[0].set_xlabel("step"); ax[0].legend(); ax[0].set_title("training loss terms")
ax[1].plot(history["dev_step"], history["deficit"], "o-")
ax[1].axhline(0, color="k", lw=1)
ax[1].set_xlabel("step"); ax[1].set_ylabel("mean high-band energy, dB")
ax[1].set_title("over-smoothing vs training (read the asymptote)")
ax[2].plot(history["dev_step"], history.get("snr", []), "o-", label="LISA")
if history.get("snr_naive"):
    ax[2].axhline(history["snr_naive"][-1], color="grey", ls="--", label="naive upsample")
ax[2].axhline(0, color="k", lw=1)
ax[2].set_xlabel("step"); ax[2].set_ylabel("waveform SNR, dB"); ax[2].legend()
ax[2].set_title("fidelity: must end above the dashed line")
plt.tight_layout(); plt.savefig(FIGS / f"training_{CFG.name}.png", dpi=130); plt.show()
""")

# ============================================================ 9 gate 2
md(r"""
## 9 · Gate 2 — is there any headroom?

Everything downstream assumes LISA's high band is energy-deficient.  Measure it before building on
it.  $\rho_b$ is the per-third-octave energy ratio on **held-out speakers**, in dB: $0$ means the
energy is right, negative means over-smoothed.

This cell trusts nothing from earlier cells that it can check itself: it reloads both speaker
splits from the persistent caches, verifies they are disjoint, and reloads the checkpoint weights.
(§7 builds a *fresh* `LISA`; only §8 loads `lisa.pt`.  Re-running §7 after a reconnect and skipping
§8 evaluates random weights.  A scratch cell that rebinds `test_utts` leaks training speakers.  Both
have happened.)

Two gates, in order:

* **2a — fidelity.**  Waveform SNR must beat naive polyphase upsampling of the same input.  If it
  does not, the model has not learned the baseband, and every high-band number below is built on
  sand.  The paper reports 24.16 dB at 12 kHz → 48 kHz with the same SNR definition.
* **2b — headroom.**  If the mean deficit above the input Nyquist is smaller than ~1–2 dB, the
  spectral loss has already eaten the headroom, and the honest result is to write that down rather
  than to correct a deficit that isn't there.  Check it against the training curve in §8 — an
  undertrained model exaggerates the deficit.

On LSD: the paper's 0.81 is **not** directly comparable to `lsd_db` here.  Their STFT basis is
n_fft 2048 / hop 1024 (from the released code; the paper states neither), and the released code's
log scaling appears to be twice the quantity in their own Eq. (5).  The paper-basis LSD is printed
for orientation only; SNR is the like-for-like number.
""")

code(r"""
# ---- pre-flight: reload what cell order could have corrupted -----------------------------------
for _tag in ("train", "test"):
    _cache = FIXTURES / f"{_tag}_{CFG.name}.npz"
    if _cache.exists():
        with np.load(_cache, allow_pickle=True) as _d:
            globals()[f"{_tag}_utts"] = list(_d["utts"])
            globals()[f"{_tag}_spk"]  = list(_d["speakers"])
    else:
        print(f"[pre-flight] no {_cache.name}; using in-memory {_tag}_utts")
_leak = set(test_spk) & set(train_spk)
if _leak:
    warnings.warn(f"SPEAKER LEAK: {sorted(_leak)} appear in both splits. Held-out numbers below are invalid.")
print(f"train {len(train_utts):>4} utts  speakers {sorted(set(train_spk))}")
print(f"test  {len(test_utts):>4} utts  speakers {sorted(set(test_spk))}")

_ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(_ck["model"]); model.eval()
print(f"weights: {CKPT_PATH.relative_to(ROOT)} @ step {_ck['step']}")

EVAL       = test_utts[:CFG.n_eval_utts]
EVAL_HAT   = [reconstruct(model, y, CFG) for y in EVAL]
EVAL_NAIVE = [naive_upsample(y, CFG) for y in EVAL]
print(f"reconstructed {len(EVAL)} held-out utterances")

bands = np.mean([band_energy_ratio(y, yh, CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop,
                                   200.0, CFG.fs_hi / 2)[:, 1]
                 for y, yh in zip(EVAL, EVAL_HAT)], axis=0)
centres = band_energy_ratio(EVAL[0], EVAL_HAT[0], CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop,
                            200.0, CFG.fs_hi / 2)[:, 0]
hb = centres >= CFG.fs_lo / 2
DEFICIT = float(np.mean(bands[hb]))

plt.figure(figsize=(7, 3.6))
plt.semilogx(centres, bands, "o-")
plt.axhline(0, color="k", lw=1)
plt.axvline(CFG.fs_lo / 2, color="crimson", ls="--", label=f"input Nyquist {CFG.fs_lo/2:.0f} Hz")
plt.axhspan(-2, 2, color="grey", alpha=0.15, label="no-headroom zone")
plt.xlabel("frequency (Hz)"); plt.ylabel(r"$\rho_b$  (dB)")
plt.title("High-band energy deficit, held-out speakers")
plt.legend(); plt.tight_layout(); plt.savefig(FIGS / f"headroom_{CFG.name}.png", dpi=130); plt.show()

SNR       = float(np.mean([snr_db(y, yh) for y, yh in zip(EVAL, EVAL_HAT)]))
SNR_NAIVE = float(np.mean([snr_db(y, yn) for y, yn in zip(EVAL, EVAL_NAIVE)]))
LSD       = float(np.mean([lsd_db(y, yh, CFG.eval_n_fft, CFG.eval_hop) for y, yh in zip(EVAL, EVAL_HAT)]))
HB_LSD    = float(np.mean([lsd_db(y, yh, CFG.eval_n_fft, CFG.eval_hop, CFG.eval_k_cut) for y, yh in zip(EVAL, EVAL_HAT)]))
LSD_PAPER_BASIS = float(np.mean([lsd_db(y, yh, 2048, 1024) for y, yh in zip(EVAL, EVAL_HAT)]))

print(f"mean deficit above {CFG.fs_lo/2:.0f} Hz: {DEFICIT:+.2f} dB")
print(f"SNR             {SNR:6.2f} dB    naive upsample {SNR_NAIVE:6.2f} dB    paper 24.16 dB")
print(f"LSD             {LSD:6.3f}       paper-basis (2048/1024) {LSD_PAPER_BASIS:.3f}    paper 0.81 (see note)")
print(f"HB-LSD          {HB_LSD:6.3f}")

SNR_OK      = SNR > SNR_NAIVE
HEADROOM_OK = DEFICIT < -1.0
print()
print("GATE 2a fidelity:", "pass -- SNR beats naive upsampling" if SNR_OK else
      "FAIL -- SNR below naive upsampling. The model has not learned the baseband; fix §8 first.")
print("GATE 2b headroom:", "headroom exists, continue" if HEADROOM_OK else
      "NO HEADROOM -- the premise is wrong for this model. Record it in RESEARCH.md and stop.")
""")

# ============================================================ 10 transport
md(r"""
## 10 · The transport ladder on held-out speakers

Maps are fitted on **train speakers** and applied to **held-out speakers**; the gap between the two
is reported, because a map that only works on the speakers it was fitted to is a lookup table.

The headline is the $\lambda$ sweep: a parametric curve in (SNR, HB-LSD).  The prediction to check
is that HB-LSD **dips and then rises** rather than falling monotonically to $\lambda=1$.  If LISA
were already at the LSD-optimal point, full marginal matching would stretch the spread *past* the
point-optimal predictor and hurt.  Where the dip sits therefore decomposes the deficit into the part
that is **bias** (a transport map can fix it) and the part that is **conditional spread** (only
sampling can fix it, §11).
""")

code(r"""
HI = slice(CFG.k_cut, None)
N_FIT_UTTS   = 8 if CFG.name == "SMOKE" else 60
N_FIT_FRAMES = 4000 if CFG.name == "SMOKE" else 40000
VAD_FLOOR    = -60.0          # dB below the loudest frame; below this a frame is treated as silence


def covariates(S, cfg):
    '''(2, n_frames): log low-band energy, low-band spectral flatness (voicing proxy).
       Both are functions of the input only -- never of the ground truth.'''
    P = np.abs(S[:, :cfg.k_cut]) ** 2 + 1e-20
    return np.stack([np.log(P.sum(1)),
                     np.exp(np.mean(np.log(P), 1)) / np.mean(P, 1)])


def pool(pairs, cfg, max_frames, label):
    '''Pooled high-band log-magnitudes and covariates over a list of (y_true, y_hat).'''
    Ls, Lt, C = [], [], []
    for y, yh in pairs:
        S_h = stft(yh, cfg.n_fft, cfg.hop)[0]
        S_t = stft(y,  cfg.n_fft, cfg.hop)[0]
        n = min(S_h.shape[0], S_t.shape[0])
        Ls.append(logmag(S_h[:n, HI]).T.astype(np.float32))
        Lt.append(logmag(S_t[:n, HI]).T.astype(np.float32))
        C.append(covariates(S_h[:n], cfg).astype(np.float32))
    Ls, Lt, C = np.hstack(Ls), np.hstack(Lt), np.hstack(C)
    if Ls.shape[1] > max_frames:
        sel = stream(f"pool/{label}").choice(Ls.shape[1], max_frames, replace=False)
        Ls, Lt, C = Ls[:, sel], Lt[:, sel], C[:, sel]
    return Ls, Lt, C


fit_utts = train_utts[:N_FIT_UTTS]
fit_pairs = [(y, reconstruct(model, y, CFG)) for y in fit_utts]
L_src, L_tgt, C_fit = pool(fit_pairs, CFG, N_FIT_FRAMES, "fit")
print(f"fitting statistics on {L_src.shape[1]:,} train-speaker frames x {L_src.shape[0]} bins")

MAPS = {
    "T1 quantile":   QuantileMap.fit(L_src, L_tgt, n_q=CFG.n_quantiles),
    "T2 diag":       BuresMap.fit(L_src, L_tgt, mode="diag"),
    "T2 block":      BuresMap.fit(L_src, L_tgt, mode="block", rho=0.1),
}
COND = ConditionalMap.fit(L_src, L_tgt, C_fit, C_fit)
print(f"T2 diag: {100*BuresMap.last_clipped:.1f}% of bins hit the sigma-ratio clip "
      f"(nearly-silent bins in the model output)")
print(f"T3: median level sensitivity dm_t/d(log energy) = "
      f"{np.median(COND.B_tgt[1]):+.3f} nats per sd -- if this is far from zero, a global "
      f"(unconditioned) map is mis-specified by construction.")


def apply_map(y_hat, fn, cfg, extra_noise=None):
    '''Modify the high band, then ALWAYS resynthesise to a waveform before any metric.'''
    S, pad, L = stft(y_hat, cfg.n_fft, cfg.hop)
    C = covariates(S, cfg)
    new = fn(logmag(S[:, HI]).T, C)
    S2 = S.copy()
    mag = np.maximum(np.exp(new.T) - 1e-8, 0.0)
    S2[:, HI] = mag * np.exp(1j * np.angle(S[:, HI]))
    if extra_noise is not None:
        S2[:, HI] = S2[:, HI] + extra_noise(mag, C)
    return istft(S2, cfg.n_fft, cfg.hop, pad, L)


def score_condition(fn, pairs=None, cfg=CFG, extra_noise=None):
    pairs = pairs or list(zip(EVAL, EVAL_HAT))
    snr, lsd, hbl, dfc = [], [], [], []
    for y, yh in pairs:
        out = apply_map(yh, fn, cfg, extra_noise)[:len(y)]
        snr.append(snr_db(y, out))
        lsd.append(lsd_db(y, out, cfg.eval_n_fft, cfg.eval_hop))
        hbl.append(lsd_db(y, out, cfg.eval_n_fft, cfg.eval_hop, cfg.eval_k_cut))
        b = band_energy_ratio(y, out, cfg.fs_hi, cfg.eval_n_fft, cfg.eval_hop,
                              cfg.fs_lo / 2, cfg.fs_hi / 2)
        dfc.append(np.mean(b[:, 1]))
    return {"snr": float(np.mean(snr)), "lsd": float(np.mean(lsd)),
            "hb_lsd": float(np.mean(hbl)), "deficit": float(np.mean(dfc))}


RESULTS = {"T0 identity": score_condition(lambda L, C: L)}
FRONTIER = {}
for name, T in list(MAPS.items()) + [("T3 conditional", COND)]:
    curve = []
    for lam in CFG.lambdas:
        if name == "T3 conditional":
            fn = lambda L, C, cm=T, lam=lam: Interpolated(Conditioned(cm, C), lam)(L)
        else:
            fn = lambda L, C, T=T, lam=lam: Interpolated(T, lam)(L)
        s = score_condition(fn); s["lambda"] = lam
        curve.append(s)
    FRONTIER[name] = curve
    RESULTS[f"{name} (lam=1)"] = curve[-1]
    best = min(curve, key=lambda r: r["hb_lsd"])
    RESULTS[f"{name} (best lam={best['lambda']:.2f})"] = best
    print(f"{name:16s} best HB-LSD {best['hb_lsd']:.3f} at lambda={best['lambda']:.2f} "
          f"(SNR {best['snr']:.2f} dB, deficit {best['deficit']:+.2f} dB)")

# train-speaker generalisation gap
gap_pairs = fit_pairs[:CFG.n_eval_utts]
gap_in  = score_condition(lambda L, C, T=MAPS["T1 quantile"]: T(L), pairs=gap_pairs)
gap_out = RESULTS["T1 quantile (lam=1)"]
print(f"\nT1 HB-LSD  train speakers {gap_in['hb_lsd']:.3f}  vs  held out {gap_out['hb_lsd']:.3f}"
      f"   (gap {gap_out['hb_lsd'] - gap_in['hb_lsd']:+.3f})")

plt.figure(figsize=(7.2, 4.6))
for name, curve in FRONTIER.items():
    plt.plot([c["snr"] for c in curve], [c["hb_lsd"] for c in curve], "o-", ms=4, label=name)
    plt.annotate(r"$\lambda$=1", (curve[-1]["snr"], curve[-1]["hb_lsd"]),
                 fontsize=8, xytext=(4, 2), textcoords="offset points")
b = RESULTS["T0 identity"]
plt.scatter([b["snr"]], [b["hb_lsd"]], c="k", zorder=5, label="T0 (LISA, $\\lambda$=0)")
plt.xlabel("SNR (dB)  -- higher is better"); plt.ylabel("HB-LSD  -- lower is better")
plt.title("The frontier: what sharpening costs in fidelity")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(FIGS / f"frontier_{CFG.name}.png", dpi=130); plt.show()
""")

# ============================================================ 11 controls
md(r"""
## 11 · Controls — the part that decides whether any of this is real

Three adversarial controls ship with the results, not as an appendix:

1. **Shaped noise.**  White noise, per-third-octave gains fitted so that high-band energy is
   proportional to low-band frame energy, gated by an energy VAD.  This is 1970s vocoder technology
   with no optimal transport anywhere in it.  If the ladder cannot beat this, the OT machinery
   bought vocabulary rather than value, and the honest question narrows to: *do quantile shape and
   cross-bin covariance buy anything beyond conditional two-moment noise shaping?*
2. **Mis-specified map.**  `T1` refitted with the target statistics rolled across frequency bins.
   If HB-LSD still improves, then LSD is rewarding generic energy inflation, and it is demoted from
   primary evidence for everything above.
3. **Frame shuffle.**  Permute the corrected high-band frames in time.  Pooled marginals are
   invariant under permutation, so a metric suite that does *not* score this much worse cannot see
   conditional structure at all — and every claim about $p(y\mid x)$ made with it would be empty.

Plus the **evaluation-basis check**: HB-LSD recomputed at a different STFT resolution.  A gain that
lives only in the basis the transport operates in is an artifact of the basis.
""")

code(r"""
# ---- control 1: classical noise-filling BWE ---------------------------------
def fit_shaped_noise(pairs, cfg, n_bands=None):
    '''alpha_b = E[high-band power in b] / E[low-band frame energy], fitted on train speakers.'''
    freqs = np.fft.rfftfreq(cfg.n_fft, 1.0 / cfg.fs_hi)[cfg.k_cut:]
    edges = third_octave_edges(cfg.fs_hi, cfg.fs_lo / 2, cfg.fs_hi / 2)
    bands = [np.where((freqs >= a) & (freqs < b))[0] for a, b in zip(edges[:-1], edges[1:])]
    bands = [b for b in bands if len(b)]
    num = np.zeros(len(bands)); den = 0.0
    for y, yh in pairs:
        S_t = stft(y, cfg.n_fft, cfg.hop)[0]
        S_h = stft(yh, cfg.n_fft, cfg.hop)[0]
        n = min(S_t.shape[0], S_h.shape[0])
        e_lb = (np.abs(S_h[:n, :cfg.k_cut]) ** 2).sum(1)
        P = np.abs(S_t[:n, cfg.k_cut:]) ** 2
        for i, b in enumerate(bands):
            num[i] += P[:, b].sum()
        den += e_lb.sum()
    return bands, num / max(den, 1e-20)


NOISE_BANDS, NOISE_ALPHA = fit_shaped_noise(fit_pairs, CFG)


def shaped_noise(mag, C, rng_label="control/noise"):
    '''Add noise so each third-octave band reaches alpha_b * (low-band frame energy).'''
    rng = stream(rng_label)
    e_lb = np.exp(C[0])
    vad = 10 * np.log10(e_lb / max(e_lb.max(), 1e-20) + 1e-20) > VAD_FLOOR
    P_have = mag ** 2
    out = np.zeros_like(mag, dtype=complex)
    for b, a in zip(NOISE_BANDS, NOISE_ALPHA):
        want = a * e_lb / len(b)                                  # per-bin target power
        deficit = np.clip(want[:, None] - P_have[:, b], 0.0, None)
        deficit *= vad[:, None]
        z = rng.standard_normal(deficit.shape) + 1j * rng.standard_normal(deficit.shape)
        out[:, b] = np.sqrt(deficit / 2.0) * z
    return out


RESULTS["control: shaped noise"] = score_condition(lambda L, C: L, extra_noise=shaped_noise)

# ---- control 2: deliberately mis-specified map ------------------------------
K = L_src.shape[0]
MISSPEC = QuantileMap(MAPS["T1 quantile"].q_src, np.roll(MAPS["T1 quantile"].q_tgt, K // 3, axis=0))
RESULTS["control: mis-specified T1"] = score_condition(lambda L, C, T=MISSPEC: T(L))

# ---- control 3: frame shuffle -----------------------------------------------
def shuffled(L, C, T=MAPS["T1 quantile"]):
    out = T(L)
    perm = stream("control/shuffle").permutation(out.shape[1])
    return out[:, perm]

RESULTS["control: frame shuffle"] = score_condition(shuffled)

# ---- basis robustness --------------------------------------------------------
alt = dataclasses.replace(CFG, eval_n_fft=CFG.eval_n_fft * 2, eval_hop=CFG.eval_hop * 2)
basis_T0 = score_condition(lambda L, C: L, cfg=alt)
basis_T1 = score_condition(lambda L, C, T=MAPS["T1 quantile"]: T(L), cfg=alt)

print(f"{'condition':<30}{'SNR':>8}{'LSD':>8}{'HB-LSD':>9}{'deficit':>9}")
for k, v in RESULTS.items():
    print(f"{k:<30}{v['snr']:8.2f}{v['lsd']:8.3f}{v['hb_lsd']:9.3f}{v['deficit']:9.2f}")
print()
print(f"basis check (n_fft={alt.eval_n_fft}):  T0 HB-LSD {basis_T0['hb_lsd']:.3f}  ->  "
      f"T1 {basis_T1['hb_lsd']:.3f}  (delta {basis_T1['hb_lsd'] - basis_T0['hb_lsd']:+.3f})")
print(f"operating basis (n_fft={CFG.eval_n_fft}): delta "
      f"{RESULTS['T1 quantile (lam=1)']['hb_lsd'] - RESULTS['T0 identity']['hb_lsd']:+.3f}")
""")

# ============================================================ 12 stochastic
md(r"""
## 12 · The stochastic rung

Everything above is a deterministic map on a point prediction, so it is still a point prediction.
It cannot be calibrated and it cannot be scored by a proper scoring rule in any meaningful way.
To actually produce samples:
$$Y[k,n] = M_{det}[k,n]\,e^{i\phi[k,n]} \;+\; \sigma_k(c[n])\,\varepsilon[k,n],
\qquad \varepsilon \sim \mathcal{CN}(0,1),$$
with $\sigma_k^2(c)$ the *residual* conditional power — the target power for that frame type minus
what the deterministic backbone already has.

This looks ad hoc and is not: $z \mapsto \sigma(c)z$ **is** the $W_2$-optimal map
$\mathcal{CN}(0,1)\to\mathcal{CN}(0,\sigma^2)$, so the sampler is reference noise pushed through a
closed-form conditional transport map — a conditional generator with zero learned parameters.

**Why this rung is the one that can win.**  Against a per-bin conditional $N(\mu,\sigma^2)$, the best
possible *point* forecast scores $\sigma\sqrt{2/\pi}\approx 0.798\sigma$ on CRPS, while a perfect
*sampler* scores $\sigma(\sqrt2-1)/\sqrt\pi\approx 0.234\sigma$.  That ~3.4× gap is unreachable by
any deterministic map, no matter how clever the transport.  And the mirror fact explains the header:
on an RMSE-type score like LSD a perfect sampler is *2× worse* than the conditional mean — so a
model tuned for LSD is being actively pushed toward the disease.
""")

code(r"""
def fit_conditional_power(pairs, cfg, cond):
    '''log target high-band power regressed on the same covariates as T3.'''
    logP, cov = [], []
    for y, yh in pairs:
        S_t = stft(y,  cfg.n_fft, cfg.hop)[0]
        S_h = stft(yh, cfg.n_fft, cfg.hop)[0]
        n = min(S_t.shape[0], S_h.shape[0])
        logP.append(np.log(np.abs(S_t[:n, HI]) ** 2 + 1e-20).T)
        cov.append(covariates(S_h[:n], cfg))
    return ConditionalPower.fit(np.hstack(logP), np.hstack(cov), cond.mu, cond.sd)


CPOW = fit_conditional_power(fit_pairs, CFG, COND)


def sample_high_band(y_hat, cfg, lam, n_samples, label):
    '''Returns (M, n_frames, K) complex high-band spectra and the reusable STFT context.'''
    S, pad, L = stft(y_hat, cfg.n_fft, cfg.hop)
    C = covariates(S, cfg)
    M_det = np.maximum(np.exp(Interpolated(Conditioned(COND, C), lam)(logmag(S[:, HI]).T).T) - 1e-8, 0.0)
    sigma2 = np.clip(CPOW(C) - M_det ** 2, 0.0, None)              # (n_frames, K)
    # Never excite a silent frame.  Hiss in the gaps is the most audible way this whole idea fails,
    # and no spectral metric in this notebook would report it.
    e_lb = np.exp(C[0])
    sigma2 *= (10 * np.log10(e_lb / max(e_lb.max(), 1e-20) + 1e-20) > VAD_FLOOR)[:, None]
    rng = stream(label)
    ph = np.angle(S[:, HI])
    out = []
    for _ in range(n_samples):
        z = rng.standard_normal(M_det.shape) + 1j * rng.standard_normal(M_det.shape)
        out.append(M_det * np.exp(1j * ph) + np.sqrt(sigma2 / 2.0) * z)
    return np.stack(out), (S, pad, L)


M_SAMPLES = 8 if CFG.name == "SMOKE" else 16
LAM_S = 1.0

crps_stoch, crps_det, ranks, sk = [], [], [], []
stoch_scores = {"snr": [], "lsd": [], "hb_lsd": [], "deficit": []}
for ui, (y, yh) in enumerate(zip(EVAL, EVAL_HAT)):
    samples, (S, pad, L) = sample_high_band(yh, CFG, LAM_S, M_SAMPLES, f"stoch/{ui}")
    waves = []
    for s in samples:
        S2 = S.copy(); S2[:, HI] = s
        waves.append(istft(S2, CFG.n_fft, CFG.hop, pad, L)[:len(y)])
    waves = np.stack(waves)

    # proper scores, on high-band log-magnitude in the evaluation basis
    lm = lambda w: logmag(stft(w, CFG.eval_n_fft, CFG.eval_hop)[0][:, CFG.eval_k_cut:])
    truth = lm(y)
    ens = np.stack([lm(w)[:truth.shape[0]] for w in waves])
    det = lm(apply_map(yh, lambda Lh, C: Interpolated(Conditioned(COND, C), LAM_S)(Lh), CFG))[:truth.shape[0]]
    crps_stoch.append(crps_ensemble(ens, truth))
    crps_det.append(float(np.mean(np.abs(det - truth))))          # CRPS of a point forecast = MAE
    ranks.append(pit_ranks(ens, truth))
    sk.append(spread_skill(ens, truth))

    w0 = waves[0]
    stoch_scores["snr"].append(snr_db(y, w0))
    stoch_scores["lsd"].append(lsd_db(y, w0, CFG.eval_n_fft, CFG.eval_hop))
    stoch_scores["hb_lsd"].append(lsd_db(y, w0, CFG.eval_n_fft, CFG.eval_hop, CFG.eval_k_cut))
    stoch_scores["deficit"].append(np.mean(band_energy_ratio(
        y, w0, CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop, CFG.fs_lo / 2, CFG.fs_hi / 2)[:, 1]))

RESULTS["S stochastic (1 draw)"] = {k: float(np.mean(v)) for k, v in stoch_scores.items()}
CRPS = {"deterministic (T3)": float(np.mean(crps_det)), "stochastic": float(np.mean(crps_stoch))}
_delta = 100 * (1 - CRPS["stochastic"] / max(CRPS["deterministic (T3)"], 1e-9))
print(f"CRPS (lower is better)  deterministic {CRPS['deterministic (T3)']:.4f}   "
      f"stochastic {CRPS['stochastic']:.4f}")
print(f"  the sampler is {abs(_delta):.1f}% {'BETTER' if _delta > 0 else 'WORSE'} than the point forecast")
if _delta <= 0:
    print("  -> the residual-variance estimate is mis-calibrated; read the PIT histogram below.")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
r = np.concatenate(ranks)
ax[0].hist(r, bins=M_SAMPLES + 1, range=(-0.5, M_SAMPLES + 0.5),
           weights=np.full(len(r), 1 / len(r)), edgecolor="k")
ax[0].axhline(1 / (M_SAMPLES + 1), color="crimson", ls="--", label="calibrated")
ax[0].set_xlabel("rank of truth among samples"); ax[0].set_title("PIT histogram")
ax[0].legend(fontsize=8)
s = np.mean(sk, axis=0)
lim = [0, max(s.max(), 1e-6) * 1.1]
ax[1].plot(s[:, 0], s[:, 1], "o-"); ax[1].plot(lim, lim, "k--", lw=1, label="ideal")
ax[1].set_xlabel("ensemble spread"); ax[1].set_ylabel("RMSE of ensemble mean")
ax[1].set_title("spread-skill"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIGS / f"calibration_{CFG.name}.png", dpi=130); plt.show()
print("U-shaped PIT = under-dispersed (too little noise); dome = over-dispersed (too much).")
""")

# ============================================================ 13 latency
md(r"""
## 13 · Latency

The worry that this correction needs a C++ or Rust rewrite is worth a measurement rather than an
argument.  The transport is one FFT pair plus an interpolation per frame; LISA is one 5-layer MLP
evaluation per *output sample*.
""")

code(r"""
def bench(fn, n_rep=20, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return np.percentile(ts, 50) * 1e3, np.percentile(ts, 95) * 1e3


y_bench = EVAL[0][:CFG.fs_hi]                     # exactly one second
yh_bench = EVAL_HAT[0][:CFG.fs_hi]
T1 = MAPS["T1 quantile"]
p50_t, p95_t = bench(lambda: apply_map(yh_bench, lambda L, C: T1(L), CFG))
p50_m, p95_m = bench(lambda: reconstruct(model, y_bench, CFG), n_rep=10)

print(f"{'stage':<34}{'p50 ms/s audio':>16}{'p95':>9}{'x real time':>14}")
print(f"{'transport (numpy, 1 core)':<34}{p50_t:16.2f}{p95_t:9.2f}{1000/p50_t:14.0f}")
print(f"{'LISA inference (' + str(DEVICE) + ')':<34}{p50_m:16.2f}{p95_m:9.2f}{1000/p50_m:14.0f}")
print(f"\ntransport is {p50_m/max(p50_t,1e-9):.0f}x cheaper than the model it corrects.")
""")

# ============================================================ 14 listen
md(r"""
## 14 · Listen

The single most likely failure of this whole idea is *"HB-LSD improved and it sounds worse"* —
deterministic transport amplifies whatever structured garbage occupies the attenuated high band
rather than restoring truth.  No metric in this notebook can adjudicate that.  Ears can.

Listen for: hiss in the silences (a global map inflating quiet frames), a metallic or "frozen"
texture on sustained vowels (deterministic sharpening), and whether fricatives — *s*, *sh*, *f* —
actually gain body.
""")

code(r"""
from IPython.display import Audio, display

y = EVAL[0]
yh = EVAL_HAT[0]
lo_up = sps.resample_poly(decimate(y, CFG.upsample), CFG.upsample, 1)[:len(y)]
samples_demo, (S_d, pad_d, L_d) = sample_high_band(yh, CFG, LAM_S, 1, "demo")
S_d2 = S_d.copy(); S_d2[:, HI] = samples_demo[0]

demos = {
    "ground truth (48k)": y,
    f"input, upsampled ({CFG.fs_lo} Hz)": lo_up,
    "LISA (T0)": yh,
    "T1 quantile, lambda=1": apply_map(yh, lambda L, C, T=T1: T(L), CFG),
    "T3 conditional, lambda=1": apply_map(yh, lambda L, C: Conditioned(COND, C)(L), CFG),
    "control: shaped noise": apply_map(yh, lambda L, C: L, CFG, extra_noise=shaped_noise),
    "S stochastic, one draw": istft(S_d2, CFG.n_fft, CFG.hop, pad_d, L_d),
}
for name, w in demos.items():
    print(name)
    display(Audio(np.clip(w[:len(y)], -1, 1), rate=CFG.fs_hi))

fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True, sharey=True)
for ax, (name, w) in zip(axes.ravel(), list(demos.items())[:6]):
    S = np.abs(stft(w[:len(y)], CFG.eval_n_fft, CFG.eval_hop)[0])
    ax.imshow(20 * np.log10(S.T + 1e-8), origin="lower", aspect="auto", cmap="magma",
              vmin=-100, vmax=0,
              extent=[0, len(y) / CFG.fs_hi, 0, CFG.fs_hi / 2000])
    ax.axhline(CFG.fs_lo / 2000, color="cyan", lw=0.8, ls="--")
    ax.set_title(name, fontsize=9)
    ax.set_ylabel("kHz")
plt.tight_layout(); plt.savefig(FIGS / f"spectrograms_{CFG.name}.png", dpi=130); plt.show()
""")

# ============================================================ 15 results
md(r"""
## 15 · Results and research log

The log records what failed, not only what worked.  A log that contains only wins is a marketing
document.
""")

code(r"""
rows = sorted(RESULTS.items(), key=lambda kv: kv[1]["hb_lsd"])
head = f"| condition | SNR dB | LSD | HB-LSD | HB deficit dB |\n|---|---|---|---|---|\n"
table = head + "".join(
    f"| {k} | {v['snr']:.2f} | {v['lsd']:.3f} | {v['hb_lsd']:.3f} | {v['deficit']:+.2f} |\n"
    for k, v in rows)

base = RESULTS["T0 identity"]
noise = RESULTS["control: shaped noise"]
best_name, best = min(((k, v) for k, v in RESULTS.items() if k.startswith(("T1", "T2", "T3", "S "))),
                      key=lambda kv: kv[1]["hb_lsd"])
beats_noise = best["hb_lsd"] < noise["hb_lsd"]
shuffle_seen = RESULTS["control: frame shuffle"]["hb_lsd"] > RESULTS["T1 quantile (lam=1)"]["hb_lsd"] + 1e-3
misspec_gain = RESULTS["control: mis-specified T1"]["hb_lsd"] < base["hb_lsd"]

log = f'''# Research log -- {CFG.name}

LISA reimplementation, {n_params:,} params (paper ~89k), {CFG.fs_lo} Hz -> {CFG.fs_hi} Hz,
{len(CFG.train_speakers)} train / {len(CFG.test_speakers)} held-out VCTK speakers, seed {SEED}.

## Gate 1 -- closed-form validation
All {len(TESTS)} tests pass: T1 reproduces the analytic log-Rayleigh shift map to 1e-12,
Bures satisfies A.Cs.A = Ct to 1e-10, McCann endpoints are exact and the W2 geodesic identity holds.

## Gate 2a -- fidelity
Waveform SNR on held-out speakers: **{SNR:.2f} dB** (naive upsampling {SNR_NAIVE:.2f} dB; paper 24.16 dB).
LSD {LSD:.3f} in this notebook's basis, {LSD_PAPER_BASIS:.3f} in the paper's STFT basis (not like-for-like; see §9).
Verdict: {"pass" if SNR_OK else "FAIL -- below naive upsampling; nothing below is a result"}.

## Gate 2b -- headroom
Mean high-band energy deficit on held-out speakers: **{DEFICIT:+.2f} dB**.
Verdict: {"headroom exists" if HEADROOM_OK else "NO HEADROOM -- premise does not hold for this model"}.

## Results
{table}
## Findings

- Best rung: **{best_name}** at HB-LSD {best['hb_lsd']:.3f} vs baseline {base['hb_lsd']:.3f},
  costing {best['snr'] - base['snr']:+.2f} dB SNR. The SNR cost is expected and required: the
  missing conditional variance has to come from somewhere.
- **Shaped-noise control**: HB-LSD {noise['hb_lsd']:.3f}. The transport ladder
  {"beats" if beats_noise else "DOES NOT beat"} 1970s noise filling.
  {"" if beats_noise else "This is the null result that matters: on this data the OT machinery adds vocabulary, not value."}
- **Frame-shuffle control**: HB-LSD {RESULTS["control: frame shuffle"]["hb_lsd"]:.3f}.
  The metric suite {"can" if shuffle_seen else "CANNOT"} see conditional structure
  {"" if shuffle_seen else "-- so no claim about p(y|x) can be supported by these metrics."}
- **Mis-specified map**: HB-LSD {RESULTS["control: mis-specified T1"]["hb_lsd"]:.3f}.
  {"WARNING: a deliberately wrong map still improves HB-LSD, so LSD is rewarding generic energy inflation and must be demoted from primary evidence." if misspec_gain else "A deliberately wrong map does not improve HB-LSD, so the gain is not generic energy inflation."}
- **CRPS**: deterministic {CRPS['deterministic (T3)']:.4f} -> stochastic {CRPS['stochastic']:.4f}.
  Only the sampler can move this; it is the metric on which the generative claim stands or falls.
- **Latency**: transport {p50_t:.2f} ms per second of audio (p95 {p95_t:.2f}), i.e.
  {1000/p50_t:.0f}x real time, {p50_m/max(p50_t,1e-9):.0f}x cheaper than LISA itself.
  There is no latency case for a C++/Rust rewrite of the correction.

## Still open

- Perceptual adjudication. No metric here settles "sharper" vs "sounds better"; §14 is by ear only.
- Post-hoc spectral transport is classical bandwidth extension in OT vocabulary. The differentiated
  claim would be transport on the 32-d latent codes, where a FULL covariance is 528 entries and is
  estimable -- exactly where it is noise-dominated across {L_src.shape[0]} STFT bins.
- LISA's decoder is a ReLU MLP, not a SIREN. ReLU spectral bias may be a second, architectural
  cause of over-smoothing, independent of the loss. One sweep of the decoder's frequency response
  would separate the two.
'''

(ROOT / f"RESEARCH_{CFG.name}.md").write_text(log)
print(log)
print(f"\nwritten to {ROOT / f'RESEARCH_{CFG.name}.md'}")
print(f"figures in {FIGS}")
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

out = pathlib.Path(__file__).parent / "lisa_rtm.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(CELLS)} cells)")
