# ============================================================ OV3-1 decoder-noise sampler, joint and ERB geometries
# Extends overnight2/c1_model.py (must be exec'd first: LISAS, LISADecoder, arm_loss, d_wave, d_logmag,
# logmag_feats, lowpass, SCALES, _win, N_NOISE).  Four additions, all cheap:
#   LISASD      LISAS plus n_dec Gaussian channels per OUTPUT sample at the decoder input.  The encoder
#               noise alone gives the conditional law only 8 channels at 12 kHz to spread with; decoder
#               noise gives it 4 more per 48 kHz sample (EnScale / DISCO Nets: noise deeper in the net).
#   es_ged      d = L1(wave) + lam * RMS over the whole log-magnitude spectrogram (Euclidean on frames x
#               bins) -- strictly proper for the JOINT law of the spectrogram (Gritsenko et al. 2020),
#               where es_marg's per-bin L1 is proper for the marginals only.
#   es_erb      d = L1(wave) + lam * L1 on log ERB-band magnitudes (32 bands to fs/2 on three STFT scales):
#               the geometry ViSQOL's audio-mode neurogram lives in.  Properness holds for the law of
#               phi(y) whenever d = ||phi(a) - phi(b)||, so this is "score the judge's own space".
#   es_marg_erb d = L1(wave) + lam * (d_logmag + d_erb) / 2.
#   logmag_ensemble_readout  mean of log|STFT| over M draws, phase of draw 0, baseband passed through.
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

N_DEC = 4


class LISASD(LISAS):
    '''LISAS with decoder-side noise.  Decoder feature = [coord, z_{i-1}, z_i, z_{i+1}, eps_dec_j], so the
    decoder's first Linear has 1 + 3C + n_dec inputs.  eps is a tuple (eps_enc (B, n_noise, L),
    eps_dec (B, R*L, n_dec)); eps=None -> zeros in both, so tau=0 is LISAS at eps=0 (copy_shared() makes
    the outputs identical).  reconstruct() in c1_model calls encode(x_lo, eps) once and then decode(z, j0, j1)
    in chunks, so encode() stores the full decoder noise on self._eps_dec and decode() slices [j0:j1].'''
    def __init__(self, cfg, n_noise=N_NOISE, n_dec=N_DEC):
        super().__init__(cfg, n_noise)
        self.n_dec = n_dec
        first = self.dec.net[0]
        self.dec.net[0] = nn.Linear(first.in_features + n_dec, first.out_features)
        self._eps_dec = None

    def sample_eps(self, x_lo, tau=1.0, seed=None):
        B, L = x_lo.shape
        g = None if seed is None else torch.Generator(device=x_lo.device).manual_seed(int(seed))
        e = tau * torch.randn(B, self.n_noise, L, device=x_lo.device, generator=g)
        d = tau * torch.randn(B, L * self.R, self.n_dec, device=x_lo.device, generator=g)
        return e, d

    def encode(self, x_lo, eps=None):
        if eps is None:
            e, d = None, None
        elif isinstance(eps, (tuple, list)):
            e, d = eps
        else:
            e, d = eps, None
        self._eps_dec = d
        return super().encode(x_lo, e)

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

        d = self._eps_dec
        d = torch.zeros(B, j1 - j0, self.n_dec, device=z.device, dtype=z.dtype) if d is None else d[:, j0:j1].to(z.dtype)
        return self.dec(torch.cat([coord, take(idx - 1), take(idx), take(idx + 1), d], -1))


def copy_shared(src, dst):
    '''Copy every weight of a LISAS into a LISASD (or LISAS); the decoder's extra noise columns are zeroed,
    so dst at eps=0 equals src at eps=0.  Returns dst.'''
    sd, dd = src.state_dict(), dst.state_dict()
    with torch.no_grad():
        for k, v in sd.items():
            if k not in dd:
                continue
            if dd[k].shape == v.shape:
                dd[k].copy_(v)
            elif k.endswith("dec.net.0.weight"):
                dd[k].zero_(); dd[k][:, : v.shape[1]].copy_(v)
    dst.load_state_dict(dd)
    return dst


# ---- distances ------------------------------------------------------------------------------------
def d_logmag_l2(fa, fb):
    '''Frobenius distance on the whole log-magnitude spectrogram divided by sqrt(F*T) (per-element RMS),
    averaged over scales.  Euclidean geometry on frames x bins: strictly proper for the joint law.  (B,)'''
    tot = 0.0
    for A, B in zip(fa, fb):
        tot = tot + (A - B).flatten(1).norm(dim=1) / math.sqrt(A.shape[1] * A.shape[2])
    return tot / len(fa)


def _erb_rate(f):
    '''ERB-rate scale (Glasberg & Moore 1990): 21.4 log10(1 + 4.37 f / 1000).'''
    return 21.4 * np.log10(1.0 + 4.37 * np.asarray(f, np.float64) / 1000.0)


def _erb_rate_inv(e):
    return (10.0 ** (np.asarray(e, np.float64) / 21.4) - 1.0) * 1000.0 / 4.37


_ERB = {}

def erb_filterbank(n_fft, fs, n_bands=32, f_lo=50.0, device=None):
    '''(n_bands, F) triangular filters with centres equally spaced on the ERB-rate scale from f_lo to fs/2.
    Every row sums to 1; every STFT bin at or above f_lo belongs to at least one band (bins no triangle
    reaches, e.g. the exact end points, go to the nearest band by centre frequency; a band too narrow to
    hold a bin takes its nearest bin).  Cached per (n_fft, fs, n_bands, f_lo, device).'''
    key = (int(n_fft), float(fs), int(n_bands), float(f_lo), str(device))
    if key not in _ERB:
        freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
        edges = _erb_rate_inv(np.linspace(_erb_rate(f_lo), _erb_rate(fs / 2), n_bands + 2))
        W = np.zeros((n_bands, len(freqs)))
        for b in range(n_bands):
            lo, c, hi = edges[b], edges[b + 1], edges[b + 2]
            W[b] = np.clip(np.minimum((freqs - lo) / max(c - lo, 1e-9), (hi - freqs) / max(hi - c, 1e-9)), 0.0, 1.0)
        centres = edges[1:-1]
        for k in np.where((W.sum(0) <= 0) & (freqs >= f_lo))[0]:          # uncovered bins -> nearest band
            W[int(np.argmin(np.abs(centres - freqs[k]))), k] = 1.0
        for b in np.where(W.sum(1) <= 0)[0]:                              # empty bands -> nearest bin
            W[b, int(np.argmin(np.abs(freqs - centres[b])))] = 1.0
        W = W / W.sum(1, keepdims=True)
        _ERB[key] = torch.tensor(W, dtype=torch.float32, device=device)
    return _ERB[key]


def erb_feats(y, fs=None):
    '''list over SCALES of (B, n_bands, T) log ERB-band RMS magnitudes: 0.5 * log(W @ |STFT|^2 + 1e-8).
    The 0.5 puts d_erb on the same scale as d_logmag (log|S|, c1_model), so one lam means one spectral
    weight across es_marg / es_erb and es_marg_erb's average is equal-weight.'''
    fs = CFG.fs_hi if fs is None else fs
    out = []
    for n, h in SCALES:
        P = torch.stft(y, n, h, window=_win(n, y.device), return_complex=True).abs() ** 2
        W = erb_filterbank(n, fs, device=y.device)
        out.append(0.5 * torch.log(torch.matmul(W, P) + 1e-8))
    return out


def d_erb(fa, fb):
    return sum((A - B).abs().mean(dim=(1, 2)) for A, B in zip(fa, fb)) / len(fa)       # (B,)


# ---- arms -------------------------------------------------------------------------------------------
NEW_KINDS = ("es_ged", "es_erb", "es_marg_erb", "es_split_marg", "es_split_marg_erb")
SPLIT_KINDS = ("es_split_marg", "es_split_marg_erb")   # waveform term on the low band only (det_split for the sampler)

def arm_loss3(kind, lam, m, x, y, spec_loss, perturb=True):
    '''c1_model's arm_loss plus the three new geometries.  Same two-draw unbiased estimator
    1/2 [d(y,y1) + d(y,y2)] - 1/2 d(y1,y2), same logged keys (wave, spec, spread).
    Any kind works with LISAS or LISASD: sample_eps / forward handle the class's own noise shape.
    perturb: anchor jitter in the decoder (True in training, as arm_loss hard-codes it; False for a
    validation loss on the same decoder path reconstruct() uses).  es_slice always delegates to arm_loss.'''
    if kind == "es_slice":
        return arm_loss(kind, lam, m, x, y, spec_loss)
    if kind in ("det", "det_split"):
        y_hat = m(x, perturb=perturb)                                          # eps = 0
        l_w = F.l1_loss(lowpass(y_hat, m.R), lowpass(y, m.R)) if kind == "det_split" else F.l1_loss(y_hat, y)
        l_s = spec_loss(y, y_hat)
        return l_w + lam * l_s, {"wave": l_w.item(), "spec": l_s.item()}
    B = x.shape[0]
    eps = m.sample_eps(x.repeat(2, 1), 1.0)
    yh = m(x.repeat(2, 1), perturb=perturb, eps=eps)
    y1, y2 = yh[:B], yh[B:]
    if kind in SPLIT_KINDS:                      # the waveform term never asks for zero above fs_lo/2
        yl, y1l, y2l = lowpass(y, m.R), lowpass(y1, m.R), lowpass(y2, m.R)
        dw = 0.5 * (d_wave(yl, y1l) + d_wave(yl, y2l)) - 0.5 * d_wave(y1l, y2l)
    else:
        dw = 0.5 * (d_wave(y, y1) + d_wave(y, y2)) - 0.5 * d_wave(y1, y2)
    if kind == "es_wave":
        return dw.mean(), {"wave": dw.mean().item(), "spec": 0.0, "spread": d_wave(y1, y2).mean().item()}
    if kind in ("es_marg", "es_split_marg"):
        feats, d = logmag_feats, d_logmag
    elif kind == "es_ged":
        feats, d = logmag_feats, d_logmag_l2
    elif kind == "es_erb":
        feats, d = erb_feats, d_erb
    elif kind in ("es_marg_erb", "es_split_marg_erb"):
        feats = lambda w: (logmag_feats(w), erb_feats(w))
        d = lambda a, b: 0.5 * (d_logmag(a[0], b[0]) + d_erb(a[1], b[1]))
    else:
        raise ValueError(kind)
    fy, f1, f2 = feats(y), feats(y1), feats(y2)
    ds = 0.5 * (d(fy, f1) + d(fy, f2)) - 0.5 * d(f1, f2)
    loss = (dw + lam * ds).mean()
    return loss, {"wave": dw.mean().item(), "spec": ds.mean().item(), "spread": d_wave(y1, y2).mean().item()}


CLASSES = {"LISAS": LISAS, "LISASD": LISASD}

def arm3(spec):
    '''(kind, lam) or (kind, lam, cls_name) -> (kind, lam, cls_name).'''
    return spec[0], spec[1], (spec[2] if len(spec) > 2 else "LISAS")


# ---- ensemble readout ---------------------------------------------------------------------------------
def _split_bands(w, fs, f_cut):
    W = np.fft.rfft(np.asarray(w, np.float64))
    k = int(round(f_cut * len(w) / fs))
    lo, hi = W.copy(), W.copy()
    lo[k:] = 0; hi[:k] = 0
    return np.fft.irfft(lo, n=len(w)), np.fft.irfft(hi, n=len(w))


def logmag_ensemble_readout(draws, y, cfg, passthrough=True):
    '''draws (M, T) waveforms of one utterance -> one waveform of length len(y).  Per-bin MEAN of log|STFT|
    across the draws in the evaluation basis (cfg.eval_n_fft, cfg.eval_hop), phase of draw 0, resynthesised.
    passthrough=True (default): brick-wall low band of naive_upsample(y) + high band of that readout, cut at
    cfg.fs_lo/2 (c7 hybrid).  passthrough=False: the raw readout.'''
    y = np.asarray(y, np.float64)
    n = len(y)
    draws = [np.pad(np.asarray(d, np.float64)[:n], (0, max(0, n - len(d)))) for d in draws]
    S0, pad, length = stft(draws[0], cfg.eval_n_fft, cfg.eval_hop)
    L = logmag(S0)
    for d in draws[1:]:
        L = L + logmag(stft(d, cfg.eval_n_fft, cfg.eval_hop)[0])
    L = L / len(draws)
    w = resynth(S0, L, slice(0, None), cfg.eval_n_fft, cfg.eval_hop, pad, length)
    w = np.pad(w[:n], (0, max(0, n - len(w))))
    if not passthrough:
        return w
    nv = naive_upsample(y, cfg)[:n]; nv = np.pad(nv, (0, n - len(nv)))
    lo, _ = _split_bands(nv, cfg.fs_hi, cfg.fs_lo / 2)
    _, hi = _split_bands(w, cfg.fs_hi, cfg.fs_lo / 2)
    return lo + hi


_p = sum(p.numel() for p in LISASD(CFG).parameters())
print(f"OV3 model/distances defined.  LISASD params: {_p:,}  (LISAS {sum(p.numel() for p in LISAS(CFG).parameters()):,}; "
      f"+{N_DEC} decoder noise channels x {CFG.dec_hidden} = {N_DEC * CFG.dec_hidden} weights)", flush=True)
del _p
