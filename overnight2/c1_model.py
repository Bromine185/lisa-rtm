# ============================================================ OV2-1 stochastic LISA + proper scoring rules
# The network becomes the transport map: reference noise (n_noise Gaussian channels at the input
# rate) is pushed through the SAME encoder/decoder as LISA.  eps = 0 recovers LISA exactly, so the
# deterministic arms and the stochastic arms share one class, one parameter count, one init.
#
# Training objectives (all paired on identical batches):
#   det       L1(wave) + lam * MSSTFT(sc + logmag)                 -- the paper's loss (lam = 1e-2)
#   det_split L1(lowpass wave) + lam * MSSTFT                      -- no waveform term above fs_lo/2
#   es_marg   energy score, d = L1(wave) + lam * L1(logmag)        -- proper for per-sample / per-bin marginals
#   es_slice  energy score, d = L1(wave) + lam * sliced-L1(logmag frames)  -- proper for the JOINT law of a frame
#   es_wave   energy score, d = L1(wave)                           -- no spectral term at all
# Energy score with two draws:  ES = (d(y,y1) + d(y,y2))/2 - d(y1,y2)/2  (unbiased, Gneiting & Raftery 2007).
import copy, math, time, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

N_NOISE = 8


class LISAS(nn.Module):
    def __init__(self, cfg, n_noise=N_NOISE):
        super().__init__()
        self.n_noise, self.R = n_noise, cfg.upsample
        layers, c_in = [], 1 + n_noise
        for i, (c, k) in enumerate(zip(cfg.enc_channels, cfg.enc_kernels)):
            layers.append(nn.Conv1d(c_in, c, k, padding=k // 2))
            if i < len(cfg.enc_channels) - 1:
                layers.append(nn.ReLU())
            c_in = c
        self.enc, self.dim = nn.Sequential(*layers), c_in
        self.dec = LISADecoder(self.dim, cfg)
        self.tau = 0.0          # default noise temperature used by reconstruct() when none is given
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


@torch.no_grad()
def reconstruct(model, y, cfg, chunk=1 << 15, tau=None, seed=None):
    '''Full-utterance inference.  One noise draw per utterance (encoded once), decoded in chunks.
    tau=None -> model.tau (0 for deterministic arms).  Overrides the notebook's reconstruct().'''
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


class HostCorpus:
    '''Corpus concatenated in host RAM (float32); batches are aligned slices moved to the GPU per step.
    Same semantics as GPUCorpus, but leaves the whole GPU to the model (a 37 h corpus is 32 GB).'''
    def __init__(self, utts, cfg, seg_hi):
        from concurrent.futures import ThreadPoolExecutor
        R = cfg.upsample
        ys = [np.asarray(u, np.float32) for u in utts if len(u) >= seg_hi]
        ys = [y[: (len(y) // R) * R] for y in ys]
        with ThreadPoolExecutor(32) as ex:
            xs = list(ex.map(lambda y: decimate(y.astype(np.float64), R).astype(np.float32), ys))
        self.n, self.R, self.seg_hi, self.seg_lo = len(ys), R, seg_hi, seg_hi // R
        self.lens = np.array([len(y) for y in ys])
        self.off = np.concatenate([[0], np.cumsum(self.lens)[:-1]])
        self.Y = np.concatenate(ys); del ys
        self.X = np.concatenate(xs); del xs
        self.hours = float(self.lens.sum()) / cfg.fs_hi / 3600
        self._ar_hi = np.arange(seg_hi)
        self._ar_lo = np.arange(seg_hi // R)
        print(f"HostCorpus: {self.n} utts, {self.hours:.2f} h, {(self.Y.nbytes + self.X.nbytes) / 1e9:.1f} GB in host RAM")

    def batch(self, rng, B):
        idx = rng.integers(self.n, size=B)
        n_pos = (self.lens[idx] - self.seg_hi) // self.R + 1
        starts = (rng.random(B) * n_pos).astype(np.int64) * self.R
        hi0 = self.off[idx] + starts
        y = self.Y[hi0[:, None] + self._ar_hi]
        x = self.X[(hi0 // self.R)[:, None] + self._ar_lo]
        return torch.from_numpy(x).to(DEVICE, non_blocking=True), torch.from_numpy(y).to(DEVICE, non_blocking=True)


# ---- distances ------------------------------------------------------------------------------------
SCALES = [(CFG.n_fft, CFG.n_fft // 4), (CFG.n_fft // 2, CFG.n_fft // 8), (CFG.n_fft // 4, CFG.n_fft // 16)]
_WIN = {}

def _win(n, device):
    k = (n, str(device))
    if k not in _WIN:
        _WIN[k] = torch.hann_window(n, device=device)
    return _WIN[k]

def logmag_feats(y):
    '''list over scales of (B, F, T) log-magnitudes.'''
    return [torch.log(torch.stft(y, n, h, window=_win(n, y.device), return_complex=True).abs() + 1e-7)
            for n, h in SCALES]

def lowpass(y, R):
    '''Brick-wall projection onto the input band [0, fs_lo/2] -- a linear projection of the error.'''
    Y = torch.fft.rfft(y)
    k_cut = Y.shape[-1] // R                     # bins <= fs_lo/2
    Y[..., k_cut + 1:] = 0
    return torch.fft.irfft(Y, n=y.shape[-1])

def d_wave(a, b):
    return (a - b).abs().mean(dim=tuple(range(1, a.ndim)))                          # (B,)

def d_logmag(fa, fb):
    return sum((A - B).abs().mean(dim=(1, 2)) for A, B in zip(fa, fb)) / len(fa)       # (B,)

def d_sliced(fa, fb, thetas):
    '''|theta . (A_t - B_t)| averaged over P random unit directions in R^F, frames t, scales.'''
    tot = 0.0
    for A, B, Th in zip(fa, fb, thetas):                       # A: (B, F, T), Th: (F, P)
        D = torch.einsum("bft,fp->bpt", A - B, Th)
        tot = tot + D.abs().mean(dim=(1, 2))
    return tot / len(fa)

def sample_thetas(device):
    out = []
    for n, _ in SCALES:
        Fbins = n // 2 + 1
        th = torch.randn(Fbins, 64, device=device)
        out.append(th / th.norm(dim=0, keepdim=True))
    return out


def arm_loss(kind, lam, m, x, y, spec_loss):
    '''Returns (loss, dict of logged terms).  ES arms run two draws in one 2B forward pass.'''
    if kind in ("det", "det_split"):
        y_hat = m(x, perturb=True)                                            # eps = 0
        l_w = F.l1_loss(lowpass(y_hat, m.R), lowpass(y, m.R)) if kind == "det_split" else F.l1_loss(y_hat, y)
        l_s = spec_loss(y, y_hat)
        return l_w + lam * l_s, {"wave": l_w.item(), "spec": l_s.item()}
    B = x.shape[0]
    eps = m.sample_eps(x.repeat(2, 1), 1.0)
    yh = m(x.repeat(2, 1), perturb=True, eps=eps)
    y1, y2 = yh[:B], yh[B:]
    dw = 0.5 * (d_wave(y, y1) + d_wave(y, y2)) - 0.5 * d_wave(y1, y2)
    if kind == "es_wave":
        loss = dw.mean()
        return loss, {"wave": dw.mean().item(), "spec": 0.0, "spread": d_wave(y1, y2).mean().item()}
    fy, f1, f2 = logmag_feats(y), logmag_feats(y1), logmag_feats(y2)
    if kind == "es_marg":
        d = d_logmag
        ds = 0.5 * (d(fy, f1) + d(fy, f2)) - 0.5 * d(f1, f2)
    elif kind == "es_slice":
        th = sample_thetas(y.device)
        d = lambda a, b: d_sliced(a, b, th)
        ds = 0.5 * (d(fy, f1) + d(fy, f2)) - 0.5 * d(f1, f2)
    else:
        raise ValueError(kind)
    loss = (dw + lam * ds).mean()
    return loss, {"wave": dw.mean().item(), "spec": ds.mean().item(), "spread": d_wave(y1, y2).mean().item()}


def probe_metrics(m, probe, cfg, naive):
    '''SNR / deficit at tau=0 and, for stochastic arms, at tau=1 (one draw).'''
    out = {}
    for tau in ((0.0,) if m.tau == 0 else (0.0, 1.0)):
        yh = reconstruct(m, probe, cfg, tau=tau, seed=0)
        b = band_energy_ratio(probe, yh, cfg.fs_hi, cfg.eval_n_fft, cfg.eval_hop, cfg.fs_lo / 2, cfg.fs_hi / 2)
        out[tau] = (snr_db(probe, yh), float(np.mean(b[:, 1])))
    return out


def train_ov2(corpus, arms, steps, batch, lr, milestones, gamma, clip, ckpt_every, tag, probe, log_every=25):
    '''arms: {name: (kind, lam)}.  One init, one batch stream, one optimiser per arm.'''
    names = list(arms)
    cls = globals().get("MODEL_CLS", LISAS)
    base = cls(CFG).to(DEVICE)
    models = {k: copy.deepcopy(base) for k in names}
    for k in names:
        models[k].tau = 0.0 if arms[k][0].startswith("det") else 1.0
    opts = {k: torch.optim.Adam(models[k].parameters(), lr=lr) for k in names}
    scheds = {k: MultiStepLR(opts[k], [int(f * steps) for f in milestones], gamma) for k in names}
    spec_loss = MultiScaleSTFTLoss(CFG.n_fft).to(DEVICE)
    hist = {k: {"step": [], "wave": [], "spec": [], "spread": [], "lr": [], "dev_step": [], "snr0": [], "def0": [],
                "snr1": [], "def1": [], "snr_naive": []} for k in names}
    rng = stream(f"{tag}/batches")
    run_dir = CKPT / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    probe = np.asarray(probe, np.float64)
    naive = snr_db(probe, naive_upsample(probe, CFG))
    logf = open(ROOT / f"train_{tag}.log", "a")
    t0 = time.time()
    for step in range(steps):
        x, y = corpus.batch(rng, batch)
        log = (step % log_every == 0)
        for k in names:
            kind, lam = arms[k]
            m = models[k]
            m.train()
            loss, terms = arm_loss(kind, lam, m, x, y, spec_loss)
            opts[k].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
            opts[k].step()
            scheds[k].step()
            if log:
                h = hist[k]
                h["step"].append(step); h["wave"].append(terms["wave"]); h["spec"].append(terms["spec"])
                h["spread"].append(terms.get("spread", 0.0)); h["lr"].append(scheds[k].get_last_lr()[0])
        if (step + 1) % ckpt_every == 0 or step + 1 == steps:
            line = f"step {step+1:>6}/{steps} [{time.time()-t0:.0f}s]"
            for k in names:
                m = models[k]
                pm = probe_metrics(m, probe, CFG, naive)
                h = hist[k]
                h["dev_step"].append(step + 1); h["snr_naive"].append(naive)
                h["snr0"].append(pm[0.0][0]); h["def0"].append(pm[0.0][1])
                s1, d1 = pm.get(1.0, (float("nan"), float("nan")))
                h["snr1"].append(s1); h["def1"].append(d1)
                save_ckpt({"model": m.state_dict(), "step": step + 1, "history": h, "arm": arms[k], "n_noise": m.n_noise,
                           "cls": type(m).__name__, "batch": batch, "seg": corpus.seg_hi, "tag": tag}, run_dir / f"{k}.pt")
                line += f" | {k}: w {h['wave'][-1]:.4f} s {h['spec'][-1]:.3f} SNR0 {pm[0.0][0]:5.2f} def0 {pm[0.0][1]:+6.2f}"
                if 1.0 in pm:
                    line += f" SNR1 {s1:5.2f} def1 {d1:+6.2f}"
            line += f"  (naive {naive:.2f})"
            print(line, flush=True)
            logf.write(line + chr(10)); logf.flush()
    logf.close()
    return models, hist


def time_ov2(corpus, arms, batch, n=20):
    names = list(arms)
    cls = globals().get("MODEL_CLS", LISAS)
    models = {k: cls(CFG).to(DEVICE) for k in names}
    opts = {k: torch.optim.Adam(models[k].parameters(), lr=1e-3) for k in names}
    spec_loss = MultiScaleSTFTLoss(CFG.n_fft).to(DEVICE)
    rng = stream("timing")
    torch.cuda.reset_peak_memory_stats()
    for i in range(n + 3):
        if i == 3:
            torch.cuda.synchronize(); t0 = time.time()
        x, y = corpus.batch(rng, batch)
        for k in names:
            loss, _ = arm_loss(arms[k][0], arms[k][1], models[k], x, y, spec_loss)
            opts[k].zero_grad(set_to_none=True); loss.backward(); opts[k].step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    print(f"{len(names)} arm(s) {list(arms)}, batch {batch} x {corpus.seg_hi}: {dt*1000:.0f} ms/step, "
          f"peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
    del models, opts
    torch.cuda.empty_cache()
    return dt


def load_arm(path, cfg=None):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    cls = globals().get(ck.get("cls", "LISAS"), LISAS)
    m = cls(cfg or CFG, n_noise=ck.get("n_noise", N_NOISE)).to(DEVICE)
    m.load_state_dict(ck["model"]); m.eval()
    m.tau = 0.0 if ck["arm"][0].startswith("det") else 1.0
    return m, ck

print("OV2 model/loss/trainer defined.  LISAS params:", sum(p.numel() for p in LISAS(CFG).parameters()))
