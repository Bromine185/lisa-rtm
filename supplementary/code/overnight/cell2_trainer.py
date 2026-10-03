# ============================================================ XL-2 trainer
# GPU-resident corpus, exact lo/hi alignment, paired multi-arm training on identical batches.
import copy, time, json, numpy as np, torch, torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from torch.optim.lr_scheduler import MultiStepLR

# speed: determinism off for training (seeds still fix the data order), TF32 on
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def naive_upsample(y, cfg):
    y = np.asarray(y, np.float64)
    return sps.resample_poly(decimate(y, cfg.upsample), cfg.upsample, 1)[:len(y)]

def hb_deficit_of(m, y, cfg):
    b = band_energy_ratio(y, reconstruct(m, y, cfg), cfg.fs_hi, cfg.eval_n_fft, cfg.eval_hop, cfg.fs_lo / 2, cfg.fs_hi / 2)
    return float(np.mean(b[:, 1])) if len(b) else float("nan")

class GPUCorpus:
    '''All utterances concatenated on the GPU; batches are aligned slices, no per-step resampling.'''
    def __init__(self, utts, cfg, seg_hi):
        R = cfg.upsample
        ys = [np.asarray(u, np.float32) for u in utts if len(u) >= seg_hi]
        ys = [y[: (len(y) // R) * R] for y in ys]
        with ThreadPoolExecutor(32) as ex:
            xs = list(ex.map(lambda y: decimate(y.astype(np.float64), R).astype(np.float32), ys))
        self.n, self.R, self.seg_hi, self.seg_lo = len(ys), R, seg_hi, seg_hi // R
        self.lens = np.array([len(y) for y in ys])
        self.off = np.concatenate([[0], np.cumsum(self.lens)[:-1]])
        self.Y = torch.from_numpy(np.concatenate(ys)).to(DEVICE)
        self.X = torch.from_numpy(np.concatenate(xs)).to(DEVICE)
        self.hours = float(self.lens.sum()) / cfg.fs_hi / 3600
        self._ar_hi = torch.arange(seg_hi, device=DEVICE)
        self._ar_lo = torch.arange(seg_hi // R, device=DEVICE)
        print(f"GPUCorpus: {self.n} utts, {self.hours:.2f} h, "
              f"{(self.Y.numel() + self.X.numel()) * 4 / 1e9:.1f} GB on {DEVICE}")

    def batch(self, rng, B):
        idx = rng.integers(self.n, size=B)
        n_pos = (self.lens[idx] - self.seg_hi) // self.R + 1
        starts = (rng.random(B) * n_pos).astype(np.int64) * self.R
        hi0 = torch.as_tensor(self.off[idx] + starts, device=DEVICE)
        y = self.Y[hi0[:, None] + self._ar_hi]
        x = self.X[(hi0 // self.R)[:, None] + self._ar_lo]
        return x, y

def train_paired(corpus, arms, steps, batch, lr, milestones, gamma, clip, ckpt_every, tag, probe, log_every=25):
    '''arms: {name: lambda_spec}.  Same init, same batches, one optimizer per arm.'''
    names = list(arms)
    base = LISA(CFG).to(DEVICE)
    models = {k: copy.deepcopy(base) for k in names}
    opts = {k: torch.optim.Adam(models[k].parameters(), lr=lr) for k in names}
    scheds = {k: MultiStepLR(opts[k], [int(f * steps) for f in milestones], gamma) for k in names}
    spec_loss = MultiScaleSTFTLoss(CFG.n_fft).to(DEVICE)
    hist = {k: {"step": [], "wave": [], "spec": [], "lr": [], "dev_step": [], "snr": [], "deficit": [], "snr_naive": []}
            for k in names}
    rng = stream(f"{tag}/batches")
    run_dir = CKPT / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    probe = np.asarray(probe, np.float64)
    naive = snr_db(probe, naive_upsample(probe, CFG))
    t0 = time.time()
    for step in range(steps):
        x, y = corpus.batch(rng, batch)
        log = (step % log_every == 0)
        for k in names:
            m = models[k]
            m.train()
            y_hat = m(x, perturb=True)
            l_w = F.l1_loss(y_hat, y)
            l_s = spec_loss(y, y_hat)
            loss = l_w + arms[k] * l_s
            opts[k].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
            opts[k].step()
            scheds[k].step()
            if log:
                h = hist[k]
                h["step"].append(step); h["wave"].append(l_w.item()); h["spec"].append(l_s.item())
                h["lr"].append(scheds[k].get_last_lr()[0])
        if (step + 1) % ckpt_every == 0 or step + 1 == steps:
            line = f"step {step+1:>6}/{steps} [{time.time()-t0:.0f}s]"
            for k in names:
                m = models[k]
                s = snr_db(probe, reconstruct(m, probe, CFG))
                d = hb_deficit_of(m, probe, CFG)
                h = hist[k]
                h["dev_step"].append(step + 1); h["snr"].append(s); h["deficit"].append(d); h["snr_naive"].append(naive)
                torch.save({"model": m.state_dict(), "step": step + 1, "history": h, "lambda": arms[k],
                            "batch": batch, "seg": corpus.seg_hi, "tag": tag}, run_dir / f"{k}.pt")
                line += f" | {k}: w {h['wave'][-1]:.4f} SNR {s:5.2f} def {d:+6.2f}"
            print(line + f"  (naive {naive:.2f})", flush=True)
    return models, hist

def time_steps(corpus, n_arms, batch, n=30):
    '''Wall time per step for n_arms models on this batch shape.'''
    arms = {f"t{i}": 1e-3 for i in range(n_arms)}
    names = list(arms)
    models = {k: LISA(CFG).to(DEVICE) for k in names}
    opts = {k: torch.optim.Adam(models[k].parameters(), lr=1e-3) for k in names}
    spec_loss = MultiScaleSTFTLoss(CFG.n_fft).to(DEVICE)
    rng = stream("timing")
    for i in range(n + 5):
        if i == 5:
            torch.cuda.synchronize(); t0 = time.time()
        x, y = corpus.batch(rng, batch)
        for k in names:
            y_hat = models[k](x, perturb=True)
            loss = F.l1_loss(y_hat, y) + 1e-3 * spec_loss(y, y_hat)
            opts[k].zero_grad(set_to_none=True); loss.backward(); opts[k].step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    print(f"{n_arms} arm(s), batch {batch} x {corpus.seg_hi} samples: {dt*1000:.0f} ms/step, "
          f"peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    del models, opts
    torch.cuda.empty_cache()
    return dt


# ---- architecture arm: Fourier-feature coordinate encoding (Tancik et al. 2020) ------------------
# Identical encoder, identical latents, identical loss and batches.  Only the decoder's view of the
# relative coordinate changes: c -> [c, sin(pi k c), cos(pi k c)]_{k=1..K}.  If this arm emits more
# high band than the ReLU arm at the same lambda, over-smoothing is partly spectral bias, not loss.
class LISAFF(LISA):
    def __init__(self, cfg, n_freq=6):
        super().__init__(cfg)
        self.n_freq = n_freq
        latent = self.enc.dim
        layers, d = [], (1 + 2 * n_freq) + 3 * latent
        for _ in range(cfg.dec_layers - 1):
            layers += [nn.Linear(d, cfg.dec_hidden), nn.ReLU()]
            d = cfg.dec_hidden
        layers.append(nn.Linear(d, 1))
        self.dec = LISADecoder.__new__(LISADecoder)
        nn.Module.__init__(self.dec)
        self.dec.net = nn.Sequential(*layers)
        self.register_buffer("freqs", math.pi * torch.arange(1, n_freq + 1, dtype=torch.float32))

    def forward(self, x_lo, j0=0, j1=None, perturb=False):
        B, L_lo = x_lo.shape
        j1 = L_lo * self.R if j1 is None else j1
        z = self.enc(x_lo).transpose(1, 2)
        q = (torch.arange(j0, j1, device=x_lo.device, dtype=torch.float32) / self.R)
        q = q.unsqueeze(0).expand(B, -1)
        anchor = q + torch.randn_like(q) * 0.5 if perturb else q
        idx = torch.floor(anchor).long().clamp(0, L_lo - 1)
        coord = (2.0 * (q - idx.float()) - 1.0).unsqueeze(-1)
        ang = coord * self.freqs
        feats = torch.cat([coord, torch.sin(ang), torch.cos(ang)], -1)

        def take(ii):
            ii = ii.clamp(0, L_lo - 1).unsqueeze(-1).expand(-1, -1, z.shape[-1])
            return torch.gather(z, 1, ii)

        return self.dec(torch.cat([feats, take(idx - 1), take(idx), take(idx + 1)], -1))


def make_arm(kind, cfg):
    return LISAFF(cfg) if kind == "ff" else LISA(cfg)


def train_arms(corpus, arms, steps, batch, lr, milestones, gamma, clip, ckpt_every, tag, probe, log_every=25):
    '''arms: {name: (kind, lambda_spec)}, kind in {"relu", "ff"}.  ReLU arms share one init.'''
    names = list(arms)
    base_relu = LISA(CFG).to(DEVICE)
    models = {}
    for k in names:
        kind, lam = arms[k]
        models[k] = copy.deepcopy(base_relu) if kind == "relu" else LISAFF(CFG).to(DEVICE)
    opts = {k: torch.optim.Adam(models[k].parameters(), lr=lr) for k in names}
    scheds = {k: MultiStepLR(opts[k], [int(f * steps) for f in milestones], gamma) for k in names}
    spec_loss = MultiScaleSTFTLoss(CFG.n_fft).to(DEVICE)
    hist = {k: {"step": [], "wave": [], "spec": [], "lr": [], "dev_step": [], "snr": [], "deficit": [], "snr_naive": []}
            for k in names}
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
            y_hat = m(x, perturb=True)
            l_w = F.l1_loss(y_hat, y)
            l_s = spec_loss(y, y_hat)
            loss = l_w + lam * l_s
            opts[k].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
            opts[k].step()
            scheds[k].step()
            if log:
                h = hist[k]
                h["step"].append(step); h["wave"].append(l_w.item()); h["spec"].append(l_s.item())
                h["lr"].append(scheds[k].get_last_lr()[0])
        if (step + 1) % ckpt_every == 0 or step + 1 == steps:
            line = f"step {step+1:>6}/{steps} [{time.time()-t0:.0f}s]"
            for k in names:
                m = models[k]
                s = snr_db(probe, reconstruct(m, probe, CFG))
                d = hb_deficit_of(m, probe, CFG)
                h = hist[k]
                h["dev_step"].append(step + 1); h["snr"].append(s); h["deficit"].append(d); h["snr_naive"].append(naive)
                save_ckpt({"model": m.state_dict(), "step": step + 1, "history": h, "arm": arms[k],
                           "batch": batch, "seg": corpus.seg_hi, "tag": tag}, run_dir / f"{k}.pt")
                line += f" | {k}: w {h['wave'][-1]:.4f} SNR {s:5.2f} def {d:+6.2f}"
            line += f"  (naive {naive:.2f})"
            print(line, flush=True)
            logf.write(line + chr(10)); logf.flush()
    logf.close()
    return models, hist


def save_ckpt(state, path, retries=4):
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
