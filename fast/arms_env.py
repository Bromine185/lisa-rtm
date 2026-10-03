# ============================================================ ENV arms: level conditional on the utterance
# Exec'd into the trainer namespace AFTER overnight3/e2b_fast.py (fast/train_env.py does this), so every
# name below -- LISAS, LISASD, ArmStack, CFG, DEVICE, SCALES, _win, erb_filterbank, copy_shared, _autocast,
# _base_index, _mlp, _OFF3, F, torch, nn, copy -- is the trainer's own.
#
# WHY.  The 26 Sep SOTA comparison (notes/2026-09-26-sota-diagnosis.md) found that es_dec_erb_l0.1 has the
# right high-band level on average (-0.7 dB over 240 utterances of the paper split) but not per utterance:
# its deficit correlates -0.6 with the truth's high-band share, so bright voices come out 6-10 dB under and
# dark ones 2-3 dB over.  Within an utterance it is +2 dB in mid-loud frames and -2 dB in quiet ones.  The
# model emits the corpus's typical level and the truth moves around it: regression to the mean, one level up,
# in loudness.  Two candidate causes, one arm each and one arm for both:
#
#   env   the objective.  d_erb averages |dlog E| over 32 ERB bands, of which ~10 sit above 6 kHz, and over
#         every frame; the high band's per-frame level is a tenth of the term.  d_env scores ONLY the high
#         band's log energy, per frame, in three sub-bands (6-9, 9-14, 14-24 kHz), at 5 ms hop.  It is an
#         L1 distance on a feature of the output, so the two-draw energy score on it is strictly proper for
#         the conditional law of that feature (Gneiting & Raftery 2007); no target-dependent weights, which
#         would break propriety.
#   ctx   the architecture.  The encoder sees 11 input samples = 0.9 ms; whether this frame is a bright /s/
#         is written in the low band's tilt and envelope over tens of ms.  LISASDW adds the residual dilated
#         stack of overnight2/c1b_wide.py (k = 3, dilations 2..64) on the 32-d latents: receptive field 263
#         input samples = 22 ms, +19,680 parameters.  ctx_out is zero-initialised, so LISASDW computes exactly LISASD
#         at initialisation; with the control's seed (fast/train_env.py) that is the control's step-0 function.
#
# PREDICTIONS, written 2026-09-27 before training (refutation criteria in the note):
#   P5  es_ctx_env: the per-utterance correlation between deficit and the truth's HB share goes from -0.6 to
#       above -0.2, and the loud-frame deficit on p236/p238 closes from -8/-6 dB to better than -3 dB.
#   P6  ctx alone moves the correlation more than env alone (the model cannot express what it cannot see).
#   P7  Neither arm changes the coherent fraction (stays < 0.03): context fixes the level, not the phase.
#
# ARMS (kind, lambda, class).  The control is es_dec_erb_l0.1: same lambda, same noise pathway, one thing
# changed per arm.  fast/train_env.py gives each arm OV50's batch stream (BATCH_TAG) and the control's seed,
# so arm and control share the batches and the step-0 function; the realised noise draws differ.
ARMS_ENV = {
    "es_env_l0.1":     ("es_marg_erb_env", 1e-1, "LISASD"),    # + d_env, same network
    "es_ctx_erb_l0.1": ("es_marg_erb",     1e-1, "LISASDW"),   # same objective, 22 ms context
    "es_ctx_env_l0.1": ("es_marg_erb_env", 1e-1, "LISASDW"),   # both
}
KIND_MAP = {"es_marg_erb_env": "es_marg_erb"}     # what ArmStack's weight table knows the env kinds as
ENV_W = 1.0                                        # weight of ES_env inside the spectral term (lm 0.5, erb 0.5, env 1.0)
ENV_NFFT, ENV_HOP = 1024, 256                      # 21 ms window, 5.3 ms hop at 48 kHz
ENV_EDGES = (1.0, 1.5, 7.0 / 3.0, 4.0)              # x (fs_lo / 2): 6, 9, 14, 24 kHz at FULL; preset-relative
CTX_DILATIONS = (2, 4, 8, 16, 32, 64)


class LISASDW(LISASD):
    '''LISASD + a residual dilated context stack on the latents (c1b_wide.py's LISASW, on the decoder-noise
    class).  encode(): LISASD.encode stores the decoder noise and returns the local latents (B, L, C); the
    stack runs on top of those.  ctx_out starts at zero, so the module equals LISASD exactly at init.'''
    def __init__(self, cfg, n_noise=N_NOISE, n_dec=N_DEC, dilations=CTX_DILATIONS):
        super().__init__(cfg, n_noise, n_dec)
        C = self.dim
        self.ctx = nn.ModuleList([nn.Conv1d(C, C, 3, padding=d, dilation=d) for d in dilations])
        self.ctx_out = nn.Conv1d(C, C, 1)
        nn.init.zeros_(self.ctx_out.weight); nn.init.zeros_(self.ctx_out.bias)
        self.rf = 11 + 2 * sum(dilations)

    def encode(self, x_lo, eps=None):
        z = super().encode(x_lo, eps).transpose(1, 2)                    # (B, C, L)
        h = z
        for conv in self.ctx:
            h = h + F.relu(conv(h))
        return (z + self.ctx_out(h)).transpose(1, 2)                     # (B, L, C)


def _env_bins(n, fs):
    cut = CFG.fs_lo / 2
    b = [int(round(e * cut * n / fs)) for e in ENV_EDGES]
    b[-1] = n // 2 + 1
    return b


def env_feats(mag):
    '''(..., F, T) STFT magnitudes at ENV_NFFT -> (..., 3, T) half-log energies of the three HB sub-bands.'''
    b = _env_bins(ENV_NFFT, CFG.fs_hi)
    P = mag.pow(2)
    E = torch.stack([P[..., a:c, :].sum(-2) for a, c in zip(b[:-1], b[1:])], -2)
    return 0.5 * torch.log(E + 1e-8)


def _env_of(w):
    '''(..., T) waveforms -> (..., 3, T') env features.'''
    lead = w.shape[:-1]
    S_ = torch.stft(w.reshape(-1, w.shape[-1]), ENV_NFFT, ENV_HOP, window=_win(ENV_NFFT, w.device), return_complex=True).abs()
    return env_feats(S_).reshape(*lead, 3, -1)


# Evaluation boots (audit/boot.py, sota/run_ours.py) exec this file for LISASDW alone and never load the
# stacked trainer, so the trainer half is defined only where ArmStack exists.
if "ArmStack" not in globals():
    ArmStack = None


class ArmStackEnv(ArmStack if ArmStack is not None else object):
    '''ArmStack with (a) the dilated context block in the stacked encoder when the base class has one and (b) the
    ES_env term for the *_env kinds.  Everything else -- the estimator, the other distances, the sub-pixel
    decoder, the per-arm clip -- is the parent's, untouched.'''
    def __init__(self, names, specs, base, cls_name, device, stream=None):
        real = {k: specs[k][0] for k in names}
        super().__init__(names, {k: (KIND_MAP.get(specs[k][0], specs[k][0]), specs[k][1], specs[k][2]) for k in names},
                         base, cls_name, device, stream)
        self.kinds = [real[k] for k in self.names]
        w_env = [ENV_W if kd.endswith("_env") else 0.0 for kd in self.kinds]
        self.w_env = torch.tensor(w_env, device=device, dtype=torch.float32).unsqueeze(1)
        self.need_env = any(w_env)
        self.ctx_plan = ([(f"ctx.{i}.weight", f"ctx.{i}.bias", int(m.dilation[0])) for i, m in enumerate(base.ctx)]
                         if hasattr(base, "ctx") else [])

    def forward(self, x_in, eps_enc=None, eps_dec=None, perturb=False, jitter=None):
        A, (S, L) = self.A, x_in.shape
        R, C, N = self.R, self.C, x_in.shape[1] * self.R
        if eps_enc is None:
            eps_enc = x_in.new_zeros(A, S, self.n_noise, L)
        h = torch.cat([x_in.unsqueeze(0).expand(A, S, L).unsqueeze(2), eps_enc], 2)
        h = h.transpose(0, 1).reshape(S, A * (1 + self.n_noise), L)
        with _autocast():
            for wk, bk, k, relu in self.enc_plan:
                W, b = self.p(wk), self.p(bk)
                h = F.conv1d(h, W.reshape(-1, W.shape[2], W.shape[3]), b.reshape(-1), padding=k // 2, groups=A)
                if relu:
                    h = F.relu(h)
            if self.ctx_plan:                                                              # the only addition
                hc = h
                for wk, bk, dil in self.ctx_plan:
                    W, b = self.p(wk), self.p(bk)
                    hc = hc + F.relu(F.conv1d(hc, W.reshape(-1, W.shape[2], W.shape[3]), b.reshape(-1),
                                              padding=dil, dilation=dil, groups=A))
                Wo, bo = self.p("ctx_out.weight"), self.p("ctx_out.bias")
                h = h + F.conv1d(hc, Wo.reshape(-1, Wo.shape[2], Wo.shape[3]), bo.reshape(-1), groups=A)
            z = h.view(S, A, C, L).permute(1, 0, 3, 2)
            zp = torch.cat([z[:, :, :1], z, z[:, :, -1:]], 2)
            q, idx0 = _base_index(L, R, x_in.device)
            Ws = [self.p(w) for w, _, _ in self.dec_plan]
            bs = [self.p(b) for _, b, _ in self.dec_plan]
            if FAST["SUBPIXEL"]:
                X = self._layer1(zp, Ws[0], bs[0], eps_dec, q, S, L, N, perturb, jitter)
                X = _mlp(X, Ws[1:], bs[1:], self.relus[1:])
            else:
                if perturb:
                    jit = torch.randn(A, S, N, device=x_in.device) * 0.5 if jitter is None else jitter
                    idx = torch.floor(q + jit).long().clamp_(0, L - 1)
                else:
                    idx = idx0.view(1, 1, N).expand(A, S, N)
                coord = (2.0 * (q - idx.float()) - 1.0).unsqueeze(-1)
                global _OFF3
                if _OFF3 is None or _OFF3.device != x_in.device:
                    _OFF3 = torch.arange(3, device=x_in.device)
                gidx = (idx.unsqueeze(-1) + _OFF3).reshape(A, S, 3 * N)
                g = torch.gather(zp, 2, gidx.unsqueeze(-1).expand(A, S, 3 * N, C)).reshape(A, S, N, 3 * C)
                parts = [coord.to(g.dtype), g]
                if self.n_dec:
                    parts.append((eps_dec if eps_dec is not None else g.new_zeros(A, S, N, self.n_dec)).to(g.dtype))
                X = torch.cat(parts, -1).reshape(A, S * N, -1)
                X = _mlp(X, Ws, bs, self.relus)
        return X.reshape(A, S, N).float()

    def losses(self, x, y, tf, eps=None, perturb=True):
        if self.is_det or not self.need_env:
            return super().losses(x, y, tf, eps, perturb)
        B, T = y.shape
        e, d = self.sample_eps(2 * B, x.shape[1]) if eps is None else eps
        yh = self.forward(x.repeat(2, 1), e, d, perturb)                                    # (A, 2B, T)
        y1, y2 = yh[:, :B], yh[:, B:]
        dwf = lambda a, b: (a - b).abs().mean(-1)
        spread = dwf(y1, y2)
        dw = 0.5 * (dwf(y, y1) + dwf(y, y2)) - 0.5 * spread
        spec = torch.zeros_like(dw)
        for s, (n, h) in enumerate(SCALES):                                                # parent's lm + erb, verbatim
            S_ = torch.stft(yh.reshape(self.A * 2 * B, T), n, h, window=_win(n, y.device), return_complex=True).abs()
            S_ = S_.view(self.A, 2 * B, S_.shape[1], S_.shape[2])
            if self.need_lm:
                Lm = torch.log(S_ + 1e-7); L1, L2 = Lm[:, :B], Lm[:, B:]; Ly = tf.lm[s]
                dl = lambda a, b: (a - b).abs().mean((-2, -1))
                spec = spec + self.w_lm * (0.5 * (dl(Ly, L1) + dl(Ly, L2)) - 0.5 * dl(L1, L2)) / len(SCALES)
            if self.need_erb:
                W = erb_filterbank(n, CFG.fs_hi, device=y.device)
                E = 0.5 * torch.log(torch.matmul(W, S_.pow(2)) + 1e-8); E1, E2 = E[:, :B], E[:, B:]; Ey = tf.erb[s]
                de = lambda a, b: (a - b).abs().mean((-2, -1))
                spec = spec + self.w_erb * (0.5 * (de(Ey, E1) + de(Ey, E2)) - 0.5 * de(E1, E2)) / len(SCALES)
        # the env term: HB sub-band log energies per frame, energy score, one scale
        env = _env_of(yh); V1, V2 = env[:, :B], env[:, B:]
        Vy = _env_of(y).unsqueeze(0)                                                       # (1, B, 3, T')
        dv = lambda a, b: (a - b).abs().mean((-2, -1))
        spec = spec + self.w_env * (0.5 * (dv(Vy, V1) + dv(Vy, V2)) - 0.5 * dv(V1, V2))
        loss = (dw + self.lams.unsqueeze(1) * spec).mean(1)
        return loss, dw.mean(1), spec.mean(1), spread.mean(1)


def make_models_env(arms):
    '''_make_models with the LISASDW base added; LISASD and LISASDW both start as copies of the LISAS init.'''
    names = list(arms)
    specs = {k: tuple(arms[k]) for k in names}
    base = {"LISAS": LISAS(CFG).to(DEVICE)}
    if any(specs[k][2] == "LISASD" for k in names):
        base["LISASD"] = copy_shared(base["LISAS"], LISASD(CFG).to(DEVICE))
    if any(specs[k][2] == "LISASDW" for k in names):
        base["LISASDW"] = copy_shared(base["LISAS"], LISASDW(CFG).to(DEVICE))
    models = {k: copy.deepcopy(base[specs[k][2]]) for k in names}
    for k in names:
        models[k].tau = 0.0 if specs[k][0].startswith("det") else 1.0
    return names, specs, base, models


def build_stacks_env(names, specs, base, device):
    '''One arm per stack (the measured-fastest configuration), no streams.'''
    return [ArmStackEnv([k], specs, base[specs[k][2]], specs[k][2], device) for k in names]


if ArmStack is None:
    ArmStackEnv = make_models_env = build_stacks_env = None    # not usable outside the trainer namespace
_m = LISASDW(CFG)
print(f"ENV arms defined: {list(ARMS_ENV)}.  LISASDW {sum(p.numel() for p in _m.parameters()):,} params "
      f"(LISASD {sum(p.numel() for p in LISASD(CFG).parameters()):,}), context {_m.rf} input samples = "
      f"{1000 * _m.rf / CFG.fs_lo:.1f} ms; env bins at {ENV_NFFT}: {_env_bins(ENV_NFFT, CFG.fs_hi)}", flush=True)
del _m
