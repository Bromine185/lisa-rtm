# ============================================================ OV3-2b stacked-arm trainer (GPU-efficient drop-in for e2)
# Same arms, same objectives, same history keys / checkpoint dict / log lines / plot as e2_trainer.py, but the
# arms are trained as STACKS instead of one after another.  Why: 7 arms at batch 32 x 1 s cost 587 ms/step on an
# A100-80GB, ~84 ms per arm-step, the same per-arm cost as batch 64 on 4 Sep -- launch/sync bound (sequential
# arms, ~28 .item() syncs per step, three gathers, fp32), while the arithmetic is ~0.7 TFLOP per arm-step.
#
# What changes (every item behind a flag, read from globals() before this file is exec'd):
#   STACK      arms of one class and one sequence count share ONE forward/backward: parameters carry a leading
#              arm axis, the encoder is one grouped conv per layer (groups = arms), the decoder MLP one batched
#              matmul per layer (baddbmm over the arm axis).  Losses are independent per arm, so backward on
#              their sum gives every arm its own gradient; one Adam over the stacked tensors IS one Adam per arm
#              (elementwise); gradient clipping is per arm (norm along the arm axis).  This is what
#              torch.func.vmap(functional_call) lowers to, written out so compile / streams / graphs see plain ops.
#   AMP        bf16 autocast on encoder + decoder; every distance / STFT in fp32 (outputs cast before the losses).
#   TF32       TF32 matmul/conv (the 4 Sep setting).
#   COMPILE    torch.compile on the decoder MLP call (try/except fallback to eager).
#   PIN        pinned-memory, non_blocking batches (same RNG consumption as HostCorpus.batch -> same batches).
#   FUSED      torch.optim.Adam(fused=True).
#   STREAMS    LISAS and LISASD groups on two CUDA streams; both synced to the default stream before the step.
#   CUDA_GRAPHS EXPERIMENTAL, default False: capture forward+backward+clip of the fixed-shape step into a CUDA
#              graph (optimiser step stays eager).  Untested on the day it was written.
#   GROUP_MAX  chunk a stack into at most this many arms (bounds activation memory).
#   index gather: base index precomputed once per (L, R); jitter added on the GPU; the three neighbours come from
#              ONE gather on a replicate-padded (A, S, L+2, C) latent.
#   STFT work: target features once per step, shared by every arm; both draws of every arm in one STFT call.
# Memory (activations saved for backward, decoder dominates): per arm and per output sequence of N = R*L
# samples ~ N * (97 + 4*144) * bytes, i.e. at batch 32 x 1 s (two draws = 64 sequences) ~8.3 GB fp32 / 4.2 GB bf16
# per arm.  A 5-arm LISAS stack at batch 32 bf16 ~21 GB; batch 64 ~42 GB; batch 128 needs GROUP_MAX 2 (~33 GB
# per chunk).  The step is no longer overhead-bound once stacked, so the batch only needs to fill the card.
# Requires in the kernel: c0 boot, c1_model, e1_model, e2_trainer (val_batches, plot_curves, _nan, _f).
import copy, math, time, json, contextlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

assert "plot_curves" in globals() and "val_batches" in globals(), "exec overnight3/e2_trainer.py before e2b_fast.py"
_CUDA = torch.cuda.is_available() and DEVICE.type == "cuda"
FAST = {"STACK": True, "AMP": _CUDA, "TF32": _CUDA, "COMPILE": _CUDA, "PIN": _CUDA, "FUSED": _CUDA,
        "STREAMS": _CUDA, "CUDA_GRAPHS": False, "GROUP_MAX": None}
for _k in list(FAST):
    if _k in globals():
        FAST[_k] = globals()[_k]
if FAST["TF32"] and _CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
_OFF3 = None
_IDX_CACHE = {}


def _autocast():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (FAST["AMP"] and _CUDA) else contextlib.nullcontext()


def _base_index(L, R, device):
    '''q = j / R for the full sequence and floor(q), computed once per (L, R, device).'''
    key = (int(L), int(R), str(device))
    if key not in _IDX_CACHE:
        q = torch.arange(L * R, device=device, dtype=torch.float32) / R
        _IDX_CACHE[key] = (q, torch.floor(q).long().clamp(0, L - 1))
    return _IDX_CACHE[key]


def _mlp_eager(X, Ws, bs, relus):
    for W, b, r in zip(Ws, bs, relus):
        X = torch.baddbmm(b.unsqueeze(1), X, W.transpose(1, 2))
        if r:
            X = F.relu(X)
    return X


_MLP = {"fn": _mlp_eager, "compiled": False}
if FAST["COMPILE"]:
    try:
        _MLP = {"fn": torch.compile(_mlp_eager, dynamic=False), "compiled": True}
    except Exception as _e:                                            # no compiler: eager
        print("torch.compile unavailable, eager MLP:", repr(_e), flush=True)


def _mlp(X, Ws, bs, relus):
    if _MLP["compiled"]:
        try:
            return _MLP["fn"](X, Ws, bs, relus)
        except Exception as e:                                          # backend failed at first call: eager
            print("torch.compile failed, falling back to eager MLP:", repr(e)[:200], flush=True)
            _MLP["fn"], _MLP["compiled"] = _mlp_eager, False
    return _mlp_eager(X, Ws, bs, relus)


_OOM = torch.cuda.OutOfMemoryError if _CUDA else MemoryError


def _pkey(k):
    return k.replace(".", "__")


class ArmStack:
    '''A arms of one class (LISAS / LISASD) and one sequence count (det: B, es: 2B), parameters stacked along a
    leading arm axis.  Exact per-arm semantics of arm_loss3: same estimator, same distances, same clipping.'''
    def __init__(self, names, specs, base, cls_name, device, stream=None):
        self.names, self.specs, self.cls_name, self.device = list(names), specs, cls_name, device
        self.A = len(self.names)
        self.kinds = [specs[k][0] for k in self.names]
        self.lams = torch.tensor([float(specs[k][1]) for k in self.names], device=device)
        self.is_det = all(k.startswith("det") for k in self.kinds)
        assert self.is_det or not any(k.startswith("det") for k in self.kinds), "mixed det/es stack"
        self.R, self.C, self.n_noise, self.n_dec = base.R, base.dim, base.n_noise, getattr(base, "n_dec", 0)
        sd = base.state_dict()
        self.keys = list(sd)
        self.P = nn.ParameterDict({_pkey(k): nn.Parameter(v.detach().to(device).unsqueeze(0).repeat(self.A, *([1] * v.dim())).clone())
                                   for k, v in sd.items()})
        mods = list(base.enc)
        self.enc_plan = [(f"enc.{i}.weight", f"enc.{i}.bias", m.kernel_size[0], i + 1 < len(mods) and isinstance(mods[i + 1], nn.ReLU))
                         for i, m in enumerate(mods) if isinstance(m, nn.Conv1d)]
        mods = list(base.dec.net)
        self.dec_plan = [(f"dec.net.{i}.weight", f"dec.net.{i}.bias", i + 1 < len(mods) and isinstance(mods[i + 1], nn.ReLU))
                         for i, m in enumerate(mods) if isinstance(m, nn.Linear)]
        self.relus = tuple(r for _, _, r in self.dec_plan)
        # per-arm loss recipe (weights on the three spectral geometries; low-band waveform term; es_wave)
        w_lm, w_erb, w_l2, split = [], [], [], []
        for kd in self.kinds:
            w_lm.append(1.0 if kd in ("es_marg", "es_split_marg") else 0.5 if kd in ("es_marg_erb", "es_split_marg_erb") else 0.0)
            w_erb.append(1.0 if kd == "es_erb" else 0.5 if kd in ("es_marg_erb", "es_split_marg_erb") else 0.0)
            w_l2.append(1.0 if kd == "es_ged" else 0.0)
            split.append(kd in ("det_split", "es_split_marg", "es_split_marg_erb"))
        t = lambda v: torch.tensor(v, device=device, dtype=torch.float32)
        self.w_lm, self.w_erb, self.w_l2 = t(w_lm).unsqueeze(1), t(w_erb).unsqueeze(1), t(w_l2).unsqueeze(1)
        self.split = torch.tensor(split, device=device).unsqueeze(1)
        self.need_lm, self.need_erb, self.need_l2 = any(w_lm), any(w_erb), any(w_l2)
        self.need_split, self.need_full = any(split), not all(split)
        self.stream = stream
        self.ema = None

    def p(self, k):
        return self.P[_pkey(k)]

    def params(self):
        return list(self.P.values())

    # ---- forward: (S, L) input shared by the arms -> (A, S, N) outputs ------------------------------------
    def forward(self, x_in, eps_enc=None, eps_dec=None, perturb=False, jitter=None):
        A, (S, L) = self.A, x_in.shape
        R, C, N = self.R, self.C, x_in.shape[1] * self.R
        if eps_enc is None:
            eps_enc = x_in.new_zeros(A, S, self.n_noise, L)
        h = torch.cat([x_in.unsqueeze(0).expand(A, S, L).unsqueeze(2), eps_enc], 2)       # (A, S, 1+n, L)
        h = h.transpose(0, 1).reshape(S, A * (1 + self.n_noise), L)
        with _autocast():
            for wk, bk, k, relu in self.enc_plan:
                W, b = self.p(wk), self.p(bk)
                h = F.conv1d(h, W.reshape(-1, W.shape[2], W.shape[3]), b.reshape(-1), padding=k // 2, groups=A)
                if relu:
                    h = F.relu(h)
            z = h.view(S, A, C, L).permute(1, 0, 3, 2)                                     # (A, S, L, C)
            zp = torch.cat([z[:, :, :1], z, z[:, :, -1:]], 2)                                # replicate pad: (A, S, L+2, C)
            q, idx0 = _base_index(L, R, x_in.device)
            if perturb:
                jit = torch.randn(A, S, N, device=x_in.device) * 0.5 if jitter is None else jitter
                idx = torch.floor(q + jit).long().clamp_(0, L - 1)                         # (A, S, N)
            else:
                idx = idx0.view(1, 1, N).expand(A, S, N)
            coord = (2.0 * (q - idx.float()) - 1.0).unsqueeze(-1)                          # (A, S, N, 1)
            global _OFF3
            if _OFF3 is None or _OFF3.device != x_in.device:
                _OFF3 = torch.arange(3, device=x_in.device)
            gidx = (idx.unsqueeze(-1) + _OFF3).reshape(A, S, 3 * N)                        # padded rows i-1, i, i+1
            g = torch.gather(zp, 2, gidx.unsqueeze(-1).expand(A, S, 3 * N, C)).reshape(A, S, N, 3 * C)
            parts = [coord.to(g.dtype), g]
            if self.n_dec:
                parts.append((eps_dec if eps_dec is not None else g.new_zeros(A, S, N, self.n_dec)).to(g.dtype))
            X = torch.cat(parts, -1).reshape(A, S * N, -1)
            X = _mlp(X, [self.p(w) for w, _, _ in self.dec_plan], [self.p(b) for _, b, _ in self.dec_plan], self.relus)
        return X.reshape(A, S, N).float()

    def sample_eps(self, S, L, gen=None):
        A, dev = self.A, self.device
        e = torch.randn(A, S, self.n_noise, L, device=dev, generator=gen)
        d = torch.randn(A, S, L * self.R, self.n_dec, device=dev, generator=gen) if self.n_dec else None
        return e, d

    # ---- losses: (loss, wave, spec, spread) each (A,) --------------------------------------------------------
    def losses(self, x, y, tf, eps=None, perturb=True):
        B, T = y.shape
        R = self.R
        if self.is_det:
            yh = self.forward(x, None, None, perturb)                                       # (A, B, T)
            l_w = (yh - y).abs().mean((1, 2))
            if self.need_split:
                l_w = torch.where(self.split[:, 0], (lowpass(yh, R) - tf.y_lo).abs().mean((1, 2)), l_w)
            l_s = 0.0
            for s, (n, h) in enumerate(SCALES):
                H = torch.stft(yh.reshape(self.A * B, T), n, h, window=_win(n, y.device), return_complex=True).abs()
                H = H.view(self.A, B, H.shape[1], H.shape[2])
                Y = tf.mag[s]
                sc = (Y - H).pow(2).sum((1, 2, 3)).sqrt() / (tf.mag_norm[s] + 1e-8)
                lm = (torch.log(H + 1e-7) - tf.lm[s]).abs().mean((1, 2, 3))
                l_s = l_s + sc + lm
            l_s = l_s / len(SCALES)
            loss = l_w + self.lams * l_s
            return loss, l_w, l_s, torch.zeros_like(l_w)
        e, d = self.sample_eps(2 * B, x.shape[1]) if eps is None else eps
        yh = self.forward(x.repeat(2, 1), e, d, perturb)                                    # (A, 2B, T)
        y1, y2 = yh[:, :B], yh[:, B:]
        dwf = lambda a, b: (a - b).abs().mean(-1)                                           # (A, B)
        spread = dwf(y1, y2)
        dw = 0.5 * (dwf(y, y1) + dwf(y, y2)) - 0.5 * spread if self.need_full else None
        if self.need_split:
            y1l, y2l = lowpass(y1, R), lowpass(y2, R)
            dws = 0.5 * (dwf(tf.y_lo, y1l) + dwf(tf.y_lo, y2l)) - 0.5 * dwf(y1l, y2l)
            dw = dws if dw is None else torch.where(self.split, dws, dw)
        spec = torch.zeros_like(dw)
        if self.need_lm or self.need_erb or self.need_l2:
            for s, (n, h) in enumerate(SCALES):
                S_ = torch.stft(yh.reshape(self.A * 2 * B, T), n, h, window=_win(n, y.device), return_complex=True).abs()
                S_ = S_.view(self.A, 2 * B, S_.shape[1], S_.shape[2])
                if self.need_lm or self.need_l2:
                    Lm = torch.log(S_ + 1e-7); L1, L2 = Lm[:, :B], Lm[:, B:]; Ly = tf.lm[s]
                    if self.need_lm:
                        dl = lambda a, b: (a - b).abs().mean((-2, -1))
                        spec = spec + self.w_lm * (0.5 * (dl(Ly, L1) + dl(Ly, L2)) - 0.5 * dl(L1, L2)) / len(SCALES)
                    if self.need_l2:
                        sq = math.sqrt(Lm.shape[2] * Lm.shape[3])
                        d2 = lambda a, b: (a - b).flatten(-2).norm(dim=-1) / sq
                        spec = spec + self.w_l2 * (0.5 * (d2(Ly, L1) + d2(Ly, L2)) - 0.5 * d2(L1, L2)) / len(SCALES)
                if self.need_erb:
                    W = erb_filterbank(n, CFG.fs_hi, device=y.device)
                    E = 0.5 * torch.log(torch.matmul(W, S_.pow(2)) + 1e-8); E1, E2 = E[:, :B], E[:, B:]; Ey = tf.erb[s]
                    de = lambda a, b: (a - b).abs().mean((-2, -1))
                    spec = spec + self.w_erb * (0.5 * (de(Ey, E1) + de(Ey, E2)) - 0.5 * de(E1, E2)) / len(SCALES)
        loss = (dw + self.lams.unsqueeze(1) * spec).mean(1)
        return loss, dw.mean(1), spec.mean(1), spread.mean(1)

    # ---- per-arm gradient clipping (clip_grad_norm_ semantics, one norm per arm) ------------------------------
    def clip_(self, clip):
        grads = [p.grad for p in self.P.values() if p.grad is not None]
        if not grads:
            return
        sq = torch.zeros(self.A, device=self.device)
        for g in grads:
            sq = sq + g.reshape(self.A, -1).pow(2).sum(1)
        coef = (clip / (sq.sqrt() + 1e-6)).clamp(max=1.0)
        for g in grads:
            g.mul_(coef.view(self.A, *([1] * (g.dim() - 1))))

    # ---- export one arm into a plain module (checkpoints, probe, reconstruct) ----------------------------------
    @torch.no_grad()
    def export(self, a, module):
        sd = module.state_dict()
        for k in self.keys:
            sd[k].copy_(self.p(k)[a])
        return module


class TargetFeats:
    '''Per-step features of the target batch y, computed once and shared by every arm of every stack.'''
    def __init__(self, y, R, need_lm, need_erb, need_mag, need_split):
        self.lm, self.erb, self.mag, self.mag_norm = [], [], [], []
        self.y_lo = lowpass(y, R) if need_split else None
        for n, h in SCALES:
            S_ = torch.stft(y, n, h, window=_win(n, y.device), return_complex=True).abs() if (need_lm or need_erb or need_mag) else None
            self.mag.append(S_ if need_mag else None)
            self.mag_norm.append(S_.pow(2).sum().sqrt() if need_mag else None)
            self.lm.append(torch.log(S_ + 1e-7) if (need_lm or need_mag) else None)
            self.erb.append(0.5 * torch.log(torch.matmul(erb_filterbank(n, CFG.fs_hi, device=y.device), S_.pow(2)) + 1e-8) if need_erb else None)


def build_stacks(names, specs, base, device):
    '''Group arms by (class, det/es) in the given order, chunk by GROUP_MAX, assign streams by class.'''
    order, groups = [], {}
    for k in names:
        kind, lam, cls = specs[k]
        groups.setdefault((cls, kind.startswith("det")), []).append(k)
    stacks = []
    streams = {}
    for (cls, det), ks in groups.items():
        if FAST["STREAMS"] and _CUDA:
            streams.setdefault(cls, torch.cuda.Stream())
        gm = FAST["GROUP_MAX"] or len(ks)
        for i in range(0, len(ks), gm):
            stacks.append(ArmStack(ks[i:i + gm], specs, base[cls], cls, device, stream=streams.get(cls)))
    return stacks


def _needs(stacks):
    return (any(s.need_lm for s in stacks), any(s.need_erb for s in stacks), any(s.is_det for s in stacks), any(s.need_split for s in stacks))


def batch_pinned(corpus, rng, B):
    '''HostCorpus.batch with the same RNG consumption (same batches as the sequential trainer), pinned + non_blocking.'''
    if not (FAST["PIN"] and _CUDA):
        return corpus.batch(rng, B)
    idx = rng.integers(corpus.n, size=B)
    n_pos = (corpus.lens[idx] - corpus.seg_hi) // corpus.R + 1
    starts = (rng.random(B) * n_pos).astype(np.int64) * corpus.R
    hi0 = corpus.off[idx] + starts
    y = corpus.Y[hi0[:, None] + corpus._ar_hi]
    x = corpus.X[(hi0 // corpus.R)[:, None] + corpus._ar_lo]
    return (torch.from_numpy(x).pin_memory().to(DEVICE, non_blocking=True),
            torch.from_numpy(y).pin_memory().to(DEVICE, non_blocking=True))


def _fwd_bwd(stacks, x, y, clip):
    '''One step's forward + backward + per-arm clip for every stack; returns the (A,4) term tensors in stack order.
    Streams: each stack runs on its stream after waiting for the default stream; the default stream waits for
    all of them before the optimiser touches the gradients.'''
    need = _needs(stacks)
    tf = TargetFeats(y, stacks[0].R, *need)
    cur = torch.cuda.current_stream() if _CUDA else None
    terms = []
    for s in stacks:
        ctx = torch.cuda.stream(s.stream) if (s.stream is not None) else contextlib.nullcontext()
        if s.stream is not None:
            s.stream.wait_stream(cur)
            for t_ in (x, y):
                t_.record_stream(s.stream)
        with ctx:
            loss, wave, spec, spread = s.losses(x, y, tf, perturb=True)
            loss.sum().backward()
            s.clip_(clip)
            terms.append(torch.stack([loss.detach(), wave.detach(), spec.detach(), spread.detach()], 1))
    if _CUDA:
        for s in stacks:
            if s.stream is not None:
                cur.wait_stream(s.stream)
    return terms


class _GraphedStep:
    '''EXPERIMENTAL (CUDA_GRAPHS=True): capture forward+backward+clip into one CUDA graph on static buffers.
    The optimiser step stays eager.  Gradients are static (never set_to_none between replays).'''
    def __init__(self, stacks, x, y, clip):
        self.stacks, self.clip = stacks, clip
        self.x, self.y = x.clone(), y.clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):                                            # warm-up (allocator, autotune)
                for st in stacks:
                    for p in st.params():
                        p.grad = None
                _fwd_bwd(stacks, self.x, self.y, clip)
        torch.cuda.current_stream().wait_stream(s)
        for st in stacks:
            for p in st.params():
                p.grad = None
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.terms = _fwd_bwd(stacks, self.x, self.y, clip)

    def __call__(self, x, y):
        self.x.copy_(x); self.y.copy_(y)
        self.graph.replay()
        return self.terms


def _make_models(arms):
    names = list(arms)
    specs = {k: arm3(arms[k]) for k in names}
    base = {"LISAS": LISAS(CFG).to(DEVICE)}
    if any(specs[k][2] == "LISASD" for k in names):
        base["LISASD"] = copy_shared(base["LISAS"], LISASD(CFG).to(DEVICE))
    models = {k: copy.deepcopy(base[specs[k][2]]) for k in names}
    for k in names:
        models[k].tau = 0.0 if specs[k][0].startswith("det") else 1.0
    return names, specs, base, models


def _make_opts(stacks, lr, steps, milestones, gamma):
    opts, scheds = [], []
    for s in stacks:
        kw = {"fused": True} if (FAST["FUSED"] and _CUDA) else {}
        opts.append(torch.optim.Adam(s.params(), lr=lr, **kw))
        scheds.append(MultiStepLR(opts[-1], [int(f * steps) for f in milestones], gamma))
    return opts, scheds


@torch.no_grad()
def val_loss_fast(stacks, vb, seed=1234):
    '''Every arm's own objective on the fixed validation batches, stacked, no_grad (+autocast), eps from a fixed
    seed, no anchor jitter.  Returns {arm: (loss, wave, spec)} after ONE host transfer.'''
    need = _needs(stacks)
    acc = [torch.zeros(s.A, 3, device=s.device) for s in stacks]
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    for x, y in vb:
        tf = TargetFeats(y, stacks[0].R, *need)
        for i, s in enumerate(stacks):
            eps = None if s.is_det else s.sample_eps(2 * x.shape[0], x.shape[1], gen)
            loss, wave, spec, _ = s.losses(x, y, tf, eps=eps, perturb=False)
            acc[i] += torch.stack([loss, wave, spec], 1)
    out = {}
    vals = torch.cat(acc, 0).div_(len(vb)).cpu().tolist()
    i = 0
    for s in stacks:
        for k in s.names:
            out[k] = tuple(vals[i]); i += 1
    return out


def train_ov3_fast(corpus, val_corpus, arms, steps, batch, lr, milestones, gamma, clip, ckpt_every, tag, probe,
                   val_every=500, n_val_batches=8, log_every=25):
    '''Drop-in for train_ov3: same signature, history keys, checkpoints, log lines, plot.  One init per class
    (LISASD shares the LISAS weights via copy_shared), one batch stream, arms stacked per class.'''
    names, specs, base, models = _make_models(arms)
    stacks = build_stacks(names, specs, base, DEVICE)
    where = {k: (si, a) for si, s in enumerate(stacks) for a, k in enumerate(s.names)}
    opts, scheds = _make_opts(stacks, lr, steps, milestones, gamma)
    hist = {k: {"step": [], "wave": [], "spec": [], "spread": [], "lr": [],
                "val_step": [], "val_loss": [], "val_wave": [], "val_spec": [], "train_loss_ema": [],
                "dev_step": [], "snr0": [], "def0": [], "snr1": [], "def1": [], "snr_naive": []} for k in names}
    vb = val_batches(val_corpus, tag, n_val_batches, batch)
    rng = stream(f"{tag}/batches")
    run_dir = CKPT / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    probe = np.asarray(probe, np.float64)
    naive = snr_db(probe, naive_upsample(probe, CFG))
    logf = open(ROOT / f"train_{tag}.log", "a")
    hist_path = ROOT / f"ov3_history_{tag}.json"
    pending = []                                                             # (step, lr, terms tensor) -> host at val/ckpt
    graphed = None
    t0 = time.time()

    def flush():
        if not pending:
            return
        T = torch.stack([t for _, _, t in pending]).cpu().tolist()           # one transfer for all pending steps
        for (st, lr_, _), rows in zip(pending, T):
            for k in names:
                si, a = where[k]
                r = rows[si][a] if isinstance(rows[0][0], list) else rows[a]
                h = hist[k]
                h["step"].append(st); h["wave"].append(r[1]); h["spec"].append(r[2]); h["spread"].append(r[3]); h["lr"].append(lr_)
        pending.clear()

    for step in range(steps):
        x, y = batch_pinned(corpus, rng, batch)
        for s in stacks:
            for p in s.params():
                if graphed is None:
                    p.grad = None
        if FAST["CUDA_GRAPHS"] and _CUDA:
            if graphed is None:
                graphed = _GraphedStep(stacks, x, y, clip)
            terms = graphed(x, y)
        else:
            terms = _fwd_bwd(stacks, x, y, clip)
        for s, o, sc in zip(stacks, opts, scheds):
            o.step(); sc.step()
        for s, t in zip(stacks, terms):
            s.ema = t[:, 0].clone() if s.ema is None else s.ema * 0.98 + t[:, 0] * 0.02
        if step % log_every == 0:
            pending.append((step, scheds[0].get_last_lr()[0], torch.cat([t.reshape(-1, 4) for t in terms], 0)))
        if (step + 1) % val_every == 0 or step + 1 == steps:
            vals = val_loss_fast(stacks, vb)
            emas = torch.cat([s.ema for s in stacks]).cpu().tolist()
            vline = f"  val {step+1:>6}"
            i = 0
            for s in stacks:
                for k in s.names:
                    vl, vw, vs = vals[k]; h = hist[k]
                    h["val_step"].append(step + 1); h["val_loss"].append(float(vl)); h["val_wave"].append(float(vw))
                    h["val_spec"].append(float(vs)); h["train_loss_ema"].append(float(emas[i]))
                    vline += f" | {k}: train {emas[i]:.4f} val {vl:.4f}"
                    i += 1
            print(vline, flush=True); logf.write(vline + chr(10)); logf.flush()
        if (step + 1) % ckpt_every == 0 or step + 1 == steps:
            flush()
            line = f"step {step+1:>6}/{steps} [{time.time()-t0:.0f}s]"
            for k in names:
                kind, lam, cls_name = specs[k]
                si, a = where[k]
                m = stacks[si].export(a, models[k])
                pm = probe_metrics(m, probe, CFG, naive)
                h = hist[k]
                h["dev_step"].append(step + 1); h["snr_naive"].append(naive)
                h["snr0"].append(pm[0.0][0]); h["def0"].append(pm[0.0][1])
                s1, d1 = pm.get(1.0, (None, None))
                h["snr1"].append(s1); h["def1"].append(d1)
                save_ckpt({"model": m.state_dict(), "step": step + 1, "history": h, "arm": (kind, lam, cls_name),
                           "n_noise": m.n_noise, "n_dec": getattr(m, "n_dec", 0), "cls": cls_name,
                           "batch": batch, "seg": corpus.seg_hi, "tag": tag}, run_dir / f"{k}.pt")
                line += f" | {k}: w {h['wave'][-1]:.4f} s {h['spec'][-1]:.3f} SNR0 {pm[0.0][0]:5.2f} def0 {pm[0.0][1]:+6.2f}"
                if 1.0 in pm:
                    line += f" SNR1 {s1:5.2f} def1 {d1:+6.2f}"
            line += f"  (naive {naive:.2f})"
            print(line, flush=True)
            logf.write(line + chr(10)); logf.flush()
            try:
                json.dump({k: hist[k] for k in names}, open(hist_path, "w"))
            except OSError as e:
                print("history write failed:", repr(e), flush=True)
            try:
                plot_curves(hist, tag)
            except Exception as e:
                print("plot_curves failed:", repr(e), flush=True)
    flush()
    logf.close()
    for k in names:
        si, a = where[k]
        stacks[si].export(a, models[k])
    return models, hist


def time_ov3_fast(corpus, arms, batch, n=20, clip=1e-3):
    names, specs, base, _ = _make_models(arms)
    stacks = build_stacks(names, specs, base, DEVICE)
    opts, _ = _make_opts(stacks, 1e-3, 1000, (0.5,), 0.5)
    rng = stream("timing")
    if _CUDA:
        torch.cuda.reset_peak_memory_stats()
    sync = torch.cuda.synchronize if _CUDA else (lambda: None)
    for i in range(n + 3):
        if i == 3:
            sync(); t0 = time.time()
        x, y = batch_pinned(corpus, rng, batch)
        for s in stacks:
            for p in s.params():
                p.grad = None
        _fwd_bwd(stacks, x, y, clip)
        for o in opts:
            o.step()
    sync()
    dt = (time.time() - t0) / n
    mem = f", peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB" if _CUDA else ""
    flags = {k: v for k, v in FAST.items() if v}
    print(f"[fast] {len(names)} arm(s) in {len(stacks)} stack(s), batch {batch} x {corpus.seg_hi}: {dt*1000:.0f} ms/step{mem}  flags {flags}", flush=True)
    del stacks, opts
    if _CUDA:
        torch.cuda.empty_cache()
    return dt


def bench_fast(corpus, arms, batch, group_max=None, streams=None, n=10):
    '''Old (time_ov3, sequential arms) vs new (time_ov3_fast) at one batch size.  OOM is reported, not raised.'''
    keep = dict(FAST)
    if group_max is not None:
        FAST["GROUP_MAX"] = group_max
    if streams is not None:
        FAST["STREAMS"] = streams
    rows = []
    for label, fn in (("sequential (e2 time_ov3)", lambda: time_ov3(corpus, arms, batch, n=n) if "time_ov3" in globals() else float("nan")),
                      ("stacked (e2b time_ov3_fast)", lambda: time_ov3_fast(corpus, arms, batch, n=n))):
        try:
            if _CUDA:
                torch.cuda.reset_peak_memory_stats()
            dt = fn()
            mem = torch.cuda.max_memory_allocated() / 1e9 if _CUDA else float("nan")
            rows.append((label, dt * 1000, mem))
        except torch.cuda.OutOfMemoryError if _CUDA else MemoryError:
            rows.append((label, float("nan"), float("nan")))
            if _CUDA:
                torch.cuda.empty_cache()
    FAST.update(keep)
    print(f"| batch {batch} GROUP_MAX {FAST['GROUP_MAX'] if group_max is None else group_max} | ms/step | peak GB |", flush=True)
    for label, ms, mem in rows:
        print(f"| {label} | {'OOM' if math.isnan(ms) else f'{ms:.0f}'} | {'-' if math.isnan(mem) else f'{mem:.1f}'} |", flush=True)
    return rows


print("OV3 fast trainer defined (train_ov3_fast, time_ov3_fast, bench_fast, val_loss_fast, ArmStack).  flags:",
      {k: v for k, v in FAST.items()}, flush=True)
