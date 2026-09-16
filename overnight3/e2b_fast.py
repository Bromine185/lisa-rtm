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
#              DEFAULT OFF: measured as no-effect-or-worse, and off removes the record_stream/wait_stream
#              bookkeeping in _fwd_bwd.
#   CUDA_GRAPHS EXPERIMENTAL, default False: capture forward+backward+clip of the fixed-shape step into a CUDA
#              graph (optimiser step stays eager).  Untested on the day it was written.
#   GROUP_MAX  chunk a stack into at most this many arms (bounds activation memory).  DEFAULT 1: measured, one
#              arm per stack is both the FASTEST and the smallest row of the table (16.6 / 16.4 audio-s per
#              compute-s at batch 16 / 32, against 13.5 / 13.6 at two per stack and 14.3 / 14.4 at seven), and
#              per-group N = 144 grouped GEMMs tile-quantise badly on an A100, so stacking arms buys nothing.
#   DETERMINISTIC  default False.  See the flag block below: 587 vs 2261 ms/step, measured.
#   SUBPIXEL   default True, EXACT: decoder layer 1 evaluated at the 12 kHz input rate as one grouped conv1d
#              over the latents plus a rank-1 coordinate term (ArmStack._layer1).  Anchor jitter stays per
#              output sample and stays bit-for-bit the gather path's; set False for the A/B.
#   JITTER_PER_CELL  default False, CHANGES MATH: one anchor draw per INPUT cell instead of per output sample.
#              Do not switch this on inside the seven-arm comparison; run it as an eighth paired arm.
#   index gather: base index precomputed once per (L, R); jitter added on the GPU; the three neighbours come from
#              ONE gather on a replicate-padded (A, S, L+2, C) latent (SUBPIXEL=False path only).
#   STFT work: target features once per step, shared by every arm; both draws of every arm in one STFT call.
# Memory (activations saved for backward, decoder dominates): per arm and per output sequence of N = R*L
# samples ~ N * (97 + 4*144) * bytes, i.e. at batch 32 x 1 s (two draws = 64 sequences) ~8.3 GB fp32 / 4.2 GB bf16
# per arm.  A 5-arm LISAS stack at batch 32 bf16 ~21 GB; batch 64 ~42 GB; batch 128 needs GROUP_MAX 2 (~33 GB
# per chunk).  The step is no longer overhead-bound once stacked, so the batch only needs to fill the card.
# NOT gradient checkpointing, deliberately.  Saved activations are ~1.35 kB bf16 per output row (4 hidden x 144
# plus the 97-wide input), which reproduces the measured 8.1 / 15.7 / 30.8 GB at 1 / 2 / 7 arms per stack.
# torch.utils.checkpoint.checkpoint(_mlp_eager, ..., use_reentrant=False, preserve_rng_state=False) would drop
# ~6x of that -- preserve_rng_state=False is safe here only because the MLP block draws no randomness, jitter
# and noise being sampled outside it -- but it costs one extra forward, i.e. +20-40% on a bandwidth-bound step.
# Throughput is FLAT in batch in every measured row, so the headroom buys nothing today, and GROUP_MAX = 1 plus
# SUBPIXEL already cut memory 2-4x.  Revisit ONLY if a re-measured roofline shows the step compute-bound AND a
# larger batch is shown to raise audio-seconds per compute-second.
# Requires in the kernel: c0 boot, c1_model, e1_model, e2_trainer (val_batches, plot_curves, _nan, _f).
import copy, math, time, json, contextlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

assert "plot_curves" in globals() and "val_batches" in globals(), "exec overnight3/e2_trainer.py before e2b_fast.py"
_CUDA = torch.cuda.is_available() and DEVICE.type == "cuda"
FAST = {"STACK": True, "AMP": _CUDA, "TF32": _CUDA, "COMPILE": _CUDA, "PIN": _CUDA, "FUSED": _CUDA,
        "STREAMS": False, "CUDA_GRAPHS": False, "GROUP_MAX": 1, "DETERMINISTIC": False,
        "COMPILE_MODE": "max-autotune", "SUBPIXEL": True, "JITTER_PER_CELL": False}
for _k in list(FAST):
    if _k in globals():
        FAST[_k] = globals()[_k]

# ============================================================ kernel selection (READ THIS BEFORE MOVING IT)
# This block must run BEFORE the torch.compile below.  torch.__init__ mirrors use_deterministic_algorithms()
# into torch._inductor.config.deterministic, which switches Inductor's on-device autotuning off, so a compiled
# object built while the flag is on stays on heuristic kernels no matter what the flag does afterwards.
#
# WHY TRAINING RUNS WITH DETERMINISM OFF.  The notebook's boot cell (lisa_rtm.ipynb section 0, seed_everything)
# sets cudnn.deterministic = True, cudnn.benchmark = False and use_deterministic_algorithms(True, warn_only=True).
# Under that flag ATen replaces the fused fastAtomicAdd kernel behind the backward of torch.gather (scatter_add_)
# with _scatter_via_index_put -> index_put_with_sort_kernel, which materialises one int64 key per gathered
# element -- ~147M keys, ~1.2 GB per buffer, per arm-draw at batch 32 -- and radix-sorts them, thirteen
# sequence-passes per step.  It also slows the encoder's conv backward.  The repo's own A/B, same sequential
# trainer, 7 arms, batch 32 x 1 s, A100-80GB:
#
#       flag ON,  cudnn.benchmark off (notebook boot cell)        2261 ms/step
#       flag OFF, cudnn.benchmark on  (overnight/cell2_trainer.py:8-12)   587 ms/step     -> 3.9x
#
# What this does NOT change: seeds, data order, noise draws, jitter draws, initialisation and arm pairing are
# all untouched, so the seven arms still see identical batches and stay comparable.  What it does change: the
# float reduction ORDER inside the gradient atomics, i.e. ~1e-7 relative in fp32, far below the bf16 rounding
# the run already carries.  Run-to-run bitwise reproducibility of the gradients is the price.
# THE EVALUATION PATH KEEPS THE REPO'S REPRODUCIBILITY.  e4_eval / e5_visqol run under no_grad, so the scatter
# backward never runs there and determinism is nearly free; the built notebook re-enables the strict pair in a
# cell above the evaluation sections (build_train_notebook.py), and seed_everything() is unchanged.
# Set DETERMINISTIC = True in a cell above this file to put the strict pair back for training too.
if FAST["DETERMINISTIC"]:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
if FAST["TF32"] and _CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
_OFF3 = None
_IDX_CACHE = {}
_PHASE_CACHE = {}


def _autocast():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (FAST["AMP"] and _CUDA) else contextlib.nullcontext()


def _base_index(L, R, device):
    '''q = j / R for the full sequence and floor(q), computed once per (L, R, device).'''
    key = (int(L), int(R), str(device))
    if key not in _IDX_CACHE:
        q = torch.arange(L * R, device=device, dtype=torch.float32) / R
        _IDX_CACHE[key] = (q, torch.floor(q).long().clamp(0, L - 1))
    return _IDX_CACHE[key]


def _phase_coord(R, device):
    '''c_p = 2p/R - 1, p = 0..R-1: the coordinate of the R outputs of one input cell when the anchor is
    floor(q), i.e. the jitter-free case.  Cached per (R, device).'''
    key = (int(R), str(device))
    if key not in _PHASE_CACHE:
        _PHASE_CACHE[key] = 2.0 * torch.arange(R, device=device, dtype=torch.float32) / R - 1.0
    return _PHASE_CACHE[key]


def _mlp_eager(X, Ws, bs, relus):
    '''(A, rows, in) -> (A, rows, out).  A == 1 (GROUP_MAX = 1, the default) takes the 2-D F.linear path:
    Inductor's mm templating is stronger than its bmm templating, and the ReLU epilogue is fused into the
    GEMM only when a Triton mm template wins autotuning -- each fused epilogue removes two of the four HBM
    traffic terms of a layer, and the decoder is bandwidth-bound (~72 FLOP/byte at bf16 vs A100's ~153).'''
    if X.shape[0] == 1:
        Y = X[0]
        for W, b, r in zip(Ws, bs, relus):
            Y = F.linear(Y, W[0], b[0])
            if r:
                Y = F.relu(Y)
        return Y.unsqueeze(0)
    for W, b, r in zip(Ws, bs, relus):
        X = torch.baddbmm(b.unsqueeze(1), X, W.transpose(1, 2))
        if r:
            X = F.relu(X)
    return X


_MLP = {"fn": _mlp_eager, "compiled": False}
if FAST["COMPILE"]:
    try:
        # _mlp is called at several shapes per step: A varies per stack, rows = S*N with S = B (det arms),
        # 2B (es arms) and the validation batch, and in_features is 97 (LISAS) or 101 (LISASD).  With
        # dynamic=False every distinct size is a fresh graph, and once the recompile limit is hit Dynamo
        # SKIPS the function and everything nested in it -- the compile silently becomes eager, which is
        # consistent with the stacked runs showing no benefit from COMPILE at all.  Raise the limit and mark
        # the row axis dynamic; the last axis stays static so the template still specialises on width.
        import torch._dynamo
        for _attr in ("recompile_limit", "cache_size_limit"):           # cache_size_limit is now an alias
            if hasattr(torch._dynamo.config, _attr):
                setattr(torch._dynamo.config, _attr, 32)
        _MLP = {"fn": torch.compile(_mlp_eager, dynamic=False, mode=FAST["COMPILE_MODE"]), "compiled": True}
    except Exception as _e:                                            # no compiler: eager
        print("torch.compile unavailable, eager MLP:", repr(_e), flush=True)


def _mlp(X, Ws, bs, relus):
    if _MLP["compiled"]:
        try:
            for _d in (0, 1):                  # never mark a size-1 axis: Dynamo specialises on 0/1 anyway
                if X.shape[_d] > 1:
                    torch._dynamo.mark_dynamic(X, _d)
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
            Ws = [self.p(w) for w, _, _ in self.dec_plan]
            bs = [self.p(b) for _, b, _ in self.dec_plan]
            if FAST["SUBPIXEL"]:
                X = self._layer1(zp, Ws[0], bs[0], eps_dec, q, S, L, N, perturb, jitter)
                X = _mlp(X, Ws[1:], bs[1:], self.relus[1:])
            else:
                if perturb:
                    jit = torch.randn(A, S, N, device=x_in.device) * 0.5 if jitter is None else jitter
                    idx = torch.floor(q + jit).long().clamp_(0, L - 1)                     # (A, S, N)
                else:
                    idx = idx0.view(1, 1, N).expand(A, S, N)
                coord = (2.0 * (q - idx.float()) - 1.0).unsqueeze(-1)                      # (A, S, N, 1)
                global _OFF3
                if _OFF3 is None or _OFF3.device != x_in.device:
                    _OFF3 = torch.arange(3, device=x_in.device)
                gidx = (idx.unsqueeze(-1) + _OFF3).reshape(A, S, 3 * N)                    # padded rows i-1, i, i+1
                g = torch.gather(zp, 2, gidx.unsqueeze(-1).expand(A, S, 3 * N, C)).reshape(A, S, N, 3 * C)
                parts = [coord.to(g.dtype), g]
                if self.n_dec:
                    parts.append((eps_dec if eps_dec is not None else g.new_zeros(A, S, N, self.n_dec)).to(g.dtype))
                X = torch.cat(parts, -1).reshape(A, S * N, -1)
                X = _mlp(X, Ws, bs, self.relus)
        return X.reshape(A, S, N).float()

    # ---- decoder layer 1 at the INPUT rate (exact; see the module note) --------------------------------
    def _layer1(self, zp, W1, b1, eps_dec, q, S, L, N, perturb, jitter):
        '''W1 = [w_c | W_z | w_d] is linear in its blocks, and the latent block depends only on the anchor i,
        so its contribution u[i] = A_-1 z_{i-1} + A_0 z_i + A_+1 z_{i+1} is ONE grouped conv1d over the
        replicate-padded latents at 12 kHz instead of a 3N-row gather at 48 kHz.  Layer-1 MACs per audio
        second fall from 48,000 x 97 x 144 to 12,000 x 96 x 144, and the 97-wide HR feature tensor and its
        backward leave the graph entirely.  (This is Shi et al.'s sub-pixel convolution: LISA's first layer
        is a sub-pixel conv whose R phase filters share one filter bank and differ only by a rank-1 bias.)

        The anchor jitter does NOT have to be disabled and does NOT break exactness.  For output j in cell
        n = floor(q) with phase p, the jittered anchor is i = clamp(floor(q + eta), 0, L-1) = n + m, and the
        coordinate is c = 2(q - i) - 1 = c_p - 2m.  The jitter only selects WHICH row of u is read; the
        arithmetic is untouched, so computing c from the gathered i keeps this bit-for-bit the gather path.
        What jitter costs is that the u gather stays at the output rate, which is why it is a FLOP win and
        roughly traffic-neutral at HR.  FAST["JITTER_PER_CELL"] (changes-math, default off) is what moves the
        gather down to 12 kHz as well.'''
        A, C, R, dev = self.A, self.C, self.R, zp.device
        H = W1.shape[1]
        # W_z columns are ordered [i-1, i, i+1] x C, so (H, 3, C) -> (H, C, 3) puts the conv taps in the
        # order (A_-1, A_0, A_+1); conv1d's out[o, n] = sum_{c,k} W[o, c, k] * in[c, n+k] then reads
        # zp[n], zp[n+1], zp[n+2] = z[n-1], z[n], z[n+1] under the replicate padding.
        W_z = W1[:, :, 1:1 + 3 * C].reshape(A, H, 3, C).transpose(2, 3).reshape(A * H, C, 3)
        u = F.conv1d(zp.permute(1, 0, 3, 2).reshape(S, A * C, L + 2), W_z, groups=A)        # (S, A*H, L)
        u = u.view(S, A, H, L).permute(1, 0, 3, 2)                                          # (A, S, L, H)
        dt = u.dtype
        w_c, bb = W1[:, :, 0].view(A, 1, 1, H).to(dt), b1.view(A, 1, 1, H).to(dt)
        if perturb and FAST["JITTER_PER_CELL"]:                 # CHANGES MATH: one draw per input cell
            jit = torch.randn(A, S, L, device=dev) * 0.5 if jitter is None else jitter
            cell = torch.arange(L, device=dev, dtype=torch.float32)
            idx = torch.floor(cell + jit).long().clamp_(0, L - 1)                           # (A, S, L)
            m = (idx.float() - cell).unsqueeze(-1).unsqueeze(-1)                            # (A, S, L, 1, 1)
            g = torch.gather(u, 2, idx.unsqueeze(-1).expand(A, S, L, H)).unsqueeze(3)       # (A, S, L, 1, H)
            cp = _phase_coord(R, dev).view(1, 1, 1, R, 1)
            X = (g + (cp - 2.0 * m).to(dt) * w_c.unsqueeze(3) + bb.unsqueeze(3)).reshape(A, S, N, H)
        elif perturb:
            jit = torch.randn(A, S, N, device=dev) * 0.5 if jitter is None else jitter
            idx = torch.floor(q + jit).long().clamp_(0, L - 1)                              # (A, S, N)
            coord = (2.0 * (q - idx.float()) - 1.0).unsqueeze(-1)                           # (A, S, N, 1)
            g = torch.gather(u, 2, idx.unsqueeze(-1).expand(A, S, N, H))
            X = g + coord.to(dt) * w_c + bb
        else:
            # perturb=False: i = floor(q) = n, so there is no gather at all -- the periodic shuffle.  expand's
            # backward is a sum-reduction (no atomics, no index tensor, no scatter), which is why this path is
            # immune to the determinism flag whichever way it is set, and why eval can keep the strict pair.
            cp = _phase_coord(R, dev).view(1, 1, 1, R, 1)
            X = (u.unsqueeze(3) + cp.to(dt) * w_c.unsqueeze(3) + bb.unsqueeze(3)).reshape(A, S, N, H)
        if self.n_dec and eps_dec is not None:      # eps_dec=None is a zero block: exactly a no-op, so skip it
            X = X + torch.matmul(eps_dec.to(dt), W1[:, :, 1 + 3 * C:].transpose(1, 2).unsqueeze(1).to(dt))
        X = X.reshape(A, S * N, H)
        return F.relu(X) if self.relus[0] else X

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


def gpu_stats():
    '''allocated / reserved / peak / total / free GB and utilisation %, all best-effort.'''
    g = {"alloc_gb": 0.0, "reserved_gb": 0.0, "peak_gb": 0.0, "total_gb": 0.0, "free_gb": 0.0, "util_pct": None}
    if not _CUDA:
        return g
    try:
        g["alloc_gb"] = torch.cuda.memory_allocated() / 1e9
        g["reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        g["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        free, total = torch.cuda.mem_get_info()
        g["free_gb"], g["total_gb"] = free / 1e9, total / 1e9
    except Exception:
        pass
    try:
        g["util_pct"] = float(torch.cuda.utilization())
    except Exception:
        pass
    return g


def train_ov3_fast(corpus, val_corpus, arms, steps, batch, lr, milestones, gamma, clip, ckpt_every, tag, probe,
                   val_every=500, n_val_batches=8, log_every=25,
                   on_step=None, on_epoch=None, steps_per_epoch=None):
    '''Drop-in for train_ov3: same signature, history keys, checkpoints, log lines, plot.  One init per class
    (LISASD shares the LISAS weights via copy_shared), one batch stream, arms stacked per class.

    on_step(info) fires every log_every steps and on_epoch(info) whenever steps_per_epoch is crossed; `info`
    carries progress, throughput, GPU memory and the per-arm terms.  The per-arm numbers reach the host in ONE
    .cpu().tolist() at the callback step, so the no-sync design between callbacks is unchanged.'''
    names, specs, base, models = _make_models(arms)
    stacks = build_stacks(names, specs, base, DEVICE)
    # (stack index, index within the stack, index in the concatenated (sum A, 4) terms tensor)
    _off, where = 0, {}
    for si, s in enumerate(stacks):
        for a, k in enumerate(s.names):
            where[k] = (si, a, _off + a)
        _off += s.A
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
    # ---- callback state: rolling step time, and the most recent val / probe numbers per arm ----------------
    _VAL_KEYS = ("val_loss", "val_wave", "val_spec")
    _DEV_KEYS = ("snr0", "def0", "snr1", "def1")
    last = {k: {kk: None for kk in _VAL_KEYS + _DEV_KEYS} for k in names}
    dt_hist, t_last, epoch_mark = [], time.time(), 0
    seg_s = corpus.seg_hi / CFG.fs_hi
    spe = float(steps_per_epoch) if steps_per_epoch else None

    def _info(step, terms, lr_):
        '''One host transfer: (sum A, 5) = [ema, loss, wave, spec, spread] per arm.'''
        rows = torch.cat([torch.cat([s.ema.unsqueeze(1) for s in stacks], 0),
                          torch.cat([t.reshape(-1, 4) for t in terms], 0)], 1).cpu().tolist()
        el = time.time() - t0
        ms = 1000 * (sum(dt_hist) / len(dt_hist)) if dt_hist else float("nan")
        sps = 1000.0 / ms if ms and ms == ms and ms > 0 else float("nan")
        done = step + 1
        arms_info = {}
        for k in names:
            r = rows[where[k][2]]
            arms_info[k] = {"loss_ema": r[0], "loss": r[1], "wave": r[2], "spec": r[3], "spread": r[4],
                            **{kk: last[k][kk] for kk in _VAL_KEYS + _DEV_KEYS}}
        return {"tag": tag, "step": done, "steps": steps, "frac": done / steps,
                "epoch": done / spe if spe else None, "epochs_total": steps / spe if spe else None,
                "t_elapsed": el, "ms_per_step": ms, "eta_s": (steps - done) * ms / 1000 if ms == ms else None,
                "samples_per_s": sps * batch, "audio_s_per_s": sps * batch * seg_s,
                "batch": batch, "seg_s": seg_s, "lr": lr_, "gpu": gpu_stats(), "arms": arms_info}

    def flush():
        if not pending:
            return
        T = torch.stack([t for _, _, t in pending]).cpu().tolist()           # one transfer for all pending steps
        for (st, lr_, _), rows in zip(pending, T):
            for k in names:
                r = rows[where[k][2]]
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
        _now = time.time(); dt_hist.append(_now - t_last); t_last = _now
        if len(dt_hist) > 50:
            del dt_hist[:-50]
        if step % log_every == 0:
            lr_now = scheds[0].get_last_lr()[0]
            pending.append((step, lr_now, torch.cat([t.reshape(-1, 4) for t in terms], 0)))
            if on_step is not None or (on_epoch is not None and spe and (step + 1) // spe > epoch_mark):
                info = _info(step, terms, lr_now)
                if on_step is not None:
                    try:
                        on_step(info)
                    except Exception as e:
                        print("on_step failed:", repr(e)[:200], flush=True)
                if on_epoch is not None and spe and (step + 1) // spe > epoch_mark:
                    epoch_mark = int((step + 1) // spe)
                    try:
                        on_epoch(info)
                    except Exception as e:
                        print("on_epoch failed:", repr(e)[:200], flush=True)
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
                    last[k].update(val_loss=float(vl), val_wave=float(vw), val_spec=float(vs))
                    vline += f" | {k}: train {emas[i]:.4f} val {vl:.4f}"
                    i += 1
            print(vline, flush=True); logf.write(vline + chr(10)); logf.flush()
        if (step + 1) % ckpt_every == 0 or step + 1 == steps:
            flush()
            line = f"step {step+1:>6}/{steps} [{time.time()-t0:.0f}s]"
            for k in names:
                kind, lam, cls_name = specs[k]
                si, a, _ = where[k]
                m = stacks[si].export(a, models[k])
                pm = probe_metrics(m, probe, CFG, naive)
                h = hist[k]
                h["dev_step"].append(step + 1); h["snr_naive"].append(naive)
                h["snr0"].append(pm[0.0][0]); h["def0"].append(pm[0.0][1])
                s1, d1 = pm.get(1.0, (None, None))
                h["snr1"].append(s1); h["def1"].append(d1)
                last[k].update(snr0=pm[0.0][0], def0=pm[0.0][1], snr1=s1, def1=d1)
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
    if steps and (on_step is not None or on_epoch is not None):      # the log grid rarely lands on the last step
        info = _info(steps - 1, terms, scheds[0].get_last_lr()[0])
        for cb in (on_step, on_epoch if (on_epoch is not None and spe and steps // spe > epoch_mark) else None):
            if cb is not None:
                try:
                    cb(info)
                except Exception as e:
                    print("final callback failed:", repr(e)[:200], flush=True)
    logf.close()
    for k in names:
        si, a, _ = where[k]
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
