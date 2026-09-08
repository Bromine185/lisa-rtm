# ============================================================ OV2-6 perceptual evaluation: LSD + ViSQOL (+ PESQ)
# Judges every condition of the 8 Sep run on the user's target metrics.  ViSQOL speech mode at 16 kHz exactly
# as evaluation.py does it (note: after resampling to 16 kHz the metric sees only 6-8 kHz of the reconstructed
# 6-24 kHz band), plus ViSQOL audio mode at 48 kHz (sees the whole band) and wideband PESQ.  Per-utterance
# values are kept so conditions can be compared paired.
#
# Requires in the kernel: boot (c0), c1_model (+ c1b_wide), the user's evaluation.py on Drive, and the
# `visqol` package (see c6_install).  Ladder rungs are included when c4 has run in this kernel (MAPS/COND/CPOW).
import json, time, math, numpy as np, torch, scipy.signal as sps
import importlib.util

OV2_DIR = Path(OV2) if not isinstance(OV2, Path) else OV2
spec_ = importlib.util.spec_from_file_location("user_evaluation", OV2_DIR / "evaluation.py")
user_eval = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(user_eval)
EVALUATOR = user_eval.Evaluator()                                   # speech mode, 16 kHz, + PESQ wb
from visqol import VisqolApi
try:
    VQ_AUDIO = VisqolApi(); VQ_AUDIO.create(mode="audio")           # 48 kHz, 32 ERB bands to 24 kHz: sees the whole band
except Exception as _e:
    print("audio mode unavailable:", repr(_e)); VQ_AUDIO = None

def visqol_speech(y, w):
    """MOS-LQO exactly as evaluation.py computes it (torchaudio resample to 16 kHz, speech mode) plus the
    mapping-independent NSIM from the same call path."""
    yt, wt = torch.from_numpy(np.asarray(y, np.float32)), torch.from_numpy(np.asarray(w, np.float32))
    hr, sr_ = EVALUATOR.sample_to_correct_rate(yt, wt, CFG.fs_hi)
    hr, sr_ = EVALUATOR.match_length(hr, sr_)
    try:
        res = EVALUATOR.visqol_api.measure_from_arrays(hr.numpy().astype(np.float64), sr_.numpy().astype(np.float64), sample_rate=EVALUATOR.target_sr)
        return float(res.moslqo), float(getattr(res, "vnsim", float("nan")))
    except Exception as e:
        return float("nan"), float("nan")

def pesq_wb(y, w):
    try:
        return float(EVALUATOR.evaluate_pesq(torch.from_numpy(np.asarray(y, np.float32))[None],
                                             torch.from_numpy(np.asarray(w, np.float32))[None], current_sr=CFG.fs_hi))
    except Exception as e:
        return float("nan")

def visqol_audio(y, w):
    if VQ_AUDIO is None:
        return float("nan"), float("nan")
    n = min(len(y), len(w))
    try:
        res = VQ_AUDIO.measure_from_arrays(np.asarray(y[:n], np.float64), np.asarray(w[:n], np.float64), sample_rate=CFG.fs_hi)
        return float(res.moslqo), float(getattr(res, "vnsim", float("nan")))
    except Exception as e:
        return float("nan"), float("nan")

def score(y, w):
    y = np.asarray(y, np.float64); w = np.asarray(w, np.float64)[:len(y)]
    w = np.pad(w, (0, len(y) - len(w)))
    vs, ns = visqol_speech(y, w); va, na = visqol_audio(y, w)
    return dict(snr=snr_db(y, w), lsd=lsd_db(y, w, CFG.eval_n_fft, CFG.eval_hop),
                hb_lsd=lsd_db(y, w, CFG.eval_n_fft, CFG.eval_hop, CFG.eval_k_cut),
                visqol_speech16k=vs, nsim_speech16k=ns, pesq_wb=pesq_wb(y, w), visqol_audio48k=va, nsim_audio48k=na)

# ---- evaluation sets ---------------------------------------------------------------------------
SETS = {"hub12": [np.asarray(u, np.float64) for u in test_utts[:12]]}
_ds = FIXTURES / "test_FULL.npz"
if _ds.exists():
    with np.load(_ds, allow_pickle=True) as d:
        _u = list(d["utts"])
    SETS["datashare36"] = [np.asarray(u, np.float64) for i, u in enumerate(_u) if i % 40 < 12][:36]
print({k: len(v) for k, v in SETS.items()}, flush=True)

# ---- models -------------------------------------------------------------------------------------
MODELS = {}
for p in sorted((CKPT / "OV2_es").glob("*.pt")) + sorted((CKPT / "OV2_wide").glob("*.pt")):
    MODELS[p.stem], _ = load_arm(p)
for lam in ("1e-3", "1e-1"):                                        # lambda frontier (plain LISA class)
    p = CKPT / "XL4_b64_1s" / f"relu_l{lam}.pt"
    if p.exists():
        ck = torch.load(p, map_location=DEVICE, weights_only=False)
        m = LISA(CFG).to(DEVICE); m.load_state_dict(ck["model"]); m.eval(); m.tau = 0.0
        MODELS[f"lam{lam}"] = m

def recon(m, y, tau, seed=0):
    if isinstance(m, LISAS):
        return reconstruct(m, y, CFG, tau=tau, seed=seed)
    # plain LISA: the notebook's chunked deterministic reconstruction
    m.eval()
    x_lo = torch.from_numpy(decimate(y, CFG.upsample)).float()[None].to(DEVICE)
    n_out = x_lo.shape[1] * CFG.upsample
    out = [m(x_lo, j0=s, j1=min(s + (1 << 15), n_out)).squeeze(0).detach().cpu().numpy() for s in range(0, n_out, 1 << 15)]
    w = np.concatenate(out).astype(np.float64)
    return np.pad(w, (0, max(0, len(y) - len(w))))[:len(y)]

# ---- conditions: name -> function(y) -> waveform -------------------------------------------------
CONDITIONS = {"naive": lambda y: naive_upsample(y, CFG)}
for k, m in MODELS.items():
    if m.tau == 0:
        CONDITIONS[k] = (lambda m: lambda y: recon(m, y, 0.0))(m)
    else:
        for tau in (0.5, 0.75, 1.0, 1.25):
            CONDITIONS[f"{k} tau={tau:g}"] = (lambda m, t: lambda y: recon(m, y, t, seed=0))(m, tau)
        CONDITIONS[f"{k} mean16"] = (lambda m: lambda y: np.mean([recon(m, y, 1.0, seed=s) for s in range(16)], 0))(m)
if "MAPS" in globals():                      # c4 ran in this kernel: add the post-hoc rungs on det
    _det = models_ov2["det"] if "models_ov2" in globals() and "det" in models_ov2 else MODELS["det"]
    def _ladder(name):
        def f(y):
            yh = recon(_det, y, 0.0)
            if name == "T1":
                return apply_map(yh, lambda L, C, T=MAPS["T1 quantile"]: T(L), CFG)[:len(y)]
            if name == "T3":
                return apply_map(yh, lambda L, C: Conditioned(COND, C)(L), CFG)[:len(y)]   # notebook's ConditionalMap
            if name == "S":
                samples, (S_, pad_, L_) = sample_high_band(yh, CFG, 1.0, 1, "visqol/S")
                S2 = S_.copy(); S2[:, HI] = samples[0]
                return istft(S2, CFG.n_fft, CFG.hop, pad_, L_)[:len(y)]
        return f
    for name in ("T1", "T3", "S"):
        CONDITIONS[f"ladder {name} on det"] = _ladder(name)
print(f"{len(CONDITIONS)} conditions:", list(CONDITIONS), flush=True)

# ---- run ------------------------------------------------------------------------------------------
OUT = ROOT / "ov2"
PER_UTT = {s: {c: [] for c in CONDITIONS} for s in SETS}
t0 = time.time()
for sname, utts in SETS.items():
    for ui, y in enumerate(utts):
        for cname, fn in CONDITIONS.items():
            PER_UTT[sname][cname].append(score(y, fn(y)))
        print(f"  {sname}: {ui+1}/{len(utts)} utts  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(PER_UTT, open(OUT / "visqol_per_utt.json", "w"))

KEYS = ("snr", "lsd", "hb_lsd", "visqol_speech16k", "nsim_speech16k", "pesq_wb", "visqol_audio48k", "nsim_audio48k")
AGG = {s: {c: {k: float(np.nanmean([r[k] for r in rows])) for k in KEYS} for c, rows in PER_UTT[s].items()} for s in SETS}
json.dump({"agg": AGG, "n": {s: len(v) for s, v in SETS.items()}}, open(OUT / "visqol_summary.json", "w"), indent=1)

for sname in SETS:
    print(f"\n=== {sname} (n={len(SETS[sname])}) ===")
    print(f"{'condition':<28}{'SNR':>7}{'LSD':>7}{'HB-LSD':>8}{'ViSQOL-sp16k':>13}{'NSIM-sp':>8}{'PESQ-wb':>9}{'ViSQOL-au48k':>13}{'NSIM-au':>8}")
    for c in sorted(AGG[sname], key=lambda c: AGG[sname][c]["lsd"]):
        a = AGG[sname][c]
        print(f"{c:<28}{a['snr']:7.2f}{a['lsd']:7.3f}{a['hb_lsd']:8.3f}{a['visqol_speech16k']:13.3f}{a['nsim_speech16k']:8.3f}{a['pesq_wb']:9.3f}{a['visqol_audio48k']:13.3f}{a['nsim_audio48k']:8.3f}")
print("VISQOL EVAL DONE", flush=True)
