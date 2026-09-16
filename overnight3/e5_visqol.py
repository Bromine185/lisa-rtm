# ============================================================ OV3-5 perceptual evaluation: LSD + ViSQOL (speech 16k, audio 48k) + PESQ
# Every OV3 condition on EVAL12 through the user's evaluation.py, the same call path as overnight2/c6_visqol_eval.py
# (torchaudio resample to 16 kHz, ViSQOL speech mode; ViSQOL audio mode at 48 kHz sees the whole band; PESQ wb).
# ASSUMES the install cells overnight2/c6_install_bg.py, c6_install_wait.py AND c6_lattice.py were run in this
# kernel (visqol-python, pesq, torchmetrics importable; ai_edge_litert for the upstream lattice speech-MOS mapping,
# which the OV2 numbers used) and that evaluation.py sits next to the ov2 / ov3 cells on Drive.  Which speech
# mapping is active is checked below and recorded in the JSON and table (NSIM is mapping-free either way).
# Conditions: naive; per arm one draw (tau=1 seed 0; tau=0 for det); per stochastic arm mean16 and logmean16;
# all of those with and without baseband passthrough; floor (passthrough + empty HB); ceiling (passthrough + true HB).
# Draws are made once per (arm, utterance) and shared by every derived condition.
# Requires: c0 boot, c1_model, e1_model.  Functions at top level; the main block honours OV3_RUN_EVAL.
import json, time, math, numpy as np, torch, importlib.util
from pathlib import Path

OV3_TAG = globals().get("OV3_TAG", "OV3_es")
M_DRAWS = int(globals().get("OV3_M", 16))

def _find_eval_py():
    cands = [Path(str(globals().get("OV3", ""))) / "evaluation.py" if globals().get("OV3") else None,
             Path(str(globals().get("OV2", ""))) / "evaluation.py" if globals().get("OV2") else None,
             ROOT / "evaluation.py", ROOT / "ov3" / "evaluation.py", ROOT / "ov2" / "evaluation.py"]
    for p in cands:
        if p is not None and p.exists():
            return p
    raise FileNotFoundError("evaluation.py not found next to the ov2/ov3 cells or under ROOT")

_spec = importlib.util.spec_from_file_location("user_evaluation", _find_eval_py())
user_eval = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(user_eval)
EVALUATOR = user_eval.Evaluator()                                   # speech mode, 16 kHz, + PESQ wb
try:
    import ai_edge_litert; VISQOL_SP_MAPPING = "lattice"
except Exception:
    VISQOL_SP_MAPPING = "polynomial"
    print("WARNING: speech MOS on the polynomial mapping; run overnight2/c6_lattice.py for OV2-comparable numbers", flush=True)
print("visqol speech16k MOS mapping:", VISQOL_SP_MAPPING, flush=True)
from visqol import VisqolApi
try:
    VQ_AUDIO = VisqolApi(); VQ_AUDIO.create(mode="audio")           # 48 kHz, 32 ERB bands to 24 kHz
except Exception as _e:
    print("audio mode unavailable:", repr(_e)); VQ_AUDIO = None

def visqol_speech(y, w):
    '''MOS-LQO exactly as evaluation.py computes it (torchaudio resample to 16 kHz, speech mode) + NSIM.'''
    yt, wt = torch.from_numpy(np.asarray(y, np.float32)), torch.from_numpy(np.asarray(w, np.float32))
    hr, sr_ = EVALUATOR.sample_to_correct_rate(yt, wt, CFG.fs_hi)
    hr, sr_ = EVALUATOR.match_length(hr, sr_)
    try:
        res = EVALUATOR.visqol_api.measure_from_arrays(hr.numpy().astype(np.float64), sr_.numpy().astype(np.float64), sample_rate=EVALUATOR.target_sr)
        return float(res.moslqo), float(getattr(res, "vnsim", float("nan")))
    except Exception:
        return float("nan"), float("nan")

def pesq_wb(y, w):
    try:
        return float(EVALUATOR.evaluate_pesq(torch.from_numpy(np.asarray(y, np.float32))[None],
                                             torch.from_numpy(np.asarray(w, np.float32))[None], current_sr=CFG.fs_hi))
    except Exception:
        return float("nan")

def visqol_audio(y, w):
    if VQ_AUDIO is None:
        return float("nan"), float("nan")
    n = min(len(y), len(w))
    try:
        res = VQ_AUDIO.measure_from_arrays(np.asarray(y[:n], np.float64), np.asarray(w[:n], np.float64), sample_rate=CFG.fs_hi)
        return float(res.moslqo), float(getattr(res, "vnsim", float("nan")))
    except Exception:
        return float("nan"), float("nan")

def score(y, w):
    y = np.asarray(y, np.float64); w = np.asarray(w, np.float64)[:len(y)]
    w = np.pad(w, (0, len(y) - len(w)))
    vs, ns = visqol_speech(y, w); va, na = visqol_audio(y, w)
    return dict(snr=snr_db(y, w), lsd=lsd_db(y, w, CFG.eval_n_fft, CFG.eval_hop),
                hb_lsd=lsd_db(y, w, CFG.eval_n_fft, CFG.eval_hop, CFG.eval_k_cut),
                visqol_speech16k=vs, nsim_speech16k=ns, pesq_wb=pesq_wb(y, w), visqol_audio48k=va, nsim_audio48k=na)

def _split_bands5(w, fs, f_cut):
    W = np.fft.rfft(np.asarray(w, np.float64))
    k = int(round(f_cut * len(w) / fs))
    lo, hi = W.copy(), W.copy()
    lo[k:] = 0; hi[:k] = 0
    return np.fft.irfft(lo, n=len(w)), np.fft.irfft(hi, n=len(w))

def passthrough5(y, w):
    y = np.asarray(y, np.float64); n = len(y)
    nv = naive_upsample(y, CFG)[:n]; nv = np.pad(nv, (0, n - len(nv)))
    w = np.asarray(w, np.float64)[:n]; w = np.pad(w, (0, n - len(w)))
    return _split_bands5(nv, CFG.fs_hi, CFG.fs_lo / 2)[0] + _split_bands5(w, CFG.fs_hi, CFG.fs_lo / 2)[1]

def conditions_for(models, y, M):
    '''name -> waveform for one utterance; draws made once per arm.'''
    y = np.asarray(y, np.float64)
    out = {"naive": naive_upsample(y, CFG)[:len(y)]}
    for k, m in models.items():
        if m.tau == 0:
            out[k] = reconstruct(m, y, CFG, tau=0.0)
            continue
        draws = np.stack([reconstruct(m, y, CFG, tau=1.0, seed=s) for s in range(M)])
        out[f"{k} tau=1"] = draws[0]
        out[f"{k} mean{M}"] = draws.mean(0)
        out[f"{k} logmean{M}"] = logmag_ensemble_readout(draws, y, CFG, passthrough=False)
    for name in list(out):
        if name != "naive":
            out[f"{name} | passthrough"] = passthrough5(y, out[name])
    out["floor: passthrough + empty HB"] = passthrough5(y, np.zeros_like(y))
    out["ceiling: passthrough + true HB"] = passthrough5(y, y)
    return out

KEYS = ("snr", "lsd", "hb_lsd", "visqol_speech16k", "nsim_speech16k", "pesq_wb", "visqol_audio48k", "nsim_audio48k")

def run_visqol(models, utts, M, tag):
    out_dir = ROOT / "ov3"; out_dir.mkdir(parents=True, exist_ok=True)
    per, t0 = {}, time.time()
    for ui, y in enumerate(utts):
        conds = conditions_for(models, y, M)
        for cname, w in conds.items():
            per.setdefault(cname, []).append(score(y, w))
        print(f"  {ui+1}/{len(utts)} utts, {len(conds)} conditions  [{time.time()-t0:.0f}s]", flush=True)
        json.dump({"per_utt": per, "n": ui + 1, "M": M, "speech_mapping": VISQOL_SP_MAPPING}, open(out_dir / f"visqol_{tag}.json", "w"))
    AGG = {c: {k: float(np.nanmean([r[k] for r in rows])) for k in KEYS} for c, rows in per.items()}
    json.dump({"agg": AGG, "per_utt": per, "n": len(utts), "M": M, "speech_mapping": VISQOL_SP_MAPPING}, open(out_dir / f"visqol_{tag}.json", "w"), indent=1)
    lines = ["| condition | SNR | LSD | HB-LSD | ViSQOL speech16k | NSIM sp | PESQ wb | ViSQOL audio48k | NSIM au |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(AGG, key=lambda c: AGG[c]["lsd"]):
        a = AGG[c]
        lines.append(f"| {c} | {a['snr']:.2f} | {a['lsd']:.3f} | {a['hb_lsd']:.3f} | {a['visqol_speech16k']:.3f} | {a['nsim_speech16k']:.3f} | "
                     f"{a['pesq_wb']:.3f} | {a['visqol_audio48k']:.3f} | {a['nsim_audio48k']:.3f} |")
    lines.append(f"\nEVAL12 n={len(utts)}, M={M}; sorted by LSD; speech16k MOS mapping: {VISQOL_SP_MAPPING}"
                 + ("" if VISQOL_SP_MAPPING == "lattice" else " (NOT the lattice mapping of the OV2 tables; compare NSIM)") + ".")
    TABLE = chr(10).join(lines)
    (out_dir / f"visqol_table_{tag}.md").write_text(TABLE)
    print(TABLE, flush=True)
    return AGG, TABLE


if globals().get("OV3_RUN_EVAL", True):
    if "models_ov3" not in globals():
        models_ov3 = {}
        for p in sorted((CKPT / OV3_TAG).glob("*.pt")):
            models_ov3[p.stem], _ = load_arm(p)
    for _m in models_ov3.values():
        _m.eval()
    EVAL12 = [np.asarray(u, np.float64) for u in test_utts[:12]]
    AGG5, TABLE5 = run_visqol(models_ov3, EVAL12, M_DRAWS, OV3_TAG)
    print("VISQOL EVAL DONE", OV3_TAG, flush=True)
