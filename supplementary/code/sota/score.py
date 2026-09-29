"""Score every model's saved draws on one set of utterances with one scorer: every OV50-datasheet metric, plus
the SNR ceiling and the calibration test. Nothing is re-run; the draws come from sota/run_models.py / run_ours.py.

    cache/visqolenv/bin/python sota/score.py --set core [--workers 6] [--no-perceptual]

visqolenv is the scratch venv of fast/paper_split_eval.py: a venv on the project interpreter whose site-packages
carry a .pth to the project venv, plus visqol-python[lattice], pesq, torchaudio 2.11 and torchmetrics.
Metric definitions come from the notebook via audit.boot (stft, logmag, lsd_db, snr_db, third_octave_edges,
crps_ensemble, pit_ranks, spread_skill, stream, logmag_ensemble_readout), so every column here is the column
of the same name in the OV50 datasheet.

INPUT. Whatever the set's lo12k/ files hold (sota/make_inputs.py): the FLowHigh / NU-Wave 2 test input,
cheby1(8, 0.05 dB) by sosfiltfilt then resample_poly to 12 kHz, on every set except ourtest_poly and demo
(plain resample_poly). The naive baseline and the passthrough low band are built from that same file.
Truth is peak 0.95.

GAIN. Each draw is rescaled so its energy below 5.5 kHz matches the truth's (the band every model is handed).
A model that copies the low band exactly gets gain 1; the high band is then judged at the level the model put
it relative to its own low band.

CONDITIONS. Every model: its draws as they come out ("raw"). Our arms also: "| pt", each draw's high band over
the low band of the naive upsample of the same Chebyshev input (brick wall at 6 kHz), and, where M >= 2,
"logmean<M> | pt", the per-bin mean of log|STFT| over the M draws, phase of draw 0 (logmag_ensemble_readout,
passthrough=False), then passthrough. The datasheet's readout is logmean16; `core` holds 8 draws, so it is
logmean8 here and says so.

COLUMNS AND IDEALS.
  Levels (per draw, mean over draws, then over utterances). High band = 6-24 kHz.
    LSD, LSD-HF, LSD-LF  2048/512 (NU-Wave 2 for_test.py), HF/LF at 6 kHz              ideal 0
    LSD, HB-LSD          1024/256, the datasheet's lsd_db and its eval_k_cut           ideal 0
    deficit              mean over third octaves of 10 log10(E_est / E_true)          ideal 0 dB
    deficit bb           energy-weighted over the whole high band                     ideal 0 dB
    loud / mid / quiet   deficit on the target's top 25 / middle 50 / bottom 25 % frames   ideal 0 dB
  SNR (per utterance, mean of dB).
    SNR, vs ceiling      ceiling = truth brick-walled at 6 kHz (true LB, empty HB)     see the note
    coh, kappa           per HB third octave then mean (e4_eval.coherent): coherent fraction Re<Y,W>/<Y,Y>
                         (the predictable fraction) and kappa Re<Y,W>/<W,W>            kappa ideal 1
    p implied            HB level the SNR alone implies if coh = 0 and the LB is perfect (an upper bound)
  Probabilistic (HB log-magnitude, 1024/256, bins >= eval_k_cut; M draws).
    CRPS                 the datasheet's crps_ensemble: spread term over all M^2 pairs  lower is better
    CRPS fair            spread term over the M(M-1) off-diagonal pairs                lower is better
    sliced CRPS          crps_ensemble on 32 random unit projections, stream("ov2/theta") lower is better
    PIT low / high / end truth below all / above all draws; ends summed               1/(M+1), 2/(M+1)
    spread-skill         8 bins of ensemble spread vs RMSE of the ensemble mean; the table shows the
                         RMSE/spread ratio in the lowest and highest spread bin      ideal ~sqrt((M+1)/M)
    gap, gap HB          10 log10(mean_i |y - x_i|^2 / |y - xbar|^2)                   10 log10(2/(1+1/M))
    spread HB            pooled s^2 (1 + 1/M) / |y - xbar|^2 over utterances           ideal 1
    corr_err             || corrcoef(truth groups) - corrcoef(pred groups) ||_F, 16 HB groups, pooled over
                         utterances; from draw 0 and from the ensemble-mean waveform   ideal 0
  Perceptual (draw 0; the readout waveform for readouts).
    ViSQOL audio, NSIM   48 kHz, 32 bands to 24 kHz                                     ceiling ~4.73, NSIM 1
    ViSQOL speech, NSIM  16 kHz (evaluation.py path); mapping recorded in _meta         ceiling ~4.5, NSIM 1
    PESQ wb              16 kHz                                                         4.64

Writes <work>/<set>/score_<set>.json and .md.
"""
import os
import argparse, json, pathlib, sys, time
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fast.paper_split_eval import lsd as lsd_paper, up_naive                      # noqa: E402

WORK = pathlib.Path(os.environ.get("BWE_CACHE", str(pathlib.Path(__file__).resolve().parents[1] / "cache")) + "/sota/work")
FS, CUT, ALIGN = 48000, 6000.0, 5500.0
MODELS = ["flowhigh", "flowhigh_std1", "nuwave2", "audiosr", "apbwe", "apbwe_sinc", "ours_es_dec_erb_l0.1", "ours_det"]
NAMES = {"flowhigh": "FLowHigh (as shipped)", "flowhigh_std1": "FLowHigh, trained prior (std 1)", "nuwave2": "NU-Wave 2",
         "audiosr": "AudioSR (speech)", "apbwe": "AP-BWE", "apbwe_sinc": "AP-BWE, its own sinc input", "ours_es_dec_erb_l0.1": "ours: es_dec_erb_l0.1", "ours_det": "ours: det",
         "naive": "naive (resample_poly)", "ceiling": "ceiling: true LB, empty HB"}
G = CFG = THETA = GROUP = HB_SEL = HBK = HB_CENTRES = None
VQ = {}


def init(perceptual):
    """Per process: the notebook's metric code, the OV2 projection directions, the HB bands, and ViSQOL."""
    global G, CFG, THETA, GROUP, HB_SEL, HBK, HB_CENTRES
    import torch
    torch.set_num_threads(1)
    from audit.boot import boot
    G = boot(); CFG = G["CFG"]
    k_hb = CFG.eval_n_fft // 2 + 1 - CFG.eval_k_cut
    th = G["stream"]("ov2/theta").standard_normal((k_hb, 32)); THETA = th / np.linalg.norm(th, axis=0, keepdims=True)
    GROUP = np.array_split(np.arange(k_hb), 16)
    fr = np.fft.rfftfreq(CFG.eval_n_fft, 1 / FS)
    edges = G["third_octave_edges"](FS, CFG.fs_lo / 2, FS / 2 - 1)
    keep = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if ((fr >= a) & (fr < b)).any()]
    HB_SEL = [(fr >= a) & (fr < b) for a, b in keep]; HB_CENTRES = [float(np.sqrt(a * b)) for a, b in keep]
    HBK = fr >= CFG.fs_lo / 2
    if perceptual:
        from visqol import VisqolApi
        import evaluation
        VQ["audio"] = VisqolApi(); VQ["audio"].create(mode="audio")
        VQ["ev"] = evaluation.Evaluator()


# ---- small pieces ---------------------------------------------------------------------------------------
def fft_split(x, lo=True):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / FS)
    X[(f >= CUT) if lo else (f < CUT)] = 0
    return np.fft.irfft(X, n=len(x))


def align(y, w):
    Y = np.fft.rfft(y); W = np.fft.rfft(w); lo = np.fft.rfftfreq(len(y), 1 / FS) < ALIGN
    g = np.sqrt(np.sum(np.abs(Y[lo]) ** 2) / max(np.sum(np.abs(W[lo]) ** 2), 1e-30))
    return w * g, float(g)


def lm_hb(w):
    return G["logmag"](G["stft"](w, CFG.eval_n_fft, CFG.eval_hop)[0][:, CFG.eval_k_cut:])


def grouped(L):
    return np.stack([L[:, g].mean(1) for g in GROUP], 1)


def crps_fair(ens, truth):
    M = ens.shape[0]
    t1 = np.abs(ens - truth[None]).mean(0)
    d = np.abs(ens[:, None] - ens[None, :]).sum((0, 1)) / (M * (M - 1))
    return float(np.mean(t1 - 0.5 * d))


def ratio_curve(Sy, Sw, frames=None):
    if frames is not None:
        Sy, Sw = Sy[frames], Sw[frames]
    Py, Pw = np.abs(Sy) ** 2, np.abs(Sw) ** 2
    c = np.array([10 * np.log10((Pw[:, s].sum() + 1e-20) / (Py[:, s].sum() + 1e-20)) for s in HB_SEL])
    return c, float(10 * np.log10((Pw[:, HBK].sum() + 1e-20) / (Py[:, HBK].sum() + 1e-20)))


def perceptual(y, w):
    import torch
    n = min(len(y), len(w)); y, w = np.asarray(y[:n], np.float64), np.asarray(w[:n], np.float64)
    out = {}
    try:
        r = VQ["audio"].measure_from_arrays(y, w, sample_rate=FS); out["visqol_audio"], out["nsim_audio"] = float(r.moslqo), float(r.vnsim)
    except Exception:
        pass
    EV = VQ["ev"]
    yt, wt = torch.from_numpy(y.astype(np.float32)), torch.from_numpy(w.astype(np.float32))
    hr, sr_ = EV.sample_to_correct_rate(yt, wt, FS); hr, sr_ = EV.match_length(hr, sr_)
    try:
        r = EV.visqol_api.measure_from_arrays(hr.numpy().astype(np.float64), sr_.numpy().astype(np.float64), sample_rate=EV.target_sr)
        out["visqol_speech"], out["nsim_speech"] = float(r.moslqo), float(r.vnsim)
    except Exception:
        pass
    try:
        out["pesq_wb"] = float(EV.evaluate_pesq(yt[None], wt[None], current_sr=FS))
    except Exception:
        pass
    return out


# ---- one waveform, one ensemble ------------------------------------------------------------------------
def one(y, w, Sy, masks):
    n = len(y); Y = np.fft.rfft(y); W = np.fft.rfft(w); f = np.fft.rfftfreq(n, 1 / FS); hb = f >= CUT; lo = f < ALIGN
    Sw = G["stft"](w, CFG.eval_n_fft, CFG.eval_hop)[0]
    curve, bb = ratio_curve(Sy, Sw)
    yy = np.array([np.sum(np.abs(Sy[:, s]) ** 2) for s in HB_SEL]); ww = np.array([np.sum(np.abs(Sw[:, s]) ** 2) for s in HB_SEL])
    yw = np.array([np.sum((Sy[:, s] * np.conj(Sw[:, s])).real) for s in HB_SEL])
    r = {"lsd": lsd_paper(y, w, 2048, 512, FS), "lsd_hf": lsd_paper(y, w, 2048, 512, FS, lo_hz=CUT),
         "lsd_lf": lsd_paper(y, w, 2048, 512, FS, hi_hz=CUT),
         "lsd_1024": G["lsd_db"](y, w, CFG.eval_n_fft, CFG.eval_hop), "hb_lsd_1024": G["lsd_db"](y, w, CFG.eval_n_fft, CFG.eval_hop, CFG.eval_k_cut),
         "snr": G["snr_db"](y, w), "deficit": float(curve.mean()), "deficit_bb": bb, "curve": curve.tolist(),
         "coh": float(np.mean(yw / np.maximum(yy, 1e-20))), "kappa": float(np.mean(yw / np.maximum(ww, 1e-20))),
         "coh_fft": float(np.sum((Y[hb] * np.conj(W[hb])).real) / np.sum(np.abs(Y[hb]) ** 2)),
         "lb_coh": float(np.sum((Y[lo] * np.conj(W[lo])).real) / np.sum(np.abs(Y[lo]) ** 2))}
    for k, fr in masks.items():
        r[k] = float(ratio_curve(Sy, Sw, fr)[0].mean())
    return r


def ensemble(y, D, truth_lm):
    M = len(D); xb = D.mean(0); out = {}
    for tag, T, X in (("", y, D), ("_hb", fft_split(y, False), np.stack([fft_split(d, False) for d in D]))):
        XB = X.mean(0)
        ed = np.mean([np.sum((T - x) ** 2) for x in X]); em = np.sum((T - XB) ** 2); s2 = np.sum((X - XB[None]) ** 2) / (M - 1)
        out["gap" + tag] = float(10 * np.log10(ed / em)); out["spread" + tag] = float(s2 * (1 + 1 / M) / em)
        out["_s2" + tag], out["_em" + tag], out["_ed" + tag] = float(s2), float(em), float(ed)
    ens = np.stack([lm_hb(d)[: truth_lm.shape[0]] for d in D])
    out["crps"] = G["crps_ensemble"](ens, truth_lm); out["crps_fair"] = crps_fair(ens, truth_lm)
    out["sliced_crps"] = G["crps_ensemble"](ens @ THETA, truth_lm @ THETA)
    rk = G["pit_ranks"](ens, truth_lm); out["_pit"] = np.bincount(rk, minlength=M + 1).tolist()
    out["_sk"] = G["spread_skill"](ens, truth_lm).tolist()
    out["snr_mean"] = G["snr_db"](y, xb)
    return out, xb


def score_utt(args):
    ws, u, models, M_cap, do_perc = args
    utt = u["utt"]; y = np.load(ws / "truth" / f"{utt}.npy"); n = len(y)
    Sy = G["stft"](y, CFG.eval_n_fft, CFG.eval_hop)[0]
    fe = 20 * np.log10(np.abs(Sy).sum(1) + 1e-12); p25, p75 = np.percentile(fe, [25, 75])
    masks = {"loud": fe >= p75, "mid": (fe >= p25) & (fe < p75), "quiet": fe < p25}
    # the naive baseline and every `| pt` low band are built from the input THIS set gave the models
    # (lo12k/<utt>.wav: Chebyshev on the paper-split sets, resample_poly on ourtest_poly), never recomputed
    import soundfile as sf
    x_lo, _ = sf.read(str(ws / "lo12k" / f"{utt}.wav"), dtype="float64"); nv = up_naive(x_lo, 4, n); nv_lo = fft_split(nv, True)
    pt = lambda w: nv_lo + fft_split(w, False)
    truth_lm = lm_hb(y)
    conds = {"naive": (np.stack([nv]), 1.0), "ceiling": (np.stack([fft_split(y, True)]), 1.0)}
    for m in models:
        fs = sorted((ws / "out" / m).glob(f"{utt}_s*.npy"), key=lambda q: int(q.stem.rsplit("_s", 1)[1]))[:M_cap]
        if not fs:
            continue
        al = [align(y, np.load(q).astype(np.float64)[:n]) for q in fs]
        D = np.stack([a[0] for a in al]); g = float(np.mean([a[1] for a in al]))
        conds[m] = (D, g)
        if m.startswith("ours_"):
            conds[f"{m} | pt"] = (np.stack([pt(d) for d in D]), g)
            if len(D) >= 2:
                lm = G["logmag_ensemble_readout"](D, y, CFG, passthrough=False)
                conds[f"{m} logmean{len(D)} | pt"] = (np.stack([pt(lm)]), g)
    rows, frames = {}, {"truth": grouped(truth_lm)}
    for name, (D, g) in conds.items():
        per = [one(y, d, Sy, masks) for d in D]
        row = {"utt": utt, "M": len(D), "gain": g}
        for k in per[0]:
            row[k] = np.mean([r[k] for r in per], 0).tolist() if k == "curve" else float(np.mean([r[k] for r in per]))
        frames[name] = {"d0": grouped(lm_hb(D[0])[: truth_lm.shape[0]])}
        if len(D) >= 2:
            e, xb = ensemble(y, D, truth_lm); row.update(e)
            frames[name]["mean"] = grouped(lm_hb(xb)[: truth_lm.shape[0]])
        else:                                   # a point forecast: CRPS is its MAE, as in the datasheet
            ens = lm_hb(D[0])[None, : truth_lm.shape[0]]
            row["crps"] = G["crps_ensemble"](ens, truth_lm); row["sliced_crps"] = G["crps_ensemble"](ens @ THETA, truth_lm @ THETA)
        if do_perc:
            row.update(perceptual(y, D[0]))
        rows[name] = row
    return utt, rows, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="core")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--M", type=int, default=None, help="first M draws only")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-perceptual", dest="perc", action="store_false", default=True)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    ws = WORK / a.set
    utts = json.load(open(ws / "utts.json"))["utts"][: a.n]
    t0 = time.time()
    jobs = [(ws, u, a.models, a.M, a.perc) for u in utts]
    if a.workers > 1:
        import multiprocessing as mp
        init(False)                                   # HB_SEL etc. in the parent, for the metadata
        with mp.get_context("fork").Pool(a.workers, initializer=init, initargs=(a.perc,)) as pool:
            res = []
            for i, r in enumerate(pool.imap(score_utt, jobs)):
                res.append(r); print(f"  [{time.time()-t0:5.0f}s] {i+1}/{len(jobs)} {r[0]}", flush=True)
    else:
        init(a.perc); res = [score_utt(j) for j in jobs]
    rows, frames = {}, {}
    for utt, rs, fr in res:
        for k, r in rs.items():
            rows.setdefault(k, []).append(r)
        for k, v in fr.items():
            frames.setdefault(k, []).append(v)
    for r in rows["ceiling"]:                          # an empty high band has no coherence or kappa to speak of
        r["coh"] = r["kappa"] = r["coh_fft"] = float("nan")
    C_u = {r["utt"]: r["snr"] for r in rows["ceiling"]}
    for rs in rows.values():
        for r in rs:
            r["vs_ceiling"] = r["snr"] - C_u[r["utt"]]
            r["p_implied_db"] = float(10 * np.log10(max(10 ** (-r["vs_ceiling"] / 10) - 1, 1e-6)))
    Ct = np.corrcoef(np.vstack(frames["truth"]).T)
    agg = {}
    for name, rs in rows.items():
        g = {"n": len(rs)}
        for k in rs[0]:
            if k == "utt" or k.startswith("_"):
                continue
            v = [r[k] for r in rs if k in r]
            if isinstance(v[0], list):
                g[k] = np.mean(np.array(v), 0).tolist()
            else:
                v = np.array(v, np.float64); g[k] = float(np.nanmean(v))
                g[k + "_se"] = float(np.nanstd(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None
        M = rs[0]["M"]
        if "_s2" in rs[0]:
            for tag in ("", "_hb"):
                S2, EM, ED = (sum(r[k + tag] for r in rs) for k in ("_s2", "_em", "_ed"))
                g["spread_pooled" + tag] = S2 * (1 + 1 / M) / EM; g["gap_pooled" + tag] = float(10 * np.log10(ED / EM))
            h = np.sum([r["_pit"] for r in rs], 0).astype(float); h /= h.sum()
            g["pit_hist"] = h.tolist(); g["pit_lo"], g["pit_hi"] = float(h[0]), float(h[-1]); g["pit_end"] = float(h[0] + h[-1])
            sk = np.mean([r["_sk"] for r in rs], 0); g["spread_skill"] = sk.tolist()
            g["sk_ratio_low"], g["sk_ratio_high"] = float(sk[0, 1] / max(sk[0, 0], 1e-12)), float(sk[-1, 1] / max(sk[-1, 0], 1e-12))
        g["corr_err_d0"] = float(np.linalg.norm(Ct - np.corrcoef(np.vstack([f["d0"] for f in frames[name]]).T)))
        if "mean" in frames[name][0]:
            g["corr_err_mean"] = float(np.linalg.norm(Ct - np.corrcoef(np.vstack([f["mean"] for f in frames[name]]).T)))
        agg[name] = g
    try:
        import ai_edge_litert  # noqa: F401
        sp_map = "lattice"
    except Exception:
        sp_map = "polynomial"
    meta = {"set": a.set, "n_utts": len(utts), "speakers": sorted({u["utt"].split("_")[0] for u in utts}),
            "seconds": float(sum(u["seconds"] for u in utts)), "band_centres_hz": HB_CENTRES,
            "gain": f"each draw scaled so its energy below {ALIGN:.0f} Hz matches the truth's",
            "input": json.load(open(ws / "utts.json")).get("input", "cheby1(8, 0.05 dB) sosfiltfilt + resample_poly to 12 kHz"),
            "lsd_2048": "2048/512, log10|X|^2, RMS over frequency, mean over frames; HF/LF at 6 kHz (NU-Wave 2 for_test.py)",
            "lsd_1024": "notebook lsd_db, 1024/256, HB from eval_k_cut", "M_cap": a.M,
            "perceptual": a.perc, "visqol_speech_mapping": sp_map if a.perc else None,
            "speech_path": "evaluation.py (torchaudio resample to 16 kHz, visqol-python speech mode, torchmetrics PESQ wb)",
            "corr_err_mean": "from the ensemble-mean WAVEFORM's HB log-magnitude", "wall_s": time.time() - t0}
    json.dump({"_meta": meta, "agg": agg,
               "per_utt": {k: [{kk: vv for kk, vv in r.items() if not kk.startswith("_")} for r in v] for k, v in rows.items()}},
              open(ws / f"score_{a.set}.json", "w"), indent=1)
    md = table(agg, meta, a.models)
    (ws / f"score_{a.set}.md").write_text(md)
    print(md)


def table(agg, meta, models):
    def f(g, k, d=2):
        v = g.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        return ">100" if k.startswith("sk_ratio") and v > 100 else f"{v:.{d}f}"
    order = ["ceiling", "naive"]
    for m in models:
        order += [k for k in agg if k == m or k.startswith(m + " ")]
    nm = lambda k: NAMES.get(k.split(" ")[0], k.split(" ")[0]) + (" " + k.split(" ", 1)[1].replace("|", "\\|") if " " in k else "")
    L = [f"# Released SOTA vs ours, one scorer ({meta['set']}: {meta['n_utts']} utterances, {meta['seconds']:.0f} s, {', '.join(meta['speakers'])})", "",
         "12 kHz -> 48 kHz. Input: Chebyshev I order 8, 0.05 dB, sosfiltfilt, resample_poly (the FLowHigh / NU-Wave 2 test input). "
         "Every draw gain-aligned on its energy below 5.5 kHz. Per-draw numbers are the mean over draws, then over utterances. "
         f"ViSQOL speech mapping: {meta['visqol_speech_mapping']}.", "",
         "## 1. Levels", "", "Ideal: 0 for every column. LSD in decades of power; deficits in dB, negative = too quiet.", "",
         "| condition | M | LSD 2048 | LSD-HF 2048 | LSD-LF 2048 | LSD 1024 | HB-LSD 1024 | deficit | deficit bb | loud | mid | quiet |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in order:
        g = agg[k]
        L.append(f"| {nm(k)} | {int(g['M'])} | {f(g,'lsd',3)} | {f(g,'lsd_hf',3)} | {f(g,'lsd_lf',3)} | {f(g,'lsd_1024',3)} | {f(g,'hb_lsd_1024',3)} | "
                 f"{f(g,'deficit')} | {f(g,'deficit_bb')} | {f(g,'loud')} | {f(g,'mid')} | {f(g,'quiet')} |")
    L += ["", "## 2. SNR against the ceiling, and coherence", "",
          "Ceiling = truth brick-walled at 6 kHz. The HB error is E(1 + p - 2 coh): an incoherent high band at the right level costs about 3 dB "
          "against the ceiling. coh: the predictable fraction (0 = unpredictable). kappa: ideal 1 (0 = hallucinated energy). p implied: the HB "
          "level the SNR allows if coh = 0 and the LB is perfect (an upper bound on the true level). LB coh: alignment check, ideal 1.", "",
          "| condition | SNR | vs ceiling | coh | kappa | p implied dB | deficit bb dB | LB coh | gain |", "|---|---|---|---|---|---|---|---|---|"]
    for k in order:
        g = agg[k]
        L.append(f"| {nm(k)} | {f(g,'snr')} ± {f(g,'snr_se')} | {f(g,'vs_ceiling')} | {f(g,'coh',3)} | {f(g,'kappa',3)} | {f(g,'p_implied_db')} | "
                 f"{f(g,'deficit_bb')} | {f(g,'lb_coh',3)} | {f(g,'gain',3)} |")
    ens = [k for k in order if "gap" in agg[k]]
    if ens:
        M = int(agg[ens[0]]["M"])
        L += ["", f"## 3. Calibration (M = {M})", "",
              f"Ideals: gap {10*np.log10(2/(1+1/M)):.2f} dB; spread 1 (<1 = under-dispersed); PIT low and high {1/(M+1):.3f} each, end {2/(M+1):.3f}; "
              f"RMSE/spread in each spread-skill bin ~{np.sqrt((M+1)/M):.2f}; corr_err 0.", "",
              "| condition | gap | gap HB | spread | spread HB (pooled) | PIT low | PIT high | PIT end | RMSE/spread, low bin | high bin | corr_err draw 0 | corr_err mean | SNR of mean |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in ens:
            g = agg[k]
            L.append(f"| {nm(k)} | {f(g,'gap')} | {f(g,'gap_hb')} | {f(g,'spread',3)} | {f(g,'spread_pooled_hb',3)} | {f(g,'pit_lo',3)} | {f(g,'pit_hi',3)} | "
                     f"{f(g,'pit_end',3)} | {f(g,'sk_ratio_low')} | {f(g,'sk_ratio_high')} | {f(g,'corr_err_d0',3)} | {f(g,'corr_err_mean',3)} | {f(g,'snr_mean')} |")
    L += ["", "## 4. Proper scores on the HB log-magnitude", "",
          "Lower is better. A point forecast (M = 1) scores its MAE: it is the reference line, not a competitor. CRPS is the datasheet's "
          "estimator (spread over all M^2 pairs, which under-credits a sampler's spread by (M-1)/M); CRPS fair uses M(M-1). corr_err ideal 0.", "",
          "| condition | M | CRPS | CRPS fair | sliced CRPS | corr_err draw 0 |", "|---|---|---|---|---|---|"]
    for k in order:
        if k == "ceiling":
            continue
        g = agg[k]
        L.append(f"| {nm(k)} | {int(g['M'])} | {f(g,'crps',4)} | {f(g,'crps_fair',4)} | {f(g,'sliced_crps',4)} | {f(g,'corr_err_d0',3)} |")
    if meta["perceptual"]:
        L += ["", "## 5. Perceptual (draw 0; the readout waveform for readouts)", "",
              "Ceilings: ViSQOL audio 4.73 (NSIM 1), ViSQOL speech ~4.5 (NSIM 1), PESQ wb 4.64. Audio mode spans 1.57-4.73 on this task; "
              "speech mode and PESQ stop at 8 kHz, so they see only 6-8 kHz of the missing band.", "",
              "| condition | ViSQOL audio | NSIM audio | ViSQOL speech | NSIM speech | PESQ wb |", "|---|---|---|---|---|---|"]
        for k in order:
            g = agg[k]
            L.append(f"| {nm(k)} | {f(g,'visqol_audio')} | {f(g,'nsim_audio',3)} | {f(g,'visqol_speech')} | {f(g,'nsim_speech',3)} | {f(g,'pesq_wb')} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
