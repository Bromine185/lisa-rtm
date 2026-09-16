# ============================================================ OV3-4 evaluation of every OV3 arm on EVAL12
# overnight2/c3_eval.py's machinery, copied (not exec'd), applied to CKPT / OV3_TAG.  Per arm: one draw and
# the 16-draw ensemble (CRPS, sliced CRPS, PIT, spread-skill, ensemble-mean metrics, coherent fraction and
# kappa), plus the log-magnitude ensemble readouts logmean16 / logmean4 (mean of log|STFT| over the draws,
# phase of draw 0), and every condition again with the baseband passed through (c7 hybrid: low band of the
# naive upsample, high band of the model).  Functions at top level; the main block runs only when
# OV3_RUN_EVAL is unset or True.  Requires: c0 boot, c1_model, e1_model (logmag_ensemble_readout).
import json, time, math, numpy as np, matplotlib.pyplot as plt, torch, soundfile as sf

OV3_TAG = globals().get("OV3_TAG", "OV3_es")
M_DRAWS = int(globals().get("OV3_M", 16))
spread = lambda xs, n: xs[:: max(1, len(xs) // n)][:n]

K_HB = CFG.eval_n_fft // 2 + 1 - CFG.eval_k_cut
_th = stream("ov2/theta").standard_normal((K_HB, 32))       # same directions as c3: sliced CRPS comparable with OV2
THETA = _th / np.linalg.norm(_th, axis=0, keepdims=True)
N_GROUPS = 16
GROUP = np.array_split(np.arange(K_HB), N_GROUPS)

def lm_hb(w):
    return logmag(stft(w, CFG.eval_n_fft, CFG.eval_hop)[0][:, CFG.eval_k_cut:])

def grouped(L):                                   # (T, K_HB) -> (T, N_GROUPS)
    return np.stack([L[:, g].mean(1) for g in GROUP], 1)

_FREQS = np.fft.rfftfreq(CFG.eval_n_fft, 1.0 / CFG.fs_hi)
_EDGES_HB = third_octave_edges(CFG.fs_hi, CFG.fs_lo / 2, CFG.fs_hi / 2 - 1)
HB_SEL = [(_FREQS >= a) & (_FREQS < b) for a, b in zip(_EDGES_HB[:-1], _EDGES_HB[1:]) if ((_FREQS >= a) & (_FREQS < b)).sum()]
HB_CENTRES = [float(np.sqrt(a * b)) for a, b in zip(_EDGES_HB[:-1], _EDGES_HB[1:]) if ((_FREQS >= a) & (_FREQS < b)).sum()]

def coherent(y, w):
    '''Per high-band third-octave: coherent fraction Re<Y,P>/<Y,Y> (= predictable fraction rho for the exact
    conditional mean; unbiased for an ensemble mean) and kappa = Re<Y,P>/<P,P> (1 = informative output energy,
    ~0 = hallucinated).  Plus the baseband pair as an alignment check.'''
    Y = stft(y, CFG.eval_n_fft, CFG.eval_hop)[0]; P = stft(w, CFG.eval_n_fft, CFG.eval_hop)[0]
    n = min(len(Y), len(P)); Y, P = Y[:n], P[:n]
    yy = np.array([np.sum(np.abs(Y[:, s]) ** 2) for s in HB_SEL]); pp = np.array([np.sum(np.abs(P[:, s]) ** 2) for s in HB_SEL])
    yp = np.array([np.sum((Y[:, s] * np.conj(P[:, s])).real) for s in HB_SEL])
    bb = _FREQS < CFG.fs_lo / 2
    bb_yp = np.sum((Y[:, bb] * np.conj(P[:, bb])).real)
    return (yp / np.maximum(yy, 1e-20), yp / np.maximum(pp, 1e-20), float(yy.sum() / max(np.sum(np.abs(Y) ** 2), 1e-20)),
            float(bb_yp / max(np.sum(np.abs(Y[:, bb]) ** 2), 1e-20)))

def wave_metrics(y, w):
    b = band_energy_ratio(y, w, CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop, 200.0, CFG.fs_hi / 2)
    hb = b[:, 0] >= CFG.fs_lo / 2
    coh, kappa, hb_frac, bb_coh = coherent(y, w)
    return dict(snr=snr_db(y, w), lsd=lsd_db(y, w, CFG.eval_n_fft, CFG.eval_hop),
                hb_lsd=lsd_db(y, w, CFG.eval_n_fft, CFG.eval_hop, CFG.eval_k_cut),
                deficit=float(b[hb, 1].mean()), baseband=float(b[~hb, 1].mean()), curve=b[:, 1].tolist(),
                hb_coh=float(np.mean(coh)), hb_kappa=float(np.mean(kappa)),
                hb_frac=hb_frac, bb_coh=bb_coh, coh=coh.tolist(), kappa=kappa.tolist())

def agg(rows, keys=None):
    keys = keys or [k for k in rows[0] if k not in ("curve", "coh", "kappa")]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    for arr in ("curve", "coh", "kappa"):
        if arr in rows[0]:
            out[arr] = np.mean([r[arr] for r in rows], 0).tolist()
    out["n"] = len(rows)
    return out

# ---- baseband passthrough (c7 hybrid on a waveform) --------------------------------------------------
def split_bands(w, fs, f_cut):
    W = np.fft.rfft(np.asarray(w, np.float64))
    k = int(round(f_cut * len(w) / fs))
    lo, hi = W.copy(), W.copy()
    lo[k:] = 0; hi[:k] = 0
    return np.fft.irfft(lo, n=len(w)), np.fft.irfft(hi, n=len(w))

def passthrough(y, w):
    '''Low band of the naive upsample of y (the given input) + high band of w, brick-wall at fs_lo/2.'''
    y = np.asarray(y, np.float64); n = len(y)
    nv = naive_upsample(y, CFG)[:n]; nv = np.pad(nv, (0, n - len(nv)))
    w = np.asarray(w, np.float64)[:n]; w = np.pad(w, (0, n - len(w)))
    return split_bands(nv, CFG.fs_hi, CFG.fs_lo / 2)[0] + split_bands(w, CFG.fs_hi, CFG.fs_lo / 2)[1]

def floor_ceiling(y):
    '''passthrough + empty high band; passthrough + true high band.'''
    y = np.asarray(y, np.float64)
    return passthrough(y, np.zeros_like(y)), passthrough(y, y)

def snr_gap_calibrated(M):
    '''Expected SNR(mean of M) - SNR(one draw) for a calibrated sampler: 10 log10(2 / (1 + 1/M)).'''
    return 10 * math.log10(2 / (1 + 1 / M))

# ---- models -----------------------------------------------------------------------------------------
def load_ov3(tag):
    models, arms = {}, {}
    for p in sorted((CKPT / tag).glob("*.pt")):
        models[p.stem], ck = load_arm(p)
        arms[p.stem] = tuple(ck["arm"])
        models[p.stem].eval()
    return models, arms

# ---- one model, M draws per utterance ----------------------------------------------------------------
def ensemble_eval(m, utts, M, tau, label):
    '''Deterministic arms: M=1 and CRPS = MAE of the point forecast.  Stochastic arms additionally get the
    ensemble mean, the logmean readouts (M draws and min(4, M) draws) and passthrough versions of everything.'''
    single, single_pt, mean_rows, mean_pt, crps, scrps, ranks, sk = [], [], [], [], [], [], [], []
    ro_spec = ([(f"logmean{M}", M)] + ([("logmean4", 4)] if M > 4 else [])) if M > 1 else []   # no duplicate keys at M <= 4
    ro_names = [n for n, _ in ro_spec]
    ro = {n: [] for n in ro_names}; ro_pt = {n: [] for n in ro_names}
    truth_frames, pred_frames = [], []
    for ui, y in enumerate(utts):
        y = np.asarray(y, np.float64)
        waves = np.stack([reconstruct(m, y, CFG, tau=tau, seed=s) for s in range(M)])
        truth = lm_hb(y)
        ens = np.stack([lm_hb(w)[:truth.shape[0]] for w in waves])
        crps.append(crps_ensemble(ens, truth))
        scrps.append(crps_ensemble(ens @ THETA, truth @ THETA))
        if M > 1:
            ranks.append(pit_ranks(ens, truth)); sk.append(spread_skill(ens, truth))
            wm = waves.mean(0)
            mean_rows.append(wave_metrics(y, wm)); mean_pt.append(wave_metrics(y, passthrough(y, wm)))
            for n, k in ro_spec:
                w_r = logmag_ensemble_readout(waves[:k], y, CFG, passthrough=False)
                ro[n].append(wave_metrics(y, w_r)); ro_pt[n].append(wave_metrics(y, passthrough(y, w_r)))
        single.append(wave_metrics(y, waves[0])); single_pt.append(wave_metrics(y, passthrough(y, waves[0])))
        truth_frames.append(grouped(truth)); pred_frames.append(grouped(ens[0]))
    Tt, Tp = np.vstack(truth_frames), np.vstack(pred_frames)
    corr_err = float(np.linalg.norm(np.corrcoef(Tt.T) - np.corrcoef(Tp.T)))
    res = {"single": agg(single), "single_pt": agg(single_pt), "crps": float(np.mean(crps)), "sliced_crps": float(np.mean(scrps)),
           "corr_err": corr_err, "tau": tau, "M": M}
    if M > 1:
        res["mean"] = agg(mean_rows); res["mean_pt"] = agg(mean_pt)
        res["readouts"] = {n: agg(ro[n]) for n in ro_names}
        res["readouts_pt"] = {n: agg(ro_pt[n]) for n in ro_names}
        r = np.concatenate(ranks)
        h = np.histogram(r, bins=M + 1, range=(-0.5, M + 0.5))[0]
        res["pit_hist"] = h.tolist()
        res["pit_end"] = float((h[0] + h[-1]) / max(h.sum(), 1))          # ideal 2/(M+1)
        res["spread_skill"] = np.mean(sk, axis=0).tolist()
        res["snr_gap"] = res["mean"]["snr"] - res["single"]["snr"]
        res["snr_gap_calibrated"] = snr_gap_calibrated(M)
    s = res["single"]
    line = (f"{label:<26} SNR {s['snr']:6.2f} LSD {s['lsd']:.3f} HB-LSD {s['hb_lsd']:.3f} def {s['deficit']:+6.2f} "
            f"| CRPS {res['crps']:.4f} sCRPS {res['sliced_crps']:.4f} | HB kappa {s['hb_kappa']:+.3f} | pt LSD {res['single_pt']['lsd']:.3f}")
    if M > 1:
        mm = res["mean"]; lm = res["readouts_pt"][ro_names[0]]
        line += (f" | mean: SNR {mm['snr']:6.2f} def {mm['deficit']:+6.2f} gap {res['snr_gap']:.2f} (cal {res['snr_gap_calibrated']:.2f}) "
                 f"PIT end {res['pit_end']:.3f} | {ro_names[0]}+pt LSD {lm['lsd']:.3f} def {lm['deficit']:+6.2f}")
    print(line, flush=True)
    return res

# ---- table / figures / audio ------------------------------------------------------------------------
def make_table(RES, M):
    ro = f"logmean{M}"
    lines = [f"| arm | SNR | LSD | HB-LSD | deficit dB | CRPS | sliced CRPS | HB kappa | PIT end-bins | mean-of-{M} SNR | mean deficit | SNR gap (calibrated {snr_gap_calibrated(M):.2f}) | one-draw passthrough LSD | {ro} LSD |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r0 in RES.items():
        if k.startswith("_"):
            continue
        r = r0["eval12"]; s = r["single"]
        row = (f"| {k} | {s['snr']:.2f} | {s['lsd']:.3f} | {s['hb_lsd']:.3f} | {s['deficit']:+.2f} | {r['crps']:.4f} | {r['sliced_crps']:.4f} | "
               f"{s['hb_kappa']:+.3f} | ")
        if "mean" in r:
            mm = r["mean"]
            row += (f"{r['pit_end']:.3f} | {mm['snr']:.2f} | {mm['deficit']:+.2f} | {r['snr_gap']:.2f} | {r['single_pt']['lsd']:.3f} | "
                    f"{r['readouts_pt'][ro]['lsd']:.3f} |")
        else:
            row += f"- | - | - | - | {r['single_pt']['lsd']:.3f} | - |"
        lines.append(row)
    lines.append(f"| floor: passthrough + empty HB | {RES['_floor']['snr']:.2f} | {RES['_floor']['lsd']:.3f} | {RES['_floor']['hb_lsd']:.3f} | {RES['_floor']['deficit']:+.2f} | - | - | - | - | - | - | - | - | - |")
    lines.append(f"| ceiling: passthrough + true HB | {RES['_ceiling']['snr']:.2f} | {RES['_ceiling']['lsd']:.3f} | {RES['_ceiling']['hb_lsd']:.3f} | {RES['_ceiling']['deficit']:+.2f} | - | - | - | - | - | - | - | - | - |")
    lines.append(f"\nEVAL12 n={RES['_n']}; PIT end-bins ideal {2/(M+1):.3f}; one-draw passthrough LSD is draw 0 (seed 0) with the baseband passed "
                 f"through; {ro} LSD is the {M}-draw log-magnitude readout with baseband passthrough (raw readout in the JSON). "
                 f"CRPS uses the notebook's crps_ensemble: all-pairs spread term over M^2 pairs incl. the diagonal (standard, not the fair "
                 f"M(M-1) estimator; spread under-credited by (M-1)/M), the same estimator as the OV2 tables.")
    return chr(10).join(lines)


def make_figures(RES, models, utts, tag, M):
    ro = f"logmean{M}"
    centres = band_energy_ratio(utts[0], utts[0], CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop, 200.0, CFG.fs_hi / 2)[:, 0]
    arms = [k for k in RES if not k.startswith("_")]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, k in enumerate(arms):
        r = RES[k]["eval12"]; c = cols[i % len(cols)]
        ax.semilogx(centres, r["single"]["curve"], "o-", ms=3, color=c, label=f"{k} (one draw)" if "mean" in r else k)
        if "mean" in r:
            ax.semilogx(centres, r["mean"]["curve"], "--", lw=1, color=c, label=f"{k} (mean of {M})")
            ax.semilogx(centres, r["readouts"][ro]["curve"], ":", lw=1.2, color=c, label=f"{k} ({ro})")
    ax.axhline(0, color="k", lw=1); ax.axvline(CFG.fs_lo / 2, color="crimson", ls="--"); ax.axhspan(-2, 2, color="grey", alpha=0.15)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("pred/target energy (dB)")
    ax.set_title(f"Energy ratio per band: one draw (solid), ensemble mean (dashed), {ro} readout (dotted)", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout(); plt.savefig(FIGS / f"ov3_spectrum_{tag}.png", dpi=130); plt.close(fig)

    stoch = [k for k in arms if "mean" in RES[k]["eval12"]]
    if stoch:
        fig, ax = plt.subplots(1, len(stoch) + 1, figsize=(3.4 * (len(stoch) + 1), 3.2))
        ax = np.atleast_1d(ax)
        for a, k in zip(ax, stoch):
            h = np.array(RES[k]["eval12"]["pit_hist"], float); h /= h.sum()
            a.bar(range(len(h)), h, edgecolor="k", lw=0.5); a.axhline(1 / len(h), color="crimson", ls="--")
            a.set_title(f"PIT {k}", fontsize=9)
        for k in stoch:
            s = np.array(RES[k]["eval12"]["spread_skill"])
            ax[-1].plot(s[:, 0], s[:, 1], "o-", ms=3, label=k)
        lim = ax[-1].get_xlim(); ax[-1].plot([0, lim[1]], [0, lim[1]], "k--", lw=1); ax[-1].set_title("spread-skill", fontsize=9); ax[-1].legend(fontsize=6)
        plt.tight_layout(); plt.savefig(FIGS / f"ov3_calibration_{tag}.png", dpi=130); plt.close(fig)


def write_audio(models, utts, out_dir, M, idx=(0, 5, 10)):
    out_dir.mkdir(parents=True, exist_ok=True)
    def w(name, x):
        sf.write(out_dir / f"{name}.wav", np.clip(np.asarray(x, np.float64), -1, 1), CFG.fs_hi)
    for ui in idx:
        if ui >= len(utts):
            continue
        y = np.asarray(utts[ui], np.float64)
        fl, ce = floor_ceiling(y)
        w(f"u{ui}_truth", y); w(f"u{ui}_naive", naive_upsample(y, CFG)); w(f"u{ui}_floor_pt", fl); w(f"u{ui}_ceiling_pt", ce)
        for k, m in models.items():
            d0 = reconstruct(m, y, CFG, tau=m.tau, seed=0)
            w(f"u{ui}_{k}", d0); w(f"u{ui}_{k}_pt", passthrough(y, d0))
            if m.tau > 0:
                d1 = reconstruct(m, y, CFG, tau=1.0, seed=1)
                w(f"u{ui}_{k}_draw2", d1); w(f"u{ui}_{k}_draw2_pt", passthrough(y, d1))
                draws = np.stack([d0, d1] + [reconstruct(m, y, CFG, tau=1.0, seed=s) for s in range(2, M)])
                w(f"u{ui}_{k}_logmean{M}", logmag_ensemble_readout(draws, y, CFG, passthrough=False))
                w(f"u{ui}_{k}_logmean{M}_pt", logmag_ensemble_readout(draws, y, CFG, passthrough=True))
    print("audio written to", out_dir, flush=True)


def run_eval(models, arms, utts, M, tag, audio=True):
    out = ROOT / "ov3"; out.mkdir(parents=True, exist_ok=True)
    utts = [np.asarray(u, np.float64) for u in utts]
    RES, t0 = {}, time.time()
    for k, m in models.items():
        stoch = m.tau > 0
        RES[k] = {"arm": list(arms[k]), "eval12": ensemble_eval(m, utts, M if stoch else 1, 1.0 if stoch else 0.0, f"{k} tau={m.tau:g}")}
        print(f"  [{time.time()-t0:.0f}s]", flush=True)
    fl, ce = zip(*[floor_ceiling(y) for y in utts])
    RES["_floor"] = agg([wave_metrics(y, f) for y, f in zip(utts, fl)])
    RES["_ceiling"] = agg([wave_metrics(y, c) for y, c in zip(utts, ce)])
    RES["_n"] = len(utts); RES["_M"] = M
    json.dump(RES, open(out / f"results_{tag}.json", "w"), indent=1)
    TABLE = make_table(RES, M)
    (out / f"table_{tag}.md").write_text(TABLE)
    make_figures(RES, models, utts, tag, M)
    if audio:
        write_audio(models, utts, out / "audio", M)
    print(TABLE, flush=True)
    return RES, TABLE


if globals().get("OV3_RUN_EVAL", True):
    models_ov3, ARMS_OV3 = load_ov3(OV3_TAG)
    print("arms:", {k: (ARMS_OV3[k], models_ov3[k].tau) for k in models_ov3}, flush=True)
    EVAL12 = [np.asarray(u, np.float64) for u in test_utts[:12]]
    RES3, TABLE3 = run_eval(models_ov3, ARMS_OV3, EVAL12, M_DRAWS, OV3_TAG)
    print("EVAL DONE", OV3_TAG, flush=True)
