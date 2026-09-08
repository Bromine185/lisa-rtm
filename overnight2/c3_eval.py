# ============================================================ OV2-3 evaluation of every arm
# Deterministic metrics (SNR, LSD, HB-LSD, deficit) on three splits; for the stochastic arms also
# the ensemble (M draws): CRPS on high-band log-magnitude, sliced CRPS (joint over bins), PIT,
# spread-skill, ensemble-mean metrics, tau sweep, cross-bin correlation structure.  Plus latency and
# audio examples to Drive.
import json, time, numpy as np, matplotlib.pyplot as plt, torch, soundfile as sf

OV2_TAG = globals().get("OV2_TAG", "OV2_es")
OUT = ROOT / "ov2"
(OUT / "audio").mkdir(parents=True, exist_ok=True)
if "models_ov2" not in globals():
    models_ov2 = {}
    for p in sorted((CKPT / OV2_TAG).glob("*.pt")):
        models_ov2[p.stem], _ = load_arm(p)
    ARMS = {k: torch.load(CKPT / OV2_TAG / f"{k}.pt", map_location="cpu", weights_only=False)["arm"] for k in models_ov2}
ARM_ORDER = list(models_ov2)
for m in models_ov2.values():
    m.eval()
M_DRAWS = 16
spread = lambda xs, n: xs[:: max(1, len(xs) // n)][:n]
EVAL12 = [np.asarray(u, np.float64) for u in test_utts[:12]]
PAPER100 = spread(paper_test_utts, 100)

K_HB = CFG.eval_n_fft // 2 + 1 - CFG.eval_k_cut
_th = stream("ov2/theta").standard_normal((K_HB, 32))
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


def ensemble_eval(m, utts, M, tau, label, corr_pool=None):
    '''One model, M draws per utterance.  Deterministic arms: M=1 and CRPS = MAE of the point forecast.'''
    single, mean_rows, crps, scrps, ranks, sk = [], [], [], [], [], []
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
            mean_rows.append(wave_metrics(y, waves.mean(0)))
        single.append(wave_metrics(y, waves[0]))
        truth_frames.append(grouped(truth)); pred_frames.append(grouped(ens[0]))
    Tt, Tp = np.vstack(truth_frames), np.vstack(pred_frames)
    corr_err = float(np.linalg.norm(np.corrcoef(Tt.T) - np.corrcoef(Tp.T)))
    res = {"single": agg(single), "crps": float(np.mean(crps)), "sliced_crps": float(np.mean(scrps)),
           "corr_err": corr_err, "tau": tau, "M": M}
    if M > 1:
        res["mean"] = agg(mean_rows)
        r = np.concatenate(ranks)
        res["pit_hist"] = np.histogram(r, bins=M + 1, range=(-0.5, M + 0.5))[0].tolist()
        res["spread_skill"] = np.mean(sk, axis=0).tolist()
    s = res["single"]
    line = (f"{label:<34} SNR {s['snr']:6.2f} LSD {s['lsd']:.3f} HB-LSD {s['hb_lsd']:.3f} def {s['deficit']:+6.2f} "
            f"| CRPS {res['crps']:.4f} sCRPS {res['sliced_crps']:.4f} corr {corr_err:.3f} | HB coh {s['hb_coh']:+.3f} kappa {s['hb_kappa']:+.3f} (BB coh {s['bb_coh']:.3f})")
    if M > 1:
        mm = res["mean"]
        line += f" | mean: SNR {mm['snr']:6.2f} def {mm['deficit']:+6.2f} HB-LSD {mm['hb_lsd']:.3f} HB coh {mm['hb_coh']:+.3f}"
    print(line, flush=True)
    return res


RES2 = {}
t0 = time.time()
for k in ARM_ORDER:
    m = models_ov2[k]
    stoch = m.tau > 0
    r = {"arm": ARMS[k]}
    r["eval12"] = ensemble_eval(m, EVAL12, M_DRAWS if stoch else 1, 1.0 if stoch else 0.0, f"{k} eval12 tau={m.tau:g}")
    if stoch:
        r["eval12_tau0"] = ensemble_eval(m, EVAL12, 1, 0.0, f"{k} eval12 tau=0")
    r["ours120"] = ensemble_eval(m, test_utts, 1, m.tau, f"{k} ours120 single")
    r["paper100"] = ensemble_eval(m, PAPER100, 1, m.tau, f"{k} paper100 single")
    r["phase12"] = [float(v) for v in phase_swap(m, EVAL12, 12, f"{k} phase-swap")]
    RES2[k] = r
    print(f"  [{time.time()-t0:.0f}s]", flush=True)
json.dump(RES2, open(OUT / f"results_{OV2_TAG}.json", "w"), indent=1)

# ---- tau sweep on the stochastic arms ------------------------------------------------------------
TAUS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
SWEEP = {}
for k in ARM_ORDER:
    if models_ov2[k].tau == 0:
        continue
    SWEEP[k] = [ensemble_eval(models_ov2[k], EVAL12, 8, t, f"{k} tau={t:g}") for t in TAUS]
json.dump(SWEEP, open(OUT / f"tausweep_{OV2_TAG}.json", "w"), indent=1)

# ---- figures -------------------------------------------------------------------------------------
centres = band_energy_ratio(EVAL12[0], EVAL12[0], CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop, 200.0, CFG.fs_hi / 2)[:, 0]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for k in ARM_ORDER:
    r = RES2[k]["eval12"]
    ax[0].semilogx(centres, r["single"]["curve"], "o-", ms=3, label=f"{k} (one draw)" if models_ov2[k].tau else k)
    if "mean" in r:
        ax[0].semilogx(centres, r["mean"]["curve"], "--", lw=1, label=f"{k} (mean of {r['M']})")
ax[0].axhline(0, color="k", lw=1); ax[0].axvline(CFG.fs_lo / 2, color="crimson", ls="--"); ax[0].axhspan(-2, 2, color="grey", alpha=0.15)
ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("pred/target energy (dB)"); ax[0].set_title("Energy ratio: samples vs ensemble mean")
ax[0].legend(fontsize=7)
for k, curve in SWEEP.items():
    ax[1].plot([c["single"]["snr"] for c in curve], [c["crps"] for c in curve], "o-", ms=4, label=k)
    for c in curve:
        ax[1].annotate(f"{c['tau']:g}", (c["single"]["snr"], c["crps"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
for k in ARM_ORDER:
    if models_ov2[k].tau == 0:
        r = RES2[k]["eval12"]
        ax[1].scatter([r["single"]["snr"]], [r["crps"]], marker="s", s=40, label=k)
ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("CRPS on HB log-magnitude"); ax[1].set_title("noise temperature tau: fidelity vs proper score")
ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIGS / f"ov2_spectrum_tau_{OV2_TAG}.png", dpi=130); plt.show()

stoch_arms = [k for k in ARM_ORDER if models_ov2[k].tau > 0]
fig, ax = plt.subplots(1, len(stoch_arms) + 1, figsize=(4 * (len(stoch_arms) + 1), 3.4))
for a, k in zip(ax, stoch_arms):
    h = np.array(RES2[k]["eval12"]["pit_hist"], float); h /= h.sum()
    a.bar(range(len(h)), h, edgecolor="k"); a.axhline(1 / len(h), color="crimson", ls="--")
    a.set_title(f"PIT {k}", fontsize=9)
for k in stoch_arms:
    s = np.array(RES2[k]["eval12"]["spread_skill"])
    ax[-1].plot(s[:, 0], s[:, 1], "o-", ms=3, label=k)
lim = ax[-1].get_xlim(); ax[-1].plot([0, lim[1]], [0, lim[1]], "k--", lw=1); ax[-1].set_title("spread-skill"); ax[-1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIGS / f"ov2_calibration_{OV2_TAG}.png", dpi=130); plt.show()

# ---- latency -------------------------------------------------------------------------------------
def bench(fn, n_rep=20, warmup=3):
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    return np.percentile(ts, 50) * 1e3, np.percentile(ts, 95) * 1e3

LAT = {}
m = models_ov2[stoch_arms[0]] if stoch_arms else models_ov2[ARM_ORDER[0]]
for name, n in [("1 s", CFG.fs_hi), ("20 ms chunk", CFG.fs_hi // 50)]:
    y_b = EVAL12[0][:n]
    x_lo = torch.from_numpy(decimate(y_b, CFG.upsample)).float()[None].to(DEVICE)
    eps = m.sample_eps(x_lo, 1.0, 0)
    p50, p95 = bench(lambda: m(x_lo, eps=eps))
    LAT[name] = (p50, p95)
    print(f"latency {name:<12} on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else DEVICE}: p50 {p50:.2f} ms  p95 {p95:.2f} ms  (stochastic forward, one draw)")
json.dump(LAT, open(OUT / f"latency_{OV2_TAG}.json", "w"))

# ---- audio examples -----------------------------------------------------------------------------
for ui in (0, 5, 10):
    y = EVAL12[ui]
    sf.write(OUT / "audio" / f"u{ui}_truth.wav", np.clip(y, -1, 1), CFG.fs_hi)
    sf.write(OUT / "audio" / f"u{ui}_naive.wav", np.clip(naive_upsample(y, CFG), -1, 1), CFG.fs_hi)
    for k in ARM_ORDER:
        m = models_ov2[k]
        sf.write(OUT / "audio" / f"u{ui}_{k}.wav", np.clip(reconstruct(m, y, CFG, tau=m.tau, seed=0), -1, 1), CFG.fs_hi)
        if m.tau > 0:
            sf.write(OUT / "audio" / f"u{ui}_{k}_draw2.wav", np.clip(reconstruct(m, y, CFG, tau=1.0, seed=1), -1, 1), CFG.fs_hi)
            sf.write(OUT / "audio" / f"u{ui}_{k}_tau0.wav", np.clip(reconstruct(m, y, CFG, tau=0.0), -1, 1), CFG.fs_hi)
print("audio written to", OUT / "audio")

# ---- markdown table -----------------------------------------------------------------------------
lines = ["| arm | split | SNR | LSD | HB-LSD | deficit dB | CRPS | sliced CRPS | corr err | mean-of-16 SNR | mean deficit |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for k in ARM_ORDER:
    for split in ("eval12", "eval12_tau0", "ours120", "paper100"):
        if split not in RES2[k]:
            continue
        r = RES2[k][split]; s = r["single"]; mm = r.get("mean")
        lines.append(f"| {k} | {split} (tau={r['tau']:g}) | {s['snr']:.2f} | {s['lsd']:.3f} | {s['hb_lsd']:.3f} | {s['deficit']:+.2f} | "
                     f"{r['crps']:.4f} | {r['sliced_crps']:.4f} | {r['corr_err']:.3f} | "
                     f"{mm['snr']:.2f} | {mm['deficit']:+.2f} |" if mm else
                     f"| {k} | {split} (tau={r['tau']:g}) | {s['snr']:.2f} | {s['lsd']:.3f} | {s['hb_lsd']:.3f} | {s['deficit']:+.2f} | "
                     f"{r['crps']:.4f} | {r['sliced_crps']:.4f} | {r['corr_err']:.3f} | - | - |")
TABLE = chr(10).join(lines)
(OUT / f"table_{OV2_TAG}.md").write_text(TABLE)
print(TABLE)
print("EVAL DONE", flush=True)
