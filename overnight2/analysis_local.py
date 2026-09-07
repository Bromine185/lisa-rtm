"""Independent CPU analysis of the OV2 arms on the Mac, from the Drive-synced checkpoints and the
ORIGINAL test_FULL.npz (DataShare-derived, 12 utterances per held-out speaker).

Produces, per arm: single-draw metrics, ensemble-mean metrics (M draws, bias-corrected), CRPS / sliced
CRPS / PIT on high-band log-magnitude, the predictability spectrum rho(f), and the deterministic SNR
bound implied by it.  Writes overnight2/analysis_OV2.json and two figures.

usage: venv/bin/python overnight2/analysis_local.py [TAG] [M] [N_UTTS]
"""
import json, os, sys, pathlib, time, math
import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
REPO = pathlib.Path(__file__).resolve().parents[1]
DRIVE = pathlib.Path.home() / "Library/CloudStorage/GoogleDrive-naghavnarna@gmail.com/My Drive/lisa_rtm"
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
TAG = _args[0] if len(_args) > 0 else "OV2_es"
M = int(_args[1]) if len(_args) > 1 else 32
N_UTTS = int(_args[2]) if len(_args) > 2 else 36
CKDIR = pathlib.Path(_args[3]) if len(_args) > 3 else DRIVE / "checkpoints" / TAG

NB = json.loads((REPO / "lisa_rtm.ipynb").read_text())
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
G = {"__name__": "__main__"}
def run(mark):
    exec(compile(next(s for s in CODE if mark in s), f"<nb {mark[:20]}>", "exec"), G)
run("import importlib, subprocess, sys")
import torch
G["DEVICE"] = torch.device("cpu")
run("@dataclasses.dataclass(frozen=True)"); G["CFG"] = G["FULL"]
run("def _hann(n):"); run("class QuantileMap:"); run("def snr_db(y, y_hat):"); run("VCTK_URL = ")
run("class LISAEncoder(nn.Module):")
exec(open(REPO / "overnight/cell2_trainer.py").read().split("def train_paired")[0], G)
exec("def save_ckpt" + open(REPO / "overnight/cell2_trainer.py").read().split("def save_ckpt")[1], G)
exec(open(REPO / "overnight/cell3_eval.py").read(), G)
torch.use_deterministic_algorithms(False)
torch.set_num_threads(os.cpu_count())
exec(open(REPO / "overnight2/c1_model.py").read(), G)
exec(open(REPO / "overnight2/c1b_wide.py").read(), G)
CFG, stft, logmag = G["CFG"], G["stft"], G["logmag"]
snr_db, lsd_db, band_energy_ratio = G["snr_db"], G["lsd_db"], G["band_energy_ratio"]
crps_ensemble, pit_ranks, spread_skill = G["crps_ensemble"], G["pit_ranks"], G["spread_skill"]
reconstruct, naive_upsample, third_octave_edges = G["reconstruct"], G["naive_upsample"], G["third_octave_edges"]

with np.load(DRIVE / "fixtures/test_FULL.npz", allow_pickle=True) as d:
    test_utts, test_spk = list(d["utts"]), list(d["speakers"])
per = max(1, N_UTTS // 3)
EVAL = [np.asarray(u, np.float64) for i, u in enumerate(test_utts) if i % 40 < per][:N_UTTS]
print(f"eval set: {len(EVAL)} utts from {sorted(set(test_spk))}, {sum(len(u) for u in EVAL)/CFG.fs_hi:.0f} s of audio", flush=True)

K_HB = CFG.eval_n_fft // 2 + 1 - CFG.eval_k_cut
_th = G["stream"]("ov2/theta").standard_normal((K_HB, 32)); THETA = _th / np.linalg.norm(_th, axis=0, keepdims=True)
def lm_hb(w):
    return logmag(stft(w, CFG.eval_n_fft, CFG.eval_hop)[0][:, CFG.eval_k_cut:])

# third-octave band powers of the target, for the SNR bound
edges = third_octave_edges(CFG.fs_hi, 200.0, CFG.fs_hi / 2 - 1)
freqs = np.fft.rfftfreq(CFG.eval_n_fft, 1.0 / CFG.fs_hi)
def band_powers(w):
    S = np.abs(stft(w, CFG.eval_n_fft, CFG.eval_hop)[0]) ** 2
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (freqs >= a) & (freqs < b)
        if sel.sum():
            out.append(S[:, sel].sum())
    return np.array(out)
# high-band third-octave bands starting exactly at the input Nyquist; total power over ALL bins
EDGES_HB = third_octave_edges(CFG.fs_hi, CFG.fs_lo / 2, CFG.fs_hi / 2 - 1)
HB_SEL = [(freqs >= a) & (freqs < b) for a, b in zip(EDGES_HB[:-1], EDGES_HB[1:]) if ((freqs >= a) & (freqs < b)).sum()]
HB_CENTRES = np.array([math.sqrt(a * b) for a, b in zip(EDGES_HB[:-1], EDGES_HB[1:]) if ((freqs >= a) & (freqs < b)).sum()])
BB_SEL = freqs < CFG.fs_lo / 2

def band_cross(y, w):
    '''Coherent fraction  Re<Y,P>/<Y,Y>  per high-band third-octave (share of target power the prediction
    reproduces coherently; equals the predictable fraction rho for the exact conditional mean and is unbiased
    for an M-draw ensemble mean), kappa = Re<Y,P>/<P,P> (1 = every unit of output power is informative,
    ~0 = the output energy is hallucinated), the target power fraction per high band (of ALL power), and the
    same two coherences for the whole baseband as an alignment check (both should be ~1).'''
    Y = stft(y, CFG.eval_n_fft, CFG.eval_hop)[0]; P = stft(w, CFG.eval_n_fft, CFG.eval_hop)[0]
    n = min(len(Y), len(P)); Y, P = Y[:n], P[:n]
    tot = np.sum(np.abs(Y) ** 2)
    yy = np.array([np.sum(np.abs(Y[:, sel]) ** 2) for sel in HB_SEL])
    pp = np.array([np.sum(np.abs(P[:, sel]) ** 2) for sel in HB_SEL])
    yp = np.array([np.sum((Y[:, sel] * np.conj(P[:, sel])).real) for sel in HB_SEL])
    bb_yy = np.sum(np.abs(Y[:, BB_SEL]) ** 2); bb_pp = np.sum(np.abs(P[:, BB_SEL]) ** 2)
    bb_yp = np.sum((Y[:, BB_SEL] * np.conj(P[:, BB_SEL])).real)
    return (yp / np.maximum(yy, 1e-20), yp / np.maximum(pp, 1e-20), yy / max(tot, 1e-20),
            bb_yp / max(bb_yy, 1e-20), bb_yp / max(bb_pp, 1e-20))

def metrics(y, w):
    b = band_energy_ratio(y, w, CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop, 200.0, CFG.fs_hi / 2)
    hb = b[:, 0] >= CFG.fs_lo / 2
    coh, kappa, pfrac, bb_coh, bb_kappa = band_cross(y, w)
    return dict(snr=snr_db(y, w), lsd=lsd_db(y, w, CFG.eval_n_fft, CFG.eval_hop),
                hb_lsd=lsd_db(y, w, CFG.eval_n_fft, CFG.eval_hop, CFG.eval_k_cut),
                deficit=float(b[hb, 1].mean()), curve=b[:, 1], coh=coh, kappa=kappa, pfrac=pfrac, bb_coh=bb_coh, bb_kappa=bb_kappa)
CENTRES = band_energy_ratio(EVAL[0], EVAL[0], CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop, 200.0, CFG.fs_hi / 2)[:, 0]
HB = CENTRES >= CFG.fs_lo / 2
P_TRUE = np.mean([band_powers(y) / band_powers(y).sum() for y in EVAL], 0)      # target power fraction per band
P_TOTAL_HB = float(P_TRUE[HB].sum())
NAIVE = float(np.mean([snr_db(y, naive_upsample(y, CFG)) for y in EVAL]))
_pf = [band_cross(y, y)[2] for y in EVAL]
P_TRUE_HB = np.mean(_pf, 0)                      # mean target power fraction per high band
NAIVE_BOUND = float(np.mean([10 * math.log10(1 / max(float(np.sum(pf)), 1e-12)) for pf in _pf]))
print(f"target power above {CFG.fs_lo/2:.0f} Hz: {100*P_TOTAL_HB:.2f} % of total   naive-upsample SNR {NAIVE:.2f} dB   "
      f"(bound with rho=0 above the cut: {NAIVE_BOUND:.2f} dB -- sanity: should sit just above naive)", flush=True)
RES_NAIVE_BOUND = NAIVE_BOUND

def wrap_lisa(path):
    '''Load a plain LISA checkpoint (XL4 run) into LISAS: noise-channel weights zero, so eps=0 is identical.'''
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    if any(k.startswith("dec.net") for k in sd) and "enc.net.0.weight" in sd and sd["enc.net.0.weight"].shape[1] == 1:
        m = G["LISAS"](CFG)
        new = {}
        for k, v in sd.items():
            nk = k.replace("enc.net.", "enc.")
            if nk == "enc.0.weight":
                w = torch.zeros_like(m.state_dict()[nk]); w[:, :1] = v; v = w
            new[nk] = v
        m.load_state_dict(new); m.eval(); m.tau = 0.0
        ck["arm"] = ("det", ck.get("arm", ("relu", ck.get("lambda", float("nan"))))[1] if isinstance(ck.get("arm"), (tuple, list)) else ck.get("lambda"))
        return m, ck
    return None, ck

EXTRA = [a for a in sys.argv if a.startswith("--extra=")]
TAU_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def avg(dicts, keys):
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}

RES = {"tag": TAG, "M": M, "n_utts": len(EVAL), "naive_snr": NAIVE, "naive_bound": NAIVE_BOUND, "centres": CENTRES.tolist(), "p_true": P_TRUE.tolist()}
t0 = time.time()
for p in sorted(CKDIR.glob("*.pt")):
    m, ck = G["load_arm"](p)
    k = p.stem
    stoch = m.tau > 0
    single, mean_rows, crps, scrps, ranks, sk, curves_mean = [], [], [], [], [], [], []
    for y in EVAL:
        draws = np.stack([reconstruct(m, y, CFG, tau=1.0 if stoch else 0.0, seed=s) for s in range(M if stoch else 1)])
        truth = lm_hb(y)
        ens = np.stack([lm_hb(w)[:truth.shape[0]] for w in draws])
        crps.append(crps_ensemble(ens, truth)); scrps.append(crps_ensemble(ens @ THETA, truth @ THETA))
        single.append(metrics(y, draws[0]))
        if stoch:
            ranks.append(pit_ranks(ens, truth)); sk.append(spread_skill(ens, truth))
            mean_rows.append(metrics(y, draws.mean(0)))
    r = {"arm": ck["arm"], "step": ck["step"], "stochastic": stoch,
         "single": avg(single, ("snr", "lsd", "hb_lsd", "deficit")), "crps": float(np.mean(crps)), "sliced_crps": float(np.mean(scrps))}
    r["single"]["curve"] = np.mean([s["curve"] for s in single], 0).tolist()
    r["single"]["coh"] = np.mean([s["coh"] for s in single], 0).tolist()
    r["single"]["kappa"] = np.mean([s["kappa"] for s in single], 0).tolist()
    best = single                                     # rows whose coherent fraction estimates rho
    if stoch:
        r["mean"] = avg(mean_rows, ("snr", "lsd", "hb_lsd", "deficit"))
        r["mean"]["curve"] = np.mean([s["curve"] for s in mean_rows], 0).tolist()
        r["mean"]["coh"] = np.mean([s["coh"] for s in mean_rows], 0).tolist()
        r["mean"]["kappa"] = np.mean([s["kappa"] for s in mean_rows], 0).tolist()
        rr = np.concatenate(ranks)
        r["pit_hist"] = np.histogram(rr, bins=M + 1, range=(-0.5, M + 0.5))[0].tolist()
        r["spread_skill"] = np.mean(sk, 0).tolist()
        # energy-based predictable fraction, bias-corrected for the M-draw mean: E[mean^2] = rho + (1-rho)/M
        ratio = 10 ** (np.array(r["mean"]["curve"]) / 10)
        r["rho_energy"] = np.clip((M * ratio - 1) / (M - 1), 1e-4, 1.0).tolist()
        # waveform calibration test: a calibrated sampler's single draw has twice the squared error of the
        # conditional mean; against an M-draw mean the expected gap is 10log10(2/(1+1/M)).
        r["snr_gap_db"] = r["mean"]["snr"] - r["single"]["snr"]
        r["snr_gap_calibrated_db"] = 10 * math.log10(2 / (1 + 1 / M))
        best = mean_rows
    # predictability spectrum = coherent fraction of the best point estimate this arm offers (unbiased in M)
    rho = np.clip(np.mean([s["coh"] for s in best], 0), 0.0, 1.0)
    r["rho"] = rho.tolist()
    # deterministic SNR bound, per utterance then averaged in dB like SNR itself: the unpredictable power is
    # sum over HIGH bands of (1 - rho_b) * P_u(b); the baseband is treated as reproducible (naive upsampling shows it is)
    bounds = [10 * math.log10(1 / max(float(np.sum((1 - rho) * s["pfrac"])), 1e-12)) for s in best]
    r["snr_bound_db"] = float(np.mean(bounds))
    r["predictable_hb_fraction"] = float(np.sum(rho * P_TRUE_HB) / P_TRUE_HB.sum())
    r["hb_kappa"] = float(np.mean(r["single"]["kappa"]))
    r["bb_coh"] = float(np.mean([s["bb_coh"] for s in single])); r["bb_kappa"] = float(np.mean([s["bb_kappa"] for s in single]))
    RES[k] = r
    s = r["single"]
    line = f"{k:<10} step {ck['step']:>6} | one draw: SNR {s['snr']:6.2f} LSD {s['lsd']:.3f} HB-LSD {s['hb_lsd']:.3f} def {s['deficit']:+6.2f} | CRPS {r['crps']:.4f} sCRPS {r['sliced_crps']:.4f}"
    if stoch:
        mm = r["mean"]
        line += f" | mean-of-{M}: SNR {mm['snr']:6.2f} def {mm['deficit']:+6.2f} HB-LSD {mm['hb_lsd']:.3f} | SNR gap {r['snr_gap_db']:.2f} dB (calibrated: {r['snr_gap_calibrated_db']:.2f})"
    line += f" | BB coh/kappa {r['bb_coh']:.3f}/{r['bb_kappa']:.3f}  HB kappa {r['hb_kappa']:+.3f}  predictable HB {100*r['predictable_hb_fraction']:.1f}%  SNR bound {r['snr_bound_db']:.1f} dB  [{time.time()-t0:.0f}s]"
    print(line, flush=True)

# ---- lambda frontier from the previous run's deterministic checkpoints -----------------------------
FRONTIER = {}
for ex in EXTRA:
    exdir = DRIVE / "checkpoints" / ex.split("=", 1)[1]
    for p in sorted(exdir.glob("relu_*.pt")):
        m, ck = wrap_lisa(p)
        if m is None:
            continue
        rows = [metrics(y, reconstruct(m, y, CFG, tau=0.0)) for y in EVAL]
        crps_pt = float(np.mean([np.abs(lm_hb(reconstruct(m, y, CFG, tau=0.0))[:lm_hb(y).shape[0]] - lm_hb(y)).mean() for y in EVAL]))
        FRONTIER[f"{exdir.name}/{p.stem}"] = {**avg(rows, ("snr", "lsd", "hb_lsd", "deficit")), "crps": crps_pt, "step": ck["step"],
                                              "curve": np.mean([r["curve"] for r in rows], 0).tolist()}
        f = FRONTIER[f"{exdir.name}/{p.stem}"]
        print(f"frontier {p.stem:<12} step {ck['step']:>6} | SNR {f['snr']:6.2f} LSD {f['lsd']:.3f} HB-LSD {f['hb_lsd']:.3f} def {f['deficit']:+6.2f} | CRPS(point) {crps_pt:.4f}", flush=True)
RES["frontier"] = FRONTIER

# ---- tau sweep for the stochastic arms (M=8) -------------------------------------------------------
SWEEP = {}
for p in sorted(CKDIR.glob("*.pt")):
    m, ck = G["load_arm"](p)
    if m.tau == 0:
        continue
    rows = []
    for tau in TAU_SWEEP:
        M8 = 8 if tau > 0 else 1
        snr, crps, dfc = [], [], []
        for y in EVAL:
            draws = np.stack([reconstruct(m, y, CFG, tau=tau, seed=s) for s in range(M8)])
            truth = lm_hb(y); ens = np.stack([lm_hb(w)[:truth.shape[0]] for w in draws])
            crps.append(crps_ensemble(ens, truth)); mt = metrics(y, draws[0]); snr.append(mt["snr"]); dfc.append(mt["deficit"])
        rows.append({"tau": tau, "snr": float(np.mean(snr)), "crps": float(np.mean(crps)), "deficit": float(np.mean(dfc))})
    SWEEP[p.stem] = rows
    print(f"tau sweep {p.stem:<9} " + "  ".join(f"tau {r['tau']:.2f}: SNR {r['snr']:.2f} CRPS {r['crps']:.3f} def {r['deficit']:+.1f}" for r in rows), flush=True)
RES["tau_sweep"] = SWEEP

out = REPO / "overnight2" / f"analysis_{TAG}.json"
json.dump(RES, open(out, "w"), indent=1)

# ---- figure: predictability spectrum ---------------------------------------------------------------
arms = [k for k in RES if isinstance(RES[k], dict) and "single" in RES[k]]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
for k in arms:
    r = RES[k]
    ax[0].semilogx(CENTRES, r["single"]["curve"], "o-", ms=3, label=f"{k}: one draw" if r["stochastic"] else f"{k} (deterministic)")
    if r["stochastic"]:
        ax[0].semilogx(CENTRES, r["mean"]["curve"], "--", lw=1.2, label=f"{k}: mean of {M} draws")
ax[0].axhline(0, color="k", lw=1); ax[0].axvline(CFG.fs_lo / 2, color="crimson", ls="--"); ax[0].axhspan(-2, 2, color="grey", alpha=0.15)
ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("pred / target energy (dB)")
ax[0].set_title(f"Samples vs ensemble mean, {len(EVAL)} held-out utterances", fontsize=10); ax[0].legend(fontsize=7)
for k in arms:
    r = RES[k]
    lab = f"{k}: mean of {M} draws" if r["stochastic"] else f"{k} (deterministic; lambda-inflated energy, coherent part only)"
    ax[1].semilogx(HB_CENTRES, 100 * np.array(r["rho"]), "o-" if r["stochastic"] else "s:", ms=3, label=lab)
ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("coherent (predictable) fraction of target power, %")
ax[1].set_title("Predictability spectrum above the input Nyquist", fontsize=10); ax[1].set_ylim(-5, 100); ax[1].axhline(0, color="k", lw=1); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(REPO / "overnight2" / f"predictability_{TAG}.png", dpi=140)

stoch_arms = [k for k in arms if RES[k]["stochastic"]]
if stoch_arms:
    fig, ax = plt.subplots(1, len(stoch_arms) + 1, figsize=(3.6 * (len(stoch_arms) + 1), 3.2))
    ax = np.atleast_1d(ax)
    for a, k in zip(ax, stoch_arms):
        h = np.array(RES[k]["pit_hist"], float); h /= h.sum()
        a.bar(range(len(h)), h, edgecolor="k", lw=0.5); a.axhline(1 / len(h), color="crimson", ls="--")
        a.set_title(f"PIT: {k}", fontsize=9); a.set_xlabel("rank of truth among draws")
    for k in stoch_arms:
        s = np.array(RES[k]["spread_skill"]); ax[-1].plot(s[:, 0], s[:, 1], "o-", ms=3, label=k)
    lim = ax[-1].get_xlim(); ax[-1].plot([0, lim[1]], [0, lim[1]], "k--", lw=1); ax[-1].set_title("spread-skill", fontsize=9)
    ax[-1].set_xlabel("ensemble spread"); ax[-1].set_ylabel("RMSE of ensemble mean"); ax[-1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(REPO / "overnight2" / f"calibration_{TAG}.png", dpi=140)
# ---- figure: lambda frontier vs tau frontier ---------------------------------------------------------
if SWEEP or FRONTIER:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, rows in SWEEP.items():
        ax[0].plot([r["snr"] for r in rows], [r["crps"] for r in rows], "o-", ms=4, label=f"{k} (tau 0 -> 1.5)")
        ax[1].plot([r["snr"] for r in rows], [r["deficit"] for r in rows], "o-", ms=4, label=k)
        for r in rows:
            ax[0].annotate(f"{r['tau']:g}", (r["snr"], r["crps"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    for k in arms:
        r = RES[k]
        if not r["stochastic"]:
            ax[0].scatter([r["single"]["snr"]], [r["crps"]], marker="s", s=45, label=f"{k} (deterministic, tonight)")
            ax[1].scatter([r["single"]["snr"]], [r["single"]["deficit"]], marker="s", s=45, label=k)
    if FRONTIER:
        fx = [f["snr"] for f in FRONTIER.values()]; fy = [f["crps"] for f in FRONTIER.values()]; fd = [f["deficit"] for f in FRONTIER.values()]
        ax[0].plot(fx, fy, "k^--", ms=6, label="deterministic lambda frontier (4 Sep run, 36k steps)")
        ax[1].plot(fx, fd, "k^--", ms=6, label="lambda frontier")
        for k, f in FRONTIER.items():
            ax[0].annotate(k.split("/")[-1].replace("relu_l", "lam="), (f["snr"], f["crps"]), fontsize=7, xytext=(3, -8), textcoords="offset points")
    ax[0].axvline(NAIVE, color="grey", ls=":", lw=1); ax[1].axvline(NAIVE, color="grey", ls=":", lw=1)
    ax[0].set_xlabel("waveform SNR (dB)   dotted = naive upsampling"); ax[0].set_ylabel("CRPS on HB log-magnitude (lower is better)")
    ax[0].set_title("Proper score vs fidelity: one sampler's tau sweep against the deterministic family", fontsize=9); ax[0].legend(fontsize=6)
    ax[1].set_xlabel("waveform SNR (dB)"); ax[1].set_ylabel("high-band deficit (dB)"); ax[1].axhline(0, color="k", lw=1)
    ax[1].set_title("Energy vs fidelity", fontsize=9); ax[1].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(REPO / "overnight2" / f"frontier_{TAG}.png", dpi=140)
print("ANALYSIS DONE ->", out)
