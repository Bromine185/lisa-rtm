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
TAG = sys.argv[1] if len(sys.argv) > 1 else "OV2_es"
M = int(sys.argv[2]) if len(sys.argv) > 2 else 32
N_UTTS = int(sys.argv[3]) if len(sys.argv) > 3 else 36
CKDIR = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else DRIVE / "checkpoints" / TAG

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
def metrics(y, w):
    b = band_energy_ratio(y, w, CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop, 200.0, CFG.fs_hi / 2)
    hb = b[:, 0] >= CFG.fs_lo / 2
    return dict(snr=snr_db(y, w), lsd=lsd_db(y, w, CFG.eval_n_fft, CFG.eval_hop),
                hb_lsd=lsd_db(y, w, CFG.eval_n_fft, CFG.eval_hop, CFG.eval_k_cut),
                deficit=float(b[hb, 1].mean()), curve=b[:, 1])
CENTRES = band_energy_ratio(EVAL[0], EVAL[0], CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop, 200.0, CFG.fs_hi / 2)[:, 0]
HB = CENTRES >= CFG.fs_lo / 2
P_TRUE = np.mean([band_powers(y) / band_powers(y).sum() for y in EVAL], 0)      # target power fraction per band
P_TOTAL_HB = float(P_TRUE[HB].sum())
NAIVE = float(np.mean([snr_db(y, naive_upsample(y, CFG)) for y in EVAL]))
print(f"target power above {CFG.fs_lo/2:.0f} Hz: {100*P_TOTAL_HB:.2f} % of total   naive-upsample SNR {NAIVE:.2f} dB", flush=True)

def avg(dicts, keys):
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}

RES = {"tag": TAG, "M": M, "n_utts": len(EVAL), "naive_snr": NAIVE, "centres": CENTRES.tolist(), "p_true": P_TRUE.tolist()}
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
    curve_for_rho = np.array(r["single"]["curve"])
    if stoch:
        r["mean"] = avg(mean_rows, ("snr", "lsd", "hb_lsd", "deficit"))
        r["mean"]["curve"] = np.mean([s["curve"] for s in mean_rows], 0).tolist()
        rr = np.concatenate(ranks)
        r["pit_hist"] = np.histogram(rr, bins=M + 1, range=(-0.5, M + 0.5))[0].tolist()
        r["spread_skill"] = np.mean(sk, 0).tolist()
        # bias-corrected predictable fraction: E[mean of M draws]^2 energy = rho + (1 - rho)/M
        ratio = 10 ** (np.array(r["mean"]["curve"]) / 10)
        rho = np.clip((M * ratio - 1) / (M - 1), 1e-4, 1.0)
        r["rho"] = rho.tolist()
        curve_for_rho = 10 * np.log10(rho)
    rho_lin = np.clip(10 ** (curve_for_rho / 10), 1e-4, 1.0)
    e_min = float(np.sum((1 - rho_lin) * P_TRUE))                    # unpredictable power fraction
    r["snr_bound_db"] = 10 * math.log10(1 / max(e_min, 1e-12))
    r["predictable_hb_fraction"] = float(np.sum(rho_lin[HB] * P_TRUE[HB]) / P_TOTAL_HB)
    RES[k] = r
    s = r["single"]
    line = f"{k:<10} step {ck['step']:>6} | one draw: SNR {s['snr']:6.2f} LSD {s['lsd']:.3f} HB-LSD {s['hb_lsd']:.3f} def {s['deficit']:+6.2f} | CRPS {r['crps']:.4f} sCRPS {r['sliced_crps']:.4f}"
    if stoch:
        mm = r["mean"]
        line += f" | mean-of-{M}: SNR {mm['snr']:6.2f} def {mm['deficit']:+6.2f} HB-LSD {mm['hb_lsd']:.3f}"
    line += f" | predictable HB {100*r['predictable_hb_fraction']:.0f}%  SNR bound {r['snr_bound_db']:.1f} dB  [{time.time()-t0:.0f}s]"
    print(line, flush=True)

out = REPO / "overnight2" / f"analysis_{TAG}.json"
json.dump(RES, open(out, "w"), indent=1)

# ---- figure: predictability spectrum ---------------------------------------------------------------
arms = [k for k in RES if isinstance(RES[k], dict) and "single" in RES[k]]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
for k in arms:
    r = RES[k]
    ax[0].semilogx(CENTRES, r["single"]["curve"], "o-", ms=3, label=f"{k}: one draw" if r["stochastic"] else f"{k} (deterministic)")
    if r["stochastic"]:
        ax[0].semilogx(CENTRES, 10 * np.log10(r["rho"]), "--", lw=1.2, label=f"{k}: mean of {M} (bias-corrected)")
ax[0].axhline(0, color="k", lw=1); ax[0].axvline(CFG.fs_lo / 2, color="crimson", ls="--"); ax[0].axhspan(-2, 2, color="grey", alpha=0.15)
ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("pred / target energy (dB)")
ax[0].set_title(f"Samples vs ensemble mean, {len(EVAL)} held-out utterances", fontsize=10); ax[0].legend(fontsize=7)
for k in arms:
    r = RES[k]
    if r["stochastic"]:
        ax[1].semilogx(CENTRES[HB], 100 * np.array(r["rho"])[HB], "o-", ms=3, label=k)
ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("predictable fraction of target power (%)")
ax[1].set_title("Predictability spectrum above the input Nyquist", fontsize=10); ax[1].set_ylim(0, 100); ax[1].legend(fontsize=7)
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
print("ANALYSIS DONE ->", out)
