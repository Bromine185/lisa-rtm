"""Figures + markdown table for notes/analysis/scale_OV3_fast.md from scale_OV3_fast.json / _sweeps.npz."""
import json, pathlib, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = pathlib.Path(__file__).resolve().parents[1]
J = json.load(open(REPO / "lisa_rtm_cache/results/scale_OV3_fast.json"))
S = np.load(REPO / "lisa_rtm_cache/results/scale_OV3_fast_sweeps.npz")
FIG = REPO / "notes/analysis/figs"; FIG.mkdir(parents=True, exist_ok=True)
ARMS = list(J["arms"]); UTT = J["utterances"]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})

# --- fig 1: agreement per arm, per utterance, tau 0 vs tau 1 -------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=120)
for k, arm in enumerate(ARMS):
    for tau, col, dx in (("0", BLUE, -0.16), ("1", ORANGE, 0.16)):
        if tau not in J["arms"][arm]["taus"]:
            continue
        per = J["arms"][arm]["taus"][tau]["per_utterance"]
        v = np.array([per[u]["agree_db"] for u in UTT])
        ax.scatter(np.full(len(v), k + dx), v, s=16, color=col, alpha=0.75, linewidths=0, zorder=3)
        ax.plot([k + dx - 0.12, k + dx + 0.12], [v.mean()] * 2, color=col, lw=2, zorder=4)
        j = int(np.argmin(v))
        if k == 0 and tau == "0":
            ax.annotate(UTT[j].split("_mic")[0], (k + dx, v[j]), xytext=(-4, -11), textcoords="offset points",
                        fontsize=7.5, color=INK2, ha="center")
ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("agreement, 8x decimated vs 4x (dB)")
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.scatter([], [], color=BLUE, s=16, label="tau = 0"); ax.scatter([], [], color=ORANGE, s=16, label="tau = 1, seed 0")
ax.plot([], [], color=INK2, lw=2, label="mean of 6 utterances")
ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=3)
ax.set_title("Same latents queried at 4x and 8x: agreement per arm, step 16000", fontsize=9.5, loc="left")
fig.tight_layout(); fig.savefig(FIG / "scale_agreement.png"); plt.close(fig)

# --- fig 2: spectrum of the 8x query, p236, 500 Hz bands, dB re total ----------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=120)
f = np.arange(0, 48000, 500) + 250
for key, col, lab, ls in (("spec|det|0", INK2, "det, tau 0", "-"), ("spec|es_erb_l0.1|0", BLUE, "es_erb_l0.1, tau 0", "-"),
                          ("spec|es_erb_l0.1|1", ORANGE, "es_erb_l0.1, tau 1", "-")):
    ax.plot(f / 1000, S[key], color=col, lw=1.4, ls=ls, label=lab)
ax.axvline(24, color=INK2, lw=0.8, ls=":"); ax.text(24.4, -12, "24 kHz = 4x Nyquist", fontsize=7.5, color=INK2)
ax.set_xlabel("frequency of the 8x query (kHz)"); ax.set_ylabel("energy per 500 Hz band (dB re total)")
ax.set_xlim(0, 48); ax.set_ylim(-100, -5)
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("What the 8x query puts above the rate it was trained at (p236_002)", fontsize=9.5, loc="left")
fig.tight_layout(); fig.savefig(FIG / "scale_spectrum.png"); plt.close(fig)

# --- fig 3: the c sweep at the loudest latent, p236, tau 0, one panel per arm --------------------------
c = np.linspace(-1, 1, 801)
fig, axes = plt.subplots(len(ARMS), 1, figsize=(7.2, 5.6), dpi=120, sharex=True)
for ax, arm in zip(axes, ARMS):
    r = S[f"{arm}|0|p236_002_mic1"].astype(float)
    ax.plot(c, r, color=BLUE, lw=1.4)
    for cv in (-1, -0.5, 0, 0.5):
        j = int(round((cv + 1) / 2 * 800)); ax.scatter([cv], [r[j]], s=18, color=BLUE, zorder=3, linewidths=0)
    for cv in (-0.75, -0.25, 0.25, 0.75):
        j = int(round((cv + 1) / 2 * 800)); ax.scatter([cv], [r[j]], s=18, facecolors=SURF, edgecolors=ORANGE, zorder=3, linewidths=1.2)
    ratio = J["arms"][arm]["taus"]["0"]["per_utterance"]["p236_002_mic1"]["curv_ratio"]
    ax.text(1.01, 0.5, f"{arm}\ncurv. ratio {ratio:.2f}", transform=ax.transAxes, fontsize=7.5, va="center", color=INK2)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    for cv in (-1, -0.5, 0, 0.5, 1):
        ax.axvline(cv, color=GRID, lw=0.6, zorder=0)
axes[-1].set_xlabel("coordinate c (filled: trained 4x phases; open: the 8x phases never trained)")
axes[0].set_title("Decoder response over c at the loudest latent triple, p236_002, tau 0", fontsize=9.5, loc="left")
fig.tight_layout(rect=(0, 0, 0.86, 1)); fig.savefig(FIG / "scale_sweep.png"); plt.close(fig)

# --- markdown table -----------------------------------------------------------------------------------
def cell(arm, tau, key, worst_is_min=True, fmt="{:.1f}"):
    T = J["arms"][arm]["taus"].get(tau)
    if T is None:
        return "—"
    s = T["summary"][key]
    return fmt.format(s["mean"]) + " / " + fmt.format(s["min"] if worst_is_min else s["max"])
print("| arm | class | agreement dB, tau 0 (mean / worst) | agreement dB, tau 1 (mean / worst) | energy above 24 kHz %, tau 0 (mean / max) | tau 1 (mean / max) | curvature ratio, tau 0 (median / max) | tau 1 (median / max) |")
print("|---|---|---|---|---|---|---|---|")
for arm in ARMS:
    A = J["arms"][arm]
    def cr(tau):
        T = A["taus"].get(tau)
        if T is None: return "—"
        v = np.array([T["per_utterance"][u]["curv_ratio"] for u in UTT])
        return f"{np.median(v):.2f} / {v.max():.2f}"
    print(f"| {arm} | {A['cls']} | {cell(arm,'0','agree_db')} | {cell(arm,'1','agree_db')} | "
          f"{cell(arm,'0','above_24k_pct',False,'{:.4f}')} | {cell(arm,'1','above_24k_pct',False,'{:.4f}')} | {cr('0')} | {cr('1')} |")
print()
print("| arm | tau | SNR vs truth 4x, mean | SNR 8x-decimated, mean | max abs difference over 6 utts |")
print("|---|---|---|---|---|")
for arm in ARMS:
    for tau, T in J["arms"][arm]["taus"].items():
        per = T["per_utterance"]
        a = np.array([per[u]["snr_4x_db"] for u in UTT]); b = np.array([per[u]["snr_8x_dec_db"] for u in UTT])
        print(f"| {arm} | {tau} | {a.mean():.2f} | {b.mean():.2f} | {np.abs(a-b).max():.2f} |")
print()
for arm in ("es_dec_l0.1", "es_dec_erb_l0.1"):
    s = J["arms"][arm]["taus"]["1"]["summary"]
    print(f"| {arm} | held: {s['held.agree_db']['mean']:.2f} / {s['held.agree_db']['min']:.2f} | fresh: {s['fresh.agree_db']['mean']:.2f} / {s['fresh.agree_db']['min']:.2f} | fresh even-sample match {s['fresh.even_match_db']['mean']:.1f} dB |")
