"""Figures for notes/2026-09-26-sota-diagnosis.md, from sota/score.py and sota/ceiling.py output.

    venv/bin/python sota/figs.py

Writes notes/analysis/figs/sota_{levels,calibration,pit,ceiling,pareto}.png. The pareto figure reads sota/cost.json
(parameters and RTFs, with their provenance in its _meta) and the `core16` set: symlinks to `core`'s truth/, lo12k/ and utts.json,
then `sota/run_ours.py --set core16 --arms es_dec_erb_l0.1 --seeds 16`.
"""
import os
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
WORK = pathlib.Path(os.environ.get("BWE_CACHE", str(pathlib.Path(__file__).resolve().parents[1] / "cache")) + "/sota/work")
OUT = REPO / "notes/analysis/figs"; OUT.mkdir(parents=True, exist_ok=True)
# fixed categorical order (dataviz reference palette, slots 1-7), colour follows the model everywhere
COL = {"flowhigh": "#2a78d6", "flowhigh_std1": "#4a3aa7", "nuwave2": "#eb6834", "audiosr": "#1baf7a",
       "apbwe": "#eda100", "ours_es_dec_erb_l0.1": "#e87ba4", "ours_det": "#008300"}
NAME = {"flowhigh": "FLowHigh (as shipped)", "flowhigh_std1": "FLowHigh, trained prior", "nuwave2": "NU-Wave 2",
        "audiosr": "AudioSR speech", "apbwe": "AP-BWE", "ours_es_dec_erb_l0.1": "ours, sampler", "ours_det": "ours, point"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def load(s):
    p = WORK / s / f"score_{s}.json"
    return json.load(open(p)) if p.exists() else None


def levels(S):
    fc = np.array(S["_meta"]["band_centres_hz"]) / 1000
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axhline(0, color=INK2, lw=1)
    for m, c in COL.items():
        if m in S["agg"]:
            ax.plot(fc, S["agg"][m]["curve"], color=c, lw=2, marker="o", ms=4, label=NAME[m])
    ax.set_xscale("log"); ax.set_xticks([6, 8, 10, 12, 16, 20]); ax.set_xticklabels(["6", "8", "10", "12", "16", "20"])
    ax.set_xlabel("third-octave band centre, kHz"); ax.set_ylabel("output / true energy, dB")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(f"High-band level, 12 → 48 kHz ({S['_meta']['n_utts']} utterances, one draw each)", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "sota_levels.png", dpi=160); plt.close(fig)


def calibration(S):
    ms = [m for m in COL if m in S["agg"] and "spread_pooled_hb" in S["agg"][m]]
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(ms) + 1.2))
    y = np.arange(len(ms))[::-1]
    v = [S["agg"][m]["spread_pooled_hb"] for m in ms]
    ax.barh(y, v, color=[COL[m] for m in ms], height=0.55)
    ax.axvline(1, color=INK, lw=1.2, ls="--"); ax.text(1.01, y[0] + 0.45, "calibrated", color=INK, fontsize=8.5, va="bottom")
    for yi, vi in zip(y, v):
        ax.text(vi + 0.015, yi, f"{vi:.2f}" if vi >= 0.01 else f"{vi:.0e}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in ms])
    ax.set_xlim(0, max(1.15, max(v) * 1.15)); ax.set_xlabel("high-band ensemble variance / variance a calibrated ensemble carries")
    M = int(round(S["agg"][ms[0]]["M"]))
    ax.set_title(f"Spread of the draws, 6-24 kHz (M = {M}, {S['_meta']['n_utts']} utterances, pooled)", loc="left", color=INK)
    fig.tight_layout(); fig.savefig(OUT / "sota_calibration.png", dpi=160); plt.close(fig)


def pit(S):
    ms = [m for m in COL if m in S["agg"] and "pit_hist" in S["agg"][m]]
    fig, axs = plt.subplots(1, len(ms), figsize=(2.0 * len(ms), 2.4), sharey=True)
    for ax, m in zip(np.atleast_1d(axs), ms):
        h = np.array(S["agg"][m]["pit_hist"]); M = len(h) - 1
        ax.bar(np.arange(M + 1), h, color=COL[m], width=0.8)
        ax.axhline(1 / (M + 1), color=INK, lw=1, ls="--")
        ax.set_title(NAME[m], fontsize=9, color=INK); ax.set_xticks([0, M]); ax.set_xticklabels(["below all", "above all"], fontsize=7.5)
    np.atleast_1d(axs)[0].set_ylabel("share of HB bins")
    fig.suptitle("Where the truth ranks among the draws (high-band log-magnitude bins); dashed = flat", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "sota_pit.png", dpi=160); plt.close(fig)


def ceiling(C, reported):
    rates = [8, 12, 16, 24]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ce = [C["by_rate"][str(r * 1000)]["empty"] for r in rates]
    ax.plot(rates, ce, color=INK, lw=2, marker="s", ms=6, label="ceiling: true low band, empty high band (measured)")
    mk = ["o", "^", "v", "D", "P", "X", "*", "h"]
    for i, (name, pts) in enumerate(reported.items()):
        xs = [r for r in rates if str(r) in pts]
        ax.scatter(xs, [pts[str(r)] for r in xs], s=40, marker=mk[i % len(mk)], color=list(COL.values())[i % 7], label=name, zorder=3)
    ax.set_xticks(rates); ax.set_xlabel("input rate, kHz (target 48 kHz)"); ax.set_ylabel("SNR, dB")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Reported SNR against the empty-band ceiling (VCTK test speakers)", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "sota_ceiling.png", dpi=160); plt.close(fig)


def pareto(K, K16, cost):
    """Quality against size and against speed, one point per model; the grey step is the non-dominated set."""
    pts = [("flowhigh", K, "flowhigh"), ("nuwave2", K, "nuwave2"), ("audiosr", K, "audiosr"), ("apbwe", K, "apbwe"),
           ("ours_es_dec_erb_l0.1", K, "ours_es_dec_erb_l0.1"),          # M = 8, like the other samplers
           ("ours_es_dec_erb_l0.1 logmean16 | pt", K16, "ours_es_dec_erb_l0.1"), ("ours_det", K, "ours_det")]
    lab = {"ours_es_dec_erb_l0.1": "ours, sampler", "ours_es_dec_erb_l0.1 logmean16 | pt": "ours, logmean16",
           "ours_det": "ours, det"}
    rows = []
    for key, S, col in pts:
        if S is None or key not in S["agg"] or key not in cost:
            continue
        g = S["agg"][key]
        rows.append(dict(key=key, col=COL[col], name=lab.get(key, NAME.get(key, key)), params=cost[key]["params"],
                         rtf=cost[key]["rtf"], visqol=g.get("visqol_audio"), crps=g.get("crps"),
                         hollow=key.endswith("| pt")))
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.4))
    for i, (ykey, ylab, better) in enumerate([("visqol", "ViSQOL audio, 48 kHz (higher is better)", "max"),
                                                ("crps", "CRPS, HB log-magnitude (lower is better)", "min")]):
        for j, (xkey, xlab) in enumerate([("params", "parameters"), ("rtf", "real-time factor, Apple M4")]):
            ax = axs[i, j]
            R = [r for r in rows if r[ykey] is not None]
            # frontier: sort by cost, keep each point that beats everything cheaper
            fr, best = [], None
            for r in sorted(R, key=lambda r: r[xkey]):
                v = r[ykey]
                if best is None or (v > best if better == "max" else v < best):
                    fr.append(r); best = v
            ax.step([r[xkey] for r in fr], [r[ykey] for r in fr], where="post", color="#b8b7b1", lw=1.5, zorder=1)
            for r in R:
                ax.scatter(r[xkey], r[ykey], s=70, color="none" if r["hollow"] else r["col"], edgecolor=r["col"],
                           linewidth=2, zorder=3)
                below = r["key"].endswith("logmean16 | pt") and ykey == "crps"
                ax.annotate(r["name"], (r[xkey], r[ykey]), xytext=(6, -11 if below else 4), textcoords="offset points", fontsize=8, color=INK)
            ax.set_xscale("log"); ax.set_xlabel(xlab); ax.set_ylabel(ylab if j == 0 else "")
            ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.suptitle(f"Quality against size and speed, 12 -> 48 kHz ({K['_meta']['n_utts']} utterances, 8 unseen speakers). "
                 "Grey: non-dominated models.", x=0.01, ha="left", fontsize=10, color=INK)
    fig.text(0.01, 0.005, "ViSQOL: draw 0 (the readout for logmean16). CRPS: 8 draws for every sampler; a one-output model scores its MAE.\n"
             "RTF: batch-run wall clock, not a latency benchmark; AudioSR on MPS, the rest on CPU. Hollow = a readout of the draws.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96)); fig.savefig(OUT / "sota_pareto.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    W, K = load("wide"), load("core")
    if W: levels(W)
    if K: calibration(K); pit(K)
    cst = pathlib.Path(__file__).with_name("cost.json")
    if K and cst.exists():
        pareto(K, load("core16"), json.load(open(cst)))
    cp = WORK / "wide" / "ceiling_wide.json"
    rp = pathlib.Path(__file__).with_name("reported_snr.json")
    if cp.exists() and rp.exists():
        ceiling(json.load(open(cp)), json.load(open(rp)))
    print("figures ->", OUT)
