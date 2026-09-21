"""Every arm's training and validation curves, and the cross-arm columns, as figures.

    python fast/plot_histories.py --src ~/lisa-results

The trainer draws nothing: it writes history_<tag>_<arm>.json per process, on eight separate VMs
that never saw each other.  This is the first place the eight curves are in the same axes.

WHICH CURVES MEAN ANYTHING ACROSS ARMS, which is the whole reason fast/compare.py exists:

    val_wave   the same waveform term for every arm.  THE cross-arm column.
    val_spec   likewise comparable -- but read notes/2026-09-21 before quoting det_paper's, which
               is AMP rounding noise counted as signal at 27.6 dB down.
    val_loss   each arm's OWN objective, and the eight objectives are different functions.  Plotted
               per arm and never overlaid, because a shared axis would invite exactly the ranking
               that ranks the arms by which loss they were handed.
    spread     the energy score's ensemble spread.  Exactly 0 for det and det_paper at every logged
               step, which is how the det/es split in ARMS is confirmed from the training record
               rather than assumed from the arm name.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fast.run_contract import ARMS, MILESTONES, RUN_TAG, STEPS

ORDER = ["det_paper", "det", "es_marg", "es_erb_l0.001", "es_erb_l0.01", "es_erb_l0.1",
         "es_dec_l0.01", "es_dec_erb_l0.1"]
COL = {"det_paper": "#444444", "det": "#888888", "es_marg": "#1f77b4", "es_erb_l0.001": "#9467bd",
       "es_erb_l0.01": "#2ca02c", "es_erb_l0.1": "#d62728", "es_dec_l0.01": "#ff7f0e",
       "es_dec_erb_l0.1": "#17becf"}


def load(src, tag):
    out = {}
    for arm in ORDER:
        p = pathlib.Path(src) / f"history_{tag}_{arm}.json"
        if p.exists():
            out[arm] = json.loads(p.read_text())[arm]
    return out


def milestones(ax):
    for m in MILESTONES:
        ax.axvline(m, color="k", lw=0.4, alpha=0.25)


def per_arm(H, figs, tag):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.2), sharex=True)
    for ax, arm in zip(axes.ravel(), ORDER):
        h = H.get(arm)
        if h is None:
            ax.set_visible(False)
            continue
        kind, lam, cls = ARMS[arm]
        milestones(ax)
        ax.plot(h["step"], h["loss"], color=COL[arm], lw=0.9, alpha=0.45, label="train (ema)")
        ax.plot(h["val_step"], h["val_loss"], color=COL[arm], lw=1.6, label="val (own objective)")
        ax.plot(h["val_step"], h["val_wave"], color="k", lw=1.1, ls="--", label="val_wave")
        ax.set_yscale("log")
        ax.set_title(f"{arm}\n{kind}  $\\lambda$={lam:g}  {cls}", fontsize=9)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.grid(alpha=0.2)
    for ax in axes[1]:
        ax.set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("loss (log)")
    fig.suptitle(f"{tag}: own objective and the shared waveform term, per arm "
                 f"(grey lines: the six MultiStepLR milestones)", fontsize=11)
    fig.tight_layout()
    p = figs / f"train_val_{tag}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def cross_arm(H, figs, tag):
    """val_wave, val_spec and spread on shared axes -- the three that compare."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for arm in ORDER:
        h = H.get(arm)
        if h is None:
            continue
        kw = dict(color=COL[arm], lw=1.4, label=arm)
        ax[0].plot(h["val_step"], h["val_wave"], **kw)
        ax[1].plot(h["val_step"], h["val_spec"], **kw)
        sp = np.array(h["spread"])
        ax[2].plot(h["step"], np.where(sp > 0, sp, np.nan), **kw)
    for a, t, yl in ((ax[0], "val_wave -- the cross-arm column", "L1 waveform on the fixed val batches"),
                     (ax[1], "val_spec -- comparable, but see det_paper", "spectral term"),
                     (ax[2], "spread -- exactly 0 for both det arms", "ensemble spread")):
        milestones(a); a.set_yscale("log"); a.set_xlabel("step")
        a.set_title(t, fontsize=10); a.set_ylabel(yl, fontsize=8); a.grid(alpha=0.2)
    ax[0].legend(fontsize=7)
    ax[2].text(0.02, 0.04, "det, det_paper: 0 at every logged step\n(not plotted on a log axis)",
               transform=ax[2].transAxes, fontsize=7, color="#444444")
    fig.suptitle(f"{tag}: the three curves that mean the same thing for every arm", fontsize=11)
    fig.tight_layout()
    p = figs / f"cross_arm_{tag}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def endgame(H, figs, tag, last=12000):
    """The last ~12k steps, linear axis: where the ranking is actually decided."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for arm in ORDER:
        h = H.get(arm)
        if h is None:
            continue
        vs, vw, vp = np.array(h["val_step"]), np.array(h["val_wave"]), np.array(h["val_spec"])
        m = vs >= STEPS - last
        ax[0].plot(vs[m], vw[m], color=COL[arm], lw=1.4, label=arm)
        ax[1].plot(vs[m], vp[m], color=COL[arm], lw=1.4)
    for a, t in ((ax[0], "val_wave, last 12k steps"), (ax[1], "val_spec, last 12k steps")):
        milestones(a); a.set_xlabel("step"); a.set_title(t, fontsize=10); a.grid(alpha=0.2)
        a.set_xlim(STEPS - last, STEPS + 400)
    ax[1].set_ylim(0.25, 0.5)
    ax[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{tag}: the endgame, linear axes (det_paper's val_spec is off-scale at 1.60)", fontsize=11)
    fig.tight_layout()
    p = figs / f"endgame_{tag}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/lisa-results")
    ap.add_argument("--tag", default=RUN_TAG)
    ap.add_argument("--out", default=None, help="default <src>/figs")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    src = pathlib.Path(a.src).expanduser().resolve()
    figs = pathlib.Path(a.out).expanduser() if a.out else src / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    H = load(src, a.tag)
    if not H:
        raise SystemExit(f"no history_{a.tag}_<arm>.json under {src}")
    print(f"{len(H)} arms: " + ", ".join(H))
    for p in (per_arm(H, figs, a.tag), cross_arm(H, figs, a.tag), endgame(H, figs, a.tag)):
        print(f"  {p}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
