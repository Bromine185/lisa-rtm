"""Training dynamics of the seven OV3_fast arms, from the trainer's own history file.

Reads   lisa_rtm_cache/results/ov3_history_OV3_fast.json
Writes  notes/analysis/figs/dyn_loss_grid.png, dyn_deficit.png, dyn_snr.png, dyn_spread.png
        lisa_rtm_cache/results/dynamics_OV3_fast.json

Per-arm objectives (val_loss / val_wave / val_spec / train_loss_ema) are never compared across arms here,
only along time within an arm.  snr*, def*, spread are comparable across arms.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "lisa_rtm_cache/results/ov3_history_OV3_fast.json"
FIGS = ROOT / "notes/analysis/figs"
OUT = ROOT / "lisa_rtm_cache/results/dynamics_OV3_fast.json"
FIGS.mkdir(parents=True, exist_ok=True)

ARMS = ["det", "es_marg", "es_marg_l0.1", "es_split_l0.1", "es_erb_l0.1", "es_dec_l0.1", "es_dec_erb_l0.1"]
# dataviz reference palette, categorical slots 1-7 in fixed order (validated light mode)
COL = {"det": "#2a78d6", "es_marg": "#eb6834", "es_marg_l0.1": "#1baf7a", "es_split_l0.1": "#eda100",
       "es_erb_l0.1": "#e87ba4", "es_dec_l0.1": "#008300", "es_dec_erb_l0.1": "#4a3aa7"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STEPS = 16000
MILESTONES = [int(f * STEPS) for f in (0.2, 0.4, 0.5, 0.6, 0.7, 0.8)]
LR0 = 1e-3
STEPS_PER_EPOCH = 2099
GAP_STEPS = [4000, 8000, 12000, 16000]
DEF_THRESH = -12.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "text.color": INK,
})

H = json.load(open(HIST))


def arr(a, k):
    return np.array([np.nan if v is None else v for v in H[a][k]], dtype=float)


def at(steps, vals, s):
    i = int(np.argmin(np.abs(steps - s)))
    assert abs(steps[i] - s) < 1e-9, (s, steps[i])
    return float(vals[i])


def pct_slope(steps, vals, lo, hi):
    """Linear fit on [lo, hi]; slope in % of the fitted value at lo, per 1000 steps."""
    m = (steps >= lo) & (steps <= hi)
    x, y = steps[m], vals[m]
    b, a = np.polyfit(x, y, 1)
    return float(100.0 * b * 1000.0 / (a + b * lo))


def window_mean(steps, vals, lo, hi):
    m = (steps >= lo) & (steps < hi)
    return float(np.nanmean(vals[m]))


# ----------------------------------------------------------------------------------------- numbers
res = {"run": "OV3_fast", "steps": STEPS, "batch": 64, "clip_seconds": 1, "lr0": LR0, "lr_gamma": 0.5,
       "lr_milestones": MILESTONES, "steps_per_epoch": STEPS_PER_EPOCH, "epochs": STEPS / STEPS_PER_EPOCH,
       "dev_probe": "p236_002 (one utterance, one draw at tau 1)", "def_threshold_db": DEF_THRESH,
       "val": "8 fixed batches x 64 x 1 s, every 500 steps; each arm's OWN objective (not comparable across arms)",
       "train_ema": "alpha 0.02 EMA of the per-step training loss, sampled at val points",
       "arms": {}}

for a in ARMS:
    st = arr(a, "step"); vs = arr(a, "val_step"); ds = arr(a, "dev_step")
    vl = arr(a, "val_loss"); vw = arr(a, "val_wave"); vsp = arr(a, "val_spec"); tr = arr(a, "train_loss_ema")
    snr0, def0, snr1, def1, naive = (arr(a, k) for k in ("snr0", "def0", "snr1", "def1", "snr_naive"))
    spread = arr(a, "spread"); wave = arr(a, "wave"); spec = arr(a, "spec"); lr = arr(a, "lr")
    stoch = a != "det"

    r = {}
    r["final_val"] = {"step": int(vs[-1]), "val_loss": float(vl[-1]), "val_wave": float(vw[-1]),
                      "val_spec": float(vsp[-1]), "train_loss_ema": float(tr[-1]),
                      "val_over_train": float(vl[-1] / tr[-1])}
    r["final_dev"] = {"step": int(ds[-1]), "snr0": float(snr0[-1]), "def0": float(def0[-1]),
                      "snr1": None if not stoch else float(snr1[-1]), "def1": None if not stoch else float(def1[-1]),
                      "snr_naive": float(naive[-1])}
    r["best_val"] = {"step": int(vs[np.argmin(vl)]), "val_loss": float(vl.min())}
    # spread: last 200 logged steps (8 logs), and the early peak
    r["spread"] = None
    if stoch:
        sm = np.convolve(spread, np.ones(8) / 8, mode="valid")  # 200-step boxcar, trailing
        sm_steps = st[7:]
        r["spread"] = {"final_200step_mean": float(np.mean(spread[-8:])),
                       "peak": float(np.nanmax(sm)), "peak_step": int(sm_steps[np.nanargmax(sm)]),
                       "at": {str(s): window_mean(st, spread, s - 200, s) for s in GAP_STEPS},
                       "slope_last4000_pct_per_1000": pct_slope(sm_steps, sm, 12000, 16000)}
    # def1 threshold crossing
    if stoch:
        idx = np.where(def1 > DEF_THRESH)[0]
        r["def1_first_above_-12dB_step"] = int(ds[idx[0]]) if len(idx) else None
        r["def1_max"] = {"value": float(np.nanmax(def1)), "step": int(ds[np.nanargmax(def1)])}
    else:
        r["def1_first_above_-12dB_step"] = "n/a (det)"
        r["def1_max"] = None
    idx0 = np.where(def0 > DEF_THRESH)[0]
    r["def0_first_above_-12dB_step"] = int(ds[idx0[0]]) if len(idx0) else None
    r["def0_max"] = {"value": float(np.nanmax(def0)), "step": int(ds[np.nanargmax(def0)])}
    # gap def1 - def0
    r["gap_def1_minus_def0"] = {str(s): (None if not stoch else at(ds, def1, s) - at(ds, def0, s)) for s in GAP_STEPS}
    r["def0_at"] = {str(s): at(ds, def0, s) for s in GAP_STEPS}
    r["def1_at"] = {str(s): (None if not stoch else at(ds, def1, s)) for s in GAP_STEPS}
    r["snr0_at"] = {str(s): at(ds, snr0, s) for s in GAP_STEPS}
    r["snr1_at"] = {str(s): (None if not stoch else at(ds, snr1, s)) for s in GAP_STEPS}
    # dev-probe noise floor: std of consecutive differences over the last 8 dev points (lr <= 3.1e-5)
    r["dev_noise_floor_last8"] = {"def0_diff_std": float(np.std(np.diff(def0[-8:]))),
                                  "def1_diff_std": None if not stoch else float(np.std(np.diff(def1[-8:]))),
                                  "snr0_diff_std": float(np.std(np.diff(snr0[-8:])))}
    # slopes over the last 4000 steps
    r["val_loss_slope_last4000_pct_per_1000"] = pct_slope(vs, vl, 12000, 16000)
    r["train_ema_slope_last4000_pct_per_1000"] = pct_slope(vs, tr, 12000, 16000)
    r["val_wave_slope_last4000_pct_per_1000"] = pct_slope(vs, vw, 12000, 16000)
    r["val_spec_slope_last4000_pct_per_1000"] = pct_slope(vs, vsp, 12000, 16000)
    r["val_loss_slope_8000_12000_pct_per_1000"] = pct_slope(vs, vl, 8000, 12000)
    r["val_loss_slope_4000_8000_pct_per_1000"] = pct_slope(vs, vl, 4000, 8000)
    # last-quarter rise
    q = vs >= 12000
    vq = vl[q]
    runmin = np.minimum.accumulate(vq)
    r["last_quarter"] = {"val_12000": float(at(vs, vl, 12000)), "val_16000": float(vl[-1]),
                         "change_pct": float(100 * (vl[-1] / at(vs, vl, 12000) - 1)),
                         "max_rise_above_running_min_pct": float(100 * np.max(vq / runmin - 1)),
                         "n_upticks_of_8": int(np.sum(np.diff(vq) > 0)),
                         "rose": bool(vl[-1] > at(vs, vl, 12000))}
    # val/train ratio trend (generalisation gap)
    r["val_over_train_at"] = {str(s): float(at(vs, vl, s) / at(vs, tr, s)) for s in GAP_STEPS}
    # lr milestones: val change over the straddling 500-step window vs the previous window;
    # and the raw train wave term, 400 steps before vs 400 after
    ms = {}
    for M in MILESTONES:
        pre = 500 * (M // 500); post = pre + 500
        v_pre, v_post, v_prev = at(vs, vl, pre), at(vs, vl, post), at(vs, vl, pre - 500)
        ms[str(M)] = {"lr_after": float(lr[st == M][0]), "val_window": [pre, post],
                      "val_change_pct": float(100 * (v_post / v_pre - 1)),
                      "prev_window_change_pct": float(100 * (v_pre / v_prev - 1)),
                      "train_wave_change_pct": float(100 * (window_mean(st, wave, M, M + 400)
                                                            / window_mean(st, wave, M - 400, M) - 1)),
                      "train_spec_change_pct": (None if np.nanmean(spec) == 0 else
                                                float(100 * (window_mean(st, spec, M, M + 400)
                                                             / window_mean(st, spec, M - 400, M) - 1)))}
    r["lr_milestones"] = ms
    res["arms"][a] = r

# cross-arm summary of milestone effects (mean over arms of val change straddling vs previous window)
res["milestone_summary"] = {}
for M in MILESTONES:
    s = [res["arms"][a]["lr_milestones"][str(M)] for a in ARMS]
    res["milestone_summary"][str(M)] = {
        "mean_val_change_pct": float(np.mean([x["val_change_pct"] for x in s])),
        "mean_prev_window_change_pct": float(np.mean([x["prev_window_change_pct"] for x in s])),
        "mean_train_wave_change_pct": float(np.mean([x["train_wave_change_pct"] for x in s]))}

json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)

# ----------------------------------------------------------------------------------------- figures
def milestones(ax, label=False):
    for i, M in enumerate(MILESTONES):
        ax.axvline(M / 1000, color=AXIS, lw=0.8, ls=(0, (3, 3)), zorder=0.5)
        if label:
            ax.text(M / 1000, 1.01, f"lr/{2 ** (i + 1)}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7.5, color=MUTED)


def end_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", ha="left",
                fontsize=8.5, color=INK2, annotation_clip=False,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    ax.plot([x], [y], "o", ms=4.5, color=color, mec=SURFACE, mew=1.2, zorder=5)


def spread_labels(ys, min_gap):
    """Push label y positions apart so none overlap; returns adjusted positions in data units."""
    order = np.argsort(ys)
    pos = np.array(ys, float)
    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        if pos[j] - pos[i] < min_gap:
            pos[j] = pos[i] + min_gap
    return pos


# 1. loss grid ------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.4), dpi=110, sharex=True)
axes = axes.ravel()
for i, a in enumerate(ARMS):
    ax = axes[i]
    vs = arr(a, "val_step") / 1000; vl = arr(a, "val_loss"); tr = arr(a, "train_loss_ema")
    ax.plot(vs, tr, color=COL[a], lw=1.6, alpha=0.45)
    ax.plot(vs, vl, color=COL[a], lw=2.0)
    ax.set_yscale("log")
    milestones(ax)
    r = res["arms"][a]
    ax.set_title(f"{a}\nval {vl[-1]:.4f} · last 4k {r['val_loss_slope_last4000_pct_per_1000']:+.2f} %/1k",
                 loc="left", color=INK, fontsize=9.5)
    lo, hi = min(vl.min(), tr.min()), max(vl.max(), tr.max())
    ax.set_ylim(lo * 0.85, hi * 1.15)
    ax.tick_params(axis="y", labelsize=8)
    if i >= 3:
        ax.set_xlabel("step (thousands)")
    if i % 4 == 0:
        ax.set_ylabel("loss (arm's own objective, log)")
# legend panel
axl = axes[7]
axl.axis("off")
axl.plot([], [], color=INK, lw=2.0, label="val loss (8 fixed batches, every 500)")
axl.plot([], [], color=INK, lw=1.6, alpha=0.45, label="train loss, EMA α = 0.02")
axl.plot([], [], color=AXIS, lw=0.8, ls=(0, (3, 3)), label="lr halved (3.2k 6.4k 8k 9.6k 11.2k 12.8k)")
axl.legend(loc="center left", fontsize=9)
axl.text(0.0, 0.12, "Each panel is its own objective.\nDo not compare heights across panels.",
         transform=axl.transAxes, fontsize=8.5, color=INK2, va="top")
fig.suptitle("OV3_fast — train and val loss per arm, 16 000 steps (7.6 epochs)", x=0.01, ha="left",
             fontsize=12, color=INK, fontweight="bold")
fig.subplots_adjust(wspace=0.32, hspace=0.42); fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIGS / "dyn_loss_grid.png")
plt.close(fig)

# 2. deficit --------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=110)
ends1, ends0 = {}, {}
for a in ARMS:
    ds = arr(a, "dev_step") / 1000; d0 = arr(a, "def0"); d1 = arr(a, "def1")
    ax.plot(ds, d0, color=COL[a], lw=1.6, ls=(0, (4, 2.5)), alpha=0.8)
    ends0[a] = d0[-1]
    if a != "det":
        ax.plot(ds, d1, color=COL[a], lw=2.0, label=a)
        ends1[a] = d1[-1]
    else:
        ax.plot([], [], color=COL[a], lw=2.0, label=a)
ax.axhline(0, color=INK2, lw=0.9)
ax.text(0.15, 0.3, "0 dB = correct high-band energy", fontsize=8.5, color=INK2, va="bottom")
ax.axhline(DEF_THRESH, color=MUTED, lw=0.9, ls=(0, (1, 2)))
ax.text(0.15, DEF_THRESH + 0.3, "−12 dB", fontsize=8.5, color=MUTED, va="bottom")
milestones(ax, label=True)
# end labels for tau 1 (solid) and, on the far right, tau 0 (dashed)
names = list(ends1)
pos = spread_labels([ends1[a] for a in names], 0.9)
for a, y in zip(names, pos):
    end_label(ax, 16, ends1[a], "", COL[a])
    ax.annotate(f"{a}  {ends1[a]:+.1f}", (16, ends1[a]), xytext=(16.35, y), textcoords="data",
                va="center", ha="left", fontsize=8.5, color=INK2, annotation_clip=False)
ax.set_xlim(0, 19.2)
ax.set_ylim(-32, 1.5)
ax.set_xlabel("step (thousands)")
ax.set_ylabel("high-band energy deficit on p236_002 (dB)")
ax.plot([], [], color=INK, lw=2.0, label="solid: one draw, τ = 1")
ax.plot([], [], color=INK, lw=1.6, ls=(0, (4, 2.5)), alpha=0.8, label="dashed: τ = 0")
ax.legend(loc="lower right", ncol=2, fontsize=8.5)
ax.set_title("OV3_fast — high-band deficit on the dev probe, noise on (solid) and off (dashed)",
             loc="left", fontsize=12, color=INK, fontweight="bold", pad=16)
fig.tight_layout()
fig.savefig(FIGS / "dyn_deficit.png")
plt.close(fig)

# 3. SNR ------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=110)
nv = float(arr("det", "snr_naive")[-1])
ax.axhline(nv, color=INK2, lw=1.0)
ax.text(0.15, nv + 0.02, f"naive sinc upsample  {nv:.2f} dB", fontsize=8.5, color=INK2, va="bottom")
ends = {}
for a in ARMS:
    ds = arr(a, "dev_step") / 1000; s0 = arr(a, "snr0"); s1 = arr(a, "snr1")
    ax.plot(ds, s0, color=COL[a], lw=1.6, ls=(0, (4, 2.5)), alpha=0.8)
    if a != "det":
        ax.plot(ds, s1, color=COL[a], lw=2.0, label=a)
        ends[a] = s1[-1]
    else:
        ax.plot([], [], color=COL[a], lw=2.0, label=a)
        ends[a] = s0[-1]
names = list(ends)
pos = spread_labels([ends[a] for a in names], 0.075)
for a, y in zip(names, pos):
    end_label(ax, 16, ends[a], "", COL[a])
    tag = "τ0" if a == "det" else "τ1"
    ax.annotate(f"{a}  {ends[a]:.2f} ({tag})", (16, ends[a]), xytext=(16.35, y), textcoords="data",
                va="center", ha="left", fontsize=8.5, color=INK2, annotation_clip=False)
milestones(ax, label=True)
ax.set_xlim(0, 19.6)
ax.set_ylim(12.4, 15.15)
ax.text(5.6, 12.47, "step-500 values of 8.1–12.9 dB are below the axis", fontsize=8, color=MUTED, va="bottom")
ax.set_xlabel("step (thousands)")
ax.set_ylabel("SNR on p236_002 (dB)")
ax.plot([], [], color=INK, lw=2.0, label="solid: one draw, τ = 1")
ax.plot([], [], color=INK, lw=1.6, ls=(0, (4, 2.5)), alpha=0.8, label="dashed: τ = 0")
ax.legend(loc="lower right", ncol=2, fontsize=8.5)
ax.set_title("OV3_fast — dev-probe SNR against its naive ceiling (SNR is maximised by adding nothing)",
             loc="left", fontsize=12, color=INK, fontweight="bold", pad=16)
fig.tight_layout()
fig.savefig(FIGS / "dyn_snr.png")
plt.close(fig)

# 4. spread ---------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=110)
ends = {}
for a in ARMS:
    if a == "det":
        continue
    st = arr(a, "step") / 1000; sp = arr(a, "spread")
    ax.plot(st, sp, color=COL[a], lw=0.7, alpha=0.22)
    sm = np.convolve(sp, np.ones(8) / 8, mode="valid")
    ax.plot(st[7:], sm, color=COL[a], lw=2.0, label=a)
    ends[a] = sm[-1]
ax.set_yscale("log")
names = list(ends)
pos = spread_labels([np.log10(ends[a]) for a in names], 0.035)
for a, y in zip(names, pos):
    end_label(ax, st[-1], ends[a], "", COL[a])
    ax.annotate(f"{a}  {ends[a]:.4f}", (st[-1], ends[a]), xytext=(16.35, 10 ** y), textcoords="data",
                va="center", ha="left", fontsize=8.5, color=INK2, annotation_clip=False)
milestones(ax, label=True)
ax.set_xlim(0, 19.6)
ax.set_xlabel("step (thousands)")
ax.set_ylabel("two-draw spread  d_wave(y₁, y₂), training batch (log)")
ax.legend(loc="upper right", ncol=3, fontsize=8.5)
ax.text(0.15, 0.0012, "thin: every 25 steps · bold: 200-step mean", fontsize=8.5, color=INK2)
ax.set_title("OV3_fast — how far two draws differ, over training (stochastic arms)",
             loc="left", fontsize=12, color=INK, fontweight="bold", pad=16)
fig.tight_layout()
fig.savefig(FIGS / "dyn_spread.png")
plt.close(fig)
print("wrote figs to", FIGS)

# ----------------------------------------------------------------------------------------- console
print()
print(f"{'arm':16} {'val16k':>8} {'trEMA':>8} {'v/t':>5} {'snr0':>6} {'def0':>7} {'snr1':>6} {'def1':>7} "
      f"{'spread':>7} {'def1>-12':>9} {'slope%/1k':>9} {'lastQ%':>7} {'rose':>5}")
for a in ARMS:
    r = res["arms"][a]; fv, fd = r["final_val"], r["final_dev"]
    sp = "-" if r["spread"] is None else f"{r['spread']['final_200step_mean']:.4f}"
    s1 = "-" if fd["snr1"] is None else f"{fd['snr1']:.2f}"
    d1 = "-" if fd["def1"] is None else f"{fd['def1']:+.2f}"
    print(f"{a:16} {fv['val_loss']:8.4f} {fv['train_loss_ema']:8.4f} {fv['val_over_train']:5.2f} {fd['snr0']:6.2f} "
          f"{fd['def0']:+7.2f} {s1:>6} {d1:>7} {sp:>7} {str(r['def1_first_above_-12dB_step']):>9} "
          f"{r['val_loss_slope_last4000_pct_per_1000']:9.2f} {r['last_quarter']['change_pct']:7.2f} "
          f"{str(r['last_quarter']['rose']):>5}")
print()
print("gap def1-def0 at", GAP_STEPS)
for a in ARMS:
    g = res["arms"][a]["gap_def1_minus_def0"]
    print(f"{a:16}", "  ".join("   -  " if g[str(s)] is None else f"{g[str(s)]:+6.2f}" for s in GAP_STEPS))
print()
print("milestone: mean val change over straddling window vs previous window (%), mean train-wave change (%)")
for M in MILESTONES:
    s = res["milestone_summary"][str(M)]
    print(f"{M:6}  {s['mean_val_change_pct']:+6.2f} vs {s['mean_prev_window_change_pct']:+6.2f}   wave {s['mean_train_wave_change_pct']:+6.2f}")
print()
for a in ARMS:
    r = res["arms"][a]
    print(f"{a:16} milestones val-change%: " + " ".join(f"{r['lr_milestones'][str(M)]['val_change_pct']:+5.1f}" for M in MILESTONES)
          + "   prev%: " + " ".join(f"{r['lr_milestones'][str(M)]['prev_window_change_pct']:+5.1f}" for M in MILESTONES))
print()
for a in ARMS:
    r = res["arms"][a]
    print(f"{a:16} v/t ratio at 4k/8k/12k/16k: " + " ".join(f"{r['val_over_train_at'][str(s)]:.3f}" for s in GAP_STEPS)
          + f"   best val step {r['best_val']['step']}   lastQ max rise {r['last_quarter']['max_rise_above_running_min_pct']:.2f}% upticks {r['last_quarter']['n_upticks_of_8']}"
          + f"   slopes 4-8k {r['val_loss_slope_4000_8000_pct_per_1000']:+.2f} 8-12k {r['val_loss_slope_8000_12000_pct_per_1000']:+.2f} 12-16k {r['val_loss_slope_last4000_pct_per_1000']:+.2f}"
          + f"   wave {r['val_wave_slope_last4000_pct_per_1000']:+.2f} spec {r['val_spec_slope_last4000_pct_per_1000']:+.2f}")
print()
for a in ARMS:
    r = res["arms"][a]
    if r["spread"]:
        print(f"{a:16} spread peak {r['spread']['peak']:.4f} @ {r['spread']['peak_step']}  at 4k/8k/12k/16k: "
              + " ".join(f"{r['spread']['at'][str(s)]:.4f}" for s in GAP_STEPS)
              + f"  slope last4k {r['spread']['slope_last4000_pct_per_1000']:+.2f}%/1k"
              + f"  noise floor def1 {r['dev_noise_floor_last8']['def1_diff_std']:.2f} def0 {r['dev_noise_floor_last8']['def0_diff_std']:.2f}")
    else:
        print(f"{a:16} noise floor def0 {r['dev_noise_floor_last8']['def0_diff_std']:.2f}")
