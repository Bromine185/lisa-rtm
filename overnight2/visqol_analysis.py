"""Read the ViSQOL/LSD evaluation written by c6_visqol_eval.py (Drive lisa_rtm/ov2/visqol_per_utt.json) and
produce: ranked tables per set (primary: LSD and ViSQOL), paired differences vs the deterministic arm with
bootstrap 95% CIs, and a figure of LSD vs ViSQOL (speech-16k and audio-48k) for every condition.

usage: venv/bin/python overnight2/visqol_analysis.py [path/to/visqol_per_utt.json]
"""
import json, sys, pathlib, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[1]
DRIVE = pathlib.Path.home() / "Library/CloudStorage/GoogleDrive-naghavnarna@gmail.com/My Drive/lisa_rtm/ov2"
SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DRIVE / "visqol_per_utt.json"
PER = json.load(open(SRC))
OUT = REPO / "overnight2" / "visqol_results"
OUT.mkdir(exist_ok=True)
KEYS = ["lsd", "visqol_speech16k", "nsim_speech16k", "visqol_audio48k", "nsim_audio48k", "pesq_wb", "hb_lsd", "snr"]
HIGHER = {"visqol_speech16k", "nsim_speech16k", "visqol_audio48k", "nsim_audio48k", "pesq_wb", "snr"}
rng = np.random.default_rng(0)

def arr(rows, k):
    return np.array([r.get(k, np.nan) for r in rows], float)

def boot_ci(d, n=4000):
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return (np.nan, np.nan)
    idx = rng.integers(len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))

md = ["# LSD + ViSQOL evaluation of every 8 Sep condition\n",
      "Primary metrics: LSD (lower is better) and ViSQOL MOS-LQO (higher is better). ViSQOL speech mode is computed exactly as `evaluation.py` does "
      "(torchaudio resample to 16 kHz; 21 ERB bands, 50 Hz–8 kHz, so it sees one band of the reconstructed 6–24 kHz); audio mode runs at 48 kHz "
      "(32 bands to 24 kHz). NSIM is the mapping-independent similarity behind each MOS. Δ columns are paired differences against `det` with bootstrap 95 % CIs.\n"]
summary = {}
for sname, conds in PER.items():
    n = len(next(iter(conds.values())))
    base = conds.get("det")
    md.append(f"\n## {sname} (n = {n} utterances)\n")
    md.append("| condition | LSD | ΔLSD vs det [95% CI] | ViSQOL sp16k | Δ [CI] | NSIM sp | ViSQOL au48k | Δ [CI] | NSIM au | PESQ wb | HB-LSD | SNR |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows_out = []
    for c, rows in conds.items():
        m = {k: float(np.nanmean(arr(rows, k))) for k in KEYS}
        d = {}
        for k in ("lsd", "visqol_speech16k", "visqol_audio48k"):
            if base is not None and c != "det":
                diff = arr(rows, k) - arr(base, k)
                lo, hi = boot_ci(diff)
                d[k] = (float(np.nanmean(diff)), lo, hi)
            else:
                d[k] = (0.0, 0.0, 0.0)
        rows_out.append((c, m, d))
        summary.setdefault(sname, {})[c] = {"mean": m, "delta_vs_det": d}
    # rank by LSD (asc) then ViSQOL speech (desc)
    rows_out.sort(key=lambda t: (t[1]["lsd"], -t[1]["visqol_speech16k"]))
    for c, m, d in rows_out:
        f = lambda k: f"{d[k][0]:+.3f} [{d[k][1]:+.3f}, {d[k][2]:+.3f}]" if c != "det" else "—"
        md.append(f"| {c} | {m['lsd']:.3f} | {f('lsd')} | {m['visqol_speech16k']:.3f} | {f('visqol_speech16k')} | {m['nsim_speech16k']:.3f} | "
                  f"{m['visqol_audio48k']:.3f} | {f('visqol_audio48k')} | {m['nsim_audio48k']:.3f} | {m['pesq_wb']:.3f} | {m['hb_lsd']:.3f} | {m['snr']:.2f} |")
    # winners on the two primary metrics
    best_lsd = min(rows_out, key=lambda t: t[1]["lsd"]); best_vs = max(rows_out, key=lambda t: t[1]["visqol_speech16k"]); best_va = max(rows_out, key=lambda t: t[1]["visqol_audio48k"])
    md.append(f"\nBest LSD: **{best_lsd[0]}** ({best_lsd[1]['lsd']:.3f}). Best ViSQOL speech-16k: **{best_vs[0]}** ({best_vs[1]['visqol_speech16k']:.3f}). "
              f"Best ViSQOL audio-48k: **{best_va[0]}** ({best_va[1]['visqol_audio48k']:.3f}).\n")
(OUT / "visqol_tables.md").write_text("\n".join(md))
json.dump(summary, open(OUT / "visqol_summary_local.json", "w"), indent=1)
print("\n".join(md))

# ---- figure: LSD vs ViSQOL per set, both modes -----------------------------------------------------
sets = list(PER)
fig, axes = plt.subplots(len(sets), 2, figsize=(13, 4.6 * len(sets)), squeeze=False)
def style(c):
    if c.startswith("es_"):   return dict(color="#B23A6F", marker="o")
    if c.startswith("wide_es"): return dict(color="#7A2A6F", marker="o")
    if c.startswith("ladder"): return dict(color="#A8720F", marker="^")
    if c == "naive":          return dict(color="#555", marker="x")
    return dict(color="#0E7C8B", marker="s")
for i, sname in enumerate(sets):
    for j, (vk, title) in enumerate((("visqol_speech16k", "ViSQOL speech mode, 16 kHz (as evaluation.py)"), ("visqol_audio48k", "ViSQOL audio mode, 48 kHz (whole band)"))):
        ax = axes[i, j]
        for c, rows in PER[sname].items():
            x, y = np.nanmean(arr(rows, "lsd")), np.nanmean(arr(rows, vk))
            st = style(c)
            ax.scatter([x], [y], s=46, **st, alpha=0.9)
            ax.annotate(c.replace(" tau=", " τ").replace("ladder ", ""), (x, y), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel("LSD (lower is better)"); ax.set_ylabel("MOS-LQO (higher is better)")
        ax.set_title(f"{sname}: {title}", fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "lsd_vs_visqol.png", dpi=140)
print("saved", OUT / "lsd_vs_visqol.png")
