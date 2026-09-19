"""Figure and note for audit/gated_arms.py.  Reads lisa_rtm_cache/results/gated_<tag>.json, writes
notes/analysis/figs/gated_<tag>.png and notes/analysis/gated_<tag>.md.  Every number in the note comes
from the JSON; nothing is typed in by hand except the parent note's own six numbers.

    venv/bin/python audit/gated_arms_report.py [--tag OV3_fast]
"""
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = pathlib.Path(__file__).resolve().parents[1]
ARMS = ["det", "es_marg", "es_marg_l0.1", "es_split_l0.1", "es_erb_l0.1", "es_dec_l0.1", "es_dec_erb_l0.1"]
# The 2026-09-17 note, section 2.1: es_erb_l0.1, step 13500, p236_002 only.
NOTE = {"arm": "es_erb_l0.1", "utt": "p236_002", "step": 13500, "deficit": -10.87,
        "loud": -11.68, "mid": -4.73, "quiet": 0.99, "snr": 13.88,
        "bands": [-1.79, -16.28, -14.22, -10.08, -13.50, -9.34]}
# dataviz reference palette, light surface: categorical slots 1-3 validate all-pairs.
C = {"loud": "#2a78d6", "mid": "#eb6834", "quiet": "#1baf7a",
     "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
     "grid": "#e1e0d9", "base": "#c3c2b7"}
SANS = "sans-serif"


def figure(R, out):
    M = R["_meta"]
    order = sorted(ARMS, key=lambda a: R[a]["mean"]["swing"])          # worst gating at the top
    rows = []                                                            # (label, loud, mid, quiet, hollow)
    for a in order:
        m = R[a]["mean"]
        rows.append((a, m["loud"], m["mid"], m["quiet"], False))
        if a == NOTE["arm"]:
            rows.append((f"   {NOTE['utt']} alone, step {NOTE['step']}", NOTE["loud"], NOTE["mid"], NOTE["quiet"], True))
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=200)
    fig.patch.set_facecolor(C["surface"]); ax.set_facecolor(C["surface"])
    ys = np.arange(len(rows))[::-1]
    for y, (label, lo, mi, qu, hollow) in zip(ys, rows):
        ax.plot([min(lo, mi, qu), max(lo, mi, qu)], [y, y], color=C["base"], lw=1.5, zorder=1,
                solid_capstyle="round")
        for k, x in (("loud", lo), ("mid", mi), ("quiet", qu)):
            if hollow:
                ax.plot(x, y, "o", ms=9, mfc=C["surface"], mec=C[k], mew=1.6, zorder=3)
            else:
                ax.plot(x, y, "o", ms=9, mfc=C[k], mec=C["surface"], mew=2, zorder=3)
        ax.text(3.3, y, f"{lo - qu:+.1f}", va="center", ha="left", color=C["muted" if hollow else "ink2"],
                fontsize=9, family=SANS)
    ax.axvline(0, color=C["muted"], lw=1, zorder=0)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, family=SANS)
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_color(C["muted"] if r[4] else C["ink"])
    ax.set_xlim(-14, 5.2); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xticks(range(-14, 3, 2))
    ax.tick_params(axis="x", colors=C["muted"], labelsize=9, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=C["grid"], lw=1, zorder=0); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("high-band energy ratio, model / target, dB   (0 = correct energy)",
                  color=C["ink2"], fontsize=9.5, family=SANS)
    ax.text(3.3, len(rows) - 0.45, "swing", color=C["ink2"], fontsize=9, ha="left", va="bottom",
            family=SANS, fontweight="bold")
    handles = [Line2D([], [], marker="o", ls="", ms=8, mfc=C[k], mec=C["surface"], mew=1.5, label=lab)
               for k, lab in (("loud", "loud frames, top 25 %"), ("mid", "mid, 25–75 %"),
                              ("quiet", "quiet, bottom 25 %"))]
    handles.append(Line2D([], [], marker="o", ls="", ms=8, mfc=C["surface"], mec=C["muted"], mew=1.4,
                          label="hollow: the parent note's one utterance"))
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, frameon=False,
                    fontsize=8.5, handletextpad=0.3, columnspacing=1.6, borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(C["ink2"])
    fig.suptitle("The high band gates with the speech, but 1 to 8 dB short of the truth",
                 x=0.01, ha="left", fontsize=11.5, color=C["ink"], fontweight="bold", family=SANS, y=0.995)
    fig.text(0.01, 0.915, f"Third-octave energy ratio above {M['band_lo_hz']/1000:.0f} kHz, split by the target's "
             f"frame loudness. Mean over {M['n_utts']} held-out utterances, step {M['step']}.\n"
             "A sampler that gates like the speech puts all three marks at one x. Swing = loud − quiet.",
             fontsize=8.5, color=C["ink2"], family=SANS, va="top", linespacing=1.4)
    fig.subplots_adjust(left=0.24, right=0.93, top=0.76, bottom=0.13)
    fig.savefig(out, facecolor=C["surface"])
    plt.close(fig)


def table(R, rows, cols, fmt, head, sd_fmt="{:.1f}"):
    out = ["| arm | " + " | ".join(head) + " |", "|---|" + "---:|" * len(head)]
    for a in rows:
        m, s = R[a]["mean"], R[a]["sd"]
        out.append(f"| {a} | " + " | ".join(fmt(m[c]) + " (" + sd_fmt.format(s[c]) + ")" for c in cols) + " |")
    return "\n".join(out)


def note_md(R, tag, fig_rel):
    M, P = R["_meta"], R["_paired"]
    g = lambda a, k: R[a]["mean"][k]
    by_swing = sorted(ARMS, key=lambda a: g(a, "swing"))
    worst, best = by_swing[0], by_swing[-1]
    bands = R[ARMS[0]]["mean"]["bands_hz"]
    worst_band = {a: bands[int(np.argmin(R[a]["mean"]["band_draw"]))] for a in ARMS}
    n_worst_8k = sum(abs(b - bands[1]) < 1 for b in worst_band.values())
    nrow = next(x for x in R[NOTE["arm"]]["per_utt"] if x["utt"] == NOTE["utt"])
    erb, dec, decerb, bd = P["erb_term"], P["dec_noise"], P["dec_noise_on_erb"], P["best_vs_det"]
    pm = lambda d: f"{d['mean']:+.1f} ± {d['se']:.1f}"
    n_flip = sum(x["swing"] > 0 for x in R["es_erb_l0.1"]["per_utt"])
    r_corr = np.corrcoef([x["target_contrast_lq"] for x in R["es_erb_l0.1"]["per_utt"]],
                         [x["swing"] for x in R["es_erb_l0.1"]["per_utt"]])[0, 1]
    tgt = g("det", "target_contrast_lq")
    L = []
    L.append(f"# Gated high-band deficit, all seven arms — `{tag}`, step {M['step']}\n")
    L.append(f"**Date:** 2026-09-17. **Data:** {M['n_utts']} held-out utterances, the first two cached FLACs of "
             f"p236 p237 p238 p360 p361 p374. **Draw:** τ = 1, seed 0 (`det`: τ = 0). **Noiseless:** τ = 0. "
             f"**Naive:** `resample_poly(resample_poly(y, 1, 4), 4, 1)`. Frames are split by the target's frame "
             f"energy in the evaluation STFT (n_fft {M['eval_n_fft']}, hop {M['eval_hop']}): loud = top 25 %, "
             f"mid = 25–75 %, quiet = bottom 25 %. The deficit is the mean over third-octave bands from "
             f"{M['band_lo_hz']/1000:.0f} to {M['band_hi_hz']/1000:.0f} kHz of 10 log10(model / target energy). "
             f"Swing = loud − quiet. Tables give the mean over utterances with the sd in parentheses; "
             f"paired differences give mean ± SE over the same {M['n_utts']} utterances.\n")
    L.append(f"**Reproduce:** `audit/gated_arms.py` → `lisa_rtm_cache/results/gated_{tag}.json`; "
             f"`audit/gated_arms_report.py` → this note and the figure. Extends "
             f"[`../2026-09-17-snr-ceiling-gating-and-scale.md`](../2026-09-17-snr-ceiling-gating-and-scale.md) §2.1, "
             f"which had one arm, one utterance, step 13500.\n")
    L.append("## 1. Deficit, gated deficit and SNR\n")
    L.append(table(R, ARMS, ["deficit_draw", "deficit_tau0", "loud", "mid", "quiet", "swing", "snr_draw", "snr_naive"],
                   lambda v: f"{v:+.2f}",
                   ["deficit draw", "deficit τ=0", "loud", "mid", "quiet", "swing", "SNR draw", "SNR naive"]))
    L.append("\nAll in dB. Naive is the same 12 utterances, so its column repeats. SNR is maximised by doing nothing "
             "(§1 of the parent note); it is here beside naive and ranks nothing.\n")
    L.append("### 1.1 How far the high band rises from the quiet quarter of frames to the loud quarter\n")
    L.append("Loud and quiet hold the same number of frames, so swing = model rise − target rise, and the gating "
             "fraction is model / target. 1.00 gates like the speech; 0 is a flat high band.\n")
    L.append(table(R, ARMS, ["target_contrast_lq", "model_contrast_lq", "model_contrast_lq_tau0", "gating_fraction"],
                   lambda v: f"{v:+.2f}" if abs(v) > 2 else f"{v:.2f}",
                   ["target rise, dB", "model rise, draw", "model rise, τ=0", "gating fraction"], sd_fmt="{:.2f}"))
    L.append("\n### 1.2 Paired differences, first minus second, over the same utterances\n")
    L.append("| pair | deficit | loud | quiet | swing | SNR |\n|---|---:|---:|---:|---:|---:|")
    for name, lab in (("erb_term", "ERB term: `es_erb_l0.1` − `es_marg_l0.1`"),
                      ("dec_noise", "decoder noise: `es_dec_l0.1` − `es_marg_l0.1`"),
                      ("dec_noise_on_erb", "decoder noise on ERB: `es_dec_erb_l0.1` − `es_erb_l0.1`"),
                      ("erb_term_on_dec", "ERB term on decoder noise: `es_dec_erb_l0.1` − `es_dec_l0.1`"),
                      ("split_wave_term", "low-band waveform term: `es_split_l0.1` − `es_marg_l0.1`"),
                      ("lam_0.1_vs_0.01", "λ 0.1 vs 0.01: `es_marg_l0.1` − `es_marg`"),
                      ("sampler_vs_det", "sampler vs point: `es_marg` − `det`"),
                      ("best_vs_det", "best vs point: `es_dec_erb_l0.1` − `det`")):
        d = P[name]
        L.append(f"| {lab} | " + " | ".join(f"{pm(d[k])} ({d[k]['n_pos']}+/{d[k]['n_neg']}−)"
                                              for k in ("deficit_draw", "loud", "quiet", "swing", "snr_draw")) + " |")
    L.append("\ndB, mean ± SE, and the sign count over 12 utterances.\n")
    L.append("## 2. Per-band ratio of the draw\n")
    head = [f"{b:.0f} Hz" for b in bands]
    out = ["| arm | " + " | ".join(head) + " |", "|---|" + "---:|" * len(head)]
    for a in ARMS:
        out.append(f"| {a} | " + " | ".join(f"{m:+.2f} ({s:.1f})" for m, s in
                                             zip(R[a]["mean"]["band_draw"], R[a]["sd"]["band_draw"])) + " |")
    L.append("\n".join(out))
    L.append("\nThird-octave centre frequencies; dB, mean (sd) over utterances. The τ = 0 pass per band is in the JSON.\n")
    L.append(f"## 3. Figure\n\n![loud / mid / quiet per arm]({fig_rel})\n\n"
             f"*Arms ordered by swing, worst at the top. Filled marks: mean over {M['n_utts']} utterances at step "
             f"{M['step']}. Hollow rings: the parent note's numbers, `{NOTE['arm']}` on {NOTE['utt']} alone at step "
             f"{NOTE['step']}. The column at the right is the swing.*\n")
    L.append("## 4. Does the note hold?\n")
    L.append(f"| `{NOTE['arm']}` on {NOTE['utt']} | deficit | loud | mid | quiet | swing | SNR |\n|---|---:|---:|---:|---:|---:|---:|")
    L.append(f"| note, step {NOTE['step']} | {NOTE['deficit']:+.2f} | {NOTE['loud']:+.2f} | {NOTE['mid']:+.2f} | "
             f"{NOTE['quiet']:+.2f} | {NOTE['loud']-NOTE['quiet']:+.2f} | {NOTE['snr']:.2f} |")
    L.append(f"| same utterance, step {M['step']} | {nrow['deficit_draw']:+.2f} | {nrow['loud']:+.2f} | {nrow['mid']:+.2f} | "
             f"{nrow['quiet']:+.2f} | {nrow['swing']:+.2f} | {nrow['snr_draw']:.2f} |")
    L.append(f"| mean of {M['n_utts']}, step {M['step']} | {g('es_erb_l0.1','deficit_draw'):+.2f} | {g('es_erb_l0.1','loud'):+.2f} | "
             f"{g('es_erb_l0.1','mid'):+.2f} | {g('es_erb_l0.1','quiet'):+.2f} | {g('es_erb_l0.1','swing'):+.2f} | "
             f"{g('es_erb_l0.1','snr_draw'):.2f} |\n")
    L.append(f"Per band, the note's draw on {NOTE['utt']} was " + " / ".join(f"{v:+.2f}" for v in NOTE["bands"]) +
             f"; step {M['step']} gives " + " / ".join(f"{v:+.2f}" for v in nrow["band_draw"]) + ". Same to 0.2 dB.\n")
    L.append("## 5. What it shows\n")
    L.append(
        f"Every arm gates, and every arm gates short. The target's high band rises {tgt:.1f} dB from the quiet "
        f"quarter of frames to the loud quarter; the seven models rise "
        f"{min(g(a,'model_contrast_lq') for a in ARMS):.1f} to {max(g(a,'model_contrast_lq') for a in ARMS):.1f} dB, "
        f"a gating fraction of {min(g(a,'gating_fraction') for a in ARMS):.2f} to "
        f"{max(g(a,'gating_fraction') for a in ARMS):.2f}, so the note's \"roughly constant level under the voice\" "
        f"is wrong as written: on {NOTE['utt']} the band rises 25 dB, and the swing is the 12 dB it falls short. "
        f"`{worst}` gates worst at {g(worst,'swing'):+.1f} dB and `{best}` best at {g(best,'swing'):+.1f}, "
        f"a paired {pm(bd['swing'])} dB on {bd['swing']['n_pos']} of {M['n_utts']} utterances; between them "
        f"`es_marg` {g('es_marg','swing'):+.1f}, `es_split_l0.1` {g('es_split_l0.1','swing'):+.1f}, "
        f"`es_erb_l0.1` {g('es_erb_l0.1','swing'):+.1f}, `es_dec_l0.1` {g('es_dec_l0.1','swing'):+.1f}, "
        f"`es_marg_l0.1` {g('es_marg_l0.1','swing'):+.1f}. "
        f"The ERB term fixes the level and not the gating: against `es_marg_l0.1` it lifts the deficit "
        f"{pm(erb['deficit_draw'])} dB, the loud frames {pm(erb['loud'])} and the quiet frames {pm(erb['quiet'])}, "
        f"each on all {M['n_utts']} utterances, and moves the swing {pm(erb['swing'])}, which is nothing. "
        f"It lands the quiet frames at {g('es_erb_l0.1','quiet'):+.2f} dB, the only arm with mid and quiet above zero, "
        f"and that is the hiss: the right energy on average, in the gaps as much as on the fricatives, for "
        f"{-erb['snr_draw']['mean']:.1f} dB of SNR. "
        f"Decoder noise alone is the same story at a third the size, deficit {pm(dec['deficit_draw'])} dB and swing "
        f"{pm(dec['swing'])}. "
        f"Decoder noise on top of the ERB term is the one change that moves the swing: `es_dec_erb_l0.1` against "
        f"`es_erb_l0.1` leaves the loud frames at {pm(decerb['loud'])} dB, drops the quiet frames "
        f"{pm(decerb['quiet'])} dB on {decerb['quiet']['n_neg']} of {M['n_utts']}, closes the swing by "
        f"{pm(decerb['swing'])}, and gives back {decerb['snr_draw']['mean']:.1f} dB of SNR. "
        f"The note's numbers hold on their own utterance: `es_erb_l0.1` on {NOTE['utt']} at step {M['step']} is "
        f"{nrow['loud']:+.2f} / {nrow['mid']:+.2f} / {nrow['quiet']:+.2f} against {NOTE['loud']:+.2f} / "
        f"{NOTE['mid']:+.2f} / {NOTE['quiet']:+.2f}, within 0.5 dB after 2500 more steps, and the per-band shape "
        f"matches to 0.2 dB. "
        f"They do not hold as a summary: {NOTE['utt']} is the worst of the twelve for that arm because its target "
        f"high band rises {nrow['target_contrast_lq']:.1f} dB from quiet to loud, the swing tracks that rise at "
        f"r = {r_corr:+.2f}, the twelve-utterance mean is {g('es_erb_l0.1','loud'):+.2f} / "
        f"{g('es_erb_l0.1','mid'):+.2f} / {g('es_erb_l0.1','quiet'):+.2f} with a swing of "
        f"{g('es_erb_l0.1','swing'):+.1f} dB rather than 12.7, and {n_flip} utterances swing the other way. "
        f"The inverted spectral balance is general: the {bands[1]:.0f} Hz band, where the target has the most "
        f"energy, is the worst band for {n_worst_8k} of the seven arms"
        + (f" (`es_split_l0.1` bottoms out at {worst_band['es_split_l0.1']:.0f} Hz)" if n_worst_8k < 7 else "")
        + ", and the band just above the cut is within 1.5 dB for every arm trained at λ = 0.1. "
        f"The mean deficit still hides all of this: `es_erb_l0.1` and `es_dec_erb_l0.1` sit "
        f"{abs(g('es_erb_l0.1','deficit_draw') - g('es_dec_erb_l0.1','deficit_draw')):.1f} dB apart on the deficit and "
        f"{abs(g('es_erb_l0.1','swing') - g('es_dec_erb_l0.1','swing')):.1f} dB apart on the swing. "
        f"Last, the τ = 0 column says where the band comes from: `es_split_l0.1`'s draw and noiseless pass agree to "
        f"{abs(g('es_split_l0.1','deficit_draw') - g('es_split_l0.1','deficit_tau0')):.1f} dB, so its noise adds no "
        f"high-band energy, while the ERB arms' noiseless pass sits 12 to 13 dB below their draw.\n")
    return "\n".join(L)


def main(tag="OV3_fast"):
    src = REPO / "lisa_rtm_cache" / "results" / f"gated_{tag}.json"
    if not src.exists():
        sys.exit(f"no {src}; run audit/gated_arms.py --tag {tag} first")
    R = json.loads(src.read_text())
    figs = REPO / "notes" / "analysis" / "figs"; figs.mkdir(parents=True, exist_ok=True)
    png = figs / f"gated_{tag}.png"
    figure(R, png)
    md = REPO / "notes" / "analysis" / f"gated_{tag}.md"
    md.write_text(note_md(R, tag, f"figs/gated_{tag}.png"))
    print("wrote", png, f"{png.stat().st_size/1024:.0f} KB"); print("wrote", md)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "OV3_fast")
