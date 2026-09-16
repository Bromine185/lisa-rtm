"""Report generator for the OV3 run: curves, tables, ViSQOL summary, prediction verdicts, figures.

    python overnight3/report_ov3.py --dir <folder with the JSONs> --out <output folder> \
        [--tag OV3_es] [--ds-tag OV3_es_datashare] [--fs 48000]

Reads ov3_history_<tag>.json, results_<tag>.json, visqol_<tag>.json (each looked up in --dir and --dir/ov3;
the visqol file is optional and every table degrades to NO DATA without it) and the same results/visqol
files for --ds-tag when present (pass --ds-tag none to skip).  Plain CLI, no Colab globals.
Schemas: overnight3/e2_trainer.py (history), e4_eval.py (results), e5_visqol.py (visqol).
Predictions P1-P7: notes/2026-09-16-score-geometry-decoder-noise-readouts.md section 2.
"""
import argparse, json, math, os, pathlib, sys
import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF, DET = "es_marg", "det"          # the 8 Sep recipe reference and the deterministic reference


# ---- loading ----------------------------------------------------------------------------------------
def find(d, name):
    for base in (d, d / "ov3", d / "figures"):
        p = base / name
        if p.exists():
            return p
    return None


def load_json(d, name):
    p = find(d, name)
    return (json.load(open(p)) if p else None), p


def arms_of(res):
    return [k for k in res if not k.startswith("_")]


def is_stoch(res, k):
    return "mean" in res[k]["eval12"]


def ro_name(res):
    return f"logmean{res['_M']}"


def f(v, nd=3, sign=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def vq_cond(k, res, kind="tau1", pt=True):
    """ViSQOL condition name for arm k: kind in tau1 | mean | logmean."""
    if not is_stoch(res, k):
        base = k
    else:
        M = res["_M"]
        base = {"tau1": f"{k} tau=1", "mean": f"{k} mean{M}", "logmean": f"{k} logmean{M}"}[kind]
    return base + (" | passthrough" if pt else "")


def esc(c):
    """Condition names contain ' | '; escape for markdown table cells."""
    return str(c).replace("|", "\\|")


def vq(visqol, cond, key="visqol_audio48k"):
    if not visqol:
        return None
    a = visqol.get("agg", {}).get(cond)
    return None if a is None else a.get(key)


# ---- 1. curves --------------------------------------------------------------------------------------
def draw_curves(hist, tag, out):
    names = list(hist)
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(2, 2, figsize=(17, 9))
    ax = ax.ravel()
    nan = lambda vs: [float("nan") if v is None else v for v in vs]
    for i, k in enumerate(names):
        h, c = hist[k], cols[i % len(cols)]
        if h.get("val_step"):
            ax[0].plot(h["val_step"], nan(h["train_loss_ema"]), "-", color=c, lw=1.3, label=f"{k} train (EMA)")
            ax[0].plot(h["val_step"], nan(h["val_loss"]), "--", color=c, lw=1.3, label=f"{k} val")
            ax[1].plot(h["val_step"], nan(h["val_wave"]), "-", color=c, lw=1.3, label=f"{k} wave")
            ax[1].plot(h["val_step"], nan(h["val_spec"]), ":", color=c, lw=1.6, label=f"{k} spec")
        if h.get("dev_step"):
            ax[2].plot(h["dev_step"], nan(h["snr0"]), "-", color=c, lw=1.3, label=f"{k} tau=0")
            if any(v is not None for v in h["snr1"]):
                ax[2].plot(h["dev_step"], nan(h["snr1"]), "-.", color=c, lw=1.1, label=f"{k} tau=1")
            d = h["def1"] if any(v is not None for v in h["def1"]) else h["def0"]
            ax[3].plot(h["dev_step"], nan(d), "o-", ms=3, color=c, lw=1.3, label=f"{k}")
    if names and hist[names[0]].get("snr_naive"):
        ax[2].axhline(hist[names[0]]["snr_naive"][-1], color="grey", ls=":", lw=1.2, label="naive upsampling")
    ax[0].set_yscale("log"); ax[0].set_title("A. objective: train EMA (solid) vs held-out val (dashed)", fontsize=10)
    ax[1].set_yscale("log"); ax[1].set_title("B. val terms: wave (solid), spec (dotted)", fontsize=10)
    ax[2].set_title("C. probe utterance SNR: tau=0 (solid), tau=1 (dash-dot)", fontsize=10); ax[2].set_ylabel("dB")
    ax[3].axhline(0, color="k", lw=1); ax[3].axhspan(-2, 2, color="grey", alpha=0.15)
    ax[3].set_title("D. probe high-band energy deficit (one draw for samplers)", fontsize=10); ax[3].set_ylabel("dB")
    for a in ax:
        a.set_xlabel("step"); a.grid(alpha=0.3); a.legend(fontsize=6, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle(f"{tag}: training and validation", fontsize=11)
    plt.tight_layout()
    p = out / f"curves_{tag}.png"; plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


# ---- 2. tables --------------------------------------------------------------------------------------
def main_table(res, visqol):
    M, ro = res["_M"], ro_name(res)
    rows = [f"| arm | SNR one draw | LSD | HB-LSD | deficit dB | CRPS | sliced CRPS | corr err | HB kappa | PIT end-bins (ideal {2/(M+1):.3f}) | SNR gap (cal {10*math.log10(2/(1+1/M)):.2f}) | mean-of-{M} SNR | mean deficit | one-draw pt LSD | {ro} pt LSD | {ro} deficit | ViSQOL-au one draw pt |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in arms_of(res):
        e = res[k]["eval12"]; s = e["single"]
        r = f"| {k} | {f(s['snr'],2)} | {f(s['lsd'])} | {f(s['hb_lsd'])} | {f(s['deficit'],2,True)} | {f(e['crps'],4)} | {f(e['sliced_crps'],4)} | {f(e['corr_err'],2)} | {f(s['hb_kappa'],3,True)} | "
        if is_stoch(res, k):
            m, rp = e["mean"], e["readouts_pt"].get(ro, {})
            r += f"{f(e['pit_end'])} | {f(e['snr_gap'],2)} | {f(m['snr'],2)} | {f(m['deficit'],2,True)} | {f(e['single_pt']['lsd'])} | {f(rp.get('lsd'))} | {f(rp.get('deficit'),2,True)} | "
        else:
            r += f"- | - | - | - | {f(e['single_pt']['lsd'])} | - | - | "
        r += f"{f(vq(visqol, vq_cond(k, res)), 3)} |"
        rows.append(r)
    for nm, key in (("floor: passthrough + empty HB", "_floor"), ("ceiling: passthrough + true HB", "_ceiling")):
        s = res[key]
        rows.append(f"| {nm} | {f(s['snr'],2)} | {f(s['lsd'])} | {f(s['hb_lsd'])} | {f(s['deficit'],2,True)} | - | - | - | - | - | - | - | - | - | - | - | {f(vq(visqol, nm), 3)} |")
    rows.append(f"\nn = {res['_n']} utterances, M = {M}. CRPS on high-band log-magnitude (notebook estimator, all M^2 pairs). Deficit: mean high-band third-octave energy ratio, dB. "
                f"pt = baseband passthrough (given low band + model high band). {ro} = per-bin mean of log|STFT| over the {M} draws, phase of draw 0.")
    return "\n".join(rows)


def readout_table(res, visqol):
    M, ro = res["_M"], ro_name(res)
    rows = ["| arm | readout | SNR | LSD | HB-LSD | deficit dB | ViSQOL-au |", "|---|---|---|---|---|---|---|"]
    for k in arms_of(res):
        if not is_stoch(res, k):
            continue
        e = res[k]["eval12"]
        items = [("one draw", e["single"], vq(visqol, vq_cond(k, res, "tau1", False))),
                 ("one draw + pt", e["single_pt"], vq(visqol, vq_cond(k, res, "tau1", True))),
                 (f"mean of {M}", e["mean"], vq(visqol, vq_cond(k, res, "mean", False))),
                 (f"mean of {M} + pt", e["mean_pt"], vq(visqol, vq_cond(k, res, "mean", True)))]
        if "logmean4" in e.get("readouts_pt", {}):
            items.append(("logmean4 + pt", e["readouts_pt"]["logmean4"], None))
        items.append((f"{ro}", e["readouts"][ro], vq(visqol, vq_cond(k, res, "logmean", False))))
        items.append((f"{ro} + pt", e["readouts_pt"][ro], vq(visqol, vq_cond(k, res, "logmean", True))))
        for nm, s, v in items:
            rows.append(f"| {k} | {nm} | {f(s['snr'],2)} | {f(s['lsd'])} | {f(s['hb_lsd'])} | {f(s['deficit'],2,True)} | {f(v)} |")
    return "\n".join(rows)


def delta_table(res, visqol):
    rows = ["| arm | dCRPS vs es_marg | dLSD(pt) vs es_marg | dViSQOL-au(pt) vs es_marg | dCRPS vs det | dLSD(pt) vs det | dViSQOL-au(pt) vs det |",
            "|---|---|---|---|---|---|---|"]
    def trip(k):
        e = res[k]["eval12"]
        return e["crps"], e["single_pt"]["lsd"], vq(visqol, vq_cond(k, res))
    refs = {r: trip(r) for r in (REF, DET) if r in res}
    for k in arms_of(res):
        c, l, v = trip(k)
        cells = []
        for r in (REF, DET):
            if r not in refs:
                cells += ["-", "-", "-"]; continue
            rc, rl, rv = refs[r]
            cells += [f(c - rc, 4, True), f(l - rl, 3, True), f(None if (v is None or rv is None) else v - rv, 3, True)]
        rows.append(f"| {k} | " + " | ".join(cells) + " |")
    rows.append("\nSign convention: dCRPS and dLSD negative = better than the reference; dViSQOL positive = better. LSD and ViSQOL are the one-draw passthrough condition (deterministic arms: the point forecast with passthrough).")
    return "\n".join(rows)


# ---- 3. visqol --------------------------------------------------------------------------------------
KEYS = ("snr", "lsd", "hb_lsd", "visqol_speech16k", "nsim_speech16k", "pesq_wb", "visqol_audio48k", "nsim_audio48k")

def visqol_md(visqol, tag):
    if not visqol:
        return f"# ViSQOL {tag}\n\nNO DATA (visqol_{tag}.json not found).\n"
    agg = visqol["agg"]
    rows = [f"# ViSQOL {tag}\n", f"n = {visqol.get('n')} utterances, M = {visqol.get('M')}, speech-mode MOS mapping: {visqol.get('speech_mapping')}. Sorted by audio-mode ViSQOL (48 kHz, 32 bands), descending.\n",
            "| condition | SNR | LSD | HB-LSD | ViSQOL speech16k | NSIM sp | PESQ wb | ViSQOL audio48k | NSIM au |", "|---|---|---|---|---|---|---|---|---|"]
    order = sorted(agg, key=lambda c: -(agg[c].get("visqol_audio48k") if agg[c].get("visqol_audio48k") == agg[c].get("visqol_audio48k") else -9))
    for c in order:
        a = agg[c]
        rows.append(f"| {esc(c)} | {f(a.get('snr'),2)} | {f(a.get('lsd'))} | {f(a.get('hb_lsd'))} | {f(a.get('visqol_speech16k'))} | {f(a.get('nsim_speech16k'))} | {f(a.get('pesq_wb'))} | {f(a.get('visqol_audio48k'))} | {f(a.get('nsim_audio48k'))} |")
    # best per family, passthrough preferred
    def fam(c):
        if c.startswith("floor") or c.startswith("ceiling") or c == "naive":
            return None
        if "logmean" in c or "mean" in c:
            return "readout"
        return "single draw" if "tau=1" in c else "deterministic"
    rows.append("\n## Best per family (passthrough conditions preferred)\n")
    rows.append("| metric | family | condition | value |", ); rows.append("|---|---|---|---|")
    for metric, better in (("lsd", min), ("visqol_audio48k", max)):
        for family in ("deterministic", "single draw", "readout"):
            cands = [c for c in agg if fam(c) == family and "passthrough" in c and agg[c].get(metric) == agg[c].get(metric)]
            if not cands:
                cands = [c for c in agg if fam(c) == family and agg[c].get(metric) == agg[c].get(metric)]
            if cands:
                b = better(cands, key=lambda c: agg[c][metric])
                rows.append(f"| {metric} | {family} | {esc(b)} | {f(agg[b][metric])} |")
    return "\n".join(rows) + "\n"


# ---- 4. predictions ---------------------------------------------------------------------------------
def predictions(res, visqol, tag):
    A = arms_of(res); M = res["_M"]; ro = ro_name(res)
    E = {k: res[k]["eval12"] for k in A}
    out = [f"# Predictions, {tag}\n", "| prediction | test | numbers | verdict |", "|---|---|---|---|"]
    verdicts = {}

    def add(pid, test, nums, verdict):
        verdicts[pid] = verdict; out.append(f"| {pid} | {test} | {nums} | **{verdict}** |")

    def have(*ks):
        return all(k in E for k in ks)

    # P1
    if have("es_marg_l0.1", REF) and is_stoch(res, "es_marg_l0.1"):
        a, b = E["es_marg_l0.1"], E[REF]
        va, vb = vq(visqol, vq_cond("es_marg_l0.1", res)), vq(visqol, vq_cond(REF, res))
        crps_ok = a["crps"] <= 1.05 * b["crps"]; pit_ok = a["pit_end"] <= 0.15
        lsd_ok = a["single_pt"]["lsd"] < b["single_pt"]["lsd"]
        vq_ok = None if (va is None or vb is None) else va > vb
        nums = f"one-draw LSD(pt) {f(a['single_pt']['lsd'])} vs {f(b['single_pt']['lsd'])}; ViSQOL-au(pt) {f(va)} vs {f(vb)}; CRPS {f(a['crps'],4)} vs {f(b['crps'],4)} ({f(100*(a['crps']/b['crps']-1),1,True)} %); PIT end {f(a['pit_end'])}"
        if not crps_ok or not pit_ok:
            v = "REFUTED"
        elif lsd_ok and vq_ok:
            v = "HOLDS"
        elif vq_ok is None:
            v = "PARTIAL" if lsd_ok else "REFUTED"
        else:
            v = "PARTIAL" if (lsd_ok or vq_ok) else "REFUTED"
        add("P1 weight", "es_marg_l0.1 vs es_marg: lower one-draw LSD(pt) and higher ViSQOL-au(pt); CRPS not up by >5 %; PIT end bins <= 0.15", nums, v)
    else:
        add("P1 weight", "-", "-", "NO DATA")
    # P2
    if have("es_split_l0.1", "es_marg_l0.1"):
        a, b = E["es_split_l0.1"], E["es_marg_l0.1"]
        da, db = abs(a["single"]["deficit"]), abs(b["single"]["deficit"])
        smaller = da < db; crps_ok = a["crps"] <= 1.01 * b["crps"]
        nums = f"|deficit| {f(da,2)} vs {f(db,2)} dB; CRPS {f(a['crps'],4)} vs {f(b['crps'],4)}"
        add("P2 low-band waveform term", "es_split_l0.1 vs es_marg_l0.1: smaller |one-draw deficit| at equal (<= +1 %) or better CRPS", nums,
            "HOLDS" if (smaller and crps_ok) else ("PARTIAL" if smaller else "REFUTED"))
    else:
        add("P2 low-band waveform term", "-", "-", "NO DATA")
    # P3
    if have("es_erb_l0.1", "es_marg_l0.1"):
        a, b = E["es_erb_l0.1"], E["es_marg_l0.1"]
        va, vb = vq(visqol, vq_cond("es_erb_l0.1", res)), vq(visqol, vq_cond("es_marg_l0.1", res))
        dv = None if (va is None or vb is None) else va - vb
        dl = a["single_pt"]["lsd"] - b["single_pt"]["lsd"]
        crps_ok = a["crps"] <= 1.03 * b["crps"]; def_ok = abs(a["single"]["deficit"]) <= 2.0
        nums = f"dViSQOL-au(pt) {f(dv,3,True)}; CRPS {f(a['crps'],4)} vs {f(b['crps'],4)}; one-draw deficit {f(a['single']['deficit'],2,True)} dB; dLSD(pt) {f(dl,3,True)}"
        if dv is None:
            v = "NO DATA"
        elif dv >= 0.10 and crps_ok and def_ok:
            v = "HOLDS"
        elif dv >= 0.10:
            v = "PARTIAL"
        else:
            v = "REFUTED"
        add("P3 judge's geometry", "es_erb_l0.1 vs es_marg_l0.1: ViSQOL-au(pt) +0.10 or more at equal CRPS (<= +3 %), |one-draw deficit| <= 2 dB", nums, v)
    else:
        add("P3 judge's geometry", "-", "-", "NO DATA")
    # P4
    if have("es_dec_l0.1") and is_stoch(res, "es_dec_l0.1"):
        a = E["es_dec_l0.1"]
        d = a["single"]["deficit"]; gap, cal = a["snr_gap"], a["snr_gap_calibrated"]
        nums = f"one-draw deficit {f(d,2,True)} dB; SNR gap {f(gap,2)} vs calibrated {f(cal,2)}"
        v = "REFUTED" if d < -5 else ("HOLDS" if (abs(d) <= 2 and abs(gap - cal) <= 0.5) else "PARTIAL")
        add("P4 decoder noise", "es_dec_l0.1: |one-draw deficit| <= 2 dB and |SNR gap - calibrated| <= 0.5 dB; refuted if deficit < -5 dB", nums, v)
    else:
        add("P4 decoder noise", "-", "-", "NO DATA")
    # P5
    def wins_both(k):
        if k == REF or REF not in E:
            return None
        e, r = E[k], E[REF]
        vk, vr = vq(visqol, vq_cond(k, res)), vq(visqol, vq_cond(REF, res))
        c = e["crps"] < r["crps"]; l = e["single_pt"]["lsd"] < r["single_pt"]["lsd"]
        v = None if (vk is None or vr is None) else vk > vr
        return c, l, v
    if have("es_dec_erb_l0.1", REF):
        c, l, v = wins_both("es_dec_erb_l0.1")
        others = [k for k in A if k not in ("es_dec_erb_l0.1", REF) and all(x is True for x in wins_both(k))]
        nums = f"vs es_marg: CRPS better {c}, LSD(pt) better {l}, ViSQOL-au(pt) better {v}; other arms improving all three: {others or 'none'}"
        if v is None:
            vv = "PARTIAL" if (c and l) else "REFUTED"
            nums += " (ViSQOL missing)"
        elif c and l and v:
            vv = "HOLDS" if not others else "PARTIAL"
        else:
            vv = "REFUTED"
        add("P5 combination", "es_dec_erb_l0.1 improves on es_marg in CRPS and in LSD(pt) and ViSQOL-au(pt); and is the only arm that does", nums, vv)
    else:
        add("P5 combination", "-", "-", "NO DATA")
    # P6
    if have(DET):
        dets = [k for k in A if not is_stoch(res, k)]
        best_s = min([E[k]["crps"] for k in A if is_stoch(res, k)] or [float("inf")])
        dc = E[DET]["crps"]
        ok = abs(dc - 1.0) <= 0.2 and all(E[k]["crps"] > best_s for k in dets)
        nums = f"det CRPS {f(dc,4)}; deterministic arms {dets}; best sampler CRPS {f(best_s,4)}"
        add("P6 deterministic ceiling", "det CRPS within 1.0 +- 0.2 and no deterministic arm below the best sampler's CRPS", nums, "HOLDS" if ok else "REFUTED")
    else:
        add("P6 deterministic ceiling", "-", "-", "NO DATA")
    # P7
    st = [k for k in A if is_stoch(res, k)]
    if st and have(DET):
        det_lsd = E[DET]["single_pt"]["lsd"]
        passed, parts = [], []
        for k in st:
            rl = E[k]["readouts_pt"][ro]["lsd"]; sl = E[k]["single_pt"]["lsd"]
            ok = rl < sl and rl < det_lsd
            passed.append(ok); parts.append(f"{k}: {f(rl)} vs draw {f(sl)}")
        nums = f"det(pt) LSD {f(det_lsd)}; " + "; ".join(parts)
        v = "HOLDS" if all(passed) else ("PARTIAL" if any(passed) else "REFUTED")
        add("P7 readout", f"{ro}(pt) LSD below the arm's one-draw(pt) LSD and below det(pt) LSD, for every stochastic arm", nums, v)
    else:
        add("P7 readout", "-", "-", "NO DATA")
    return "\n".join(out) + "\n", verdicts


# ---- 5. figures from results ------------------------------------------------------------------------
def band_centres(n, fs):
    edges = 200.0 * 2.0 ** (np.arange(n + 1) / 3.0)
    return np.sqrt(edges[:-1] * edges[1:])


def draw_spectrum(res, tag, out, fs):
    M, ro = res["_M"], ro_name(res)
    A = arms_of(res)
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, k in enumerate(A):
        e = res[k]["eval12"]; c = cols[i % len(cols)]
        cur = e["single"]["curve"]; x = band_centres(len(cur), fs)
        ax.semilogx(x, cur, "o-", ms=3, color=c, lw=1.3, label=f"{k} (one draw)" if is_stoch(res, k) else k)
        if is_stoch(res, k):
            ax.semilogx(x, e["mean"]["curve"], "--", lw=1.0, color=c, label=f"{k} (mean of {M})")
            ax.semilogx(x, e["readouts"][ro]["curve"], ":", lw=1.4, color=c, label=f"{k} ({ro})")
    ax.axhline(0, color="k", lw=1); ax.axvline(fs / 8, color="crimson", ls="--", lw=1); ax.axhspan(-2, 2, color="grey", alpha=0.15)
    ax.set_xlabel("frequency (Hz), third-octave bands from 200 Hz"); ax.set_ylabel("prediction / target energy (dB)")
    ax.set_title(f"{tag}: energy ratio per band -- one draw (solid), ensemble mean (dashed), {ro} readout (dotted)", fontsize=9)
    ax.legend(fontsize=6, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False); ax.grid(alpha=0.3)
    plt.tight_layout(); p = out / f"spectrum_{tag}.png"; plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def draw_pit(res, tag, out):
    st = [k for k in arms_of(res) if is_stoch(res, k)]
    if not st:
        return None
    n = len(st) + 1
    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.3))
    ax = np.atleast_1d(ax)
    for a, k in zip(ax, st):
        h = np.array(res[k]["eval12"]["pit_hist"], float); h /= max(h.sum(), 1)
        a.bar(range(len(h)), h, edgecolor="k", lw=0.5); a.axhline(1 / len(h), color="crimson", ls="--", lw=1)
        a.set_title(f"PIT {k}", fontsize=8); a.set_xlabel("rank of truth", fontsize=8); a.tick_params(labelsize=7)
    for k in st:
        s = np.array(res[k]["eval12"]["spread_skill"])
        ax[-1].plot(s[:, 0], s[:, 1], "o-", ms=3, label=k)
    lim = ax[-1].get_xlim(); ax[-1].plot([0, lim[1]], [0, lim[1]], "k--", lw=1)
    ax[-1].set_title("spread-skill", fontsize=8); ax[-1].set_xlabel("ensemble spread", fontsize=8); ax[-1].set_ylabel("RMSE of mean", fontsize=8)
    ax[-1].legend(fontsize=6); ax[-1].tick_params(labelsize=7)
    fig.suptitle(f"{tag}: calibration (flat PIT = calibrated)", fontsize=10)
    plt.tight_layout(); p = out / f"pit_{tag}.png"; plt.savefig(p, dpi=130); plt.close(fig)
    return p


# ---- 6. summary -------------------------------------------------------------------------------------
def summary(res, visqol, verdicts, tag):
    A = arms_of(res); E = {k: res[k]["eval12"] for k in A}; ro = ro_name(res)
    lines = [f"== {tag}: n={res['_n']} utterances, M={res['_M']} =="]
    if REF in E:
        r = E[REF]; rv = vq(visqol, vq_cond(REF, res))
        lines.append(f"reference es_marg: CRPS {f(r['crps'],4)}  one-draw pt LSD {f(r['single_pt']['lsd'])}  ViSQOL-au pt {f(rv)}")
        winners = []
        for k in A:
            if k == REF:
                continue
            e = E[k]; v = vq(visqol, vq_cond(k, res))
            c_ok = e["crps"] < r["crps"]; l_ok = e["single_pt"]["lsd"] < r["single_pt"]["lsd"]
            v_ok = None if (v is None or rv is None) else v > rv
            if c_ok and l_ok and (v_ok or (v_ok is None)):
                winners.append((k, e["crps"], e["single_pt"]["lsd"], v, v_ok))
        if winners:
            w = min(winners, key=lambda t: t[1])
            lines.append(f"WINNER under the both-metrics rule: {w[0]}  (CRPS {f(w[1],4)}, LSD pt {f(w[2])}, ViSQOL-au pt {f(w[3])}{' -- ViSQOL NO DATA' if w[4] is None else ''})")
            if len(winners) > 1:
                lines.append("also beating es_marg on every counted metric: " + ", ".join(t[0] for t in winners if t[0] != w[0]))
        else:
            lines.append("WINNER under the both-metrics rule: none (no arm beats es_marg on CRPS and LSD(pt) and ViSQOL-au(pt) together)")
    best_c = min(A, key=lambda k: E[k]["crps"]); lines.append(f"best CRPS: {best_c} {f(E[best_c]['crps'],4)}")
    cand = [(E[k]["single_pt"]["lsd"], f"{k} one draw + pt") for k in A]
    cand += [(E[k]["readouts_pt"][ro]["lsd"], f"{k} {ro} + pt") for k in A if is_stoch(res, k)]
    cand += [(E[k]["mean_pt"]["lsd"], f"{k} mean + pt") for k in A if is_stoch(res, k)]
    b = min(cand); lines.append(f"best LSD (any readout, passthrough): {b[1]} {f(b[0])}")
    if visqol:
        agg = visqol["agg"]
        cs = [c for c in agg if agg[c].get("visqol_audio48k") == agg[c].get("visqol_audio48k") and not c.startswith("ceiling")]
        if cs:
            bv = max(cs, key=lambda c: agg[c]["visqol_audio48k"]); lines.append(f"best audio ViSQOL (excluding the ceiling): {bv} {f(agg[bv]['visqol_audio48k'])}")
        fl, ce = agg.get("floor: passthrough + empty HB", {}).get("visqol_audio48k"), agg.get("ceiling: passthrough + true HB", {}).get("visqol_audio48k")
        if fl is not None and ce is not None and cs:
            lines.append(f"floor {f(fl)} / ceiling {f(ce)}: best model recovers {100*(agg[bv]['visqol_audio48k']-fl)/(ce-fl):.0f} % of the range")
    else:
        lines.append("ViSQOL: NO DATA")
    st = [k for k in A if is_stoch(res, k)]
    if st:
        lines.append("one-draw deficit dB: " + ", ".join(f"{k} {f(E[k]['single']['deficit'],1,True)}" for k in st))
        lines.append("PIT end bins (ideal %.3f): " % (2 / (res["_M"] + 1)) + ", ".join(f"{k} {f(E[k]['pit_end'])}" for k in st))
        lines.append("SNR gap vs calibrated: " + ", ".join(f"{k} {f(E[k]['snr_gap'],2)}/{f(E[k]['snr_gap_calibrated'],2)}" for k in st))
    lines.append("predictions: " + ", ".join(f"{p.split()[0]} {v}" for p, v in verdicts.items()))
    return "\n".join(lines)


# ---- main -------------------------------------------------------------------------------------------
def run_tag(d, out, tag, fs, with_history, with_predictions):
    hist, hp = load_json(d, f"ov3_history_{tag}.json")
    res, rp = load_json(d, f"results_{tag}.json")
    visqol, vp = load_json(d, f"visqol_{tag}.json")
    made = []
    if with_history and hist:
        made.append(draw_curves(hist, tag, out))
    elif with_history:
        print(f"[{tag}] history NO DATA (ov3_history_{tag}.json not found)")
    if not res:
        print(f"[{tag}] results NO DATA (results_{tag}.json not found)")
        return made, None
    md = [f"# Tables, {tag}\n", f"sources: {rp}, {vp or 'visqol NO DATA'}\n", "## Main table\n", main_table(res, visqol),
          "\n## Readouts per stochastic arm\n", readout_table(res, visqol), "\n## Deltas against the references\n", delta_table(res, visqol)]
    p = out / f"tables_{tag}.md"; p.write_text("\n".join(md) + "\n"); made.append(p)
    p = out / f"visqol_{tag}.md"; p.write_text(visqol_md(visqol, tag)); made.append(p)
    verdicts = {}
    if with_predictions:
        txt, verdicts = predictions(res, visqol, tag)
        p = out / f"predictions_{tag}.md"; p.write_text(txt); made.append(p)
    made.append(draw_spectrum(res, tag, out, fs))
    pp = draw_pit(res, tag, out)
    if pp:
        made.append(pp)
    print(summary(res, visqol, verdicts, tag))
    return made, verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="OV3_es"); ap.add_argument("--ds-tag", default="OV3_es_datashare")
    ap.add_argument("--fs", type=int, default=48000, help="target sample rate, for the band axis and the Nyquist line")
    a = ap.parse_args()
    d, out = pathlib.Path(a.dir), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    made, _ = run_tag(d, out, a.tag, a.fs, with_history=True, with_predictions=True)
    if a.ds_tag and a.ds_tag.lower() != "none":
        m2, _ = run_tag(d, out, a.ds_tag, a.fs, with_history=False, with_predictions=True)
        made += m2
    print("written:", ", ".join(str(m.name) for m in made))


if __name__ == "__main__":
    main()
