"""Every table in the paper and the technical appendix, rebuilt from results.json, plus a check of every
number the paper prints against the field it comes from.

    python results/make_tables.py            # writes results/tables/*.csv, *.tex and results/number_check.md

Pure standard library. Nothing here is typed in by hand except the paper's printed values in CHECKS,
which are what is being checked.
"""
import csv, json, math, pathlib

HERE = pathlib.Path(__file__).resolve().parent
D = json.load(open(HERE / "results.json"))
S = D["sota"]
OUT = HERE / "tables"; OUT.mkdir(exist_ok=True)

ARMS = ["det_paper", "det", "es_marg", "es_dec_l0.01", "es_erb_l0.01", "es_erb_l0.1", "es_erb_l0.001", "es_dec_erb_l0.1"]
ARM_LABEL = {"det_paper": "det\\_paper", "det": "det", "es_marg": "es\\_marg", "es_dec_l0.01": "es\\_dec",
             "es_erb_l0.01": "es\\_erb", "es_erb_l0.1": "es\\_erb, $\\lambda{=}0.1$", "es_erb_l0.001": "es\\_erb, $\\lambda{=}0.001$",
             "es_dec_erb_l0.1": "es\\_dec\\_erb, $\\lambda{=}0.1$"}
ARM_CHANGE = {"det_paper": ("LISA recipe (L1, $\\lambda{=}0$)", None), "det": ("+ log-magnitude term, $\\lambda{=}0.01$", "det_paper"),
              "es_marg": ("point $\\to$ draw (energy score)", "det"), "es_dec_l0.01": ("+ decoder noise", "es_marg"),
              "es_erb_l0.01": ("+ ERB-band term", "es_marg"), "es_erb_l0.1": ("$\\lambda \\times 10$", "es_erb_l0.01"),
              "es_erb_l0.001": ("$\\lambda / 10$", "es_erb_l0.01"), "es_dec_erb_l0.1": ("decoder noise + ERB, $\\lambda{=}0.1$", None)}
REL = ["flowhigh", "apbwe", "nuwave2", "audiosr"]
REL_LABEL = {"flowhigh": "FLowHigh", "flowhigh_std1": "FLowHigh, prior restored", "apbwe": "AP-BWE",
             "apbwe_sinc": "AP-BWE, own sinc input", "nuwave2": "NU-Wave 2", "audiosr": "AudioSR (speech)"}


def arm(a, s):
    return S["ours"][a]["sets"].get(s) or {}


def rel(m, s):
    if m in S["models"]:
        return S["models"][m]["sets"].get(s) or {}
    for base, v in S["models"].items():
        if m in (v.get("extra") or {}):
            return v["extra"][m]["sets"].get(s) or {}
    return {}


def crps_of(g):
    """The column the paper reports: the fair estimator for an ensemble, the MAE (= CRPS) for one output."""
    return g.get("crps_fair") if g.get("crps_fair") is not None else g.get("crps")


def f(v, d=2, sign=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.lstrip("-+") if float(s) == 0 else s          # no "-0.0"


def tex_minus(s):
    """A minus sign in front of a number becomes a math minus; hyphens in words and the "--" placeholder stay."""
    import re
    return s if s == "--" else re.sub(r"(?<![A-Za-z])-(?=\d)", "$-$", s)


def write(name, header, rows, tex_cols=None, caption=None):
    with open(OUT / f"{name}.csv", "w", newline="") as fh:
        plain = lambda c: (str(c).replace("\\_", "_").replace("$\\lambda{=}", "lambda=").replace("$\\lambda", "lambda")
                           .replace("\\times", "x").replace("$\\to$", "->").replace("\\kappa", "kappa").replace("\\", "").replace("$", ""))
        w = csv.writer(fh); w.writerow([plain(h) for h in header]); w.writerows([[plain(c) for c in r] for r in rows])
    cols = tex_cols or ("l" + "r" * (len(header) - 1))
    L = ["\\begin{tabular}{" + cols + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        L.append(" & ".join(tex_minus(str(c)) if i else str(c) for i, c in enumerate(r)) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(L) + "\n")


# ---- Table 1 of the paper ----------------------------------------------------------------------------
t1 = []
for a in ARMS:
    w, c = arm(a, "wide"), arm(a, "core")
    chg, base = ARM_CHANGE[a]
    step = f(w["deficit"] - arm(base, "wide")["deficit"], 1) if base else "--"
    t1.append([ARM_LABEL[a], chg, f(w["deficit"], 2), step + (f" [{ARM_LABEL[base]}]" if base else ""),
               f(c.get("spread_hb"), 3), f(crps_of(c), 3)])
write("table1_arms", ["arm", "one change", "deficit (dB)", "step", "spread", "CRPS"], t1, "llrlrr")

# ---- Table 2 of the paper ----------------------------------------------------------------------------
def params(m):
    p = S["models"][m]["params"]
    return f"{p/1e6:.1f} M" if p >= 1e6 else f"{p/1e3:.0f} k"


t2 = []
for m in REL:
    w, c = rel(m, "wide"), rel(m, "core")
    cc = rel("flowhigh_std1", "core") if m == "flowhigh" else c        # FLowHigh's spread/CRPS: its trained prior restored
    t2.append([REL_LABEL[m], params(m), f(w["deficit"], 2), f(w["lsd"], 3), f(w.get("visqol_audio"), 2),
               f(cc.get("spread_hb"), 3), f(crps_of(cc), 3), f(S["models"][m]["slope"]["slope"], 2)])
for a in ("det", "es_dec_erb_l0.1"):
    w, c = arm(a, "wide"), arm(a, "core")
    t2.append([ARM_LABEL[a], "88 k", f(w["deficit"], 2), f(w["lsd"], 3), f(w.get("visqol_audio"), 2),
               f(c.get("spread_hb"), 3), f(crps_of(c), 3), f(S["ours"][a]["slope"]["slope"], 2)])
write("table2_released", ["model", "params", "deficit (dB)", "LSD", "ViSQOL", "spread", "CRPS", "slope"], t2)

# ---- Appendix tables ---------------------------------------------------------------------------------
ROWS = [("flowhigh", "rel"), ("flowhigh_std1", "rel"), ("apbwe", "rel"), ("apbwe_sinc", "rel"), ("nuwave2", "rel"),
        ("audiosr", "rel")] + [(a, "arm") for a in ARMS]
lab = lambda k, t: REL_LABEL[k] if t == "rel" else ARM_LABEL[k]
get = lambda k, t, s: rel(k, s) if t == "rel" else arm(k, s)

for s in ("wide", "core", "ourtest"):
    rows = []
    for k, t in ROWS:
        g = get(k, t, s)
        if not g:
            continue
        rows.append([lab(k, t), f(g.get("deficit"), 2), f(g.get("deficit_bb"), 2), f(g.get("loud"), 2), f(g.get("mid"), 2),
                     f(g.get("quiet"), 2), f(g.get("lsd"), 3), f(g.get("lsd_hf"), 3), f(g.get("lsd_lf"), 3)])
    write(f"levels_{s}", ["condition", "deficit", "deficit bb", "loud", "mid", "quiet", "LSD", "LSD-HF", "LSD-LF"], rows)
    rows = []
    for k, t in ROWS:
        g = get(k, t, s)
        if not g:
            continue
        rows.append([lab(k, t), f(g.get("snr"), 2), f(g.get("vs_ceiling"), 2), f(g.get("coh"), 4), f(g.get("kappa"), 3)])
    write(f"snr_{s}", ["condition", "SNR (dB)", "vs ceiling", "coherent frac.", "$\\kappa$"], rows)

M = 8
target_gap = 10 * math.log10(2 / (1 + 1 / M))
rows = []
for k, t in ROWS:
    g = get(k, t, "core")
    if not g or g.get("pit_hi") is None:
        continue
    rows.append([lab(k, t), f(g.get("gap_hb"), 2), f(g.get("spread_hb"), 3), f(g.get("pit_lo"), 3), f(g.get("pit_hi"), 3),
                 f(g.get("crps"), 3), f(g.get("crps_fair"), 3), f(g.get("sliced_crps"), 3), f(g.get("corr_err"), 3)])
write("calibration_core", ["condition", "gap HB (dB)", "spread", "PIT low", "PIT high", "CRPS", "CRPS fair", "sliced", "corr err"], rows)
(OUT / "calibration_core_ideals.txt").write_text(
    f"M = {M}: gap HB {target_gap:.2f} dB; spread 1; PIT low = PIT high = 1/(M+1) = {1/(M+1):.3f}; corr_err 0.\n")

rows = []
for k, t in ROWS:
    g = get(k, t, "core")
    if not g or g.get("pit_hi") is not None:
        continue
    rows.append([lab(k, t), f(g.get("crps"), 3), f(g.get("sliced_crps"), 3), f(g.get("corr_err"), 3)])
write("point_models_core", ["condition (one output)", "CRPS = MAE", "sliced", "corr err"], rows)

rows = []
for k, t in ROWS:
    g = get(k, t, "wide")
    if not g:
        continue
    rows.append([lab(k, t), f(g.get("visqol_audio"), 2), f(g.get("nsim_audio"), 3), f(g.get("visqol_speech"), 2), f(g.get("pesq"), 2)])
write("perceptual_wide", ["condition", "ViSQOL audio", "NSIM audio", "ViSQOL speech", "PESQ wb"], rows)

rows = [[REL_LABEL[m], f(S["models"][m]["slope"]["slope"], 3), f(S["models"][m]["slope"]["r"], 2), S["models"][m]["slope"]["n"]] for m in REL]
rows += [[ARM_LABEL[a], f(S["ours"][a]["slope"]["slope"], 3), f(S["ours"][a]["slope"]["r"], 2), S["ours"][a]["slope"]["n"]] for a in ARMS]
write("slope_wide", ["model", "slope (dB/dB)", "r", "n"], rows)

rows = [[REL_LABEL[m], params(m), f(S["models"][m].get("rtf_m4"), 3), S["models"][m].get("rtf_device") or "--",
         S["models"][m]["arch"].get("steps", "--")] for m in REL]
write("cost", ["model", "params", "RTF", "device", "sampling"], rows, "lrrll")

rows = [[f"{r} kHz", f(S["ceiling"][f"snr_{r}k"], 2)] for r in (8, 12, 16, 24)]
write("ceiling", ["input rate", "empty-band SNR ceiling (dB)"], rows)

# ---- the paper's printed numbers, checked --------------------------------------------------------------
wide = lambda a: arm(a, "wide"); core = lambda a: arm(a, "core")
CHECKS = [
    # (where, printed, value, decimals, source)
    ("T1 det_paper deficit", "-21.78", wide("det_paper")["deficit"], 2, "sota.ours.det_paper.sets.wide.deficit"),
    ("T1 det deficit", "-6.18", wide("det")["deficit"], 2, "sota.ours.det.sets.wide.deficit"),
    ("T1 es_marg deficit", "-1.05", wide("es_marg")["deficit"], 2, "sota.ours.es_marg.sets.wide.deficit"),
    ("T1 es_dec deficit", "-1.12", wide("es_dec_l0.01")["deficit"], 2, "sota.ours.es_dec_l0.01.sets.wide.deficit"),
    ("T1 es_erb deficit", "-1.08", wide("es_erb_l0.01")["deficit"], 2, "sota.ours.es_erb_l0.01.sets.wide.deficit"),
    ("T1 es_erb l0.1 deficit", "-0.63", wide("es_erb_l0.1")["deficit"], 2, "sota.ours.es_erb_l0.1.sets.wide.deficit"),
    ("T1 es_erb l0.001 deficit", "-5.73", wide("es_erb_l0.001")["deficit"], 2, "sota.ours.es_erb_l0.001.sets.wide.deficit"),
    ("T1 es_dec_erb deficit", "-0.70", wide("es_dec_erb_l0.1")["deficit"], 2, "sota.ours.es_dec_erb_l0.1.sets.wide.deficit"),
    ("T1 step det", "15.6", wide("det")["deficit"] - wide("det_paper")["deficit"], 1, "difference of wide deficits"),
    ("T1 step es_marg", "5.1", wide("es_marg")["deficit"] - wide("det")["deficit"], 1, "difference of wide deficits"),
    ("T1 step es_dec", "0.0", wide("es_dec_l0.01")["deficit"] - wide("es_marg")["deficit"], 1, "difference of wide deficits"),
    ("T1 step es_erb", "0.0", wide("es_erb_l0.01")["deficit"] - wide("es_marg")["deficit"], 1, "difference of wide deficits"),
    ("T1 step lambda x10", "0.5", wide("es_erb_l0.1")["deficit"] - wide("es_erb_l0.01")["deficit"], 1, "difference of wide deficits"),
    ("T1 step lambda /10", "-4.6", wide("es_erb_l0.001")["deficit"] - wide("es_erb_l0.01")["deficit"], 1, "difference of wide deficits"),
    ("T1 es_marg spread", "0.526", core("es_marg")["spread_hb"], 3, "sota.ours.es_marg.sets.core.spread_hb"),
    ("T1 es_dec spread", "0.499", core("es_dec_l0.01")["spread_hb"], 3, "...core.spread_hb"),
    ("T1 es_erb spread", "0.442", core("es_erb_l0.01")["spread_hb"], 3, "...core.spread_hb"),
    ("T1 es_erb l0.1 spread", "0.479", core("es_erb_l0.1")["spread_hb"], 3, "...core.spread_hb"),
    ("T1 es_erb l0.001 spread", "0.225", core("es_erb_l0.001")["spread_hb"], 3, "...core.spread_hb"),
    ("T1 es_dec_erb spread", "0.464", core("es_dec_erb_l0.1")["spread_hb"], 3, "...core.spread_hb"),
    ("T1 det_paper CRPS (=MAE)", "2.85", core("det_paper")["crps"], 2, "sota.ours.det_paper.sets.core.crps"),
    ("T1 det CRPS (=MAE)", "0.889", core("det")["crps"], 3, "sota.ours.det.sets.core.crps"),
    ("T1 es_marg CRPS", "0.537", core("es_marg")["crps_fair"], 3, "...core.crps_fair"),
    ("T1 es_dec CRPS", "0.535", core("es_dec_l0.01")["crps_fair"], 3, "...core.crps_fair"),
    ("T1 es_erb CRPS", "0.512", core("es_erb_l0.01")["crps_fair"], 3, "...core.crps_fair"),
    ("T1 es_erb l0.1 CRPS", "0.515", core("es_erb_l0.1")["crps_fair"], 3, "...core.crps_fair"),
    ("T1 es_erb l0.001 CRPS", "0.683", core("es_erb_l0.001")["crps_fair"], 3, "...core.crps_fair"),
    ("T1 es_dec_erb CRPS", "0.514", core("es_dec_erb_l0.1")["crps_fair"], 3, "...core.crps_fair"),
    ("T2 FLowHigh deficit", "-1.60", rel("flowhigh", "wide")["deficit"], 2, "sota.models.flowhigh.sets.wide.deficit"),
    ("T2 FLowHigh LSD", "0.761", rel("flowhigh", "wide")["lsd"], 3, "...wide.lsd"),
    ("T2 FLowHigh ViSQOL", "3.61", rel("flowhigh", "wide")["visqol_audio"], 2, "...wide.visqol_audio"),
    ("T2 FLowHigh spread (prior restored)", "0.023", rel("flowhigh_std1", "core")["spread_hb"], 3, "...flowhigh_std1.sets.core.spread_hb"),
    ("T2 FLowHigh CRPS (prior restored)", "0.682", rel("flowhigh_std1", "core")["crps_fair"], 3, "...flowhigh_std1.sets.core.crps_fair"),
    ("T2 FLowHigh slope", "-0.06", S["models"]["flowhigh"]["slope"]["slope"], 2, "sota.models.flowhigh.slope (wide, n=240)"),
    ("T2 AP-BWE deficit", "-1.38", rel("apbwe", "wide")["deficit"], 2, "sota.models.apbwe.sets.wide.deficit"),
    ("T2 AP-BWE LSD", "0.780", rel("apbwe", "wide")["lsd"], 3, "...wide.lsd"),
    ("T2 AP-BWE ViSQOL", "3.48", rel("apbwe", "wide")["visqol_audio"], 2, "...wide.visqol_audio"),
    ("T2 AP-BWE CRPS (=MAE)", "0.777", rel("apbwe", "core")["crps"], 3, "...core.crps"),
    ("T2 AP-BWE slope", "-0.13", S["models"]["apbwe"]["slope"]["slope"], 2, "sota.models.apbwe.slope (wide, n=240)"),
    ("T2 NU-Wave 2 deficit", "-7.96", rel("nuwave2", "wide")["deficit"], 2, "sota.models.nuwave2.sets.wide.deficit"),
    ("T2 NU-Wave 2 LSD", "0.989", rel("nuwave2", "wide")["lsd"], 3, "...wide.lsd"),
    ("T2 NU-Wave 2 ViSQOL", "2.57", rel("nuwave2", "wide")["visqol_audio"], 2, "...wide.visqol_audio"),
    ("T2 NU-Wave 2 spread", "0.094", rel("nuwave2", "core")["spread_hb"], 3, "...core.spread_hb"),
    ("T2 NU-Wave 2 CRPS", "0.583", rel("nuwave2", "core")["crps_fair"], 3, "...core.crps_fair"),
    ("T2 NU-Wave 2 slope", "-0.31", S["models"]["nuwave2"]["slope"]["slope"], 2, "sota.models.nuwave2.slope (wide, n=240)"),
    ("T2 AudioSR deficit", "-0.60", rel("audiosr", "wide")["deficit"], 2, "sota.models.audiosr.sets.wide.deficit"),
    ("T2 AudioSR LSD", "1.561", rel("audiosr", "wide")["lsd"], 3, "...wide.lsd"),
    ("T2 AudioSR ViSQOL", "2.41", rel("audiosr", "wide")["visqol_audio"], 2, "...wide.visqol_audio"),
    ("T2 AudioSR spread", "0.765", rel("audiosr", "core")["spread_hb"], 3, "...core.spread_hb"),
    ("T2 AudioSR CRPS", "1.243", rel("audiosr", "core")["crps_fair"], 3, "...core.crps_fair"),
    ("T2 AudioSR slope", "-0.34", S["models"]["audiosr"]["slope"]["slope"], 2, "sota.models.audiosr.slope (wide, n=240)"),
    ("T2 det LSD", "0.864", wide("det")["lsd"], 3, "sota.ours.det.sets.wide.lsd"),
    ("T2 det ViSQOL", "3.13", wide("det")["visqol_audio"], 2, "...wide.visqol_audio"),
    ("T2 det slope", "-0.28", S["ours"]["det"]["slope"]["slope"], 2, "sota.ours.det.slope (wide, n=240)"),
    ("T2 es_dec_erb LSD", "0.895", wide("es_dec_erb_l0.1")["lsd"], 3, "...wide.lsd"),
    ("T2 es_dec_erb ViSQOL", "2.83", wide("es_dec_erb_l0.1")["visqol_audio"], 2, "...wide.visqol_audio"),
    ("T2 es_dec_erb slope", "-0.29", S["ours"]["es_dec_erb_l0.1"]["slope"]["slope"], 2, "sota.ours.es_dec_erb_l0.1.slope (wide, n=240)"),
    ("Text: SNR ceiling 12 kHz", "22.02", S["ceiling"]["snr_12k"], 2, "sota.ceiling.snr_12k"),
    ("Text: SNR ceiling 8 kHz", "19.0", S["ceiling"]["snr_8k"], 1, "sota.ceiling.snr_8k"),
    ("Text: SNR ceiling 16 kHz", "24.7", S["ceiling"]["snr_16k"], 1, "sota.ceiling.snr_16k"),
    ("Text: SNR ceiling 24 kHz", "30.0", S["ceiling"]["snr_24k"], 1, "sota.ceiling.snr_24k"),
    ("Text: ERB corr_err before", "0.374", core("es_marg")["corr_err"], 3, "sota.ours.es_marg.sets.core.corr_err"),
    ("Text: ERB corr_err after", "0.289", core("es_erb_l0.01")["corr_err"], 3, "sota.ours.es_erb_l0.01.sets.core.corr_err"),
    ("Text: NU-Wave 2 loud frames", "-9", rel("nuwave2", "wide")["loud"], 0, "sota.models.nuwave2.sets.wide.loud"),
    ("Text: NU-Wave 2 quiet frames", "4.7", rel("nuwave2", "wide")["quiet"], 1, "sota.models.nuwave2.sets.wide.quiet"),
    ("Text: sampler on p236-p238", "-4.3", arm("es_dec_erb_l0.1", "ourtest")["deficit"], 1, "sota.ours.es_dec_erb_l0.1.sets.ourtest.deficit"),
    ("Text: FLowHigh on p236-p238", "-1.0", rel("flowhigh", "ourtest")["deficit"], 1, "sota.models.flowhigh.sets.ourtest.deficit"),
]
lines = ["# The paper's printed numbers, checked against results.json", "",
         "Generated by `make_tables.py`. A mismatch means the printed value is not what the committed pipeline",
         "produces; the note column says which value to print.", "",
         "| where | printed | from results.json | match | source |", "|---|---|---|---|---|"]
n_ok = 0
for where, printed, v, d, src in CHECKS:
    got = f"{v:.{d}f}" if d else f"{v:.0f}"
    ok = float(got) == float(printed) or (got in ("-0.0", "0.0") and float(printed) == 0.0)
    n_ok += ok
    lines.append(f"| {where} | {printed} | {got} | {'yes' if ok else '**NO**'} | `{src}` |")
lines += ["", f"{n_ok} of {len(CHECKS)} match.", "",
          "Pooled calibration claim in the text (\"for our arms the truth lies above all eight draws in 21 to 24% of "
          "high-band bins, against 11%\"): PIT high on `core`, one tail, ideal 1/9 = 0.111:", ""]
for a in ARMS[2:]:
    g = core(a)
    lines.append(f"- {a}: PIT high {g['pit_hi']:.3f}, PIT low {g['pit_lo']:.3f}")
(HERE / "number_check.md").write_text("\n".join(lines) + "\n")
print(f"tables -> {OUT}\nnumber check: {n_ok}/{len(CHECKS)} match -> {HERE / 'number_check.md'}")
