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

# ---- level against dispersion (core, M = 8) -----------------------------------------------------------
# When the high band is incoherent with the truth (c ~ 0), draws that are mutually independent, with truth HB
# energy E and mean draw HB energy P, give in expectation
#     gap    = 10 log10((E + P) / (E + P/M))          spread = P (1 + 1/M) / (E + P/M)
# so both waveform measures are set by level, not only by diversity. The reference is computed by the scorer
# per utterance from the measured E and P (gap_indep_hb: mean of dB over utterances, as gap_hb is;
# spread_indep_hb: pooled, as spread_hb is). No single level p stands in for a model whose bias varies by band.
rows = []
for k, t in ROWS:
    g = get(k, t, "core")
    if not g or g.get("gap_hb") is None:
        continue
    rows.append([lab(k, t), f(g["deficit_bb"], 1), f(g["gap_hb"], 2), f(g["gap_indep_hb"], 2),
                 f(g["spread_hb"], 3), f(g["spread_indep_hb"], 3), f(g["pit_lo"], 3), f(g["pit_hi"], 3)])
write("level_dispersion_core", ["condition", "HB level (dB)", "gap", "gap if indep.", "spread", "spread if indep.",
                                "PIT low", "PIT high"], rows)

# ---- does the harness reproduce the released models' own numbers? (wide, 240 utterances) ---------------
# Published values are citations, typed from the papers: FLowHigh Table I and AP-BWE Table V (12 -> 48 kHz,
# VCTK, 8 test speakers); NU-Wave 2's own Table 1 reports SNR, and its LSD/ViSQOL are as re-run in AP-BWE's
# Table V. LSD on 2048/512 (AP-BWE's basis; FLowHigh's paper uses hop 480); ViSQOL v3 audio mode, 48 kHz.
PUBLISHED = {"flowhigh": ("0.75", "3.61", "FLowHigh, Tab. I"), "apbwe": ("0.78", "3.46", "AP-BWE, Tab. V"),
             "nuwave2": ("0.94", "2.75", "AP-BWE, Tab. V (re-run)")}
rows = [[REL_LABEL[m], pub[0], f(rel(m, "wide")["lsd"], 3), pub[1], f(rel(m, "wide")["visqol_audio"], 2), pub[2]]
        for m, pub in PUBLISHED.items()]
write("fidelity", ["model", "LSD reported", "LSD ours", "ViSQOL reported", "ViSQOL ours", "reported in"], rows, "lrrrrl")

# ---- the paper's printed numbers, checked --------------------------------------------------------------
wide = lambda a: arm(a, "wide"); core = lambda a: arm(a, "core")
def indep_gap(g):
    return g["gap_indep_hb"]
fh1 = rel("flowhigh_std1", "core"); nw = rel("nuwave2", "core"); asr = rel("audiosr", "core")
nw_c = rel("nuwave2", "wide")["coh"]
max_c = max(get(k, t, "wide")["coh"] for k, t in ROWS if k != "apbwe_sinc" and get(k, t, "wide"))
CHECKS = [
    # (where, printed, value, decimals, source)
    ("Setup: FLowHigh LSD", "0.761", rel("flowhigh", "wide")["lsd"], 3, "sota.models.flowhigh.sets.wide.lsd"),
    ("Setup: FLowHigh ViSQOL", "3.61", rel("flowhigh", "wide")["visqol_audio"], 2, "...wide.visqol_audio"),
    ("Setup: AP-BWE LSD", "0.780", rel("apbwe", "wide")["lsd"], 3, "sota.models.apbwe.sets.wide.lsd"),
    ("Setup: AP-BWE ViSQOL", "3.48", rel("apbwe", "wide")["visqol_audio"], 2, "...wide.visqol_audio"),
    ("Setup: NU-Wave 2 LSD", "0.989", rel("nuwave2", "wide")["lsd"], 3, "sota.models.nuwave2.sets.wide.lsd"),
    ("Setup: NU-Wave 2 ViSQOL", "2.57", rel("nuwave2", "wide")["visqol_audio"], 2, "...wide.visqol_audio"),
    ("Level: SNR ceiling, 12 kHz", "22.02", S["ceiling"]["snr_12k"], 2, "sota.ceiling.snr_12k"),
    ("Level: NU-Wave 2 bound from its own table (C-S = 0.5 dB, + 2c)", "-8.5",
     10 * math.log10(10 ** (0.5 / 10) - 1 + 2 * nw_c), 1, "derived; c = sota.models.nuwave2.sets.wide.coh"),
    ("Level: NU-Wave 2 broadband deficit", "-8.9", rel("nuwave2", "wide")["deficit_bb"], 1, "...nuwave2.sets.wide.deficit_bb"),
    ("Level: largest coherent fraction, any model (< 0.02)", "0.02", math.ceil(max_c * 100) / 100, 2, "max of sets.wide.coh"),
    ("Level: calibrated gap at M = 8", "2.50", 10 * math.log10(2 / (1 + 1 / 8)), 2, "10 log10(2/(1+1/M))"),
    ("T1 FLowHigh deficit", "-1.60", rel("flowhigh", "wide")["deficit"], 2, "sota.models.flowhigh.sets.wide.deficit"),
    ("T1 FLowHigh gap (prior restored)", "0.08", fh1["gap_hb"], 2, "...flowhigh_std1.sets.core.gap_hb"),
    ("T1 FLowHigh gap if independent", "1.77", indep_gap(fh1), 2, "...flowhigh_std1.sets.core.gap_indep_hb"),
    ("T1 FLowHigh CRPS (prior restored)", "0.682", fh1["crps_fair"], 3, "...flowhigh_std1.sets.core.crps_fair"),
    ("T1 AP-BWE deficit", "-1.38", rel("apbwe", "wide")["deficit"], 2, "sota.models.apbwe.sets.wide.deficit"),
    ("T1 AP-BWE CRPS (= MAE)", "0.777", rel("apbwe", "core")["crps"], 3, "...apbwe.sets.core.crps"),
    ("T1 NU-Wave 2 deficit", "-7.96", rel("nuwave2", "wide")["deficit"], 2, "sota.models.nuwave2.sets.wide.deficit"),
    ("T1 NU-Wave 2 gap", "0.57", nw["gap_hb"], 2, "...nuwave2.sets.core.gap_hb"),
    ("T1 NU-Wave 2 gap if independent", "0.63", indep_gap(nw), 2, "...nuwave2.sets.core.gap_indep_hb"),
    ("T1 NU-Wave 2 CRPS", "0.583", nw["crps_fair"], 3, "...nuwave2.sets.core.crps_fair"),
    ("T1 AudioSR deficit", "-0.60", rel("audiosr", "wide")["deficit"], 2, "sota.models.audiosr.sets.wide.deficit"),
    ("T1 AudioSR LSD", "1.561", rel("audiosr", "wide")["lsd"], 3, "...wide.lsd"),
    ("T1 AudioSR ViSQOL", "2.41", rel("audiosr", "wide")["visqol_audio"], 2, "...wide.visqol_audio"),
    ("T1 AudioSR gap", "2.62", asr["gap_hb"], 2, "...audiosr.sets.core.gap_hb"),
    ("T1 AudioSR gap if independent", "3.92", indep_gap(asr), 2, "...audiosr.sets.core.gap_indep_hb"),
    ("T1 AudioSR CRPS", "1.243", asr["crps_fair"], 3, "...audiosr.sets.core.crps_fair"),
    ("T1 ours point: deficit", "-6.18", wide("det")["deficit"], 2, "sota.ours.det.sets.wide.deficit"),
    ("T1 ours point: LSD", "0.864", wide("det")["lsd"], 3, "...wide.lsd"),
    ("T1 ours point: ViSQOL", "3.13", wide("det")["visqol_audio"], 2, "...wide.visqol_audio"),
    ("T1 ours point: CRPS (= MAE)", "0.889", core("det")["crps"], 3, "...core.crps"),
    ("T1 ours sampler: deficit", "-0.70", wide("es_dec_erb_l0.1")["deficit"], 2, "sota.ours.es_dec_erb_l0.1.sets.wide.deficit"),
    ("T1 ours sampler: LSD", "0.895", wide("es_dec_erb_l0.1")["lsd"], 3, "...wide.lsd"),
    ("T1 ours sampler: ViSQOL", "2.83", wide("es_dec_erb_l0.1")["visqol_audio"], 2, "...wide.visqol_audio"),
    ("T1 ours sampler: gap", "2.07", core("es_dec_erb_l0.1")["gap_hb"], 2, "...core.gap_hb"),
    ("T1 ours sampler: gap if independent", "2.37", indep_gap(core("es_dec_erb_l0.1")), 2, "...core.gap_indep_hb"),
    ("T1 ours sampler: CRPS", "0.514", core("es_dec_erb_l0.1")["crps_fair"], 3, "...core.crps_fair"),
    ("Audit: FLowHigh truth below all draws (%)", "41", 100 * fh1["pit_lo"], 0, "...flowhigh_std1.sets.core.pit_lo"),
    ("Audit: FLowHigh truth above all draws (%)", "46", 100 * fh1["pit_hi"], 0, "...flowhigh_std1.sets.core.pit_hi"),
    ("Audit: NU-Wave 2 loud frames", "-9.0", rel("nuwave2", "wide")["loud"], 1, "...nuwave2.sets.wide.loud"),
    ("Audit: NU-Wave 2 quiet frames", "4.7", rel("nuwave2", "wide")["quiet"], 1, "...nuwave2.sets.wide.quiet"),
    ("Audit: NU-Wave 2 spread", "0.094", nw["spread_hb"], 3, "...nuwave2.sets.core.spread_hb"),
    ("Audit: AudioSR truth above all draws (%)", "52", 100 * asr["pit_hi"], 0, "...audiosr.sets.core.pit_hi"),
    ("Audit: slope FLowHigh", "-0.07", S["models"]["flowhigh"]["slope"]["slope"], 2, "sota.models.flowhigh.slope"),
    ("Audit: slope AP-BWE", "-0.12", S["models"]["apbwe"]["slope"]["slope"], 2, "sota.models.apbwe.slope"),
    ("Audit: slope NU-Wave 2", "-0.32", S["models"]["nuwave2"]["slope"]["slope"], 2, "sota.models.nuwave2.slope"),
    ("Audit: slope AudioSR", "-0.34", S["models"]["audiosr"]["slope"]["slope"], 2, "sota.models.audiosr.slope"),
    ("Audit: shallowest slope of our arms", "-0.20", max(S["ours"][a]["slope"]["slope"] for a in ARMS), 2, "sota.ours.*.slope"),
    ("Audit: steepest slope of our arms", "-0.35", min(S["ours"][a]["slope"]["slope"] for a in ARMS), 2, "sota.ours.*.slope"),
    ("T2 det_paper deficit", "-21.78", wide("det_paper")["deficit"], 2, "sota.ours.det_paper.sets.wide.deficit"),
    ("T2 det_paper CRPS (= MAE)", "2.853", core("det_paper")["crps"], 3, "...core.crps"),
    ("T2 es_marg deficit", "-1.05", wide("es_marg")["deficit"], 2, "sota.ours.es_marg.sets.wide.deficit"),
    ("T2 es_marg gap", "2.21", core("es_marg")["gap_hb"], 2, "...core.gap_hb"),
    ("T2 es_marg CRPS", "0.537", core("es_marg")["crps_fair"], 3, "...core.crps_fair"),
    ("T2 es_erb deficit", "-1.08", wide("es_erb_l0.01")["deficit"], 2, "sota.ours.es_erb_l0.01.sets.wide.deficit"),
    ("T2 es_erb gap", "2.03", core("es_erb_l0.01")["gap_hb"], 2, "...core.gap_hb"),
    ("T2 es_erb CRPS", "0.512", core("es_erb_l0.01")["crps_fair"], 3, "...core.crps_fair"),
    ("T2 es_erb lambda=0.001 deficit", "-5.73", wide("es_erb_l0.001")["deficit"], 2, "sota.ours.es_erb_l0.001.sets.wide.deficit"),
    ("T2 es_erb lambda=0.001 gap", "1.16", core("es_erb_l0.001")["gap_hb"], 2, "...core.gap_hb"),
    ("T2 es_erb lambda=0.001 CRPS", "0.683", core("es_erb_l0.001")["crps_fair"], 3, "...core.crps_fair"),
    ("Deficit: det_paper shortfall", "21.8", -wide("det_paper")["deficit"], 1, "sota.ours.det_paper.sets.wide.deficit"),
    ("Deficit: log-magnitude term recovers", "15.6", wide("det")["deficit"] - wide("det_paper")["deficit"], 1, "difference of wide deficits"),
    ("Deficit: energy score recovers", "5.1", wide("es_marg")["deficit"] - wide("det")["deficit"], 1, "difference of wide deficits"),
    ("Deficit: band-correlation error before", "0.374", core("es_marg")["corr_err"], 3, "sota.ours.es_marg.sets.core.corr_err"),
    ("Deficit: band-correlation error after", "0.289", core("es_erb_l0.01")["corr_err"], 3, "sota.ours.es_erb_l0.01.sets.core.corr_err"),
    ("Deficit: sampler truth above all draws (%)", "21", 100 * core("es_dec_erb_l0.1")["pit_hi"], 0, "...es_dec_erb_l0.1.sets.core.pit_hi"),
    ("Deficit: ideal tail at M = 8 (%)", "11", 100 / 9, 0, "1/(M+1)"),
    ("Deficit: sampler lower tail (%, at ideal)", "11", 100 * core("es_dec_erb_l0.1")["pit_lo"], 0, "...es_dec_erb_l0.1.sets.core.pit_lo"),
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
lines += ["", f"{n_ok} of {len(CHECKS)} match."]
(HERE / "number_check.md").write_text("\n".join(lines) + "\n")
print(f"tables -> {OUT}\nnumber check: {n_ok}/{len(CHECKS)} match -> {HERE / 'number_check.md'}")
