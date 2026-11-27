"""Every number the paper prints, from the score files, with its source key. No inference.

    python sota/paper_numbers.py vctk     levels from wide (240 utt.), calibration from core (32 x 8): the Sep 29 paper
    python sota/paper_numbers.py ears     levels and calibration from ears (96 utt. x 8)

Files are read from $BWE_CACHE/sota/work/<set>/ (score_<set>.json, bootstrap_<set>.json, level_checks*.json,
draw_agreement_<set>.json), or from --dir for the EARS files copied back from Colab. Prints name = value [source]
and writes paper_numbers_<corpus>.json next to this script's results folder.
"""
import argparse, json, os, pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
WORK = pathlib.Path(os.environ.get("BWE_CACHE", str(REPO / "lisa_rtm_cache"))) / "sota" / "work"
FH, FH1, NW, AS, AP = "flowhigh", "flowhigh_std1", "nuwave2", "audiosr", "apbwe"
DET, DP, EM, EE, EE3, ED = ("ours_det", "ours_det_paper", "ours_es_marg", "ours_es_erb_l0.01", "ours_es_erb_l0.001",
                            "ours_es_dec_erb_l0.1")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("corpus", choices=["vctk", "ears"]); ap.add_argument("--dir")
    a = ap.parse_args()
    if a.corpus == "vctk":
        lev_set, cal_set = "wide", "core"
        get = lambda s, f: json.loads((WORK / s / f).read_text())
        LEV, CAL = get("wide", "score_wide.json"), get("core", "score_core.json")
        BS, SL, DA = get("core", "bootstrap_core.json"), get("wide", "level_checks.json"), get("core", "draw_agreement_core.json")
        ceiling = json.loads((WORK / "wide" / "ceiling_wide.json").read_text())["by_rate"]["12000"]["empty"]
        ceiling_src = "wide/ceiling_wide.json by_rate.12000.empty"
    else:
        d = pathlib.Path(a.dir) if a.dir else WORK / "ears"
        get = lambda f: json.loads((d / f).read_text())
        lev_set = cal_set = "ears"
        LEV = CAL = get("score_ears.json")
        BS, SL, DA = get("bootstrap_ears.json"), get("level_checks_ears.json"), get("draw_agreement_ears.json")
        ceiling, ceiling_src = LEV["agg"]["ceiling"]["snr"], "ears agg.ceiling.snr"
    L, C = LEV["agg"], CAL["agg"]
    out = {}

    def put(name, v, src):
        out[name] = {"value": v, "source": src}

    def crps(m):
        return (C[m]["crps_fair"], f"{cal_set} agg.{m}.crps_fair") if "crps_fair" in C[m] else (C[m]["crps"], f"{cal_set} agg.{m}.crps (MAE)")

    # Table 1: released models and ours
    for row, lm, cm in (("FLowHigh", FH, FH1), ("AP-BWE", AP, AP), ("NU-Wave 2", NW, NW), ("AudioSR", AS, AS),
                        ("ours, point", DET, DET), ("ours, sampler", ED, ED)):
        put(f"T1 {row} deficit", L[lm]["deficit"], f"{lev_set} agg.{lm}.deficit")
        put(f"T1 {row} LSD", L[lm]["lsd"], f"{lev_set} agg.{lm}.lsd")
        put(f"T1 {row} ViSQOL", L[lm].get("visqol_audio"), f"{lev_set} agg.{lm}.visqol_audio")
        if "gap_hb" in C[cm]:
            put(f"T1 {row} gap", C[cm]["gap_hb"], f"{cal_set} agg.{cm}.gap_hb")
            put(f"T1 {row} gap indep", C[cm]["gap_indep_hb"], f"{cal_set} agg.{cm}.gap_indep_hb")
        v, s = crps(cm); put(f"T1 {row} CRPS", v, s)
    # Table 2: the six arms
    for row, m in (("det_paper", DP), ("det", DET), ("es_marg", EM), ("es_erb", EE), ("es_erb l0.001", EE3), ("es_dec_erb", ED)):
        put(f"T2 {row} deficit", L[m]["deficit"], f"{lev_set} agg.{m}.deficit")
        if "gap_hb" in C[m]:
            put(f"T2 {row} gap", C[m]["gap_hb"], f"{cal_set} agg.{m}.gap_hb")
        v, s = crps(m); put(f"T2 {row} CRPS", v, s)
    # text
    put("ceiling SNR (12 kHz input)", ceiling, ceiling_src)
    put("det_paper SNR below ceiling", -L[DP]["vs_ceiling"], f"{lev_set} agg.{DP}.vs_ceiling")
    put("det_paper band short", -L[DP]["deficit"], f"{lev_set} agg.{DP}.deficit")
    put("log-mag term recovers", L[DET]["deficit"] - L[DP]["deficit"], f"{lev_set} deficit {DET} - {DP}")
    put("energy score recovers more", L[EM]["deficit"] - L[DET]["deficit"], f"{lev_set} deficit {EM} - {DET}")
    put("NU-Wave 2 broadband deficit", L[NW]["deficit_bb"], f"{lev_set} agg.{NW}.deficit_bb")
    put("max coherent fraction c", max(L[m]["coh"] for m in (FH, FH1, NW, AS, AP, DET, ED)), f"{lev_set} max agg.*.coh")
    put("NU-Wave 2 loud frames", L[NW]["loud"], f"{lev_set} agg.{NW}.loud")
    put("NU-Wave 2 quiet frames", L[NW]["quiet"], f"{lev_set} agg.{NW}.quiet")
    put("NU-Wave 2 spread", C[NW]["spread_pooled_hb"], f"{cal_set} agg.{NW}.spread_pooled_hb")
    cen = LEV["_meta"]["band_centres_hz"]
    for i in (2, 5):
        put(f"AudioSR level at {cen[i]/1000:.1f} kHz", L[AS]["curve"][i], f"{lev_set} agg.{AS}.curve[{i}]")
    put("ERB corr err es_marg", C[EM]["corr_err_d0"], f"{cal_set} agg.{EM}.corr_err_d0")
    put("ERB corr err es_erb", C[EE]["corr_err_d0"], f"{cal_set} agg.{EE}.corr_err_d0")
    for m, key in ((FH1, "pit_lo"), (FH1, "pit_hi"), (AS, "pit_hi"), (ED, "pit_hi"), (ED, "pit_lo"), (NW, "pit_lo"), (NW, "pit_hi")):
        p, ci = BS[m]["point"][key], BS[m]["utterance_ci"][key]
        put(f"{m} {key} %", [100 * p, 100 * ci[0], 100 * ci[1]], f"bootstrap {m} {key} point, 95% CI")
    for m in (FH1, NW, AS, ED, EM, EE, EE3):
        p, ci = BS[m]["point"]["gap_minus_indep"], BS[m]["utterance_ci"]["gap_minus_indep"]
        put(f"{m} gap - indep", [p, ci[0], ci[1]], f"bootstrap {m} gap_minus_indep point, 95% CI")
    put("FLowHigh as released: draw spread, waveform rel. RMS", DA[FH]["agg"]["wave_rel"]["median"], f"draw_agreement {FH} wave_rel median")
    put("FLowHigh prior restored: HB rel. RMS", DA[FH1]["agg"]["hb_rel"]["median"], f"draw_agreement {FH1} hb_rel median")
    put("NU-Wave 2: HB rel. RMS", DA[NW]["agg"]["hb_rel"]["median"], f"draw_agreement {NW} hb_rel median")
    for row, m in (("FLowHigh", FH), ("AP-BWE", AP), ("NU-Wave 2", NW), ("AudioSR", AS)):
        put(f"slope {row}", SL["slope"][m]["slope"], f"level_checks slope.{m}")
    ours = [SL["slope"][m]["slope"] for m in SL["slope"] if m.startswith("ours_") and " | " not in m
            and m.replace("ours_", "") in ("det_paper", "det", "es_marg", "es_erb_l0.01", "es_erb_l0.001", "es_dec_erb_l0.1")]
    put("slope ours range", [max(ours), min(ours)], "level_checks slope.ours_<six Table 2 arms>")
    put("n levels / calibration", [LEV["_meta"]["n_utts"], CAL["_meta"]["n_utts"]], "_meta.n_utts")

    for k, v in out.items():
        val = v["value"]; txt = (f"{val:.4g}" if isinstance(val, float) and abs(val) < 1e-3 else f"{val:.4f}" if isinstance(val, float) else
                                 "[" + ", ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in val) + "]" if isinstance(val, list) else str(val))
        print(f"{k:52s} = {txt:32s} [{v['source']}]")
    dst = REPO / "supplementary" / "results" / "camera_ready" / f"paper_numbers_{a.corpus}.json"
    dst.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
