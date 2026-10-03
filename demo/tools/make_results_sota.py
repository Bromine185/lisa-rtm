"""Add the SOTA harness's numbers to web/public/assets/results.json under one key, `sota`.

    venv/bin/python demo/tools/make_results_sota.py [--results web/public/assets/results.json]

Sources, every one written by code and none typed in:
    sota/work/wide/score_wide.json        240 utterances, 8 paper-split speakers, 1 draw:  levels, LSD, SNR, ViSQOL
    sota/work/core/score_core.json        32 utterances, 8 draws: calibration, CRPS
    sota/work/ourtest/score_ourtest.json  p236-p238, 30 utterances, 16 draws
    sota/work/core/out/<m>/_run.json      RTF per released model (sota/run_models.py, the scored core run)
    sota/params.json                      parameters per model and per block (sota/count_params.py)
    demo/tools/sota_models.json           architecture facts (papers + code; no measured numbers)
    per-utterance slope of deficit vs the truth's high-band share on `wide`, computed here from the per_utt rows

Shape (read by web/lib/types.ts `SotaResults`):
    sota.sets[set] = {n_utts, seconds, speakers, M, input, note}   M = the most draws any model has on the set
    sota.ceiling   = {snr_12k, ...}  the empty-band SNR ceiling on `wide`
    sota.models[m] = {name, family, params, det, rtf_m4, rtf_device, machine, arch: {...sota_models.json entry,
                      blocks[].params from sota/params.json...}, sets: {wide, core, ourtest}, slope: {slope, r, n, set},
                      extra: {cond: {note, sets}}}   extra = a variant of the same model scored beside it
    sota.ours[arm] = the same `sets` and `slope` for every OV50 arm on the SAME utterances
    sota.avg       = null (the mean-of-M table has no score file of its own yet; nothing is typed in here)
Per-set metric keys: lsd, lsd_hf, lsd_lf, deficit, deficit_bb, loud, mid, quiet, snr, vs_ceiling, coh, kappa,
visqol_audio, nsim_audio, visqol_speech, pesq, crps, crps_fair, sliced_crps, corr_err, gap, gap_hb, gap_indep_hb,
spread, spread_hb, spread_indep_hb, pit_lo, pit_hi, M. Missing -> null, and the page shows a dash.
gap_indep_hb / spread_indep_hb: independent draws at the model's own measured energies (sota/score.py, or
sota/gap_indep.py for a run scored before those fields existed).
"""
import argparse, json, pathlib, platform, subprocess

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
WORK = pathlib.Path("/Users/raghavsharma/projects/lisa-rtm/lisa_rtm_cache/sota/work")
SETS = {"wide": "240 utterances of the paper-split speakers, one draw",
        "core": "32 utterances of the paper-split speakers, 8 draws",
        "ourtest": "p236-p238, 30 utterances, 16 draws"}
RELEASED = ["flowhigh", "apbwe", "nuwave2", "audiosr"]
# variants scored beside a released model, and what each one changes (prose, no numbers)
EXTRA = {"flowhigh": {"flowhigh_std1": "the same checkpoint with the trained prior restored (std_1 = std_2 = 1 passed)"},
         "apbwe": {"apbwe_sinc": "the same checkpoint on its own paper's input filter (torchaudio sinc), not the Chebyshev one"}}
SLOPE_SET = "wide"   # every model has all of `wide`, so every slope shares one pool of utterances
OURS = ["det_paper", "det", "es_marg", "es_dec_l0.01", "es_erb_l0.001", "es_erb_l0.01", "es_erb_l0.1", "es_dec_erb_l0.1"]
KEYS = ["lsd", "lsd_hf", "lsd_lf", "deficit", "deficit_bb", "loud", "mid", "quiet", "snr", "vs_ceiling", "coh", "kappa",
        "visqol_audio", "nsim_audio", "visqol_speech", "pesq_wb", "crps", "crps_fair", "sliced_crps", "corr_err_d0",
        "gap", "gap_hb", "gap_indep_hb", "spread", "spread_pooled_hb", "spread_indep_pooled_hb", "pit_lo", "pit_hi", "M"]
RENAME = {"pesq_wb": "pesq", "corr_err_d0": "corr_err", "spread_pooled_hb": "spread_hb", "spread_indep_pooled_hb": "spread_indep_hb"}


def num(v):
    return float(v) if isinstance(v, (int, float)) and v == v else None


def row(agg, cond):
    g = agg.get(cond)
    if not g:
        return None
    return {RENAME.get(k, k): num(g.get(k)) for k in KEYS}


def run_meta(m):
    """RTF, device and machine. First choice: sota/rtf.json (sota/bench_rtf.py: the model alone on an idle
    machine, warm, median of repeats). Fallback: the scored core run's _run.json (sota/run_models.py), whose
    wall clock includes file I/O and may have shared the machine; a _run.json with no audio in it (a resumed
    run that found every draw on disk) carries no timing: None."""
    b = pathlib.Path(__file__).resolve().parents[2] / "sota" / "rtf.json"
    if b.exists():
        r = json.loads(b.read_text()).get(m)
        if r and r.get("rtf"):
            return round(float(r["rtf"]), 4), r.get("device"), r.get("machine") or this_machine()
    q = WORK / "core" / "out" / m / "_run.json"
    if not q.exists():
        return None, None, None
    r = json.loads(q.read_text())
    ok = (r.get("audio_s") or 0) > 0 and (r.get("draws_run") or 0) > 0
    return (round(float(r["rtf"]), 4) if ok else None), r.get("device"), r.get("machine") or this_machine()


def this_machine():
    # _run.json files written before run_models.py recorded the machine: the cache is this Mac's, so name it
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return platform.processor() or None


def hb_share(set_, utt):
    y = np.load(WORK / set_ / "truth" / f"{utt}.npy"); Y = np.abs(np.fft.rfft(y)) ** 2; f = np.fft.rfftfreq(len(y), 1 / 48000)
    return 10 * np.log10(Y[f >= 6000].sum() / Y.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "web/public/assets/results.json"))
    a = ap.parse_args()
    rp = pathlib.Path(a.results)
    res = json.loads(rp.read_text())
    scores = {s: json.loads((WORK / s / f"score_{s}.json").read_text()) for s in SETS if (WORK / s / f"score_{s}.json").exists()}
    counts = json.loads((REPO / "sota/params.json").read_text())
    facts = json.loads((REPO / "demo/tools/sota_models.json").read_text())["models"]

    sets = {}
    for s, S in scores.items():
        m = S["_meta"]
        sets[s] = {"n_utts": m["n_utts"], "seconds": round(m["seconds"]), "speakers": m["speakers"], "M": None,
                   "input": m["input"], "note": SETS[s], "gain": m["gain"], "lsd_basis": m.get("lsd_2048")}
    # per-utterance slope of the deficit against the truth's HB share, on `wide` only: every model has
    # all of it, so every slope is fit to the same utterances (ourtest has only some models, and seen speakers)
    share = {}
    for s in (SLOPE_SET,):
        if s in scores:
            for r in scores[s]["per_utt"]["ceiling"]:
                share[(s, r["utt"])] = hb_share(s, r["utt"])

    def slope(cond):
        x, y = [], []
        for s in (SLOPE_SET,):
            if s in scores and cond in scores[s]["per_utt"]:
                for r in scores[s]["per_utt"][cond]:
                    x.append(share[(s, r["utt"])]); y.append(r["deficit"])
        if len(x) < 10:
            return None
        x, y = np.array(x), np.array(y); b, _ = np.polyfit(x, y, 1)
        return {"slope": round(float(b), 3), "r": round(float(np.corrcoef(x, y)[0, 1]), 3), "n": len(x), "set": SLOPE_SET}

    def entry(cond, name):
        e = {"sets": {s: row(scores[s]["agg"], cond) for s in scores}, "slope": slope(cond)}
        for s in scores:
            if e["sets"][s] and e["sets"][s].get("M") is not None:
                sets[s]["M"] = max(sets[s]["M"] or 0, int(e["sets"][s]["M"]))
        return e

    models = {}
    for m in RELEASED:
        f, c = facts[m], counts[m]
        arch = {k: v for k, v in f.items() if k not in ("blocks", "count_whole")}
        arch["blocks"] = [{"id": b["id"], "label": b["label"], "detail": b["detail"], "params": c["blocks"][b["id"]]} for b in f["blocks"]]
        arch["params"] = c["total"]
        rtf, dev, mach = run_meta(m)
        extra = {v: {"note": note, "sets": {s: row(scores[s]["agg"], v) for s in scores if v in scores[s]["agg"]}}
                 for v, note in EXTRA.get(m, {}).items()}
        models[m] = {"name": f["name"], "family": f["family"], "params": c["total"], "det": f["det"],
                     "rtf_m4": rtf, "rtf_device": dev, "machine": mach, "arch": arch, "extra": extra or None, **entry(m, m)}
    ours = {arm: entry(f"ours_{arm}", arm) for arm in OURS}
    ceiling = {"snr_12k": num(scores["wide"]["agg"]["ceiling"]["snr"]) if "wide" in scores else None,
               "note": "true low band, empty high band; per-utterance mean of dB on `wide`"}
    cp = WORK / "wide" / "ceiling_wide.json"
    if cp.exists():
        c = json.loads(cp.read_text())["by_rate"]
        ceiling.update({f"snr_{int(k) // 1000}k": round(v["empty"], 2) for k, v in c.items()})
    avg = None   # no score file carries the mean-of-M table; never write literals here
    res["sota"] = {"sets": sets, "ceiling": ceiling, "models": models, "ours": ours, "avg": avg,
                   "sources": {"scores": [str(WORK / s / f"score_{s}.json") for s in scores], "rtf": str(WORK / "core/out/<m>/_run.json"),
                               "params": "sota/params.json", "arch": "demo/tools/sota_models.json"}}
    rp.write_text(json.dumps(res, indent=1))
    print("wrote sota block:", list(models), "| sets", {s: (v["n_utts"], v["M"]) for s, v in sets.items()}, "->", rp)


if __name__ == "__main__":
    main()
