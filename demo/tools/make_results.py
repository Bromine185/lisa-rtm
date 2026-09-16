"""Fold the evaluation JSONs into the one file the web app reads: web/public/assets/results.json.

Sources (each optional except the first):
  results_<tag>.json    overnight3/e4_eval.py     CRPS, PIT, coherence, deficit, readouts, floor / ceiling
  visqol_<tag>.json     overnight3/e5_visqol.py   ViSQOL speech / audio, PESQ, per condition
  gated_<tag>.json      notes/analysis (audit)    deficit split by frame loudness, tau=0 deficit for the samplers

    venv/bin/python demo/tools/make_results.py --dir lisa_rtm_cache/results --tag OV3_fast --out web/public/assets/results.json

Nothing is typed in here. If a number is missing from its source it is null and the page shows a dash.
"""
import argparse, json, pathlib

ARMS = ["det", "es_marg", "es_marg_l0.1", "es_split_l0.1", "es_erb_l0.1", "es_dec_l0.1", "es_dec_erb_l0.1"]
KIND = {"det": "L1 + λ·STFT", "es_marg": "energy score, log-mag", "es_split_marg": "energy score, split",
        "es_marg_erb": "energy score, +ERB"}


def load(d, name):
    p = d / name
    return json.load(open(p)) if p.exists() else None


def g(o, *ks):
    for k in ks:
        if not isinstance(o, dict) or k not in o:
            return None
        o = o[k]
    return o


def num(v):
    return float(v) if isinstance(v, (int, float)) and v == v else None


def readout_lsd(r):
    """readouts['logmean16'] is a metrics dict in e4; tolerate a bare number."""
    if isinstance(r, dict):
        return num(r.get("lsd"))
    return num(r)


def visqol_row(vq, cond):
    agg = g(vq, "agg") or {}
    return agg.get(cond) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="lisa_rtm_cache/results")
    ap.add_argument("--tag", default="OV3_fast")
    ap.add_argument("--out", default="web/public/assets/results.json")
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    res = load(d, f"results_{a.tag}.json")
    if res is None:
        raise SystemExit(f"no results_{a.tag}.json in {d}")
    vq = load(d, f"visqol_{a.tag}.json")
    gated = load(d, f"gated_{a.tag}.json")

    out = {"tag": a.tag, "n_utts": res.get("_n"), "M": res.get("_M"), "arms": {}}
    for arm in ARMS:
        r = res.get(arm)
        if not r:
            continue
        kind, lam, cls = r["arm"]
        e = r["eval12"]
        det = kind.startswith("det")
        s, s_pt, m = e.get("single", {}), e.get("single_pt", {}), e.get("mean", {})
        cond = f"{arm} tau=0" if det else f"{arm} tau=1"
        v_raw, v_pt = visqol_row(vq, cond), visqol_row(vq, cond + " | passthrough")
        gt = (gated or {}).get(arm, {})
        gm = gt.get("mean", {}) if isinstance(gt, dict) else {}
        out["arms"][arm] = {
            "cls": cls, "kind": KIND.get(kind, kind), "lam": lam, "det": det,
            "snr_draw": num(s.get("snr")), "snr_mean16": None if det else num(m.get("snr")),
            "snr_draw_pt": num(s_pt.get("snr")),
            "lsd_draw": num(s.get("lsd")), "lsd_draw_pt": num(s_pt.get("lsd")),
            "lsd_logmean16": None if det else readout_lsd(g(e, "readouts_pt", "logmean16")),
            "hb_lsd": num(s.get("hb_lsd")),
            "deficit_draw": num(s.get("deficit")),
            "deficit_tau0": num(s.get("deficit")) if det else num(gm.get("deficit_tau0")),
            "deficit_mean16": None if det else num(m.get("deficit")),
            "crps": num(e.get("crps")), "sliced_crps": num(e.get("sliced_crps")),
            "kappa": num(s.get("hb_kappa")), "coherent": num(s.get("hb_coh")),
            "pit_end": None if det else num(e.get("pit_end")),
            "snr_gap": None if det else num(e.get("snr_gap")),
            "snr_gap_calibrated": None if det else num(e.get("snr_gap_calibrated")),
            "visqol_speech": num(v_pt.get("visqol_speech16k")), "visqol_speech_raw": num(v_raw.get("visqol_speech16k")),
            "visqol_audio": num(v_pt.get("visqol_audio48k")), "visqol_audio_raw": num(v_raw.get("visqol_audio48k")),
            "nsim_audio": num(v_pt.get("nsim_audio48k")),
            "pesq": num(v_pt.get("pesq_wb")),
            "gated": ({"loud": num(gm.get("loud")), "mid": num(gm.get("mid")), "quiet": num(gm.get("quiet"))}
                      if gm and gm.get("loud") is not None else None),
        }
    fl, ce = res.get("_floor") or {}, res.get("_ceiling") or {}
    nv = visqol_row(vq, "naive")
    out["naive"] = {"snr": num(fl.get("snr")), "lsd": num(fl.get("lsd")),
                    "visqol_audio": num(nv.get("visqol_audio48k")), "visqol_speech": num(nv.get("visqol_speech16k"))}
    out["floor"] = {"snr": num(fl.get("snr")), "lsd": num(fl.get("lsd"))}
    out["ceiling"] = {"snr": num(ce.get("snr")), "lsd": num(ce.get("lsd"))}
    out["sources"] = {"results": True, "visqol": vq is not None, "gated": gated is not None}
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    have = [k for k in ("visqol", "gated") if out["sources"][k]]
    print(f"wrote {p} : {len(out['arms'])} arms; extra sources: {', '.join(have) or 'none yet'}")
    for k, v in out["arms"].items():
        print(f"  {k:<17} deficit {v['deficit_draw']:+6.2f}  crps {v['crps']:.3f}  lsd {v['lsd_draw']:.3f}"
              f"  logmean16 {v['lsd_logmean16'] if v['lsd_logmean16'] is None else round(v['lsd_logmean16'], 3)}"
              f"  visqol_audio {v['visqol_audio']}")


if __name__ == "__main__":
    main()
