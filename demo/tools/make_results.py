"""Fold the evaluation JSONs into the one file the web app reads: web/public/assets/results.json.

Sources (each optional except the first):
  results_<tag>.json    fast/run_e4_eval.py       CRPS, PIT, coherence, deficit, readouts, floor / ceiling
  visqol_<tag>.json     fast/run_e5_visqol.py     ViSQOL speech / audio, PESQ, per condition
  latency_<tag>.json    fast/bench_latency.py     one-pass and shipped wall time per arm, on the bench machine
  gated_<tag>.json      notes/analysis (audit)    deficit split by frame loudness, tau=0 deficit for the samplers

    venv/bin/python demo/tools/make_results.py --dir overnight3/results_OV50 --tag OV50 --out web/public/assets/results.json

Nothing is typed in here. If a number is missing from its source it is null and the page shows a dash.
"""
import argparse, json, pathlib

# The eight arms of the OV50 run, in display order. det_paper is LISA as published (plain L1, tau = 0).
ARMS = ["det_paper", "det", "es_marg", "es_dec_l0.01", "es_erb_l0.001", "es_erb_l0.01", "es_erb_l0.1", "es_dec_erb_l0.1"]
KIND = {"det": "L1 + λ·STFT", "es_marg": "energy score, log-mag", "es_split_marg": "energy score, split",
        "es_marg_erb": "energy score, +ERB"}


def kind_label(kind, lam):
    # a deterministic arm with no spectral term is the paper's model, and the page should say so
    if kind == "det" and not lam:
        return "plain L1 (the paper)"
    return KIND.get(kind, kind)


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


def pit_end(e):
    """Mass in the two end bins of the PIT histogram. e4 writes it; older runs only wrote the histogram."""
    v = num(e.get("pit_end"))
    if v is not None:
        return v
    h = e.get("pit_hist")
    if isinstance(h, list) and len(h) >= 2 and sum(h) > 0:
        return (h[0] + h[-1]) / sum(h)
    return None


def snr_gap(e):
    """SNR of the 16-draw mean over the single draw, both with passthrough. e4 writes it; derive if not."""
    v = num(e.get("snr_gap"))
    if v is not None:
        return v
    a, b = num(g(e, "mean_pt", "snr")), num(g(e, "single_pt", "snr"))
    return None if a is None or b is None else a - b


def visqol_row(vq, cond):
    agg = g(vq, "agg") or {}
    return agg.get(cond) or {}


# e5 names the deterministic arm's condition plainly, and each readout gets its own row.
def conds(arm, det):
    base = arm if det else f"{arm} tau=1"
    out = {"draw": base, "draw_pt": base + " | passthrough"}
    if not det:
        for r in ("mean16", "logmean16"):
            out[r] = f"{arm} {r}"
            out[r + "_pt"] = f"{arm} {r} | passthrough"
    return out


def perceptual(vq, arm, det):
    """Every readout of one arm, raw and with the baseband passed through."""
    out = {}
    for key, cond in conds(arm, det).items():
        r = visqol_row(vq, cond)
        if not r:
            continue
        out[key] = {"snr": num(r.get("snr")), "lsd": num(r.get("lsd")), "hb_lsd": num(r.get("hb_lsd")),
                    "visqol_audio": num(r.get("visqol_audio48k")), "nsim_audio": num(r.get("nsim_audio48k")),
                    "visqol_speech": num(r.get("visqol_speech16k")), "nsim_speech": num(r.get("nsim_speech16k")),
                    "pesq": num(r.get("pesq_wb"))}
    return out


def latency_env(lt):
    """The bench machine, so the page can say where its milliseconds came from."""
    if not lt:
        return None
    look, fs = num(lt.get("lookahead_samples_12k")), num(lt.get("fs_lo"))
    return {"cpu": g(lt, "machine", "cpu"), "torch": lt.get("torch"), "threads": lt.get("threads"),
            "utt_seconds": num(g(lt, "utt", "seconds")), "passthrough_ms": num(lt.get("passthrough_ms")),
            "lookahead_ms": None if look is None or not fs else 1000.0 * look / fs,
            # bench_latency.py runs batch 1, fp32, eager PyTorch (its docstring) and does not write that
            # into its JSON; this is the description of the measurement, not a measurement.
            "precision": "fp32 eager, batch 1"}


def latency_arm(lt, arm, det):
    """One arm's wall time: one pass per second of audio, one 20 ms frame, and the shipped pipeline."""
    if not lt:
        return None
    L = g(lt, "latency", arm) or {}
    if not L:
        return None
    one, frame, utt = L.get("cpu/1s") or {}, L.get("cpu/20ms") or {}, L.get(f"cpu/{g(lt, 'utt', 'key')}") or {}
    ship1, ship16 = L.get("cpu/shipped_one_pt") or {}, L.get("cpu/shipped_logmean16_pt") or {}
    return {"params": g(lt, "info", arm, "params"),
            "one_pass_ms_per_s": num(one.get("median_ms")), "rtf_one_pass": num(one.get("rtf")),
            "frame_20ms_ms": num(frame.get("median_ms")),
            "utt_ms": num(utt.get("median_ms")),
            "shipped_one_ms": num(ship1.get("median_ms")),
            "shipped_logmean16_ms": None if det else num(ship16.get("median_ms")),
            "rtf_logmean16": None if det else num(ship16.get("rtf"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="overnight3/results_OV50")
    ap.add_argument("--tag", default="OV50")
    ap.add_argument("--out", default="web/public/assets/results.json")
    # EVAL12 is the first twelve utterances of the sorted held-out set, and those are all p236, so every
    # EVAL12 number on the page is one speaker. Comma-separate if a later run's slice spans more.
    ap.add_argument("--eval-speakers", default="p236")
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    res = load(d, f"results_{a.tag}.json")
    if res is None:
        raise SystemExit(f"no results_{a.tag}.json in {d}")
    vq = load(d, f"visqol_{a.tag}.json")
    gated = load(d, f"gated_{a.tag}.json")
    lt = load(d, f"latency_{a.tag}.json")

    out = {"tag": a.tag, "n_utts": res.get("_n"), "M": res.get("_M"),
           "eval_speakers": [s.strip() for s in a.eval_speakers.split(",") if s.strip()], "arms": {}}
    for arm in ARMS:
        r = res.get(arm)
        if not r:
            continue
        kind, lam, cls = r["arm"]
        e = r["eval12"]
        det = kind.startswith("det")
        s, s_pt, m = e.get("single", {}), e.get("single_pt", {}), e.get("mean", {})
        perc = perceptual(vq, arm, det)
        v_raw, v_pt = perc.get("draw", {}), perc.get("draw_pt", {})
        # the readout that wins the perceptual judges is the one the page should lead with
        best = max((k for k in perc if k.endswith("_pt") and perc[k].get("visqol_audio") is not None),
                   key=lambda k: perc[k]["visqol_audio"], default=None)
        gt = (gated or {}).get(arm, {})
        gm = gt.get("mean", {}) if isinstance(gt, dict) else {}
        out["arms"][arm] = {
            "cls": cls, "kind": kind_label(kind, lam), "lam": lam, "det": det,
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
            "pit_end": None if det else pit_end(e),
            "snr_gap": None if det else snr_gap(e),
            "snr_gap_calibrated": None if det else num(e.get("snr_gap_calibrated")),
            "visqol_speech": v_pt.get("visqol_speech"), "visqol_speech_raw": v_raw.get("visqol_speech"),
            "visqol_audio": v_pt.get("visqol_audio"), "visqol_audio_raw": v_raw.get("visqol_audio"),
            "nsim_audio": v_pt.get("nsim_audio"),
            "pesq": v_pt.get("pesq"),
            "readouts": perc, "best_readout": best,
            "gated": ({"loud": num(gm.get("loud")), "mid": num(gm.get("mid")), "quiet": num(gm.get("quiet"))}
                      if gm and gm.get("loud") is not None else None),
            "latency": latency_arm(lt, arm, det),
        }
    fl, ce = res.get("_floor") or {}, res.get("_ceiling") or {}
    nv, flv, cev = visqol_row(vq, "naive"), visqol_row(vq, "floor: passthrough + empty HB"), visqol_row(vq, "ceiling: passthrough + true HB")
    out["naive"] = {"snr": num(nv.get("snr")) or num(fl.get("snr")), "lsd": num(nv.get("lsd")) or num(fl.get("lsd")),
                    "visqol_audio": num(nv.get("visqol_audio48k")), "visqol_speech": num(nv.get("visqol_speech16k"))}
    out["floor"] = {"snr": num(fl.get("snr")), "lsd": num(fl.get("lsd")),
                    "visqol_audio": num(flv.get("visqol_audio48k")), "nsim_audio": num(flv.get("nsim_audio48k"))}
    out["ceiling"] = {"snr": num(ce.get("snr")), "lsd": num(ce.get("lsd")),
                      "visqol_audio": num(cev.get("visqol_audio48k")), "nsim_audio": num(cev.get("nsim_audio48k"))}
    # audio-mode ViSQOL is bounded below by an empty high band and above by the true one; report the share
    # of that range each arm's best readout captures, which is the only scale on which the judge is readable.
    lo, hi = out["floor"]["visqol_audio"], out["ceiling"]["visqol_audio"]
    if lo is not None and hi is not None and hi > lo:
        out["visqol_audio_range"] = {"floor": lo, "ceiling": hi}
        for k, v in out["arms"].items():
            va = v.get("visqol_audio")
            b = v.get("best_readout")
            vb = v["readouts"][b]["visqol_audio"] if b else None
            v["visqol_audio_share"] = None if va is None else (va - lo) / (hi - lo)
            v["visqol_audio_best"] = vb
            v["visqol_audio_best_share"] = None if vb is None else (vb - lo) / (hi - lo)
    out["latency_env"] = latency_env(lt)
    out["sources"] = {"results": True, "visqol": vq is not None, "gated": gated is not None, "latency": lt is not None}
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    have = [k for k in ("visqol", "gated", "latency") if out["sources"][k]]
    print(f"wrote {p} : {len(out['arms'])} arms; extra sources: {', '.join(have) or 'none yet'}")
    for k, v in out["arms"].items():
        b = v.get("best_readout") or "draw_pt"
        r = v["readouts"].get(b, {})
        lat = v.get("latency") or {}
        ms = lat.get("one_pass_ms_per_s")
        print(f"  {k:<17} deficit {v['deficit_draw']:+6.2f}  CRPS {v['crps']:.4f}  "
              f"LSD {r.get('lsd') or float('nan'):.3f}  ViSQOL-audio {r.get('visqol_audio') or float('nan'):.3f}"
              f"  ({b.replace('_pt','')} + passthrough, {100*(v.get('visqol_audio_best_share') or 0):.0f} % of range)"
              + (f"  one pass {ms:.1f} ms/s" if ms is not None else ""))


if __name__ == "__main__":
    main()
