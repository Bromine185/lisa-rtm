"""Score the predictions in notes/2026-11-27-ears-predictions.md against the EARS results. No tuning, no inference.

    python sota/ears_predictions.py <dir holding score_ears.json, bootstrap_ears.json, level_checks_ears.json>

Prints each prediction, the numbers it rests on, and PASS / FAIL, then writes ears_predictions.json beside the
inputs. The thresholds are copied from the note, which was committed before any model saw an EARS file.
"""
import json, pathlib, sys

D = pathlib.Path(sys.argv[1])
S = json.loads((D / "score_ears.json").read_text())["agg"]
B = json.loads((D / "bootstrap_ears.json").read_text())
L = json.loads((D / "level_checks_ears.json").read_text())["slope"]
OURS = "ours_es_dec_erb_l0.1"


def bt(m, k):
    return B[m]["point"][k], B[m]["utterance_ci"][k]


res = []


def check(n, text, ok, facts):
    res.append({"n": n, "prediction": text, "pass": bool(ok), "facts": facts})
    print(f"{n}. {'PASS' if ok else 'FAIL'}  {text}\n     {facts}")


g, gi = S["flowhigh_std1"]["gap_hb"], S["flowhigh_std1"]["gap_indep_hb"]
lo, hi = S["flowhigh_std1"]["pit_lo"], S["flowhigh_std1"]["pit_hi"]
check(1, "FLowHigh (prior restored) is still a regressor: gap < indep/2, tails sum > 50%",
      g < gi / 2 and lo + hi > 0.5, f"gap {g:.3f}, indep {gi:.3f}, tails {100*lo:.1f}% + {100*hi:.1f}%")

d, dci = bt("nuwave2", "gap_minus_indep"); df = S["nuwave2"]["deficit"]
check(2, "NU-Wave 2 samples at its own level: |diff| < 0.2 dB, deficit < -4 dB",
      abs(d) < 0.2 and df < -4, f"diff {d:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}], deficit {df:.2f}")

d, dci = bt("audiosr", "gap_minus_indep"); hi = S["audiosr"]["pit_hi"]
check(3, "AudioSR is under-dispersed: diff < -0.5 dB with interval below 0, upper tail > 30%",
      d < -0.5 and dci[1] < 0 and hi > 0.30, f"diff {d:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}], upper tail {100*hi:.1f}%")

d, dci = bt(OURS, "gap_minus_indep"); lo, hi = S[OURS]["pit_lo"], S[OURS]["pit_hi"]
check(4, "Our sampler: -0.6 < diff < 0 dB, upper tail > lower tail, lower tail in [7%, 15%]",
      -0.6 < d < 0 and hi > lo and 0.07 <= lo <= 0.15,
      f"diff {d:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}], tails {100*lo:.1f}% / {100*hi:.1f}%")

dp, dt = S["ours_det_paper"]["deficit"], S["ours_det"]["deficit"]
em, ed = S["ours_es_marg"]["deficit"], S[OURS]["deficit"]
check(5, "Loss domain sets the deficit: det_paper < -15 dB, det >= det_paper + 8, es_marg and es_dec_erb within 3 dB of 0",
      dp < -15 and dt >= dp + 8 and abs(em) <= 3 and abs(ed) <= 3,
      f"det_paper {dp:.2f}, det {dt:.2f}, es_marg {em:.2f}, es_dec_erb {ed:.2f}")

bad = {m: v for m, v in L.items() if not v["ci"][1] < 0}
check(6, "Level regresses per utterance: every slope negative, interval below 0",
      not bad, "; ".join(f"{m} {v['slope']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]" for m, v in L.items()))

g = S["flowhigh"]["gap_hb"]
check(7, "FLowHigh as released is a point estimate: gap < 0.01 dB", g < 0.01, f"gap {g:.2e}")

(D / "ears_predictions.json").write_text(json.dumps(res, indent=1))
print(f"\n{sum(r['pass'] for r in res)} of {len(res)} predictions hold")
