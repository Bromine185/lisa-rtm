"""Unrounded level numbers for the camera-ready, read from the stored score files (no inference).

    venv/bin/python sota/level_checks.py

1. Per-utterance level regression on `wide` (240 utterances, paper split): slope of each utterance's
   band-averaged deficit (dB) on the truth's HB energy share (10 log10 of energy above 6 kHz over total),
   by np.polyfit, as demo/tools/make_results_sota.py does, but unrounded. Also a 95% utterance bootstrap.
2. det_paper against the empty band on `wide`: SNR, the ceiling (true low band, empty high band), deficit.
Writes <work>/wide/level_checks.json.
"""
import json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from score import WORK                                                              # noqa: E402

B, SEED = 10000, 0


def hb_share(ws, utt):
    y = np.load(ws / "truth" / f"{utt}.npy"); Y = np.abs(np.fft.rfft(y)) ** 2; f = np.fft.rfftfreq(len(y), 1 / 48000)
    return 10 * np.log10(Y[f >= 6000].sum() / Y.sum())


def main():
    ws = WORK / "wide"; S = json.loads((ws / "score_wide.json").read_text())
    share = {r["utt"]: hb_share(ws, r["utt"]) for r in S["per_utt"]["ceiling"]}
    rng = np.random.default_rng(SEED); out = {"_meta": {"set": "wide", "n_utts": S["_meta"]["n_utts"], "B": B,
                                                        "seed": SEED, "doc": __doc__.strip()}, "slope": {}}
    idx = None
    for cond, rows in S["per_utt"].items():
        if cond in ("ceiling", "naive") or " | " in cond:
            continue
        x = np.array([share[r["utt"]] for r in rows]); y = np.array([r["deficit"] for r in rows])
        b = float(np.polyfit(x, y, 1)[0])
        if idx is None:
            idx = rng.integers(0, len(x), (B, len(x)))
        bs = [np.polyfit(x[i], y[i], 1)[0] for i in idx]
        out["slope"][cond] = {"slope": b, "ci": [float(v) for v in np.percentile(bs, [2.5, 97.5])],
                              "r": float(np.corrcoef(x, y)[0, 1]), "n": len(x)}
        print(f"{cond:26s} slope {b:+.4f}  [{out['slope'][cond]['ci'][0]:+.3f}, {out['slope'][cond]['ci'][1]:+.3f}]  n {len(x)}")
    a = S["agg"]; dp = a["ours_det_paper"]; ce = a["ceiling"]
    C = json.loads((ws / "ceiling_wide.json").read_text())["by_rate"]["12000"]["empty"]
    out["det_paper"] = {"snr": dp["snr"], "snr_se": dp["snr_se"], "ceiling_snr_score": ce["snr"],
                        "ceiling_snr_12k": C if isinstance(C, float) else C, "vs_ceiling": dp["vs_ceiling"],
                        "deficit": dp["deficit"], "deficit_bb": dp["deficit_bb"]}
    print(json.dumps(out["det_paper"], indent=1))
    (ws / "level_checks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
