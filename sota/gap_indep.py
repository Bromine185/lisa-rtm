"""Add the independent-draw reference to an existing score file, without re-running the perceptual metrics.

    venv/bin/python sota/gap_indep.py --set core

For every model with an ensemble on the set, per utterance and with score.py's own gain and high-band split:
    gap_indep_hb           10 log10((E + P)/(E + P/M)), E = truth HB energy, P = mean draw HB energy; mean over
                           utterances, as gap_hb is
    spread_indep_pooled_hb sum P (1 + 1/M) / sum (E + P/M), pooled as spread_pooled_hb is
These are what draws independent of one another and of the truth would score at the model's own measured energies,
utterance by utterance. score.py computes the same two fields on a fresh run; this script writes them into
<work>/<set>/score_<set>.json for a run already scored. It recomputes gap_hb on the way and stops if it does not
match the stored value, so the reference and the gap it is compared with come from the same draws.
"""
import argparse, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import score                                                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="core")
    a = ap.parse_args()
    ws = score.WORK / a.set; path = ws / f"score_{a.set}.json"
    S = json.loads(path.read_text()); M_cap = S["_meta"].get("M_cap")
    utts = [u["utt"] for u in json.load(open(ws / "utts.json"))["utts"]][: S["_meta"]["n_utts"]]
    for m in sorted(p.name for p in (ws / "out").iterdir() if p.is_dir()):
        g = S["agg"].get(m)
        if not g or "gap_hb" not in g:
            continue
        gap, ind, SE, SP = [], [], 0.0, 0.0
        for utt in utts:
            y = np.load(ws / "truth" / f"{utt}.npy"); n = len(y)
            fs = sorted((ws / "out" / m).glob(f"{utt}_s*.npy"), key=lambda q: int(q.stem.rsplit("_s", 1)[1]))[:M_cap]
            D = np.stack([score.align(y, np.load(q).astype(np.float64)[:n])[0] for q in fs]); M = len(D)
            T = score.fft_split(y, False); X = np.stack([score.fft_split(d, False) for d in D]); XB = X.mean(0)
            gap.append(10 * np.log10(np.mean([np.sum((T - x) ** 2) for x in X]) / np.sum((T - XB) ** 2)))
            E, P = float(np.sum(T ** 2)), float(np.mean([np.sum(x ** 2) for x in X]))
            ind.append(10 * np.log10((E + P) / (E + P / M))); SE += E; SP += P
        if abs(np.mean(gap) - g["gap_hb"]) > 1e-6:
            sys.exit(f"{m}: recomputed gap_hb {np.mean(gap):.6f} != stored {g['gap_hb']:.6f}; draws changed since scoring")
        g["gap_indep_hb"] = float(np.mean(ind)); g["gap_indep_hb_se"] = float(np.std(ind, ddof=1) / np.sqrt(len(ind)))
        g["spread_indep_pooled_hb"] = SP * (1 + 1 / M) / (SE + SP / M)
        print(f"{m:24s} gap_hb {g['gap_hb']:.2f}  gap_indep_hb {g['gap_indep_hb']:.2f}  "
              f"spread_hb {g['spread_pooled_hb']:.3f}  spread_indep_hb {g['spread_indep_pooled_hb']:.3f}")
    path.write_text(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
