"""Utterance-level bootstrap intervals for the calibration numbers, from the saved draws (no inference).

    venv/bin/python sota/bootstrap.py --set core [--B 10000] [--seed 0]

score.py keeps per-utterance rank counts and energies in private fields it does not write. This recomputes
them, per utterance and per model, with score.py's own gain alignment, HB split, HB log-magnitude and
pit_ranks, and stops unless the aggregates match the stored score file to 1e-6:
    pit_lo, pit_hi   truth below / above all M draws, as a share of HB time-frequency bins, pooled over utterances
    gap_hb           10 log10(mean_i |y - x_i|^2 / |y - xbar|^2) on the band above 6 kHz, mean over utterances
    gap_indep_hb     10 log10((E + P)/(E + P/M)) at each utterance's measured HB energies, mean over utterances
Then it resamples the utterances (not the bins, which are correlated) B times with replacement and reports
95% percentile intervals: tails as pooled shares of the resampled utterances, gaps as their means, and the
paired difference gap_hb - gap_indep_hb. A speaker-level resample (whole speakers, with their utterances) is
written beside it as a sensitivity check. Writes <work>/<set>/bootstrap_<set>.json.
"""
import argparse, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import score                                                                        # noqa: E402

TOL = 1e-6


def per_utt(ws, m, utts):
    rows = []
    for utt in utts:
        y = np.load(ws / "truth" / f"{utt}.npy"); n = len(y)
        fs = sorted((ws / "out" / m).glob(f"{utt}_s*.npy"), key=lambda q: int(q.stem.rsplit("_s", 1)[1]))
        D = np.stack([score.align(y, np.load(q).astype(np.float64)[:n])[0] for q in fs]); M = len(D)
        T = score.fft_split(y, False); X = np.stack([score.fft_split(d, False) for d in D]); XB = X.mean(0)
        gap = 10 * np.log10(np.mean([np.sum((T - x) ** 2) for x in X]) / np.sum((T - XB) ** 2))
        E, P = float(np.sum(T ** 2)), float(np.mean([np.sum(x ** 2) for x in X]))
        truth_lm = score.lm_hb(y)
        ens = np.stack([score.lm_hb(d)[: truth_lm.shape[0]] for d in D])
        pit = np.bincount(score.G["pit_ranks"](ens, truth_lm), minlength=M + 1)
        rows.append({"utt": utt, "M": M, "gap_hb": float(gap), "gap_indep_hb": float(10 * np.log10((E + P) / (E + P / M))),
                     "pit": pit.tolist()})
    return rows


def stats(rows, idx):
    """The four statistics on one resample; idx is an (B, n) array of row indices."""
    pit = np.array([r["pit"] for r in rows], np.float64)                         # (n, M+1)
    g = np.array([r["gap_hb"] for r in rows]); gi = np.array([r["gap_indep_hb"] for r in rows])
    h = pit[idx].sum(1)                                                          # (B, M+1)
    tot = h.sum(1)
    return {"pit_lo": h[:, 0] / tot, "pit_hi": h[:, -1] / tot, "gap_hb": g[idx].mean(1),
            "gap_indep_hb": gi[idx].mean(1), "gap_minus_indep": (g - gi)[idx].mean(1)}


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5]); return [float(lo), float(hi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="core")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()
    score.init(False)
    ws = score.WORK / a.set
    S = json.loads((ws / f"score_{a.set}.json").read_text())
    utts = [u["utt"] for u in json.load(open(ws / "utts.json"))["utts"]][: S["_meta"]["n_utts"]]
    spk = np.array([u.split("_")[0] for u in utts]); speakers = sorted(set(spk))
    models = a.models or [m for m, g in S["agg"].items() if "pit_lo" in g and " | " not in m]
    rng = np.random.default_rng(a.seed); n = len(utts)
    idx_u = rng.integers(0, n, (a.B, n))
    by_spk = [np.flatnonzero(spk == s) for s in speakers]
    pick = rng.integers(0, len(speakers), (a.B, len(speakers)))
    idx_s = [np.concatenate([by_spk[j] for j in row]) for row in pick]           # ragged when speakers differ in size
    out = {"_meta": {"set": a.set, "n_utts": n, "speakers": speakers, "B": a.B, "seed": a.seed,
                     "interval": "95% percentile", "doc": __doc__.strip()}}
    for m in models:
        rows = per_utt(ws, m, utts)
        point = stats(rows, np.arange(n)[None])
        point = {k: float(v[0]) for k, v in point.items()}
        g = S["agg"][m]
        for k in ("pit_lo", "pit_hi", "gap_hb", "gap_indep_hb"):
            if k in g and abs(point[k] - g[k]) > TOL:
                sys.exit(f"{m}: recomputed {k} {point[k]:.8f} != stored {g[k]:.8f}; draws or code changed since scoring")
        bu = stats(rows, idx_u)
        bs = {k: np.array([stats(rows, i[None])[k][0] for i in idx_s]) for k in point}
        out[m] = {"point": point, "utterance_ci": {k: ci(v) for k, v in bu.items()},
                  "speaker_ci": {k: ci(v) for k, v in bs.items()}, "per_utt": rows}
        u = out[m]["utterance_ci"]
        print(f"{m:24s} lo {100*point['pit_lo']:4.1f}% [{100*u['pit_lo'][0]:.1f}, {100*u['pit_lo'][1]:.1f}]  "
              f"hi {100*point['pit_hi']:4.1f}% [{100*u['pit_hi'][0]:.1f}, {100*u['pit_hi'][1]:.1f}]  "
              f"gap {point['gap_hb']:.2f} [{u['gap_hb'][0]:.2f}, {u['gap_hb'][1]:.2f}]  "
              f"indep {point['gap_indep_hb']:.2f} [{u['gap_indep_hb'][0]:.2f}, {u['gap_indep_hb'][1]:.2f}]  "
              f"diff {point['gap_minus_indep']:+.2f} [{u['gap_minus_indep'][0]:+.2f}, {u['gap_minus_indep'][1]:+.2f}]",
              flush=True)
    (ws / f"bootstrap_{a.set}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
