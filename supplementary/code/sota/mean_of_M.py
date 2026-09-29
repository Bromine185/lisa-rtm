"""Averaging draws rebuilds the deficit: the mean-of-M table, from the draws sota/run_ours.py saved.

    python sota/mean_of_M.py --set core --model ours_es_dec_erb_l0.1 [--Ms 1 2 4 8 16]

Added for the supplementary material: the paper's "Averaging rebuilds the deficit" paragraph had no committed
score file (demo/tools/make_results_sota.py writes `sota.avg = null` for exactly this reason). This script is
that score file. It reuses sota/score.py's own pieces -- init (the notebook's metric code), align (gain on
the band below 5.5 kHz), one (deficit, LSD) and the notebook's logmag_ensemble_readout -- so every column
means what the column of the same name means in score_<set>.json.

For each M it scores three waveforms per utterance, built from the first M draws:
  draw      draw 0 alone (M-independent; the reference row)
  wavemean  the sample mean of the M waveforms. For draws whose high bands are mutually incoherent,
            E|mean|^2 = E|draw|^2 / M in that band, so the deficit falls by about 10 log10 M dB.
            A regressor trained on squared error outputs (an estimate of) exactly this quantity at M -> inf.
  logmean   the per-bin mean of log|STFT| over the M draws, phase of draw 0 (the notebook's
            logmag_ensemble_readout, passthrough=False), then the low band of the naive upsample of this
            set's own input -- exactly score.py's "logmean<M> | pt". The Bayes readout for LSD.
and reports, averaged over utterances: deficit (mean over third octaves, dB), deficit_bb, LSD (2048/512,
NU-Wave 2 basis), LSD-HF, and the expected deficit of `wavemean` for incoherent draws, draw deficit - 10 log10 M.

Writes <work>/<set>/mean_of_M_<model>.json and prints a table.
"""
import os
import argparse, importlib.util, json, pathlib, sys
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("score", REPO / "sota" / "score.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="core")
    ap.add_argument("--model", default="ours_es_dec_erb_l0.1")
    ap.add_argument("--Ms", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--n", type=int, default=None, help="first n utterances only")
    a = ap.parse_args()
    ws = S.WORK / a.set
    utts = json.load(open(ws / "utts.json"))["utts"][: a.n]
    S.init(False)
    G, CFG = S.G, S.CFG
    rows = {}
    for u in utts:
        utt = u["utt"]; y = np.load(ws / "truth" / f"{utt}.npy"); n = len(y)
        fs = sorted((ws / "out" / a.model).glob(f"{utt}_s*.npy"), key=lambda q: int(q.stem.rsplit("_s", 1)[1]))
        D = np.stack([S.align(y, np.load(q).astype(np.float64)[:n])[0] for q in fs])
        Sy = G["stft"](y, CFG.eval_n_fft, CFG.eval_hop)[0]
        # the passthrough low band comes from the input THIS set gave the models, as in score.py's `| pt`
        import soundfile as sf
        x_lo, _ = sf.read(str(ws / "lo12k" / f"{utt}.wav"), dtype="float64")
        nv_lo = S.fft_split(S.up_naive(x_lo, 4, n), True)
        for M in a.Ms:
            if M > len(D):
                continue
            conds = {"draw": D[0], "wavemean": D[:M].mean(0)}
            if M >= 2:
                lm = G["logmag_ensemble_readout"](D[:M], y, CFG, passthrough=False)
                conds["logmean"] = nv_lo + S.fft_split(lm, False)
            for name, w in conds.items():
                r = S.one(y, w, Sy, {})
                rows.setdefault((M, name), []).append({k: r[k] for k in ("deficit", "deficit_bb", "lsd", "lsd_hf")})
    out, lines = {}, [f"# mean of M draws: {a.model} on {a.set} ({len(utts)} utterances)", "",
                      "| M | readout | deficit dB | deficit bb dB | LSD 2048 | LSD-HF 2048 | incoherent prediction dB |",
                      "|---|---|---|---|---|---|---|"]
    d0 = None
    for (M, name), rs in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        g = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}; g["n"] = len(rs)
        if name == "draw" and d0 is None:
            d0 = g["deficit"]
        g["incoherent_prediction"] = (d0 - 10 * np.log10(M)) if (name == "wavemean" and d0 is not None) else None
        out[f"M{M}/{name}"] = g
        pred = f"{g['incoherent_prediction']:+.2f}" if g["incoherent_prediction"] is not None else "--"
        lines.append(f"| {M} | {name} | {g['deficit']:+.2f} | {g['deficit_bb']:+.2f} | {g['lsd']:.3f} | {g['lsd_hf']:.3f} | {pred} |")
    json.dump({"set": a.set, "model": a.model, "n_utts": len(utts), "rows": out}, open(ws / f"mean_of_M_{a.model}.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
