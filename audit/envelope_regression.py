"""Is the high-band ENERGY ENVELOPE regressing to its mean?

The gated deficit says every arm is too quiet on loud frames and too loud in the gaps.  Two
different failures produce that, and they need different fixes:

  (a) compressed variance -- the model's per-frame high-band energy is pulled toward the average.
      Regression to the mean, one level up from the waveform.  Signature: slope < 1 WITH high r.
  (b) wrong assignment  -- the model emits roughly the right spread of energies but does not know
      which frame needs which.  Signature: r near 0, slope near 0, sd ratio near 1.

So regress log E_hb(model) on log E_hb(truth) across frames and read off the slope, the correlation
and the sd ratio.  A correct sampler sits at slope 1, r 1, sd ratio 1.

Also measured: how predictable the high-band envelope is FROM THE LOW BAND in the first place
(r between log E_hb and log E_lb of the truth, and the R^2 of a least-squares fit on the log
low-band ERB profile).  That is the headroom -- the part of the high band that is NOT a coin flip
and therefore should be regressed rather than sampled.

    venv/bin/python audit/envelope_regression.py [--tag OV3_fast] [--n 12]

Writes lisa_rtm_cache/results/envelope_<tag>.json.
"""
import argparse, json, pathlib, sys
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SPEAKERS = ("p236", "p237", "p238", "p360", "p361", "p374")
FLOOR_DB = 60.0          # frames more than this far below the loudest frame are digital silence


def frame_energies(G, y, CFG):
    """(log10 E_hb, log10 E_lb, frame energy in dB) per frame, in the evaluation STFT basis."""
    S, _, _ = G["stft"](y, CFG.eval_n_fft, CFG.eval_hop)
    P = np.abs(S) ** 2
    f = np.fft.rfftfreq(CFG.eval_n_fft, 1.0 / CFG.fs_hi)
    hb, lb = f >= CFG.fs_lo / 2, f < CFG.fs_lo / 2
    e_hb, e_lb = P[:, hb].sum(1), P[:, lb].sum(1)
    tot = P.sum(1)
    db = 10 * np.log10(tot + 1e-20)
    return np.log10(e_hb + 1e-20), np.log10(e_lb + 1e-20), db - db.max(), P, f


def ols(x, y):
    """slope, intercept, r of y on x."""
    if len(x) < 8:
        return float("nan"), float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx <= 0:
        return float("nan"), float("nan"), float("nan")
    b = float(((x - xm) * (y - ym)).sum() / sxx)
    r = float(np.corrcoef(x, y)[0, 1])
    return b, float(ym - b * xm), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="OV3_fast")
    ap.add_argument("--ckpt-dir", default="lisa_rtm_cache/ckpt/final")
    ap.add_argument("--n", type=int, default=2, help="utterances per speaker")
    a = ap.parse_args()
    ckpt_dir = REPO / a.ckpt_dir if not pathlib.Path(a.ckpt_dir).is_absolute() else pathlib.Path(a.ckpt_dir)

    from audit.boot import boot
    from audit.vctk_fixtures import fetch, load
    G = boot()
    CFG = G["CFG"]
    paths = []
    for spk in SPEAKERS:
        paths += fetch((spk,), per_speaker=a.n)[: a.n]
    print(f"{len(paths)} utterances, {len(SPEAKERS)} speakers\n", flush=True)

    arms = {p.stem.split("_step")[0]: p for p in sorted(ckpt_dir.glob("*.pt"))}
    out = {"_meta": {"tag": a.tag, "utts": [p.name for p in paths], "floor_db": FLOOR_DB,
                     "n_fft": CFG.eval_n_fft, "hop": CFG.eval_hop, "cut_hz": CFG.fs_lo / 2}}

    # ---- how predictable is the high-band envelope from the low band at all? -----------------
    r_lb, r2_fit = [], []
    for p in paths:
        y = load(p, CFG.fs_hi)
        lh, ll, rel, P, f = frame_energies(G, y, CFG)
        m = rel > -FLOOR_DB
        r_lb.append(float(np.corrcoef(lh[m], ll[m])[0, 1]))
        # least squares from the log low-band ERB profile (16 bands under 6 kHz) -> log E_hb
        lo = f < CFG.fs_lo / 2
        edges = np.linspace(0, lo.sum(), 17).astype(int)
        prof = np.stack([np.log10(P[:, lo][:, e0:e1].sum(1) + 1e-20) for e0, e1 in zip(edges[:-1], edges[1:])], 1)
        X = np.c_[prof[m], np.ones(m.sum())]
        beta, *_ = np.linalg.lstsq(X, lh[m], rcond=None)
        resid = lh[m] - X @ beta
        r2_fit.append(float(1 - resid.var() / lh[m].var()))
    out["_envelope_predictability"] = {"r_hb_vs_lb": float(np.mean(r_lb)),
                                       "r2_from_lowband_profile": float(np.mean(r2_fit))}
    print(f"high-band envelope predictability from the low band, truth only:")
    print(f"   r(log E_hb, log E_lb)            {np.mean(r_lb):+.3f}")
    print(f"   R^2 from a 16-band log profile   {np.mean(r2_fit):+.3f}\n", flush=True)

    # ---- each arm's envelope against the truth's ---------------------------------------------
    print(f"{'arm':<18} {'slope':>7} {'r':>7} {'sd ratio':>9} {'deficit':>8}   reading")
    for arm, ck_path in arms.items():
        m_, ck = G["load_arm"](ck_path, CFG)
        det = ck["arm"][0].startswith("det")
        sl, rr, sd, df = [], [], [], []
        for p in paths:
            y = load(p, CFG.fs_hi)
            yh = G["reconstruct"](m_, y, CFG, tau=0.0 if det else 1.0, seed=0)
            lh_t, _, rel, _, _ = frame_energies(G, y, CFG)
            lh_m, _, _, _, _ = frame_energies(G, np.asarray(yh, np.float64), CFG)
            k = min(len(lh_t), len(lh_m))
            msk = rel[:k] > -FLOOR_DB
            b, _, r = ols(lh_t[:k][msk], lh_m[:k][msk])
            sl.append(b); rr.append(r)
            sd.append(float(lh_m[:k][msk].std() / max(lh_t[:k][msk].std(), 1e-9)))
            df.append(float(10 * (lh_m[:k][msk] - lh_t[:k][msk]).mean()))
        S, R, D, F = np.mean(sl), np.mean(rr), np.mean(sd), np.mean(df)
        out[arm] = {"slope": float(S), "r": float(R), "sd_ratio": float(D), "deficit_db": float(F),
                    "per_utt": {"slope": sl, "r": rr, "sd_ratio": sd, "deficit_db": df}, "det": bool(det)}
        why = ("compressed variance" if R > 0.5 and S < 0.8 else
               "wrong assignment" if R < 0.35 else
               "calibrated" if 0.8 <= S <= 1.2 and R > 0.5 else "mixed")
        print(f"{arm:<18} {S:7.3f} {R:7.3f} {D:9.3f} {F:8.2f}   {why}", flush=True)

    p = REPO / "lisa_rtm_cache" / "results" / f"envelope_{a.tag}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")
    print("slope 1, r 1, sd ratio 1 is a correctly conditioned envelope.")


if __name__ == "__main__":
    main()
