"""What is envelope correction worth?  Rescale the high band, post hoc, and remeasure.

audit/envelope_regression.py found every arm sitting at slope 0.75-0.86 with r 0.86-0.93: the models
know which frames need high-band energy and systematically under-deliver on the loud ones.  It also
found the high-band envelope is ~89 % predictable from a log profile of the low band.  So the high
band splits in two -- an ENVELOPE that is largely deterministic and should be regressed, and a fine
structure that is a coin flip and must be sampled.  We sample both.

This applies the split after the fact, as a per-(ERB band, frame) gain on the draw's high band, phase
and fine structure untouched, and remeasures.  Three gains, in order of how much they know:

  global   one gain per band for the whole utterance.  Fixes the average level only.  The control:
           if this recovers the loud-frame deficit, the problem was never the envelope's shape.
  fitted   log band energy predicted from a 16-band log profile of the low band by least squares,
           fitted LEAVE-ONE-SPEAKER-OUT.  The honest number -- no truth, held-out speaker.
  fit+inf  the same prediction with its deviations inflated to the truth's per-band variance, the
           inflation measured on the training fold.  Least squares returns a CONDITIONAL MEAN, so the
           predictor regresses to the mean exactly as the network does; this asks whether that is what
           is left.
  fit+samp the prediction plus a draw from its own residual spread -- SAMPLING the envelope instead of
           regressing it, which is the honest fix if the residual is genuinely unpredictable.
  oracle   the truth's own band energy.  The ceiling: perfect envelope, the model's own fine structure.

And one control that runs the swap the other way -- the TRUTH's fine structure and phase carrying the
MODEL's envelope -- so the high band's error splits cleanly in two.

    venv/bin/python audit/envelope_rescale.py [--tag OV3_fast]

Writes lisa_rtm_cache/results/envelope_rescale_<tag>.json.
"""
import argparse, json, pathlib, sys, time
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from audit.boot import boot                                   # noqa: E402
from audit.vctk_fixtures import load                          # noqa: E402

ARMS = ["det", "es_marg", "es_marg_l0.1", "es_split_l0.1", "es_erb_l0.1", "es_dec_l0.1", "es_dec_erb_l0.1"]
SPEAKERS = ["p236", "p237", "p238", "p360", "p361", "p374"]
CKPT = REPO / "lisa_rtm_cache" / "ckpt" / "final"
AUDIO = REPO / "lisa_rtm_cache" / "audit"
OUT = REPO / "lisa_rtm_cache" / "results"
N_LO = 16                 # low-band profile features
EPS = 1e-20


def _erb_rate(f):
    return 21.4 * np.log10(1.0 + 4.37 * np.asarray(f, np.float64) / 1000.0)


def _erb_rate_inv(e):
    return (10.0 ** (np.asarray(e, np.float64) / 21.4) - 1.0) * 1000.0 / 4.37


def hi_erb_bank(n_fft, fs, f_cut, n_bands=16):
    """(n_bands, F) triangular ERB filters spanning f_cut..fs/2, plus the column-normalised copy used
    to spread a per-band gain back over bins.  Rows sum to 1 (energy read-out); columns of Wn sum to 1
    (so a constant per-band gain comes back as that constant on every bin it touches)."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    edges = _erb_rate_inv(np.linspace(_erb_rate(f_cut), _erb_rate(fs / 2), n_bands + 2))
    W = np.zeros((n_bands, len(freqs)))
    for b in range(n_bands):
        lo, c, hi = edges[b], edges[b + 1], edges[b + 2]
        W[b] = np.clip(np.minimum((freqs - lo) / max(c - lo, 1e-9), (hi - freqs) / max(hi - c, 1e-9)), 0, 1)
    W[:, freqs < f_cut] = 0.0                                  # never touch the baseband
    centres = edges[1:-1]
    for k in np.where((W.sum(0) <= 0) & (freqs >= f_cut))[0]:   # uncovered high bins -> nearest band
        W[int(np.argmin(np.abs(centres - freqs[k]))), k] = 1.0
    keep = W.sum(1) > 0
    W = W[keep]
    Wn = W / np.maximum(W.sum(0, keepdims=True), 1e-12)         # column-normalised
    return W / W.sum(1, keepdims=True), Wn, freqs >= f_cut


def lo_profile(P, freqs, f_cut):
    """(T, N_LO) log10 energies of equal-bin-count slices of the low band -- the predictor features."""
    lo = freqs < f_cut
    e = np.linspace(0, int(lo.sum()), N_LO + 1).astype(int)
    Plo = P[:, lo]
    return np.stack([np.log10(Plo[:, a:b].sum(1) + EPS) for a, b in zip(e[:-1], e[1:])], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="OV3_fast")
    ap.add_argument("--bands", type=int, default=16)
    a = ap.parse_args()

    G = boot()
    CFG = G["CFG"]
    stft, istft, snr_db = G["stft"], G["istft"], G["snr_db"]
    NF, HOP, FS, FC = CFG.eval_n_fft, CFG.eval_hop, CFG.fs_hi, CFG.fs_lo / 2
    W, Wn, hb = hi_erb_bank(NF, FS, FC, a.bands)
    NB = W.shape[0]
    ber = lambda y, yh, m=None: G["band_energy_ratio"](y, yh, FS, NF, HOP, FC, FS / 2, frame_mask=m)
    lsd = lambda y, yh, k0=0: G["lsd_db"](y, yh, NF, HOP, k_from=k0)
    k_cut = int(np.searchsorted(np.fft.rfftfreq(NF, 1.0 / FS), FC))

    utts = [(s, p) for s in SPEAKERS for p in sorted((AUDIO / s).glob("*.flac"))[:2]]
    assert len(utts) == 12, utts
    print(f"{len(utts)} utterances, {NB} ERB bands over {FC:.0f}-{FS/2:.0f} Hz\n", flush=True)

    # ---- truth side, once: spectra, masks, band energies, low-band features --------------------
    T = {}
    for spk, p in utts:
        y = load(p, FS)
        S, pad, L = stft(y, NF, HOP)
        P = np.abs(S) ** 2
        fe = 20 * np.log10(np.abs(S).sum(1) + 1e-12)
        p25, p75 = np.percentile(fe, 25), np.percentile(fe, 75)
        T[p.stem] = {"spk": spk, "y": y, "n": L,
                     "masks": {"loud": fe >= p75, "mid": (fe >= p25) & (fe < p75), "quiet": fe < p25},
                     "E": W @ P.T,                                   # (NB, T) truth band energy
                     "X": lo_profile(P, np.fft.rfftfreq(NF, 1.0 / FS), FC)}

    # ---- leave-one-speaker-out linear map: low-band log profile -> log high-band band energies --
    beta, r2 = {}, {}
    for held in SPEAKERS:
        X = np.concatenate([np.c_[v["X"], np.ones(len(v["X"]))] for v in T.values() if v["spk"] != held])
        Y = np.concatenate([np.log10(v["E"] + EPS).T for v in T.values() if v["spk"] != held])
        beta[held] = np.linalg.lstsq(X, Y, rcond=None)[0]
        Xh = np.concatenate([np.c_[v["X"], np.ones(len(v["X"]))] for v in T.values() if v["spk"] == held])
        Yh = np.concatenate([np.log10(v["E"] + EPS).T for v in T.values() if v["spk"] == held])
        r2[held] = float(1 - ((Yh - Xh @ beta[held]) ** 2).sum() / ((Yh - Yh.mean(0)) ** 2).sum())
    print(f"leave-one-speaker-out R^2 of the linear envelope predictor: "
          f"{np.mean(list(r2.values())):.3f}  (per speaker {', '.join(f'{v:.2f}' for v in r2.values())})\n",
          flush=True)

    def rescale(S_m, gain_b):
        """gain_b: (NB, T) amplitude gain per band per frame -> spread over bins, high band only."""
        g = np.ones_like(np.abs(S_m))
        g[:, hb] = (gain_b.T @ Wn)[:, hb]
        return S_m * g

    # per-band training-fold statistics for the inflation, one set per held-out speaker
    infl = {}
    for held in SPEAKERS:
        Xtr = np.concatenate([np.c_[v["X"], np.ones(len(v["X"]))] for v in T.values() if v["spk"] != held])
        Ytr = np.concatenate([np.log10(v["E"] + EPS).T for v in T.values() if v["spk"] != held])
        Ptr = Xtr @ beta[held]
        infl[held] = (Ptr.mean(0), Ytr.std(0) / np.maximum(Ptr.std(0), 1e-9),
                      (Ytr - Ptr).std(0))                            # per-band residual sd of log10 E
    print("inflation factors sd(truth)/sd(pred), mean over bands: "
          + ", ".join(f"{s_:.2f}" for _, s_ in (infl[k] for k in SPEAKERS)).replace("nan", "-")
          if False else "inflation factor sd(truth)/sd(pred), mean over bands and speakers: "
          f"{np.mean([s_.mean() for _, s_, _ in infl.values()]):.3f}   "
          f"residual sd {np.mean([r.mean() for _, _, r in infl.values()]) * 10:.2f} dB\n", flush=True)

    GAINS = ("raw", "global", "fitted", "fit+inf", "fit+samp", "oracle", "swap")
    rows, t0 = {}, time.time()
    for arm in ARMS:
        m_, ck = G["load_arm"](CKPT / f"{arm}.pt", CFG)
        is_det = arm.startswith("det")
        per = {k: [] for k in GAINS}
        for spk, p in utts:
            t = T[p.stem]
            y = t["y"]
            yh = G["reconstruct"](m_, y, CFG, tau=0.0 if is_det else 1.0, seed=0)
            S_m, pad, L = stft(np.asarray(yh, np.float64), NF, HOP)
            Em = W @ (np.abs(S_m) ** 2).T                            # (NB, T)
            Et = t["E"][:, :Em.shape[1]]
            Em = Em[:, :Et.shape[1]]
            nT = Et.shape[1]
            S_m = S_m[:nT]
            Xh = np.c_[t["X"][:nT], np.ones(nT)]
            lp = Xh @ beta[spk]                                      # (T, NB) predicted log10 energy
            mu, sc, sd_r = infl[spk]
            rng = np.random.default_rng(abs(hash(p.stem)) % (2 ** 31))
            pred = np.maximum(10.0 ** lp.T, EPS)                     # (NB, T) predicted band energy
            pinf = np.maximum(10.0 ** (mu + sc * (lp - mu)).T, EPS)  # variance-inflated
            gains = {
                "raw":     np.ones((NB, nT)),
                "global":  np.sqrt((Et.sum(1) + EPS) / (Em.sum(1) + EPS))[:, None] * np.ones((1, nT)),
                "fitted":  np.sqrt(pred / (Em + EPS)),
                "fit+inf": np.sqrt(pinf / (Em + EPS)),
                "fit+samp": np.sqrt(np.maximum(
                    10.0 ** (lp + sd_r * rng.standard_normal(lp.shape)).T, EPS) / (Em + EPS)),
                "oracle":  np.sqrt((Et + EPS) / (Em + EPS)),
            }
            S_t = stft(y, NF, HOP)[0][:nT]                            # truth's fine structure...
            for k, gb in gains.items():
                yk = istft(rescale(S_m, gb), NF, HOP, pad, L)[:len(y)]
                mk = {n: v[:nT] for n, v in t["masks"].items()}
                g = {n: float(np.mean(ber(y, yk, mk[n])[:, 1])) for n in ("loud", "mid", "quiet")}
                per[k].append({"deficit": float(ber(y, yk)[:, 1].mean()), **g,
                               "swing": g["loud"] - g["quiet"],
                               "lsd": lsd(y, yk), "hb_lsd": lsd(y, yk, k_cut), "snr": snr_db(y, yk)})
            ys_ = istft(rescale(S_t, np.sqrt((Em + EPS) / (Et + EPS))), NF, HOP, pad, L)[:len(y)]
            mk = {n: v[:nT] for n, v in t["masks"].items()}           # ...carrying the model's envelope
            g = {n: float(np.mean(ber(y, ys_, mk[n])[:, 1])) for n in ("loud", "mid", "quiet")}
            per["swap"].append({"deficit": float(ber(y, ys_)[:, 1].mean()), **g,
                                "swing": g["loud"] - g["quiet"], "lsd": lsd(y, ys_),
                                "hb_lsd": lsd(y, ys_, k_cut), "snr": snr_db(y, ys_)})
        rows[arm] = {k: {f: float(np.mean([d[f] for d in v])) for f in v[0]} for k, v in per.items()}
        r = rows[arm]
        print(f"{arm}")
        print(f"   {'gain':<8} {'deficit':>8} {'loud':>7} {'mid':>7} {'quiet':>7} {'swing':>7} "
              f"{'LSD':>6} {'HB-LSD':>7} {'SNR':>6}")
        for k in GAINS:
            d = r[k]
            print(f"   {k:<8} {d['deficit']:8.2f} {d['loud']:7.2f} {d['mid']:7.2f} {d['quiet']:7.2f} "
                  f"{d['swing']:7.2f} {d['lsd']:6.3f} {d['hb_lsd']:7.3f} {d['snr']:6.2f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"envelope_rescale_{a.tag}.json"
    out.write_text(json.dumps({"_meta": {"tag": a.tag, "n_bands": int(NB), "f_cut_hz": FC,
                                         "n_lo_features": N_LO, "utts": [p.stem for _, p in utts],
                                         "predictor": "leave-one-speaker-out OLS, log10 band energy "
                                                      "from a 16-band log10 low-band profile + bias",
                                         "loso_r2": r2, "loso_r2_mean": float(np.mean(list(r2.values()))),
                                         "gain": "per (ERB band, frame) amplitude gain on the draw's "
                                                 "high band; phase and fine structure untouched"},
                               **rows}, indent=1))
    print(f"\nwrote {out}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
