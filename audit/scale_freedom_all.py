"""Scale freedom (note 2026-09-17 §3) for ALL seven OV3_fast arms at step 16000, six held-out utterances,
tau = 0 and tau = 1 (seed 0).  Same measurements as audit/scale_freedom.py:

  * the same latents queried at 4x and at 8x; the 8x output decimated by 2 (resample_poly) and compared
    to the 4x output -> agreement SNR in dB.  Also an ideal brick-wall version of the same comparison and
    the filter's own round-trip ceiling on a perfectly band-limited signal.
  * the share of the 8x query's energy above 24 kHz (%).
  * the curvature ratio of the decoder response over c in [-1, 1] at the loudest latent triple:
    mean |d2 f| in the gaps between the trained coordinates / mean |d2 f| at them.

Decoder-noise arms (LISASD) draw one noise vector per OUTPUT sample, so the 8x lattice needs a rule:
  held   eps8[2k] = eps8[2k+1] = eps4[k]   -- the 8x query is a strict superset of the 4x query
  fresh  a new draw at 8x                  -- what a naive call at R = 8 would do
Both are measured; `held` is the headline.

    venv/bin/python audit/scale_freedom_all.py
"""
import sys, json, pathlib, time
import numpy as np, torch, scipy.signal as sps
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from audit.boot import boot                                   # noqa: E402
from audit.vctk_fixtures import load, DATA                    # noqa: E402

ARMS = ["det", "es_marg", "es_marg_l0.1", "es_split_l0.1", "es_erb_l0.1", "es_dec_l0.1", "es_dec_erb_l0.1"]
SPK = ["p236", "p237", "p238", "p360", "p361", "p374"]
CKPT = REPO / "lisa_rtm_cache/ckpt/final"
OUT_JSON = REPO / "lisa_rtm_cache/results/scale_OV3_fast.json"
OUT_NPZ = REPO / "lisa_rtm_cache/results/scale_OV3_fast_sweeps.npz"
SEEN = np.array([-1.0, -0.5, 0.0, 0.5])
NC = 801
CHUNK = 1 << 15


def decode_chunked(m, z, rate, n_out):
    m.R = rate
    out = []
    with torch.no_grad():
        for s in range(0, n_out, CHUNK):
            out.append(m.decode(z, s, min(s + CHUNK, n_out)).squeeze(0).numpy().astype(np.float64))
    return np.concatenate(out)


def above_nyq(y8, fs8):
    X = np.fft.rfft(y8)
    fh = np.fft.rfftfreq(len(y8), 1 / fs8)
    P = np.abs(X) ** 2
    return float(P[fh >= fs8 / 4].sum() / P.sum())


def ideal_decimate(y8):
    """Brick-wall at fs8/4, then every other sample."""
    X = np.fft.rfft(y8)
    k = len(X) // 2
    X[k:] = 0
    return np.fft.irfft(X, n=len(y8))[::2]


def ideal_upsample2(y4):
    X = np.fft.rfft(y4)
    Y = np.zeros(len(y4) + 1, complex)
    Y[:len(X)] = X
    return np.fft.irfft(Y, n=2 * len(y4)) * 2


def sweep(m, z, i, eps_row):
    c = torch.linspace(-1.0, 1.0, NC)
    parts = [c.view(1, -1, 1), z[:, i-1:i].expand(1, NC, -1), z[:, i:i+1].expand(1, NC, -1),
             z[:, i+1:i+2].expand(1, NC, -1)]
    if eps_row is not None:
        parts.append(eps_row.view(1, 1, -1).expand(1, NC, -1))
    with torch.no_grad():
        r = m.dec(torch.cat(parts, -1)).squeeze(0).numpy().astype(np.float64)
    d2 = np.abs(np.diff(r, 2))
    near = np.zeros(len(d2), bool)
    for cv in SEEN:
        j = int(round((cv + 1) / 2 * (NC - 1)))
        near[max(0, j - 8):min(len(d2), j + 8)] = True
    at, gap = float(d2[near].mean()), float(d2[~near].mean())
    return r, at, gap, gap / max(at, 1e-30)


def main():
    gone = [a for a in ARMS if not (CKPT / f"{a}.pt").exists()]
    if gone:
        sys.exit(f"no checkpoints under {CKPT}: {' '.join(gone)}")
    empty = [s for s in SPK if not any((DATA / s).glob("*.flac"))]
    if empty:                                 # vctk_fixtures.fetch(SPK, 1) fills DATA
        sys.exit(f"no cached FLACs under {DATA} for {' '.join(empty)}")
    G = boot()
    CFG, snr_db, decimate = G["CFG"], G["snr_db"], G["decimate"]
    R, fs = CFG.upsample, CFG.fs_hi
    utts = [sorted((DATA / s).glob("*.flac"))[0] for s in SPK]
    ys = {p.stem: load(p, fs) for p in utts}
    print("utterances:", {k: f"{len(v)/fs:.2f}s" for k, v in ys.items()}, flush=True)

    res = {"run": "OV3_fast", "step": None, "fs_hi": fs, "upsample": R, "seed": 0, "n_sweep": NC,
           "utterances": [p.stem for p in utts],
           "definitions": {
               "agree_db": "snr_db(y4, resample_poly(y8, 1, 2)): the 8x query decimated by 2 against the 4x query",
               "agree_ideal_db": "same with a brick-wall decimator (rfft zero above 24 kHz, take every other sample)",
               "filter_ceiling_db": "snr_db(y4, resample_poly(ideal_upsample2(y4), 1, 2)): the decimator's own limit on a band-limited signal",
               "even_match_db": "snr_db(y4, y8[::2]): the 8x query at the 4x lattice points vs the 4x query (should be ~fp32 precision)",
               "above_24k_pct": "share of the 8x query's energy at or above 24 kHz, percent",
               "above_24k_db": "the same in dB relative to the total",
               "snr_4x_db": "SNR of the 4x query against the truth",
               "snr_8x_dec_db": "SNR of the decimated 8x query against the truth",
               "curv_at": "mean |d2 f| within +-8 sweep points of c in {-1, -0.5, 0, 0.5}",
               "curv_gap": "mean |d2 f| elsewhere in [-1, 1]",
               "curv_ratio": "curv_gap / curv_at; a spiky decoder is >> 1",
               "latent_index": "loudest latent (200-sample moving |x_lo| argmax) used for the sweep",
               "dec_noise": "for LISASD arms at tau=1: 'held' repeats each 4x decoder-noise row for two 8x samples; 'fresh' draws new noise at 8x",
           },
           "arms": {}}
    sweeps = {}
    t0 = time.time()
    for arm in ARMS:
        m, ck = G["load_arm"](CKPT / f"{arm}.pt", CFG)
        res["step"] = int(ck["step"])
        cls = ck.get("cls", "LISAS")
        n_dec = int(getattr(m, "n_dec", 0))
        taus = [0.0] if arm.startswith("det") else [0.0, 1.0]
        A = {"cls": cls, "n_dec": n_dec, "step": int(ck["step"]), "taus": {}}
        for tau in taus:
            per = {}
            for name, y in ys.items():
                x_lo = torch.from_numpy(decimate(y, R)).float()[None]
                L = x_lo.shape[1]
                e = np.convolve(np.abs(x_lo.numpy()[0]), np.ones(200) / 200, "same")
                i = int(np.clip(np.argmax(e), 1, L - 2))
                with torch.no_grad():
                    eps = None if tau == 0 else m.sample_eps(x_lo, tau, 0)
                    z = m.encode(x_lo, eps)
                eps4 = getattr(m, "_eps_dec", None) if n_dec else None        # (1, L*R, n_dec) or None
                y4 = decode_chunked(m, z, R, L * R)
                variants = {"none": None} if (n_dec == 0 or tau == 0) else {
                    "held": eps4.repeat_interleave(2, dim=1),
                    "fresh": tau * torch.randn(1, L * 2 * R, n_dec, generator=torch.Generator().manual_seed(1)),
                }
                rec = {"latent_index": i, "n_lo": L, "snr_4x_db": float(snr_db(y[:len(y4)], y4))}
                y8_keep = None
                for vname, eps8 in variants.items():
                    if n_dec:
                        m._eps_dec = eps8
                    y8 = decode_chunked(m, z, 2 * R, L * 2 * R)
                    if n_dec:
                        m._eps_dec = eps4
                    y8_dn = sps.resample_poly(y8, 1, 2)[:len(y4)]
                    y8_id = ideal_decimate(y8)[:len(y4)]
                    up = sps.resample_poly(ideal_upsample2(y4), 1, 2)[:len(y4)]
                    frac = above_nyq(y8, 2 * fs)
                    d = {"agree_db": float(snr_db(y4, y8_dn)),
                         "agree_ideal_db": float(snr_db(y4, y8_id)),
                         "filter_ceiling_db": float(snr_db(y4, up)),
                         "even_match_db": float(snr_db(y4, y8[::2])),
                         "above_24k_pct": 100 * frac,
                         "above_24k_db": float(10 * np.log10(max(frac, 1e-30))),
                         "snr_8x_dec_db": float(snr_db(y[:len(y8_dn)], y8_dn))}
                    if vname == "none":
                        rec.update(d)
                    else:
                        rec[vname] = d
                        if vname == "held":
                            rec.update(d)
                    if y8_keep is None:
                        y8_keep = y8
                m.R = R
                # sweep at the loudest latent triple; decoder noise row = the one at output sample i*R
                eps_row = None
                if n_dec:
                    eps_row = torch.zeros(n_dec) if tau == 0 else eps4[0, i * R]
                r, at, gap, ratio = sweep(m, z, i, eps_row)
                rec.update({"curv_at": at, "curv_gap": gap, "curv_ratio": ratio,
                            "sweep_range": [float(r.min()), float(r.max())]})
                sweeps[f"{arm}|{tau:g}|{name}"] = r.astype(np.float32)
                if name == "p236_002_mic1":
                    X = np.fft.rfft(y8_keep); P = np.abs(X) ** 2
                    fh = np.fft.rfftfreq(len(y8_keep), 1 / (2 * fs))
                    edges = np.arange(0, 2 * fs / 2 + 1, 500)
                    band = [float(10 * np.log10(max(P[(fh >= a) & (fh < b)].sum() / P.sum(), 1e-30)))
                            for a, b in zip(edges[:-1], edges[1:])]
                    sweeps[f"spec|{arm}|{tau:g}"] = np.array(band, np.float32)
                per[name] = rec
                print(f"{arm:16s} tau={tau:g} {name:14s} agree {rec['agree_db']:6.2f} dB  ideal {rec['agree_ideal_db']:6.2f} "
                      f"ceil {rec['filter_ceiling_db']:6.2f}  even {rec['even_match_db']:6.1f}  >24k {rec['above_24k_pct']:.4f}%  "
                      f"snr4 {rec['snr_4x_db']:5.2f} snr8 {rec['snr_8x_dec_db']:5.2f}  curv {ratio:.3f}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
            keys = ["agree_db", "agree_ideal_db", "filter_ceiling_db", "even_match_db", "above_24k_pct",
                    "above_24k_db", "snr_4x_db", "snr_8x_dec_db", "curv_ratio", "curv_at", "curv_gap"]
            summ = {}
            for k in keys:
                v = np.array([per[u][k] for u in per])
                worst = float(v.min()) if k in ("agree_db", "agree_ideal_db", "filter_ceiling_db", "even_match_db") else float(v.max())
                summ[k] = {"mean": float(v.mean()), "worst": worst, "min": float(v.min()), "max": float(v.max())}
            for vname in ("held", "fresh"):
                if vname in next(iter(per.values())):
                    for k in ("agree_db", "agree_ideal_db", "above_24k_pct", "above_24k_db", "snr_8x_dec_db", "even_match_db"):
                        v = np.array([per[u][vname][k] for u in per])
                        summ[f"{vname}.{k}"] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
            A["taus"][f"{tau:g}"] = {"per_utterance": per, "summary": summ}
        res["arms"][arm] = A
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1))
    np.savez_compressed(OUT_NPZ, **sweeps)
    print("wrote", OUT_JSON, OUT_NPZ, f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
