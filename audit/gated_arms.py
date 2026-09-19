"""Does the high band gate with the speech, for every arm?

Repeats the measurement of audit/band_gating.py (notes/2026-09-17-snr-ceiling-gating-and-scale.md
section 2.1) on all seven FINAL checkpoints and twelve held-out utterances.  Per arm and utterance:
one draw (tau=1, seed=0; tau=0 for det) and the noiseless pass (tau=0); the mean third-octave
deficit; the same deficit restricted to loud / mid / quiet frames of the target; SNR of the draw,
the tau=0 pass and naive sinc upsampling; and the per-band ratios.

    venv/bin/python audit/gated_arms.py [--tag OV3_fast]

Writes lisa_rtm_cache/results/gated_<tag>.json.  The figure and note come from audit/gated_arms_report.py.
"""
import json, pathlib, sys, time
import numpy as np
import scipy.signal as sps

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from audit.boot import boot                                   # noqa: E402
from audit.vctk_fixtures import load                          # noqa: E402

ARMS = ["det", "es_marg", "es_marg_l0.1", "es_split_l0.1", "es_erb_l0.1", "es_dec_l0.1", "es_dec_erb_l0.1"]
SPEAKERS = ["p236", "p237", "p238", "p360", "p361", "p374"]
CKPT = REPO / "lisa_rtm_cache" / "ckpt" / "final"
AUDIO = REPO / "lisa_rtm_cache" / "audit"
OUT = REPO / "lisa_rtm_cache" / "results"

SCALARS = ["deficit_draw", "deficit_tau0", "loud", "mid", "quiet", "swing",
           "loud_tau0", "mid_tau0", "quiet_tau0", "swing_tau0",
           "snr_draw", "snr_tau0", "snr_naive", "snr_draw_minus_naive",
           "hb_share_loud_pct", "hb_share_mid_pct", "hb_share_quiet_pct", "hb_share_pct",
           "target_contrast_lq", "model_contrast_lq", "model_contrast_lq_tau0", "gating_fraction"]


def frame_masks(G, y, CFG):
    """Loud / mid / quiet frame masks from the TARGET's frame energy in the evaluation STFT basis,
    exactly as audit/band_gating.py builds them."""
    S, _, _ = G["stft"](y, CFG.eval_n_fft, CFG.eval_hop)
    fe = 20 * np.log10(np.abs(S).sum(1) + 1e-12)
    p25, p75 = np.percentile(fe, 25), np.percentile(fe, 75)
    masks = {"loud": fe >= p75, "mid": (fe >= p25) & (fe < p75), "quiet": fe < p25}
    f = np.fft.rfftfreq(CFG.eval_n_fft, 1 / CFG.fs_hi)
    Sy = np.abs(S)
    hb = f >= CFG.fs_lo / 2
    share = {k: float(100 * (Sy[m][:, hb] ** 2).sum() / (Sy[m] ** 2).sum()) for k, m in masks.items()}
    share["all"] = float(100 * (Sy[:, hb] ** 2).sum() / (Sy ** 2).sum())
    hb_db = {k: float(10 * np.log10((Sy[m][:, hb] ** 2).sum() + 1e-20)) for k, m in masks.items()}
    share["target_contrast_lq"] = hb_db["loud"] - hb_db["quiet"]      # dB the target's HB rises, quiet -> loud
    share["target_contrast_lm"] = hb_db["loud"] - hb_db["mid"]
    return masks, share


def main(tag="OV3_fast"):
    gone = [a for a in ARMS if not (CKPT / f"{a}.pt").exists()]
    if gone:
        sys.exit(f"no checkpoints under {CKPT}: {' '.join(gone)}")
    utts = [p for s in SPEAKERS for p in sorted((AUDIO / s).glob("*.flac"))[:2]]
    if len(utts) != 12:                       # vctk_fixtures.fetch(SPEAKERS, 2) fills AUDIO
        sys.exit(f"{len(utts)} of 12 cached FLACs under {AUDIO}")
    G = boot()
    CFG, snr_db = G["CFG"], G["snr_db"]
    R = CFG.upsample
    ber = lambda y, yh, msk=None: G["band_energy_ratio"](y, yh, CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop,
                                                         CFG.fs_lo / 2, CFG.fs_hi / 2, frame_mask=msk)
    ys = {p.stem.replace("_mic1", ""): load(p, CFG.fs_hi) for p in utts}
    naive = {k: sps.resample_poly(sps.resample_poly(y, 1, R), R, 1)[:len(y)] for k, y in ys.items()}
    masks = {k: frame_masks(G, y, CFG) for k, y in ys.items()}

    results = {"_meta": {"tag": tag, "step": None, "n_utts": len(utts), "utts": list(ys),
                         "fs_lo": CFG.fs_lo, "fs_hi": CFG.fs_hi, "eval_n_fft": CFG.eval_n_fft,
                         "eval_hop": CFG.eval_hop, "band_lo_hz": CFG.fs_lo / 2, "band_hi_hz": CFG.fs_hi / 2,
                         "draw": "tau=1 seed=0 (det: tau=0)", "noiseless": "tau=0",
                         "naive": "resample_poly(resample_poly(y,1,4),4,1)[:len(y)]",
                         "frame_energy": "20*log10(sum_bins |S_target|) in the eval STFT; loud = top 25 %, "
                                         "mid = 25-75 %, quiet = bottom 25 %",
                         "contrast": "target_contrast_lq = 10log10(HB energy, loud frames / HB energy, quiet "
                                     "frames) of the target; model_contrast_lq = swing + target_contrast_lq "
                                     "(equal frame counts); gating_fraction = model / target contrast, 1 = gates "
                                     "like the speech, 0 = flat high band",
                         "deficit": "mean over third-octave bands from fs_lo/2 to fs_hi/2 of "
                                    "10*log10(sum|S_hat|^2 / sum|S|^2), frames restricted by the mask"}}
    t0 = time.time()
    for arm in ARMS:
        m, ck = G["load_arm"](CKPT / f"{arm}.pt", CFG)
        step = int(ck["step"])
        results["_meta"]["step"] = step
        is_det = arm.startswith("det")
        rows = []
        for utt, y in ys.items():
            yh0 = G["reconstruct"](m, y, CFG, tau=0.0, seed=0)
            yh = yh0 if is_det else G["reconstruct"](m, y, CFG, tau=1.0, seed=0)
            mk, share = masks[utt]
            b_draw, b_t0 = ber(y, yh), ber(y, yh0)
            g = {k: float(np.mean(ber(y, yh, mk[k])[:, 1])) for k in ("loud", "mid", "quiet")}
            g0 = {k: float(np.mean(ber(y, yh0, mk[k])[:, 1])) for k in ("loud", "mid", "quiet")}
            row = {"utt": utt, "seconds": len(y) / CFG.fs_hi,
                   "deficit_draw": float(b_draw[:, 1].mean()), "deficit_tau0": float(b_t0[:, 1].mean()),
                   "loud": g["loud"], "mid": g["mid"], "quiet": g["quiet"], "swing": g["loud"] - g["quiet"],
                   "loud_tau0": g0["loud"], "mid_tau0": g0["mid"], "quiet_tau0": g0["quiet"],
                   "swing_tau0": g0["loud"] - g0["quiet"],
                   "snr_draw": float(snr_db(y, yh)), "snr_tau0": float(snr_db(y, yh0)),
                   "snr_naive": float(snr_db(y, naive[utt])),
                   "hb_share_loud_pct": share["loud"], "hb_share_mid_pct": share["mid"],
                   "hb_share_quiet_pct": share["quiet"], "hb_share_pct": share["all"],
                   "bands_hz": [float(c) for c in b_draw[:, 0]],
                   "band_draw": [float(d) for d in b_draw[:, 1]],
                   "band_tau0": [float(d) for d in b_t0[:, 1]]}
            row["snr_draw_minus_naive"] = row["snr_draw"] - row["snr_naive"]
            # loud and quiet hold the same number of frames, so swing = model contrast - target contrast
            row["target_contrast_lq"] = share["target_contrast_lq"]
            row["model_contrast_lq"] = row["swing"] + share["target_contrast_lq"]
            row["model_contrast_lq_tau0"] = row["swing_tau0"] + share["target_contrast_lq"]
            row["gating_fraction"] = row["model_contrast_lq"] / share["target_contrast_lq"]
            rows.append(row)
        mean = {k: float(np.mean([r[k] for r in rows])) for k in SCALARS}
        sd = {k: float(np.std([r[k] for r in rows], ddof=1)) for k in SCALARS}
        mean["band_draw"] = [float(v) for v in np.mean([r["band_draw"] for r in rows], 0)]
        sd["band_draw"] = [float(v) for v in np.std([r["band_draw"] for r in rows], 0, ddof=1)]
        mean["band_tau0"] = [float(v) for v in np.mean([r["band_tau0"] for r in rows], 0)]
        sd["band_tau0"] = [float(v) for v in np.std([r["band_tau0"] for r in rows], 0, ddof=1)]
        mean["bands_hz"] = rows[0]["bands_hz"]
        results[arm] = {"arm_tuple": [str(a) for a in ck["arm"]], "step": step, "cls": ck.get("cls"),
                        "per_utt": rows, "mean": mean, "sd": sd}
        print(f"{arm:<16} step {step}  deficit {mean['deficit_draw']:+7.2f}  loud {mean['loud']:+7.2f}  "
              f"mid {mean['mid']:+7.2f}  quiet {mean['quiet']:+7.2f}  swing {mean['swing']:6.2f}  "
              f"SNR {mean['snr_draw']:6.2f} (naive {mean['snr_naive']:6.2f})   {time.time()-t0:5.1f}s", flush=True)
    # paired differences over the same utterances: the questions the note asks
    def paired(a, b, k):
        d = np.array([ra[k] - rb[k] for ra, rb in zip(results[a]["per_utt"], results[b]["per_utt"])])
        return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum())}
    PAIRS = {"erb_term": ("es_erb_l0.1", "es_marg_l0.1"), "dec_noise": ("es_dec_l0.1", "es_marg_l0.1"),
             "dec_noise_on_erb": ("es_dec_erb_l0.1", "es_erb_l0.1"), "erb_term_on_dec": ("es_dec_erb_l0.1", "es_dec_l0.1"),
             "lam_0.1_vs_0.01": ("es_marg_l0.1", "es_marg"), "split_wave_term": ("es_split_l0.1", "es_marg_l0.1"),
             "sampler_vs_det": ("es_marg", "det"), "best_vs_det": ("es_dec_erb_l0.1", "det")}
    results["_paired"] = {name: {"first": a, "second": b,
                                 **{k: paired(a, b, k) for k in ("deficit_draw", "loud", "mid", "quiet", "swing",
                                                                 "snr_draw", "gating_fraction")}}
                          for name, (a, b) in PAIRS.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"gated_{tag}.json"
    out.write_text(json.dumps(results, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "OV3_fast")
