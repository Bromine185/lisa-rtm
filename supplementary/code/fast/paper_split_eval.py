"""Every OV50 arm on the literature's VCTK test split, in the literature's LSD basis.

    venv/bin/python fast/paper_split_eval.py --ckpt-dir ~/results/ckpt/OV50 [--per-speaker 10]
        [--basis 2048,512] [--our-basis 1024,256] [--resampler poly|cheby|sinc] [--visqol pt|all|none] [--M 16]

WHY A SECOND EVALUATION.  EVAL12 is twelve utterances of ONE speaker (p236) scored in this repo's
1024/256 STFT basis.  The papers this model will be read against (NVSR, AP-BWE, AERO, FLowHigh)
evaluate on VCTK's last speakers -- LISA's split, speaker id >= 350 -- in a 2048/512 basis, and quote
LSD over the whole band plus the high and low bands separately.  fast/stage_corpus.py excluded every
speaker >= p350 from the OV50 training corpus (manifest.json: n_paper_test = 2961), so those speakers
are genuinely unseen and the comparison is legitimate on the data side.  This script makes it
legitimate on the metric side too: same basis, same band bounds, same downsampling filter, stated.

WHAT IS NOT MATCHED, and is printed rather than hidden: the number of utterances (the cached audit
fixtures, 8-10 per speaker, unless more are fetched), peak normalisation to 0.95 (LSD and NSIM are
invariant to a gain common to reference and estimate; nothing here changes the estimate's gain
relative to its reference), and the anti-aliasing filter of the OTHER paper's pipeline, which is
reproduced only as far as --resampler allows.  Three are offered: poly (scipy resample_poly, what the
model was trained on), cheby (the FLowHigh / NU-Wave 2 evaluation input, which sota/ uses throughout) and
sinc (torchaudio's windowed sinc); run more than one and read the gap as the domain shift it is.

LSD here is the conventional one: d = log10(|S_gt|^2 + eps) - log10(|S_est|^2 + eps), RMS over
frequency within a frame, mean over frames (audit/lisa_paper_protocol.py::lsd_standard).  LSD-HF and
LSD-LF restrict the frequency axis to above / below the input Nyquist (--hf-cut, default fs_lo / 2).
Decades of power, NOT dB, like every LSD in this repo.

Output: <out>/paperset_<tag>_<resampler>.json (per-utterance rows and aggregates) and the .md table.
"""
import argparse, json, os, pathlib, re, sys, time
import numpy as np
import scipy.signal as sps

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from audit.vctk_fixtures import DATA, PAPER_TEST, load      # noqa: E402

ORDER = ["det_paper", "det", "es_marg", "es_dec_l0.01", "es_erb_l0.001", "es_erb_l0.01", "es_erb_l0.1", "es_dec_erb_l0.1"]


def arm_name(p):
    return re.sub(r"_step\d+$", "", pathlib.Path(p).stem)


# ---- the literature's LSD -------------------------------------------------------------------------
def stft_mag2(x, n_fft, hop):
    w = np.hanning(n_fft + 1)[:-1]
    x = np.asarray(x, np.float64)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_fr = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_fr)[:, None]
    S = np.fft.rfft(x[idx] * w, axis=1)
    return (np.abs(S) ** 2).T                                   # (F, T)


def lsd(gt, est, n_fft, hop, fs, lo_hz=None, hi_hz=None, eps=1e-10):
    n = min(len(gt), len(est))
    G, E = stft_mag2(gt[:n], n_fft, hop), stft_mag2(est[:n], n_fft, hop)
    f = np.fft.rfftfreq(n_fft, 1.0 / fs)
    sel = np.ones(len(f), bool)
    if lo_hz is not None:
        sel &= f >= lo_hz
    if hi_hz is not None:
        sel &= f < hi_hz
    d = np.log10(G[sel] + eps) - np.log10(E[sel] + eps)
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=0))))


def snr_db(y, e):
    y = np.asarray(y, np.float64); e = np.asarray(e, np.float64)[:len(y)]
    return float(10 * np.log10(np.sum(y ** 2) / max(np.sum((y - e) ** 2), 1e-20)))


# ---- input pipelines --------------------------------------------------------------------------------
def down_poly(y, R):
    return sps.resample_poly(y, 1, R)                           # the repo's decimate(): what the model trained on


def down_cheby(y, R, fs_hi):
    """FLowHigh's evaluation input (data.py; issue #3): cheby1(8, 0.05 dB, (sr_in/2)/(fs_hi/2)) applied with
    sosfiltfilt, then scipy resample_poly down.  The extra low-pass sits inside resample_poly's own
    anti-alias filter, so the difference from --resampler poly is confined to the transition band."""
    sos = sps.cheby1(8, 0.05, (fs_hi / R / 2) / (fs_hi / 2), btype="lowpass", output="sos")
    return sps.resample_poly(sps.sosfiltfilt(sos, np.asarray(y, np.float64)), 1, R)


def down_sinc(y, R, fs_hi):
    """torchaudio.functional.resample defaults (sinc_interp_hann, width 6, rolloff 0.99), the filter
    NU-Wave / NVSR-style pipelines that call torchaudio or Kaldi resampling apply."""
    import torch, torchaudio.functional as F
    return F.resample(torch.from_numpy(np.asarray(y, np.float64)), fs_hi, fs_hi // R).numpy()


def up_naive(x_lo, R, n):
    return sps.resample_poly(x_lo, R, 1)[:n]


def split_bands(w, fs, f_cut):
    W = np.fft.rfft(np.asarray(w, np.float64)); k = int(round(f_cut * len(w) / fs))
    lo, hi = W.copy(), W.copy(); lo[k:] = 0; hi[:k] = 0
    return np.fft.irfft(lo, n=len(w)), np.fft.irfft(hi, n=len(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="~/results/ckpt/OV50")
    ap.add_argument("--tag", default="OV50")
    ap.add_argument("--out", default="~/results/ov3")
    ap.add_argument("--per-speaker", type=int, default=10, help="cached audit fixtures per speaker (8-10 are cached)")
    ap.add_argument("--speakers", nargs="*", default=list(PAPER_TEST))
    ap.add_argument("--basis", default="2048,512", help="n_fft,hop of the paper's LSD")
    ap.add_argument("--our-basis", default="1024,256")
    ap.add_argument("--hf-cut", type=float, default=None, help="Hz; default fs_lo / 2")
    ap.add_argument("--resampler", default="poly", choices=["poly", "cheby", "sinc"],
                    help="poly: resample_poly, what the model trained on; cheby: FLowHigh's order-8 Chebyshev I "
                         "(0.05 dB ripple) low-pass by sosfiltfilt, then resample_poly; sinc: torchaudio's windowed sinc")
    ap.add_argument("--visqol", default="pt", choices=["pt", "all", "none"])
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--n", type=int, default=None, help="first n utterances only (smoke)")
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()
    out_dir = pathlib.Path(a.out).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = pathlib.Path(a.ckpt_dir).expanduser().resolve()
    nf, hp = map(int, a.basis.split(",")); onf, ohp = map(int, a.our_basis.split(","))

    # Cached fixtures only.  vctk_fixtures.fetch() would go to the 11 GB VCTK archive over HTTP ranges
    # for any speaker with fewer than per_speaker files cached, silently and for minutes; an evaluation
    # script must never do that on its own.  Fetch more with audit/vctk_fixtures.py first if you want more.
    paths = [q for spk in a.speakers for q in sorted((DATA / spk).glob("*.flac"))[: a.per_speaker]]
    missing = [spk for spk in a.speakers if not (DATA / spk).is_dir()]
    if missing:
        print(f"no cached audio for {missing} under {DATA}; skipping them", flush=True)
    if not paths:
        raise SystemExit(f"nothing cached under {DATA} for {a.speakers}")
    if a.n:
        paths = paths[: a.n]
    from audit.boot import boot                                  # chdirs to a temp dir: all paths above are absolute
    G = boot()
    CFG = G["CFG"]; R, fs = CFG.upsample, CFG.fs_hi; fs_lo = fs // R
    hf = a.hf_cut if a.hf_cut else fs_lo / 2
    reconstruct, readout = G["reconstruct"], G["logmag_ensemble_readout"]
    # The high-band energy ratio, from the repo's own band_energy_ratio, so `deficit` here is the same
    # quantity as in the datasheet's other sections and directly comparable with fast/flowhigh_compare.py.
    # It is the column LSD cannot express: LSD charges a model for energy in the wrong place, deficit
    # asks whether it put any back, and on this task the two rank models differently.
    def deficit(y, w):
        b = G["band_energy_ratio"](y, w, CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop, CFG.fs_lo / 2, CFG.fs_hi / 2)
        return float(np.mean(b[:, 1])) if len(b) else None
    import torch
    models = {arm_name(p): G["load_arm"](p, CFG)[0] for p in sorted(ckpt_dir.glob("*.pt"))}
    if a.arms:
        models = {k: v for k, v in models.items() if k in a.arms}
    order = [k for k in ORDER if k in models] + sorted(set(models) - set(ORDER))
    print(f"{len(paths)} utterances from {sorted(set(p.parent.name for p in paths))}; arms {order}; "
          f"basis {nf}/{hp} (ours {onf}/{ohp}); HF cut {hf:.0f} Hz; resampler {a.resampler}; visqol {a.visqol}", flush=True)

    VQ = None
    if a.visqol != "none":
        from visqol import VisqolApi
        VQ = VisqolApi(); VQ.create(mode="audio")
        EV = None
        try:
            sys.path.insert(0, str(REPO)); import evaluation; EV = evaluation.Evaluator()
        except Exception as e:
            print("speech-mode evaluator unavailable:", repr(e), flush=True)
        try:
            import ai_edge_litert; sp_map = "lattice"
        except Exception:
            sp_map = "polynomial"

    def vq_audio(y, w):
        n = min(len(y), len(w))
        try:
            r = VQ.measure_from_arrays(np.asarray(y[:n], np.float64), np.asarray(w[:n], np.float64), sample_rate=fs)
            return float(r.moslqo), float(getattr(r, "vnsim", np.nan))
        except Exception:
            return np.nan, np.nan

    def vq_speech_fallback(y, w):
        """evaluation.py needs torchaudio; without it, the same measurement with scipy's resampler:
        48 -> 16 kHz by resample_poly, ViSQOL speech mode, pesq wb.  Speech-mode numbers made this way
        differ from OV2's in the third decimal (a different anti-aliasing filter), and _meta says so."""
        from pesq import pesq as _pesq
        if not hasattr(vq_speech_fallback, "api"):
            vq_speech_fallback.api = VisqolApi(); vq_speech_fallback.api.create(mode="speech")
        r16 = sps.resample_poly(np.asarray(y, np.float64), 1, 3); d16 = sps.resample_poly(np.asarray(w[:len(y)], np.float64), 1, 3)
        n = min(len(r16), len(d16)); r16, d16 = r16[:n], d16[:n]
        try:
            r = vq_speech_fallback.api.measure_from_arrays(r16, d16, sample_rate=16000); mos, ns = float(r.moslqo), float(getattr(r, "vnsim", np.nan))
        except Exception:
            mos, ns = np.nan, np.nan
        try:
            pq = float(_pesq(16000, r16.astype(np.float32), d16.astype(np.float32), "wb"))
        except Exception:
            pq = np.nan
        return mos, ns, pq

    def vq_speech(y, w):
        if EV is None:
            return vq_speech_fallback(y, w)
        yt, wt = torch.from_numpy(np.asarray(y, np.float32)), torch.from_numpy(np.asarray(w[:len(y)], np.float32))
        hr, sr_ = EV.sample_to_correct_rate(yt, wt, fs); hr, sr_ = EV.match_length(hr, sr_)
        try:
            r = EV.visqol_api.measure_from_arrays(hr.numpy().astype(np.float64), sr_.numpy().astype(np.float64), sample_rate=EV.target_sr)
            mos, ns = float(r.moslqo), float(getattr(r, "vnsim", np.nan))
        except Exception:
            mos, ns = np.nan, np.nan
        try:
            pq = float(EV.evaluate_pesq(yt[None], wt[None], current_sr=fs))
        except Exception:
            pq = np.nan
        return mos, ns, pq

    def score(y, w, perceptual):
        w = np.asarray(w, np.float64)[:len(y)]; w = np.pad(w, (0, len(y) - len(w)))
        row = {"lsd": lsd(y, w, nf, hp, fs), "lsd_hf": lsd(y, w, nf, hp, fs, lo_hz=hf), "lsd_lf": lsd(y, w, nf, hp, fs, hi_hz=hf),
               "lsd_ours": lsd(y, w, onf, ohp, fs), "snr": snr_db(y, w), "deficit": deficit(y, w)}
        if perceptual and VQ is not None:
            row["visqol_audio48k"], row["nsim_audio48k"] = vq_audio(y, w)
            row["visqol_speech16k"], row["nsim_speech16k"], row["pesq_wb"] = vq_speech(y, w)
        return row

    rows, t0 = {}, time.time()
    for i, p in enumerate(paths):
        utt = p.stem.replace("_mic1", "")
        y = load(p, fs); y = y[: (len(y) // R) * R]
        x_lo = {"poly": lambda: down_poly(y, R), "cheby": lambda: down_cheby(y, R, fs), "sinc": lambda: down_sinc(y, R, fs)}[a.resampler]()
        nv = up_naive(x_lo, R, len(y))
        lo_nv = split_bands(nv, fs, fs_lo / 2)[0]
        pt = lambda w: lo_nv + split_bands(np.pad(np.asarray(w, np.float64)[:len(y)], (0, max(0, len(y) - len(w[:len(y)])))), fs, fs_lo / 2)[1]
        # the model's own input path is decimate() inside reconstruct(); to feed it x_lo from another
        # resampler we go through encode/decode directly, exactly as reconstruct() does after decimation
        def run(m, tau, seed):
            xt = torch.from_numpy(np.asarray(x_lo, np.float32))[None]
            with torch.no_grad():
                eps = None if tau == 0 else m.sample_eps(xt, tau, seed)
                z = m.encode(xt, eps); n_out = xt.shape[1] * m.R
                w = torch.cat([m.decode(z, s, min(s + (1 << 15), n_out)) for s in range(0, n_out, 1 << 15)], 1)[0].numpy().astype(np.float64)
            return np.pad(w, (0, max(0, len(y) - len(w))))[: len(y)]
        conds = {"naive": nv, "floor: pt + empty HB": pt(np.zeros_like(y)), "ceiling: pt + true HB": pt(y)}
        for k in order:
            m = models[k]
            if m.tau == 0:
                w0 = run(m, 0.0, 0); conds[f"{k}"] = w0; conds[f"{k} | pt"] = pt(w0)
            else:
                draws = np.stack([run(m, 1.0, s) for s in range(a.M)])
                lm = readout(draws, y, CFG, passthrough=False)
                conds[f"{k} draw"] = draws[0]; conds[f"{k} draw | pt"] = pt(draws[0])
                conds[f"{k} logmean{a.M}"] = lm; conds[f"{k} logmean{a.M} | pt"] = pt(lm)
                conds[f"{k} tau0"] = run(m, 0.0, 0)
        for name, w in conds.items():
            perc = a.visqol == "all" or (a.visqol == "pt" and (name in ("naive",) or "pt" in name))
            rows.setdefault(name, []).append({"utt": utt, "seconds": len(y) / fs, **score(y, w, perc)})
        print(f"  [{time.time()-t0:5.0f}s] {i+1}/{len(paths)} {utt} {len(y)/fs:.2f}s", flush=True)

    keys = ["lsd", "lsd_hf", "lsd_lf", "lsd_ours", "snr", "deficit", "visqol_audio48k", "nsim_audio48k", "visqol_speech16k", "nsim_speech16k", "pesq_wb"]
    agg = {}
    for name, rs in rows.items():
        agg[name] = {}
        for k in keys:
            v = np.array([r.get(k, np.nan) for r in rs], np.float64); v = v[~np.isnan(v)]
            agg[name][k] = None if not len(v) else float(v.mean())
            agg[name][k + "_se"] = None if len(v) < 2 else float(v.std(ddof=1) / np.sqrt(len(v)))
    meta = {"tag": a.tag, "speakers": sorted(set(p.parent.name for p in paths)), "n_utts": len(paths),
            "seconds": float(sum(r["seconds"] for r in rows["naive"])), "basis": [nf, hp], "our_basis": [onf, ohp],
            "hf_cut_hz": hf, "resampler": a.resampler, "fs_lo": fs_lo, "fs_hi": fs, "M": a.M, "peak_norm": 0.95,
            "visqol": a.visqol, "visqol_speech_mapping": (sp_map if VQ is not None else None),
            "speech_resampler": (None if VQ is None else ("torchaudio (evaluation.py)" if EV is not None else "scipy resample_poly (fallback)")),
            "lsd": "log10(|S|^2+1e-10) difference, RMS over frequency within a frame, mean over frames; decades of power, not dB",
            "step": int(G["load_arm"](next(iter(sorted(ckpt_dir.glob('*.pt')))), CFG)[1]["step"])}
    stem = f"paperset_{a.tag}_{a.resampler}"
    json.dump({"_meta": meta, "agg": agg, "per_utt": rows}, open(out_dir / f"{stem}.json", "w"), indent=1)
    L = [f"# {a.tag} on the paper split ({', '.join(meta['speakers'])}; n = {meta['n_utts']}, {meta['seconds']:.0f} s)", "",
         f"LSD basis {nf}/{hp} (conventional, decades of power); HF/LF split at {hf:.0f} Hz; input by {a.resampler}; "
         f"M = {a.M}; ViSQOL audio mode at 48 kHz, speech mode ({meta['visqol_speech_mapping']}) at 16 kHz.", "",
         "| condition | LSD | LSD-HF | LSD-LF | deficit dB | LSD (1024/256) | SNR | ViSQOL audio | NSIM audio | ViSQOL speech | PESQ |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    f2 = lambda v, n=3: "--" if v is None else f"{v:.{n}f}"
    for name in rows:
        g = agg[name]
        cell = name.replace('|', '\\|')
        L.append(f"| {cell} | {f2(g['lsd'])} ± {f2(g['lsd_se'])} | {f2(g['lsd_hf'])} | {f2(g['lsd_lf'])} | {f2(g['deficit'], 2)} | {f2(g['lsd_ours'])} | "
                 f"{f2(g['snr'], 2)} | {f2(g['visqol_audio48k'], 2)} | {f2(g['nsim_audio48k'])} | {f2(g['visqol_speech16k'], 2)} | {f2(g['pesq_wb'], 2)} |")
    (out_dir / f"{stem}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print(f"\n-> {out_dir / (stem + '.json')}", flush=True)


if __name__ == "__main__":
    main()
