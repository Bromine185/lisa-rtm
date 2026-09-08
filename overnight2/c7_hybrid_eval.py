# ============================================================ OV2-7 baseband passthrough hybrids
# For bandwidth extension the band below the input Nyquist is GIVEN.  Hybrid = lowpass(naive upsampled
# input) + highpass(model output), brick-wall at fs_lo/2.  Separates two causes of the samplers' low
# speech-mode ViSQOL / PESQ: leakage of the noise pathway into the baseband (fixed by passthrough) versus
# the metric penalising sampled high-band texture (unchanged by passthrough).  Same metrics as c6.
import json, time, numpy as np, torch

def split_bands(w, fs, f_cut):
    W = np.fft.rfft(np.asarray(w, np.float64))
    k = int(round(f_cut * len(w) / fs))
    lo, hi = W.copy(), W.copy()
    lo[k:] = 0; hi[:k] = 0
    return np.fft.irfft(lo, n=len(w)), np.fft.irfft(hi, n=len(w))

def hybrid(y, cond_fn):
    nv = naive_upsample(y, CFG)[:len(y)]
    out = np.asarray(cond_fn(y), np.float64)[:len(y)]
    out = np.pad(out, (0, len(y) - len(out)))
    lo, _ = split_bands(nv, CFG.fs_hi, CFG.fs_lo / 2)
    _, hi = split_bands(out, CFG.fs_hi, CFG.fs_lo / 2)
    return lo + hi

PICK = ["naive", "det", "lam1e-1", "wide_det", "det_split", "es_marg tau=1", "es_marg tau=0.75", "es_marg mean16",
        "ladder S on det", "ladder T1 on det"]
PICK = [c for c in PICK if c in CONDITIONS]
HYB = {f"{c} | passthrough": (lambda fn: lambda y: hybrid(y, fn))(CONDITIONS[c]) for c in PICK}
# plus the lower bound: passthrough baseband and NOTHING above (= brick-wall naive) and the upper bound: truth high band
HYB["passthrough + zero HB"] = lambda y: split_bands(naive_upsample(y, CFG)[:len(y)], CFG.fs_hi, CFG.fs_lo / 2)[0]
HYB["passthrough + TRUE HB (oracle)"] = lambda y: split_bands(naive_upsample(y, CFG)[:len(y)], CFG.fs_hi, CFG.fs_lo / 2)[0] + split_bands(y, CFG.fs_hi, CFG.fs_lo / 2)[1]
print(f"{len(HYB)} hybrid conditions:", list(HYB), flush=True)

PER_H = {s: {c: [] for c in HYB} for s in SETS}
t0 = time.time()
for sname, utts in SETS.items():
    for ui, y in enumerate(utts):
        for cname, fn in HYB.items():
            PER_H[sname][cname].append(score(y, fn(y)))
        print(f"  {sname}: {ui+1}/{len(utts)} utts  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(PER_H, open(OUT / "visqol_hybrid_per_utt.json", "w"))
for sname in SETS:
    print(f"\n=== hybrids, {sname} (n={len(SETS[sname])}) ===")
    print(f"{'condition':<36}{'SNR':>7}{'LSD':>7}{'HB-LSD':>8}{'ViSQOL-sp16k':>13}{'NSIM-sp':>8}{'PESQ-wb':>9}{'ViSQOL-au48k':>13}{'NSIM-au':>8}")
    for c, rows in sorted(PER_H[sname].items(), key=lambda kv: np.nanmean([r["lsd"] for r in kv[1]])):
        a = {k: float(np.nanmean([r[k] for r in rows])) for k in KEYS}
        print(f"{c:<36}{a['snr']:7.2f}{a['lsd']:7.3f}{a['hb_lsd']:8.3f}{a['visqol_speech16k']:13.3f}{a['nsim_speech16k']:8.3f}{a['pesq_wb']:9.3f}{a['visqol_audio48k']:13.3f}{a['nsim_audio48k']:8.3f}")
print("HYBRID EVAL DONE", flush=True)
