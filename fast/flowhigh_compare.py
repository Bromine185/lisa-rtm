"""Run FLowHigh's released checkpoint on OUR utterances, and score it with OUR scorer.

    venv/bin/python fast/flowhigh_compare.py --fh-dir <FLowHigh_code> --ckpt <FLowHigh_indep_adaptive_400k.pt>
        [--resampler cheby] [--speakers p360 ...] [--per-speaker 10] [--device cpu] [--n 4]

WHY.  `fast/paper_split_eval.py` puts our arms on the literature's speakers, basis and input filter, but
the FLowHigh column in that comparison is still a number typed from their paper, measured on their
utterances, their ViSQOL build and an LSD whose STFT they never state.  Three unknowns sit between the
two columns.  This removes all three: their released weights, our utterances, one scorer -- the very
`lsd()` this file imports from paper_split_eval, so the two tables cannot drift.

WHAT IS THEIRS AND UNTOUCHED.  Their `inference.py` is run as a subprocess with the flags their README
gives, so their mel front end, their single Euler step, their BigVGAN and their STFT-domain
post-processing all run as shipped.  The only edit is the device: the repo hard-codes `.cuda()` in five
files and `map_location="cuda"` in one, so `patch_repo()` rewrites those to a device read from
FLOWHIGH_DEVICE.  It is idempotent, it prints every line it changes, and it touches nothing else.

THE GAIN TRAP, and why two LSD columns.  LSD is NOT scale invariant: scaling an estimate by g shifts
every bin of `log10|S|^2` by `2 log10 g`, which does not cancel in the per-frame RMS.  Their
post-processing ends with `audio / |audio|.max() * 0.99` while our references are peak-normalised to
0.95 by the corpus loader, so a raw comparison charges them a constant this repo's own conditions never
pay.  Both are reported: `lsd` as the file comes out, and `lsd_gain` after rescaling the estimate to the
reference's RMS.  Quote `lsd_gain` when comparing models; quote `lsd` only to show the offset is small.

WHAT IS STILL NOT MATCHED, and cannot be from here: FLowHigh held out the NU-Wave 2 test speakers
(p360-p376, s5) and ours held out every speaker >= p350, so these speakers are unseen by both; their reported numbers use their
full test set including s5, which is not cached here.  This script compares the two MODELS on one set of
utterances.  It does not reproduce their published table.

Output: <out>/flowhigh_<tag>_<resampler>.json and .md, beside paperset_<tag>_<resampler>.json.
"""
import argparse, json, os, pathlib, re, shutil, subprocess, sys, tempfile, time
import numpy as np
import scipy.signal as sps
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from audit.vctk_fixtures import DATA, PAPER_TEST, load                      # noqa: E402
from fast.paper_split_eval import down_cheby, down_poly, lsd, snr_db, split_bands, up_naive   # noqa: E402

# (file, pattern, replacement) -- every CUDA-bound site in the released tree, listed rather than
# globbed so a future version of their repo fails loudly here instead of silently running unpatched.
PATCHES = [
    ("cfm_superresolution.py", r"\.cuda\(\)", ".to(_FH_DEVICE)"),
    ("utils.py",               r"\.cuda\(\)", ".to(_FH_DEVICE)"),
    ("postprocessing.py",      r"\.cuda\(rank\)", ".to(_FH_DEVICE)"),
    ("inference.py",           r"\.cuda\(\)", ".to(_FH_DEVICE)"),
    ("init_vocoder.py",        r"\.cuda\(\)", ".to(_FH_DEVICE)"),
    ("init_vocoder.py",        r'map_location="cuda"', "map_location=_FH_DEVICE"),
    ("inference.py",           r"torch\.device\('cuda' if torch\.cuda\.is_available\(\) else 'cpu'\)", "_FH_DEVICE"),
    ("cfm_superresolution.py", r"device = self\.device\)\.to\(_FH_DEVICE\)", "device = _FH_DEVICE)"),
]
# API compatibility, NOT behaviour: librosa made filters.mel keyword-only in 0.10 and their
# requirements pin 0.9.2. The filterbank is identical; only the call spelling changes. Kept separate
# from the device list so it is obvious nothing numeric is being altered.
COMPAT = [
    ("cfm_superresolution.py",
     r"librosa_mel_fn\(self\.sampling_rate, self\.n_fft, self\.n_mels, self\.f_min, self\.f_max\)",
     "librosa_mel_fn(sr=self.sampling_rate, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)"),
    ("vocoder/BIGVGAN/bigvgan/meldataset.py",
     r"librosa_mel_fn\(sampling_rate, n_fft, num_mels, fmin, fmax\)",
     "librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)"),
]
SHIM = '''"""Device for this checkout, written by fast/flowhigh_compare.py. The upstream code hard-codes
.cuda(); every such call now goes through this. FLOWHIGH_DEVICE selects it (default cpu)."""
import os, torch
_FH_DEVICE = torch.device(os.environ.get("FLOWHIGH_DEVICE", "cpu"))
'''


def patch_repo(fh, verbose=True):
    """Rewrite the hard-coded CUDA calls to a configurable device. Idempotent."""
    (fh / "_fh_device.py").write_text(SHIM)
    changed = {}
    for name, pat, rep in PATCHES + COMPAT:
        p = fh / name
        if not p.exists():
            raise SystemExit(f"{p} missing: is --fh-dir a FLowHigh_code checkout?")
        src = p.read_text()
        new, n = re.subn(pat, rep, src)
        if n:
            changed[name] = changed.get(name, 0) + n
        if "from _fh_device import _FH_DEVICE" not in new and "_FH_DEVICE" in new:
            lines = new.split("\n")
            i = max((j for j, l in enumerate(lines[:40]) if l.startswith(("import ", "from "))), default=0)
            lines.insert(i + 1, "from _fh_device import _FH_DEVICE")
            new = "\n".join(lines)
        if new != src:
            p.write_text(new)
    if verbose:
        print("device patch:", ", ".join(f"{k} x{v}" for k, v in changed.items()) or "already applied", flush=True)
    return changed


def rms_align(ref, est):
    """Scale est to the reference's RMS. LSD is not scale invariant and their pipeline peak-normalises
    to 0.99 while our references sit at 0.95, so this removes a constant they would otherwise be charged."""
    n = min(len(ref), len(est))
    a, b = np.asarray(ref[:n], np.float64), np.asarray(est[:n], np.float64)
    g = np.sqrt(np.sum(a ** 2) / max(np.sum(b ** 2), 1e-20))
    return b * g, float(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fh-dir", required=True, help="a clone of github.com/jjunak-yun/FLowHigh_code")
    ap.add_argument("--ckpt", required=True, help="FLowHigh_indep_adaptive_400k.pt")
    ap.add_argument("--vocoder", default=None, help="default: <fh-dir>/vocoder/BIGVGAN/checkpoint/g_48_00850000")
    ap.add_argument("--vocoder-config", default=None)
    ap.add_argument("--python", default=sys.executable, help="interpreter with their deps installed")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--resampler", default="cheby", choices=["cheby", "poly"])
    ap.add_argument("--speakers", nargs="*", default=list(PAPER_TEST))
    ap.add_argument("--per-speaker", type=int, default=10)
    ap.add_argument("--n", type=int, default=None, help="first n utterances only")
    ap.add_argument("--tag", default="OV50")
    ap.add_argument("--out", default="~/lisa-results/ov3")
    ap.add_argument("--basis", default="2048,512")
    ap.add_argument("--hf-cut", type=float, default=6000.0)
    ap.add_argument("--time-step", type=int, default=1)
    ap.add_argument("--ode-method", default="euler", choices=["euler", "midpoint"])
    ap.add_argument("--keep-audio", default=None, help="directory to keep the generated wavs in")
    ap.add_argument("--visqol", action="store_true", default=True)
    ap.add_argument("--no-visqol", dest="visqol", action="store_false")
    ap.add_argument("--score-only", metavar="DIR", default=None,
                    help="skip inference and score the wavs already in DIR/out/output (a previous --keep-audio)")
    a = ap.parse_args()

    fh = pathlib.Path(a.fh_dir).expanduser().resolve()
    ckpt = pathlib.Path(a.ckpt).expanduser().resolve()
    voc = pathlib.Path(a.vocoder).expanduser() if a.vocoder else fh / "vocoder/BIGVGAN/checkpoint/g_48_00850000"
    vcfg = pathlib.Path(a.vocoder_config).expanduser() if a.vocoder_config else fh / "vocoder/BIGVGAN/config/bigvgan_48khz_256band_config.json"
    for p in (ckpt, voc, vcfg):
        if not p.exists():
            raise SystemExit(f"missing: {p}")
    out_dir = pathlib.Path(a.out).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)
    nf, hp = map(int, a.basis.split(","))
    FS, R = 48000, 4

    if not a.score_only:
        patch_repo(fh)

    paths = [q for spk in a.speakers for q in sorted((DATA / spk).glob("*.flac"))[: a.per_speaker]]
    if a.n:
        paths = paths[: a.n]
    if not paths:
        raise SystemExit(f"nothing cached under {DATA} for {a.speakers}")
    print(f"{len(paths)} utterances; input by {a.resampler}; device {a.device}; "
          f"{a.time_step} step(s), {a.ode_method}", flush=True)

    work = (pathlib.Path(a.score_only).expanduser() if a.score_only
            else pathlib.Path(a.keep_audio).expanduser() if a.keep_audio
            else pathlib.Path(tempfile.mkdtemp(prefix="flowhigh_")))
    lo_dir = work / "input_12k"; hi_dir = work / "out"
    truth, names = {}, []
    if not a.score_only:
        lo_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        utt = p.stem.replace("_mic1", "")
        y = load(p, FS); y = y[: (len(y) // R) * R]
        truth[utt] = y; names.append(utt)
        if a.score_only:
            continue
        x_lo = down_cheby(y, R, FS) if a.resampler == "cheby" else down_poly(y, R)
        sf.write(str(lo_dir / f"{utt}.wav"), np.clip(x_lo, -1, 1), FS // R, subtype="PCM_16")

    cmd = [a.python, "inference.py", "--input_path", str(lo_dir), "--output_path", str(hi_dir),
           "--target_sampling_rate", "48000", "--up_sampling_method", "scipy", "--architecture", "transformer",
           "--time_step", str(a.time_step), "--cfm_method", "independent_cfm_adaptive", "--ode_method", a.ode_method,
           "--sigma", "0.0001", "--model_path", str(ckpt), "--n_layers", "2", "--n_heads", "16", "--dim_head", "64",
           "--n_mels", "256", "--f_max", "24000", "--n_fft", "2048", "--win_length", "2048", "--hop_length", "480",
           "--vocoder", "bigvgan", "--vocoder_path", str(voc), "--vocoder_config_path", str(vcfg)]
    env = {**os.environ, "FLOWHIGH_DEVICE": a.device, "PYTHONPATH": str(fh)}
    params, dt = None, float("nan")
    if a.score_only:
        print(f"scoring the wavs already in {hi_dir / 'output'}; no inference run", flush=True)
    else:
        print("running their inference.py ...", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(fh), env=env, capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode != 0:
            print((r.stdout or "")[-1500:] + (r.stderr or "")[-2500:], flush=True)
            raise SystemExit(f"their inference.py exited {r.returncode}")
        m = re.search(r"Total number of parameters: ([\d.]+) million", r.stdout or "")
        if m:
            params = float(m.group(1)) * 1e6
        print(f"done in {dt:.0f} s ({dt / max(sum(len(v) for v in truth.values()) / FS, 1e-9):.3f} RTF overall)"
              + (f"; their script reports {params / 1e6:.2f}M parameters" if params else ""), flush=True)

    # the high-band energy ratio, from the repo's own band_energy_ratio, so the `deficit` column here is
    # the same quantity as everywhere else in the datasheet. LSD punishes a model for putting energy back
    # in the wrong place; deficit says whether it put any back at all, and the two disagree by design.
    from audit.boot import boot
    G = boot()
    CFG = G["CFG"]
    def deficit(y, w):
        b = G["band_energy_ratio"](y, w, CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop, CFG.fs_lo / 2, CFG.fs_hi / 2)
        return float(np.mean(b[:, 1])) if len(b) else None

    VQ = None
    if a.visqol:
        try:
            from visqol import VisqolApi
            VQ = VisqolApi(); VQ.create(mode="audio")
        except Exception as e:
            print("ViSQOL unavailable:", repr(e), flush=True)

    def vq_audio(y, w):
        if VQ is None:
            return None, None
        n = min(len(y), len(w))
        try:
            res = VQ.measure_from_arrays(np.asarray(y[:n], np.float64), np.asarray(w[:n], np.float64), sample_rate=FS)
            return float(res.moslqo), float(getattr(res, "vnsim", np.nan))
        except Exception:
            return None, None

    gen_dir = hi_dir / "output"
    rows = []
    for utt in names:
        f = gen_dir / f"{utt}.wav"
        if not f.exists():
            print(f"  missing output for {utt}", flush=True); continue
        w, sr = sf.read(str(f), dtype="float64")
        if sr != FS:
            raise SystemExit(f"{f}: {sr} Hz, expected {FS}")
        y = truth[utt]
        n = min(len(y), len(w)); y2, w = y[:n], np.asarray(w)[:n]
        wg, g = rms_align(y2, w)
        row = {"utt": utt, "seconds": n / FS, "gain": g,
               "lsd": lsd(y2, w, nf, hp, FS), "lsd_hf": lsd(y2, w, nf, hp, FS, lo_hz=a.hf_cut), "lsd_lf": lsd(y2, w, nf, hp, FS, hi_hz=a.hf_cut),
               "lsd_gain": lsd(y2, wg, nf, hp, FS), "lsd_hf_gain": lsd(y2, wg, nf, hp, FS, lo_hz=a.hf_cut),
               "lsd_lf_gain": lsd(y2, wg, nf, hp, FS, hi_hz=a.hf_cut), "snr": snr_db(y2, wg),
               "deficit": deficit(y2, w), "deficit_gain": deficit(y2, wg)}
        va, na = vq_audio(y2, w); row["visqol_audio48k"], row["nsim_audio48k"] = va, na
        vag, nag = vq_audio(y2, wg); row["visqol_audio48k_gain"], row["nsim_audio48k_gain"] = vag, nag
        rows.append(row)
        print(f"  {utt} {row['seconds']:.2f}s  LSD {row['lsd']:.3f} (gain-aligned {row['lsd_gain']:.3f})"
              + f"  deficit {row['deficit']:+.2f} dB" + (f"  ViSQOL {va:.2f}" if va else ""), flush=True)

    keys = [k for k in rows[0] if k not in ("utt",)] if rows else []
    agg = {}
    for k in keys:
        v = np.array([r.get(k) if r.get(k) is not None else np.nan for r in rows], np.float64); v = v[~np.isnan(v)]
        agg[k] = None if not len(v) else float(v.mean())
        agg[k + "_se"] = None if len(v) < 2 else float(v.std(ddof=1) / np.sqrt(len(v)))
    meta = {"tag": a.tag, "model": "FLowHigh indep_adaptive 400k", "checkpoint": ckpt.name, "vocoder": voc.name,
            "device": a.device, "time_step": a.time_step, "ode_method": a.ode_method, "params": params,
            "wall_seconds": dt, "audio_seconds": float(sum(r["seconds"] for r in rows)),
            "rtf": dt / max(sum(r["seconds"] for r in rows), 1e-9),
            "speakers": sorted({r["utt"].split("_")[0] for r in rows}), "n_utts": len(rows),
            "resampler": a.resampler, "basis": [nf, hp], "hf_cut_hz": a.hf_cut,
            "scorer": "fast/paper_split_eval.py::lsd -- identical to the paperset tables",
            "gain": "their post-processing peak-normalises to 0.99, our references to 0.95; LSD is not scale "
                    "invariant, so *_gain rescales the estimate to the reference RMS. Compare with *_gain.",
            "source": "github.com/jjunak-yun/FLowHigh_code, device-patched by fast/flowhigh_compare.py"}
    stem = f"flowhigh_{a.tag}_{a.resampler}"
    json.dump({"_meta": meta, "agg": agg, "per_utt": rows}, open(out_dir / f"{stem}.json", "w"), indent=1)
    f3 = lambda k, n=3: "--" if agg.get(k) is None else f"{agg[k]:.{n}f}"
    L = [f"# FLowHigh on our {len(rows)} utterances ({', '.join(meta['speakers'])})", "",
         f"Their released `{ckpt.name}` and BigVGAN, their `inference.py` ({a.time_step} step, {a.ode_method}), "
         f"device-patched only. Input by {a.resampler}. Scored by `fast/paper_split_eval.py::lsd`, basis {nf}/{hp}, "
         f"HF/LF at {a.hf_cut:.0f} Hz.", "",
         "| | LSD | LSD-LF | LSD-HF | deficit dB | ViSQOL audio | SNR |", "|---|---|---|---|---|---|---|",
         f"| as output (peak 0.99) | {f3('lsd')} | {f3('lsd_lf')} | {f3('lsd_hf')} | {f3('deficit', 2)} | {f3('visqol_audio48k', 2)} | -- |",
         f"| gain-aligned to the reference | {f3('lsd_gain')} ± {f3('lsd_gain_se')} | {f3('lsd_lf_gain')} | {f3('lsd_hf_gain')} | {f3('deficit_gain', 2)} | {f3('visqol_audio48k_gain', 2)} | {f3('snr', 2)} |",
         "", f"Mean applied gain {f3('gain', 4)}. Wall {dt:.0f} s for {meta['audio_seconds']:.0f} s of audio on "
         f"{a.device}, RTF {meta['rtf']:.2f}" + (f"; {params/1e6:.2f}M parameters plus the vocoder." if params else "."), ""]
    (out_dir / f"{stem}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print(f"-> {out_dir / (stem + '.json')}", flush=True)
    # Only a directory this run created may be removed. --score-only is handed an existing directory and
    # must never delete it; an early version did, and destroyed the audio it had just been asked to score.
    if a.keep_audio or a.score_only:
        print(f"   audio kept in {work}", flush=True)
    else:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
