"""Add a x8 render (96 kHz) of every arm to the demo fixtures.

The decoder is continuous in its coordinate: q = j / R, so the same weights answer any R.  The
training and evaluation ran at R = 4 (12 -> 48 kHz).  The demo lets a listener ask for x8, which
the browser engine computes live; this precomputes the same thing so it can be played and A/B'd
without waiting, and records the one number that says whether x8 is honest: the share of output
energy above 24 kHz.  Nothing in training ever showed the decoder a query above 24 kHz, so any
energy there is invention, and the page prints the measured share rather than a claim.

Per speaker and arm: one draw at tau = 1, seed 0 (tau = 0 for a deterministic arm), R = 8, written
as <arm>_x8.wav at 96 kHz, merged into the audio manifest under files.arms[<arm>].x8 with the
above-24 kHz share under x8_above24.

    venv/bin/python demo/tools/add_x8.py --ckpt-dir ~/lisa-results/ckpt/OV50 --out web/public/assets/audio
"""
import argparse, json, pathlib, re, sys, time
import numpy as np
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
R8, CHUNK = 8, 1 << 15


def arm_name(p):
    return re.sub(r"_step\d+$", "", pathlib.Path(p).stem)


def render_x8(G, m, y, tau, seed):
    """encode once at 12 kHz, decode at R = 8: len(y) / 4 * 8 output samples at 96 kHz."""
    import torch
    CFG, DEVICE = G["CFG"], G["DEVICE"]
    x_lo = torch.from_numpy(G["decimate"](np.asarray(y, np.float64), CFG.upsample)).float()[None].to(DEVICE)
    # R is set BEFORE the noise is drawn: LISASD samples one decoder-noise vector per OUTPUT sample,
    # sized L * m.R, so a draw made at R = 4 is two chunks short at R = 8 (a shape error, caught once).
    R0, m.R = m.R, R8
    try:
        with torch.no_grad():
            eps = None if tau == 0 else m.sample_eps(x_lo, tau, seed)
            z = m.encode(x_lo, eps)
            n_out = x_lo.shape[1] * R8
            out = [m.decode(z, s, min(s + CHUNK, n_out)).squeeze(0).cpu().numpy() for s in range(0, n_out, CHUNK)]
    finally:
        m.R = R0
    return np.concatenate(out).astype(np.float64)


def above(w, fs, f_cut):
    W = np.abs(np.fft.rfft(w)) ** 2
    f = np.fft.rfftfreq(len(w), 1.0 / fs)
    return float(W[f >= f_cut].sum() / max(W.sum(), 1e-20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="~/lisa-results/ckpt/OV50")
    ap.add_argument("--out", default="web/public/assets/audio")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out = out if out.is_absolute() else (REPO / out).resolve()
    ckpt_dir = pathlib.Path(a.ckpt_dir).expanduser().resolve()
    man_path = out / "manifest.json"; man = json.loads(man_path.read_text())
    from audit.boot import boot
    from audit.vctk_fixtures import load as load_audio
    G = boot(); CFG = G["CFG"]; fs8 = CFG.fs_lo * R8
    models = {arm_name(p): G["load_arm"](p, CFG)[0] for p in sorted(ckpt_dir.glob("*.pt"))}
    t0 = time.time()
    for spk in man["speakers"]:
        d = out / spk["id"]; y = load_audio(d / "truth.wav", CFG.fs_hi)
        for arm, m in models.items():
            tau = 0.0 if m.tau == 0 else 1.0
            w = render_x8(G, m, y, tau, man.get("seed", 0))
            sf.write(d / f"{arm}_x8.wav", np.clip(w, -1, 1), fs8, subtype="PCM_16")
            e = spk["files"]["arms"].setdefault(arm, {})
            e["x8"] = f"{spk['id']}/{arm}_x8.wav"; e["x8_above24"] = round(above(w, fs8, 24000), 6)
        print(f"  {spk['id']}  {len(models)} arms at x8  above-24k share "
              + " ".join(f"{k}:{spk['files']['arms'][k]['x8_above24']*100:.3f}%" for k in models) + f"  [{time.time()-t0:.0f}s]", flush=True)
    man["x8"] = {"fs": fs8, "R": R8, "tau": man.get("tau_draw", 1.0), "seed": man.get("seed", 0),
                 "note": "same weights queried at R = 8; x8_above24 is the share of output energy above 24 kHz, "
                         "which training never asked for"}
    man_path.write_text(json.dumps(man, indent=1) + "\n")
    print(f"manifest updated: {sum(1 for _ in out.rglob('*.wav'))} wav files, {sum(p.stat().st_size for p in out.rglob('*.wav')):,} bytes")


if __name__ == "__main__":
    main()
