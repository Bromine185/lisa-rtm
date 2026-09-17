#!/usr/bin/env python
"""Write PyTorch reference outputs for demo/tools/validate_engine.mjs.

    venv/bin/python demo/tools/make_testvec.py [--ckpt lisa_rtm_cache/ckpt/es_erb_l0.1_step13500.pt]
                                               [--flac lisa_rtm_cache/audit/p236/p236_002_mic1.flac]
                                               [--seconds 0.5] [--no-synth]

Writes demo/tools/testvec_<arm>.json with
    x12k          0.5 s of the utterance at 12 kHz (float32 values, the engine's input)
    y_r4_tau0     m.decode(m.encode(x, None), 0, 4L)  at R = 4, tau = 0, perturb = False
    y_r8_tau0     the same latents with m.R = 8, m.decode(z, 0, 8L)   (audit/scale_freedom.py)
    y_r4_tau1_s0  tau = 1, seed = 0, with the noise the JS engine draws (mulberry32 + Box-Muller, ported below)
    fs, fs_out, R, arm, source, offset

Unless --no-synth: also builds a synthetic LISASD (the LISAS weights plus random decoder-noise columns) in
lisa_rtm_cache/demo_synth/, exports it there with export_weights.py, and writes a matching test vector,
so the n_dec > 0 path is checked even though no trained LISASD checkpoint is on this machine.
"""
import argparse, json, math, pathlib, subprocess, sys

import numpy as np
import scipy.signal as sps
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from audit.boot import boot            # noqa: E402
from audit.vctk_fixtures import load   # noqa: E402


# ---- the engine's RNG, ported bit for bit (see engine.js mulberry32 / gaussFill) ----
def mulberry32(seed):
    a = int(seed) & 0xFFFFFFFF

    def imul(x, y):
        return (x * y) & 0xFFFFFFFF

    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = imul(a ^ (a >> 15), (1 | a) & 0xFFFFFFFF)
        t = ((t + imul(t ^ (t >> 7), (61 | t) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return (t ^ (t >> 14)) / 4294967296.0
    return nxt


def gauss_fill(rng, n, tau):
    out = np.empty(n, np.float32)
    for i in range(0, n, 2):
        u1, u2 = 1.0 - rng(), rng()
        r, th = math.sqrt(-2.0 * math.log(u1)), 6.283185307179586 * u2
        out[i] = np.float32(tau * r * math.cos(th))
        if i + 1 < n:
            out[i + 1] = np.float32(tau * r * math.sin(th))
    return out


def engine_noise(seed, tau, n_noise, L, R, n_dec):
    rng = mulberry32(seed)
    e = gauss_fill(rng, n_noise * L, tau).reshape(n_noise, L)
    d = gauss_fill(rng, L * R * n_dec, tau).reshape(L * R, n_dec) if n_dec > 0 else None
    return e, d


# ---- compact float lists ----
def f32_list(a):
    '''Shortest decimal that round-trips each float32 (JS parses to double, stores to Float32Array).'''
    return [float(np.format_float_positional(np.float32(v), unique=True, trim="-")) for v in a]


def f7_list(a):
    return [float("%.7g" % v) for v in a]


def dump(path, obj):
    path.write_text(json.dumps(obj, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


@torch.no_grad()
def references(m, x12k, R, seed=0):
    x = torch.from_numpy(x12k.astype(np.float32))[None]
    L = x.shape[1]
    R0 = m.R
    n_dec = getattr(m, "n_dec", 0)
    z = m.encode(x, None)
    m.R = R;     y4 = m.decode(z, 0, L * R).squeeze(0).numpy()
    m.R = 2 * R; y8 = m.decode(z, 0, L * 2 * R).squeeze(0).numpy()
    m.R = R
    e, d = engine_noise(seed, 1.0, m.n_noise, L, R, n_dec)
    eps = torch.from_numpy(e)[None]
    if n_dec > 0:
        eps = (eps, torch.from_numpy(d)[None])
    y1 = m.decode(m.encode(x, eps), 0, L * R).squeeze(0).numpy()
    m.R = R0
    return y4, y8, y1


def pick_window(x12k, n):
    '''The n-sample window with the most energy, aligned to 4 samples.'''
    e = np.convolve(x12k ** 2, np.ones(n), "valid")
    i = int(np.argmax(e)) & ~3
    return x12k[i:i + n], i


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(REPO / "lisa_rtm_cache/ckpt/es_erb_l0.1_step13500.pt"))
    ap.add_argument("--flac", default=str(REPO / "lisa_rtm_cache/audit/p236/p236_002_mic1.flac"))
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--no-synth", action="store_true")
    args = ap.parse_args()

    G = boot()
    CFG = G["CFG"]
    R, fs_hi = CFG.upsample, CFG.fs_hi
    fs_lo = fs_hi // R
    ckpt = pathlib.Path(args.ckpt)
    m, ck = G["load_arm"](ckpt, CFG)
    arm = pathlib.Path(ckpt).stem
    arm = arm[:arm.rfind("_step")] if "_step" in arm else arm

    y = load(args.flac, fs_hi)
    x_all = sps.resample_poly(y, 1, R)
    n = int(round(args.seconds * fs_lo))
    x12k, off = pick_window(x_all, n)
    x12k = x12k.astype(np.float32)
    print(f"{arm}: {type(m).__name__} n_noise {m.n_noise} n_dec {getattr(m, 'n_dec', 0)}; "
          f"window {off / fs_lo:.3f}-{(off + n) / fs_lo:.3f} s of {pathlib.Path(args.flac).name}")

    y4, y8, y1 = references(m, x12k, R)
    dump(REPO / "demo/tools" / f"testvec_{arm}.json", {
        "arm": arm, "cls": type(m).__name__, "n_noise": m.n_noise, "n_dec": getattr(m, "n_dec", 0),
        "fs": fs_lo, "fs_out": fs_hi, "R": R, "source": str(pathlib.Path(args.flac).relative_to(REPO)),
        "offset": off, "seed": 0,
        "x12k": f32_list(x12k), "y_r4_tau0": f7_list(y4), "y_r8_tau0": f7_list(y8), "y_r4_tau1_s0": f7_list(y1),
    })

    if args.no_synth:
        return
    # ---- synthetic LISASD: the trained LISAS plus random (small) decoder-noise columns ----
    synth = REPO / "lisa_rtm_cache/demo_synth"
    synth.mkdir(parents=True, exist_ok=True)
    msd = G["LISASD"](CFG, n_noise=m.n_noise, n_dec=G["N_DEC"])
    G["copy_shared"](m, msd)
    with torch.no_grad():
        g = torch.Generator().manual_seed(1234)
        W1 = msd.dec.net[0].weight
        W1[:, 1 + 3 * msd.dim:] = 0.05 * torch.randn(W1.shape[0], msd.n_dec, generator=g)
    msd.eval()
    sd_path = synth / "synth_lisasd_step0.pt"
    torch.save({"model": msd.state_dict(), "step": 0, "arm": ("es_synth", 0.1, "LISASD"), "n_noise": msd.n_noise,
                "n_dec": msd.n_dec, "cls": "LISASD", "tag": "demo_synth"}, sd_path)
    subprocess.run([sys.executable, str(REPO / "demo/tools/export_weights.py"), str(sd_path),
                    "--out", str(synth / "weights")], check=True)
    xs = x12k[: (n // 2) & ~3]
    y4, y8, y1 = references(msd, xs, R)
    # tau = 0 must equal the LISAS output on the same input (copy_shared's contract)
    y4_lisas = references(m, xs, R)[0]
    assert np.max(np.abs(y4 - y4_lisas)) < 1e-6, "copy_shared broke tau=0 equality"
    dump(synth / "testvec_synth_lisasd.json", {
        "arm": "synth_lisasd", "cls": "LISASD", "n_noise": msd.n_noise, "n_dec": msd.n_dec,
        "fs": fs_lo, "fs_out": fs_hi, "R": R, "source": str(pathlib.Path(args.flac).relative_to(REPO)),
        "offset": off, "seed": 0,
        "x12k": f32_list(xs), "y_r4_tau0": f7_list(y4), "y_r8_tau0": f7_list(y8), "y_r4_tau1_s0": f7_list(y1),
    })


if __name__ == "__main__":
    main()
