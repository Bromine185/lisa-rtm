"""Is a model trained only at 4x actually fixed to 4x?

The decoder saw four coordinate values and no others: c = 2(q - i) - 1 with q = j/4 gives
c in {-1, -0.5, 0, +0.5}, and the anchor jitter shifts the anchor by an INTEGER, so it only
supplies c - 2m -- the same four phases displaced.  c = +-0.25 and +-0.75 were never evaluated.
Query the same latents at 8x: half the samples land exactly there.

    venv/bin/python audit/scale_freedom.py <checkpoint.pt> [utterance.flac]

See notes/2026-09-17-snr-ceiling-gating-and-scale.md §3.
"""
import sys, pathlib
import numpy as np, torch, scipy.signal as sps
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from audit.boot import boot                                   # noqa: E402
from audit.vctk_fixtures import fetch, load, OUR_TEST         # noqa: E402


def main(ckpt, utt=None):
    G = boot()
    CFG, snr_db, decimate = G["CFG"], G["snr_db"], G["decimate"]
    y = load(utt if utt else fetch(OUR_TEST[:1], per_speaker=1)[0], CFG.fs_hi)
    m, ck = G["load_arm"](pathlib.Path(ckpt), CFG)
    print(f"arm {ck['arm']}  step {ck['step']}\n")

    x_lo = torch.from_numpy(decimate(y, CFG.upsample)).float()[None]      # the only input rate it knows
    L, R = x_lo.shape[1], CFG.upsample
    with torch.no_grad():
        z = m.encode(x_lo, None)                                          # tau = 0: same latents both ways
        def at(rate):
            m.R = rate
            return m.decode(z, 0, L * rate).squeeze(0).numpy().astype(np.float64)
        y4, y8 = at(R), at(2 * R)
    m.R = R
    y8_dn = sps.resample_poly(y8, 1, 2)[:len(y4)]
    n = min(len(y4), len(y8_dn))
    print("the same latents queried at 4x and at 8x, the 8x brought back to 48 kHz:")
    print(f"   agreement, 8x-then-decimate vs 4x    {snr_db(y4[:n], y8_dn[:n]):6.2f} dB")
    print(f"   SNR against the truth, 4x            {snr_db(y[:len(y4)], y4):6.2f} dB")
    print(f"   SNR against the truth, 8x-decimated  {snr_db(y[:n], y8_dn[:n]):6.2f} dB")
    X = np.fft.rfft(y8)
    fh = np.fft.rfftfreq(len(y8), 1 / (CFG.fs_hi * 2))
    above = (np.abs(X[fh >= CFG.fs_hi / 2]) ** 2).sum() / (np.abs(X) ** 2).sum()
    print(f"   energy the 8x query invents above {CFG.fs_hi//2000} kHz: {100*above:.2f} % of its total")
    print("   (so the decoder's continuous function is band-limited; querying ABOVE 4x is safe,")
    print("    querying BELOW it aliases, because the decoder has no idea what rate you asked for)")

    e = np.convolve(np.abs(x_lo.numpy()[0]), np.ones(200) / 200, "same")
    i = int(np.argmax(e))
    c = torch.linspace(-1.0, 1.0, 801)
    feat = torch.cat([c.view(1, -1, 1), z[:, i-1:i].expand(1, 801, -1),
                      z[:, i:i+1].expand(1, 801, -1), z[:, i+1:i+2].expand(1, 801, -1)], -1)
    with torch.no_grad():
        r = m.dec(feat).squeeze(0).numpy()
    seen = np.array([-1.0, -0.5, 0.0, 0.5])
    print(f"\ndecoder output sweeping c over [-1, 1] at latent {i} (loudest region):")
    for cv in np.arange(-1.0, 1.0, 0.25):
        j = int(round((cv + 1) / 2 * 800))
        print(f"   c = {cv:+5.2f}   f = {r[j]:+.5f}   "
              f"{'trained' if np.any(np.isclose(cv, seen)) else 'never seen'}")
    d2 = np.abs(np.diff(r, 2))
    near = np.zeros(len(d2), bool)
    for cv in seen:
        j = int(round((cv + 1) / 2 * 800))
        near[max(0, j - 8):min(len(d2), j + 8)] = True
    print(f"\ncurvature |d2f| near the trained coordinates {d2[near].mean():.3e}")
    print(f"curvature |d2f| in the gaps between them     {d2[~near].mean():.3e}")
    print(f"ratio (a spiky decoder would be >> 1)        {d2[~near].mean()/max(d2[near].mean(),1e-30):.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
