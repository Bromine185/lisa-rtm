"""Does the high band gate with the speech, and would a low-pass help?  (It would not.)

`deficit` is the MEAN over third-octave bands and over frames.  Both averages hide things a
listener hears.  This splits the frames by loudness -- real high-band energy fires on fricatives
and vanishes in the gaps -- and prices a brick-wall low-pass on both SNR and deficit.

    venv/bin/python audit/band_gating.py <checkpoint.pt> [utterance.flac]

See notes/2026-09-17-snr-ceiling-gating-and-scale.md §2.
"""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from audit.boot import boot                                   # noqa: E402
from audit.vctk_fixtures import fetch, load, OUR_TEST         # noqa: E402


def main(ckpt, utt=None):
    if not pathlib.Path(ckpt).exists():       # boot() execs the whole notebook; fail before that
        sys.exit(f"no checkpoint at {ckpt}")
    G = boot()
    CFG, snr_db = G["CFG"], G["snr_db"]
    y = load(utt if utt else fetch(OUR_TEST[:1], per_speaker=1)[0], CFG.fs_hi)
    m, ck = G["load_arm"](pathlib.Path(ckpt), CFG)
    print(f"arm {ck['arm']}  step {ck['step']}  on {len(y)/CFG.fs_hi:.2f} s\n")
    draw = G["reconstruct"](m, y, CFG, tau=1.0, seed=0)
    det = G["reconstruct"](m, y, CFG, tau=0.0, seed=0)
    ber = lambda yh, msk=None: G["band_energy_ratio"](y, yh, CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop,
                                                      CFG.fs_lo / 2, CFG.fs_hi / 2, frame_mask=msk)

    S, _, _ = G["stft"](y, CFG.eval_n_fft, CFG.eval_hop)
    fe = 20 * np.log10(np.abs(S).sum(1) + 1e-12)
    f = np.fft.rfftfreq(CFG.eval_n_fft, 1 / CFG.fs_hi)
    masks = [("loud   (top 25%)", fe >= np.percentile(fe, 75)),
             ("mid    (25-75%)", (fe >= np.percentile(fe, 25)) & (fe < np.percentile(fe, 75))),
             ("quiet  (bottom 25%)", fe < np.percentile(fe, 25))]
    print(f"{'frames':<22} {'HB energy ratio':>16}   {'target HB share of frame':>26}")
    for label, msk in masks:
        Sy = np.abs(S[msk])
        share = (Sy[:, f >= CFG.fs_lo / 2] ** 2).sum() / (Sy ** 2).sum()
        print(f"{label:<22} {np.mean(ber(draw, msk)[:, 1]):+13.2f} dB {100*share:24.3f} %")
    print("A sampler that gates correctly shows the SAME ratio in every row.  A ratio that RISES as")
    print("the frames get quieter is energy where the truth has none -- the hiss, which no filter removes.")

    print(f"\n{'band (Hz)':>12}  {'draw':>8}  {'tau=0':>8}   target share of total energy")
    P = np.abs(np.fft.rfft(y)) ** 2
    ff = np.fft.rfftfreq(len(y), 1 / CFG.fs_hi)
    for (c, d), (_, d0) in zip(ber(draw), ber(det)):
        lo, hi = c / 2 ** (1 / 6), c * 2 ** (1 / 6)
        print(f"{c:11.0f}   {d:+7.2f}  {d0:+7.2f}   {100*P[(ff>=lo)&(ff<hi)].sum()/P.sum():9.4f} %")

    def lp(x, fc):
        X = np.fft.rfft(x)
        X[np.fft.rfftfreq(len(x), 1 / CFG.fs_hi) >= fc] = 0
        return np.fft.irfft(X, len(x))
    quiet = masks[2][1]
    print(f"\nlow-passing the draw:\n{'cutoff':>10} {'SNR':>8} {'deficit':>9} {'HB ratio, quiet frames':>24}")
    for fc in (CFG.fs_hi // 2, 20000, 16000, 12000, 8000):
        z = draw if fc >= CFG.fs_hi // 2 else lp(draw, fc)
        tag = "  (unfiltered)" if fc >= CFG.fs_hi // 2 else ""
        print(f"{fc/1000:9.0f}k {snr_db(y, z):7.2f} {np.mean(ber(z)[:,1]):+8.2f} "
              f"{np.mean(ber(z, quiet)[:,1]):+23.2f}{tag}")
    print("\nfiltering moves SNR UP and the deficit DOWN: the wrong direction on both axes at once.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
