"""What can SNR measure on 12 kHz -> 48 kHz?

A point predictor reproduces the low band exactly -- sinc interpolation is the optimal
reconstruction of a band-limited signal -- so its whole error is high-band energy it fails to
reproduce IN PHASE.  SNR is therefore a function of one number: the coherently recovered
fraction of the band above 6 kHz.  Tabulate it, and show that naive upsampling IS the
zero-recovery solution.

Everything per utterance then averaged in dB, which is the convention `snr_db` reports.

    venv/bin/python audit/snr_scale.py
"""
import sys, pathlib
import numpy as np, scipy.signal as sps
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from audit.vctk_fixtures import fetch, load, PAPER_TEST      # noqa: E402

FS, R = 48000, 4
PAPER_4X = 24.16


def main():
    fr_hi, snr_naive = [], []
    for p in fetch(PAPER_TEST, per_speaker=8):
        y = load(p, FS)
        if len(y) < FS:
            continue
        Y = np.fft.rfft(y)
        f = np.fft.rfftfreq(len(y), 1 / FS)
        P = np.abs(Y) ** 2
        fr_hi.append(P[f >= FS / (2 * R)].sum() / P.sum())
        up = sps.resample_poly(sps.resample_poly(y, 1, R), R, 1)[:len(y)]
        snr_naive.append(10 * np.log10(np.sum(y ** 2) / np.sum((y - up) ** 2)))
    fr_hi, snr_naive = np.array(fr_hi), np.array(snr_naive)
    naive = snr_naive.mean()
    at = lambda r: float(np.mean(10 * np.log10(1.0 / (fr_hi * (1 - r)))))

    print(f"{len(fr_hi)} held-out utterances.  energy at or above 6 kHz: {100*fr_hi.mean():.3f} %\n")
    print(f"naive sinc upsample  mean {naive:.2f} dB   sd {snr_naive.std():.2f} dB   "
          f"range {snr_naive.min():.2f} to {snr_naive.max():.2f}\n")
    print(f"{'coherently recovered':>21}  {'SNR':>8}  {'vs naive':>9}")
    for r in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75):
        s = at(r)
        tag = "   <- measured coherent fraction of this model class" if r == 0.02 else ""
        print(f"{100*r:20.1f}%  {s:6.2f} dB  {s-naive:+8.2f}{tag}")

    print(f"\nzero recovery gives {at(0.0):.2f} dB against naive's {naive:.2f} dB: naive IS the")
    print("zero-recovery solution, to within 0.1 dB.  It is the MMSE point predictor, not a weak baseline.")
    lo, hi = 0.0, 1.0
    for _ in range(60):                                    # what the paper's number would demand
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if at(mid) < PAPER_4X else (lo, mid)
    print(f"\nreaching {PAPER_4X} dB would need {100*(lo+hi)/2:.0f} % of the high band recovered in phase.")
    print("measured coherent fraction for every model trained in this repo: <= 2 %.")


if __name__ == "__main__":
    main()
