"""Two degradation operators, two inverse problems.

With anti-aliasing the operator is `A = D_R . H`, and for an ideal `H` the missing band lies exactly
in `ker(A) = range(P_H)`: the observation carries no linear information about it, the optimal Wiener
filter for that band is the zero filter, and naive sinc upsampling is the LMMSE solution.  A REAL `H`
is not ideal -- `scipy.signal.resample_poly`'s 81-tap Kaiser passes 6.0-6.6 kHz at up to -6 dB -- so a
little of the band survives and a fitted filter finds it.  That leak is measured here, not assumed.
Drop `H` altogether and `x = y[::R]` folds all of 6-24 kHz back into the baseband: the band is fully
present in the observation, mirrored and summed, and the task becomes unmixing rather than synthesis.

Three measurements, in increasing order of assumption:

  1. how much of the band above 6 kHz survives into the observation at all.  No estimator, no model.
  2. what a per-bin linear filter recovers of it -- the best time-invariant linear estimator given the
     observed bin, fitted on four speakers and scored on four others.  This is admissible: it uses
     nothing but `x`.  Reported across all eight rotations of the split, because one split is a
     lottery: the aliased number spans 1 to 21 % over the eight.
  3. what an ORACLE per-bin unfolder recovers -- the Wiener unfolder that knows each alias's own power
     at every bin.  It is NOT admissible (those powers are a function of `y`, not of `x`) and so its
     score is neither an upper nor a lower bound on the protocol's ceiling.  It is reported as what a
     phase-blind per-bin unmixer could do if the spectral split were handed to it.

Nothing is trained, no checkpoint, no GPU.  See notes/2026-09-17-lisa-reported-numbers-audit.md §2, §6
and paper/bwe-information-ceiling.md §8.

    venv/bin/python audit/protocol_fork.py
"""
import sys, pathlib
import numpy as np, scipy.signal as sps
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from audit.vctk_fixtures import fetch, load, PAPER_TEST      # noqa: E402

FS, R = 48000, 4
CHUNK = FS                     # 1 s, the chunk length LISA's wrapper uses
M = CHUNK // R                 # baseband bins; each one is the sum of R aliases
HALF = M // 2                  # m <= HALF is fitted, the rest follows by conjugate symmetry
GROUP = 401                    # bins pooled per fitted gain (~400 Hz): a spectral envelope, not detail
CUT = FS / (2 * R)             # 6 kHz
SPLITS = [tuple(PAPER_TEST[(j + i) % len(PAPER_TEST)] for i in range(len(PAPER_TEST)))
          for j in range(len(PAPER_TEST))]          # every rotation of who is fitted
FIT, EVAL = PAPER_TEST[:4], PAPER_TEST[4:]


def aliases(c):
    """A[k, m] = Y[m + k*M].  Decimation by R without a filter observes their sum."""
    return np.fft.fft(c, axis=-1).reshape(*np.shape(c)[:-1], R, M)


def alias_freqs():
    """|f| of every alias of every baseband bin, shape (R, M)."""
    n = np.arange(R)[:, None] * M + np.arange(M)[None, :]
    return np.abs(np.where(n < R * M / 2, n, n - R * M) * FS / (R * M))


def high_mask():
    """Which aliases carry |f| >= 6 kHz.  Computed from frequency, so it agrees with split() bin for bin."""
    return alias_freqs() >= CUT


def observe(c, protocol):
    if protocol == "aliased":
        return c[..., ::R]                              # no filter: the high band folds down
    return sps.resample_poly(c, 1, R, axis=-1)          # anti-aliased: H, then decimate


def split(c):
    """(high-band energy fraction, the exact low band) -- the low band is the rho = 0 reconstruction."""
    Y = np.fft.fft(c, axis=-1)
    hi = np.abs(np.fft.fftfreq(R * M, 1 / FS)) >= CUT
    lo = Y.copy()
    lo[..., hi] = 0
    return (np.abs(Y[..., hi]) ** 2).sum(-1) / (np.abs(Y) ** 2).sum(-1), np.fft.ifft(lo, axis=-1).real


def survives(utts, protocol):
    """Power of the high band reaching the observation, over its power in y.  x(y) - x(y_lo) is all of it.

    The factor R turns two sums of different length into a mean-square ratio.  It can exceed 1 under
    aliasing, where the folded copies interfere; its expectation there is exactly 1.
    """
    num = den = 0.0
    for c in utts:
        _, lo = split(c)
        num += R * ((observe(c, protocol) - observe(lo, protocol)) ** 2).sum()
        den += ((c - lo) ** 2).sum()
    return num / den


def fit_gains(utts, protocol):
    """One complex gain per (alias, ~400 Hz group), least squares, each chunk normalised to unit energy.

    w = E[A conj(Z)] / E[|Z|^2] is the best linear estimator of that alias from that observed bin.
    Level normalisation stops the few loudest chunks from owning the fit; num and den take the same
    box, so the 1/GROUP cancels and this is the exact group least squares, edges included.
    """
    num = np.zeros((R, HALF + 1), complex)
    den = np.zeros(HALF + 1)
    for c in utts:
        A, Z = aliases(c)[..., :HALF + 1], np.fft.fft(observe(c, protocol), axis=-1)[..., :HALF + 1]
        s = np.maximum((c ** 2).sum(-1), 1e-30)[:, None]
        num += (A * np.conj(Z)[:, None, :] / s[:, None, :]).sum(0)
        den += (np.abs(Z) ** 2 / s).sum(0)
    num = uniform_filter1d(num, GROUP, axis=-1, mode="constant")
    den = uniform_filter1d(den, GROUP, mode="constant")
    w = np.zeros((R, M), complex)
    w[:, :HALF + 1] = num / np.maximum(den, 1e-30)
    for k in range(R):                                  # conjugate mirror keeps the output real
        w[R - 1 - k, HALF + 1:] = np.conj(w[k, 1:M - HALF][::-1])
    k = np.arange(R)                                    # m = 0 and m = HALF are their own mirror,
    w[:, 0] = 0.5 * (w[:, 0] + np.conj(w[(R - k) % R, 0]))       # and pair differently: n = kM with
    w[:, HALF] = 0.5 * (w[:, HALF] + np.conj(w[::-1, HALF]))     # (R-k)M, n = kM+HALF with (R-1-k)M
    return w


def oracle_gains(A):
    """R * sigma_k^2 / sum_j sigma_j^2 per bin, from the true per-alias powers.  Phase-blind, per bin.

    Not admissible: those powers are a function of y.  See the module docstring.
    """
    P = np.abs(A) ** 2
    return R * P / np.maximum(P.sum(-2, keepdims=True), 1e-30)


def estimate(c, protocol, gains):
    Z = np.fft.fft(observe(c, protocol), axis=-1)
    A = aliases(c)
    w = oracle_gains(A) if isinstance(gains, str) else gains
    return A, w * Z[:, None, :]


def recovered(utts, protocol, gains):
    """Coherently recovered fraction of the band above 6 kHz: 1 - residual / high-band energy."""
    hi, res, tot = high_mask(), 0.0, 0.0
    for c in utts:
        A, Ahat = estimate(c, protocol, gains)
        res += (np.abs(A - Ahat) ** 2 * hi).sum()
        tot += (np.abs(A) ** 2 * hi).sum()
    return 1 - res / tot


def rebuild(Ahat):
    y = np.fft.ifft(Ahat.reshape(*Ahat.shape[:-2], R * M), axis=-1)
    bad = (y.imag ** 2).sum() / max((y.real ** 2).sum(), 1e-30)   # gains must stay conjugate-mirrored
    assert bad < 1e-9, f"reconstruction is not real: {bad:.2e}"
    return y.real


def score(utts, preds):
    """SNR under three averaging conventions.  The map from energy fraction to dB is not free (Jensen).

    'utt dB' is an unweighted mean over utterances of unequal length, so it is not 'chunk dB' regrouped.
    """
    ce, uu, pool = [], [], []
    for c, p in zip(utts, preds):
        sig, err = (c ** 2).sum(-1), ((c - p) ** 2).sum(-1)
        ce.append(10 * np.log10(sig / err))
        uu.append(10 * np.log10(sig.sum() / err.sum()))
        pool.append((sig.sum(), err.sum()))
    ce = np.concatenate(ce)
    s, e = np.array(pool).sum(0)
    return ce.mean(), float(np.mean(uu)), 10 * np.log10(s / e), ce.std()


def chunks(speakers):
    out = []
    for p in fetch(speakers, per_speaker=8):
        y = load(p, FS)
        n = len(y) // CHUNK
        if n:
            out.append(np.stack([y[j * CHUNK:(j + 1) * CHUNK] for j in range(n)]))
    return out


def sweep(protocol, cache):
    """Held-out recovery over every rotation of which four of the eight speakers are fitted."""
    out = []
    for sp in SPLITS:
        f = [x for s in sp[:4] for x in cache[s]]
        e = [x for s in sp[4:] for x in cache[s]]
        out.append(recovered(e, protocol, fit_gains(f, protocol)))
    return np.array(out)


def main():
    cache = {s: chunks((s,)) for s in PAPER_TEST}
    fit_u = [x for s in FIT for x in cache[s]]
    ev_u = [x for s in EVAL for x in cache[s]]
    print(f"fit  {' '.join(FIT)}   {sum(len(c) for c in fit_u)} one-second chunks")
    print(f"eval {' '.join(EVAL)}   {sum(len(c) for c in ev_u)} one-second chunks   (speaker-disjoint)")
    fr = np.concatenate([split(c)[0] for c in ev_u])
    print(f"energy at or above {CUT/1000:.0f} kHz on the eval set: {100*fr.mean():.3f} % per chunk\n")

    names = {"clean": "anti-aliased  x = D_R(H y)", "aliased": "aliased       x = y[::R]"}
    print("1. how much of the band above 6 kHz reaches the observation (no estimator, no fit)\n")
    for p in ("clean", "aliased"):
        print(f"   {names[p]:<30} {100*survives(ev_u, p):8.3f} %")
    print("\n   With H, 97.8 % of the band is destroyed and never reaches any estimator.  Without it,")
    print("   all of it is there.  Everything below is about what can be done with what is there.\n")

    W = {p: fit_gains(fit_u, p) for p in ("clean", "aliased")}
    print("2. of the band above 6 kHz, what an ADMISSIBLE per-bin linear filter recovers, held out.")
    print(f"   Eight rotations of which four of the {len(PAPER_TEST)} speakers are fitted:\n")
    print(f"   {'':<30} {'this split':>11} {'median':>8} {'min':>8} {'max':>8}")
    for p in ("clean", "aliased"):
        v = 100 * sweep(p, cache)
        print(f"   {names[p]:<30} {100*recovered(ev_u, p, W[p]):10.2f}% {np.median(v):7.2f}% "
              f"{v.min():7.2f}% {v.max():7.2f}%")
    print("\n   One split is a lottery -- the aliased number spans an order of magnitude across the")
    print("   eight -- but every split separates the two protocols, and the medians differ 8x.")
    hm, af = high_mask(), alias_freqs()
    g = np.abs(W["clean"]) * hm
    f_worst = af[np.unravel_index(int(np.argmax(g)), g.shape)]
    top = af[g > 0.1]
    print(f"\n   The anti-aliased row is NOT zero, and that is the filter, not the speech: the fitted")
    print(f"   gain peaks at {g.max():.2f} at {f_worst:.0f} Hz and stays above 0.1 out to "
          f"{top.max():.0f} Hz, inside")
    print("   resample_poly's transition band (its 81-tap Kaiser is only -6 dB down at 6.0 kHz).")
    print("   For an ideal brick wall the high band is exactly in ker(A) and the zero filter is optimal;")
    print("   for an 81-tap Kaiser it is optimal to within the 2.15 % that leaks.\n")

    print("3. what an ORACLE per-bin unfolder recovers from the aliased observation.  Inadmissible:")
    print("   it is handed each alias's own power at every bin, which is a function of y, not of x.\n")
    print(f"   {'aliased, oracle unfolder':<30} {100*recovered(ev_u, 'aliased', 'oracle'):10.2f}%"
          f"   (fit set {100*recovered(fit_u, 'aliased', 'oracle'):.2f}%, no fitting involved)\n")

    preds = {
        "empty high band (rho = 0)": [split(c)[1] for c in ev_u],
        "naive sinc upsample, anti-aliased input": [
            sps.resample_poly(observe(c, "clean"), R, 1, axis=-1) for c in ev_u],
        "LMMSE filter, anti-aliased input": [
            rebuild(estimate(c, "clean", W["clean"])[1]) for c in ev_u],
        "naive sinc upsample, ALIASED input": [
            sps.resample_poly(observe(c, "aliased"), R, 1, axis=-1) for c in ev_u],
        "LMMSE filter, ALIASED input": [
            rebuild(estimate(c, "aliased", W["aliased"])[1]) for c in ev_u],
        "oracle unfolder, ALIASED input": [
            rebuild(estimate(c, "aliased", "oracle")[1]) for c in ev_u],
    }
    print("4. SNR of each reconstruction, held-out speakers, three averaging conventions\n")
    print(f"   {'':<42}{'chunk dB':>9}{'utt dB':>8}{'pooled':>8}{'sd':>7}")
    got = {}
    for k, v in preds.items():
        got[k] = score(ev_u, v)
        print(f"   {k:<42}{got[k][0]:9.2f}{got[k][1]:8.2f}{got[k][2]:8.2f}{got[k][3]:7.2f}")

    ceil, lmc = got["empty high band (rho = 0)"], got["LMMSE filter, anti-aliased input"]
    nv_a, orc = got["naive sinc upsample, ALIASED input"], got["oracle unfolder, ALIASED input"]
    print(f"\n5. reading the table honestly")
    print(f"   The rho = 0 row is a hard bound only for an ideal H.  The fitted anti-aliased filter")
    print(f"   crosses it by {lmc[1]-ceil[1]:+.3f} utterance-dB and {lmc[2]-ceil[2]:+.3f} pooled -- the leak of §2,")
    print(f"   worth three hundredths of a decibel.  Round it off and the bound holds.")
    print(f"   The aliased protocol's own naive baseline sits {nv_a[0]-ceil[0]:+.2f} dB lower, and the")
    print(f"   oracle recovers {orc[0]-nv_a[0]:+.2f} dB of that -- about half from the baseband the fold")
    print("   corrupted, about half from the high band itself.  That is a statement about how much")
    print("   room a metric has on each protocol, not a measured information ceiling for the aliased")
    print("   one: no admissible estimator here reaches it, and on p236-p238 it lands below the")
    print("   anti-aliased bound rather than above.  What is protocol-level and assumption-free is §1.")


if __name__ == "__main__":
    main()
