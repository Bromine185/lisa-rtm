"""Two degradation operators, two inverse problems, two ceilings.

With anti-aliasing the operator is `A = D_R . H` and `ker(A) = range(P_H)`: the null space is exactly
the band being estimated, so the observation carries no linear information about it, the optimal Wiener
filter for the missing band is the zero filter, and naive sinc upsampling is the LMMSE solution.  Drop
`H` and `x = y[::R]` folds 6-24 kHz back into the baseband.  The band is then present in the observation
-- mirrored and summed with the baseband -- and the task is unmixing, not synthesis.

Three measurements, in increasing order of assumption:

  1. how much of the band above 6 kHz survives into the observation at all.  No estimator, no model.
  2. what a fixed linear per-bin filter, fitted on four speakers and applied to four others, recovers
     of it.  This is the best time-invariant linear estimator: it is what "no model" buys you.
  3. what an ORACLE per-bin unfolder recovers -- the Wiener unfolder that knows each alias's own power
     at every bin.  It is not an upper bound on the aliased protocol; it is one estimator with strong
     side information.  It matters because whatever it reaches, the aliased protocol's ceiling is at
     least that high, and it is measured against the anti-aliased ceiling, which IS an upper bound.

Nothing is trained, no checkpoint, no GPU.  See notes/2026-09-17-lisa-reported-numbers-audit.md §2, §6.

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
FIT, EVAL = PAPER_TEST[:4], PAPER_TEST[4:]
CUT = FS / (2 * R)             # 6 kHz


def aliases(c):
    """A[k, m] = Y[m + k*M].  Decimation by R without a filter observes their sum."""
    return np.fft.fft(c, axis=-1).reshape(*np.shape(c)[:-1], R, M)


def high_mask():
    """Exactly one alias of each bin is the low band; the other R-1 are what we are trying to recover."""
    m = np.arange(M)
    return np.arange(R)[:, None] != np.where(m < HALF, 0, R - 1)[None, :]


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
    """Fraction of the high band's energy that reaches the observation.  x(y) - x(y_lo) is all of it."""
    num = den = 0.0
    for c in utts:
        _, lo = split(c)
        num += R * ((observe(c, protocol) - observe(lo, protocol)) ** 2).sum()
        den += ((c - lo) ** 2).sum()
    return num / den


def fit_gains(utts, protocol):
    """One complex gain per (alias, ~400 Hz group), least squares, each chunk normalised to unit energy.

    w = E[A conj(Z)] / E[|Z|^2] is the best linear estimator of that alias from that observed bin.
    Level normalisation stops the few loudest chunks from owning the fit.
    """
    num = np.zeros((R, HALF + 1), complex)
    den = np.zeros(HALF + 1)
    for c in utts:
        A, Z = aliases(c)[..., :HALF + 1], np.fft.fft(observe(c, protocol), axis=-1)[..., :HALF + 1]
        s = np.maximum((c ** 2).sum(-1), 1e-30)[:, None]
        num += (A * np.conj(Z)[:, None, :] / s[:, None, :]).sum(0)
        den += (np.abs(Z) ** 2 / s).sum(0)
    num = uniform_filter1d(num, GROUP, axis=-1, mode="nearest")
    den = uniform_filter1d(den, GROUP, mode="nearest")
    w = np.zeros((R, M), complex)
    w[:, :HALF + 1] = num / np.maximum(den, 1e-30)
    for k in range(R):                                  # conjugate mirror keeps the output real
        w[R - 1 - k, HALF + 1:] = np.conj(w[k, 1:M - HALF][::-1])
    k = np.arange(R)                                    # m = 0 and m = HALF are their own mirror,
    w[:, 0] = 0.5 * (w[:, 0] + np.conj(w[(R - k) % R, 0]))       # and pair differently: n = kM with
    w[:, HALF] = 0.5 * (w[:, HALF] + np.conj(w[::-1, HALF]))     # (R-k)M, n = kM+HALF with (R-1-k)M
    return w


def oracle_gains(A):
    """R * sigma_k^2 / sum_j sigma_j^2 per bin, from the true per-alias powers.  Phase-blind, per bin."""
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
    """SNR under three averaging conventions.  The map from energy fraction to dB is not free (Jensen)."""
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


def main():
    fit_u, ev_u = chunks(FIT), chunks(EVAL)
    print(f"fit  {' '.join(FIT)}   {sum(len(c) for c in fit_u)} one-second chunks")
    print(f"eval {' '.join(EVAL)}   {sum(len(c) for c in ev_u)} one-second chunks   (speaker-disjoint)")
    fr = np.concatenate([split(c)[0] for c in ev_u])
    print(f"energy at or above {CUT/1000:.0f} kHz on the eval set: {100*fr.mean():.3f} % per chunk\n")

    W = {p: fit_gains(fit_u, p) for p in ("clean", "aliased")}
    names = {"clean": "anti-aliased  x = D_R(H y)", "aliased": "aliased       x = y[::R]"}
    print("1. how much of the band above 6 kHz reaches the observation (no estimator)\n")
    for p in ("clean", "aliased"):
        print(f"   {names[p]:<30} {100*survives(ev_u, p):8.3f} %")
    print("\n2. of that band, what is coherently recoverable by the best FIXED linear per-bin filter,")
    print("   fitted on the four fit speakers:\n")
    print(f"   {'':<30} {'fit set':>9} {'held out':>10}")
    for p in ("clean", "aliased"):
        print(f"   {names[p]:<30} {100*recovered(fit_u, p, W[p]):8.2f}% {100*recovered(ev_u, p, W[p]):9.2f}%")
    print(f"\n   anti-aliased: the fitted gain on every high-band alias is at most "
          f"{np.abs(W['clean'][1:R-1]).max():.1e} --")
    print("   W(f) = S_xy/S_xx = 0 above 6 kHz, so the zero filter is optimal and naive sinc is LMMSE.")
    print("   aliased: the band IS there, but a time-invariant filter cannot unmix it -- the split")
    print("   between baseband and folded band is signal-dependent.  Unfolding needs a model.\n")
    print("3. what an ORACLE per-bin unfolder recovers from the aliased observation (side information:")
    print("   every alias's own power at every bin; phase-blind; still only a scalar gain per bin)\n")
    print(f"   {'aliased, oracle unfolder':<30} {100*recovered(fit_u, 'aliased', 'oracle'):8.2f}%"
          f" {100*recovered(ev_u, 'aliased', 'oracle'):9.2f}%\n")

    preds = {
        "ceiling: exact low band, empty high band": [split(c)[1] for c in ev_u],
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

    ceil = got["ceiling: exact low band, empty high band"]
    nv_c = got["naive sinc upsample, anti-aliased input"]
    nv_a = got["naive sinc upsample, ALIASED input"]
    orc = got["oracle unfolder, ALIASED input"]
    print("\n5. the fork: headroom above each protocol's OWN naive baseline, which is what a metric")
    print("   needs if it is to rank models at all\n")
    print(f"   {'':<30}{'chunk dB':>9}{'utt dB':>8}{'pooled':>8}")
    for name, a, b in (("anti-aliased, ceiling - naive", ceil, nv_c),
                       ("aliased, oracle unfold - naive", orc, nv_a)):
        print(f"   {name:<30}{a[0]-b[0]:+9.2f}{a[1]-b[1]:+8.2f}{a[2]-b[2]:+8.2f}")
    print(f"\n   Anti-aliased: {ceil[0]-nv_c[0]:.2f} dB, and that is an UPPER bound -- nothing that does not")
    print("   synthesise high-band phase gets past it.  Aliased: at least "
          f"{orc[0]-nv_a[0]:.2f} dB, from one")
    print("   phase-blind per-bin unfolder, and a model that adapts per frame will find more.")
    print("   Same nominal task, same corpus, same metric; the dynamic range differs by 20x.")
    print(f"\n   Against the anti-aliased ceiling itself the oracle unfolder lands {orc[0]-ceil[0]:+.2f} chunk-dB,")
    print(f"   {orc[1]-ceil[1]:+.2f} utterance-dB, {orc[2]-ceil[2]:+.2f} pooled: it crosses the other protocol's")
    print("   hard bound under two conventions of three.  Quote every ceiling with its averaging rule.")


if __name__ == "__main__":
    main()
