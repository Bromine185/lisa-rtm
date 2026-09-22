// The offline fixtures make `input.wav` with scipy's resample_poly(y, 1, 4) and `naive.wav` with
// resample_poly(x_lo, 4, 1) (demo/tools/make_fixtures.py:135-136). This is the same filter — a
// 2·10·max(up,down)+1 tap windowed sinc, Kaiser β = 5, cutoff 1/max(up,down) of Nyquist, unity DC
// gain, zero phase — so a clip recorded in the browser is band-limited exactly the way the corpus
// was. Anything the network then does to it is the network's, not the resampler's.
import type { Signal } from "./types";

function besselI0(x: number): number {
  // series for the modified Bessel function of the first kind, order 0; converges fast for |x| < 30
  let s = 1, term = 1;
  const q = (x * x) / 4;
  for (let k = 1; k < 60; k++) { term *= q / (k * k); s += term; if (term < s * 1e-12) break; }
  return s;
}

const taps = new Map<number, Float64Array>();
/** scipy.signal.firwin(2·halfLen+1, 1/rate, window=('kaiser', 5.0)) — unity gain at DC. */
function lowpass(rate: number): Float64Array {
  const c = taps.get(rate);
  if (c) return c;
  const half = 10 * rate, n = 2 * half + 1, fc = 1 / rate, beta = 5;
  const h = new Float64Array(n);
  const i0b = besselI0(beta);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const m = i - half;
    const sinc = m === 0 ? 1 : Math.sin(Math.PI * fc * m) / (Math.PI * fc * m);
    const r = (2 * i) / (n - 1) - 1;
    const w = besselI0(beta * Math.sqrt(Math.max(0, 1 - r * r))) / i0b;
    h[i] = fc * sinc * w;
    sum += h[i];
  }
  for (let i = 0; i < n; i++) h[i] /= sum;
  taps.set(rate, h);
  return h;
}

/** resample_poly(x, 1, R): zero-phase lowpass at fs/(2R), then every R-th sample. */
export function decimate(sig: Signal, R: number): Signal {
  const h = lowpass(R), half = (h.length - 1) >> 1, x = sig.data, L = x.length;
  const n = Math.ceil(L / R);
  const y = new Float32Array(n);
  for (let m = 0; m < n; m++) {
    const c = m * R;
    let acc = 0;
    const j0 = Math.max(-half, -c), j1 = Math.min(half, L - 1 - c);
    for (let j = j0; j <= j1; j++) acc += h[half + j] * x[c + j];
    y[m] = acc;
  }
  return { data: y, fs: sig.fs / R };
}

/** resample_poly(x, R, 1): zero-stuff by R, then the same lowpass scaled by R. Nothing above fs/2 of the input. */
export function upsample(sig: Signal, R: number, outLen?: number): Signal {
  const h = lowpass(R), half = (h.length - 1) >> 1, x = sig.data, L = x.length;
  const n = outLen ?? L * R;
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    // y[i] = Σ_k h[k] · xu[i + half − k], and xu is non-zero only at multiples of R
    let acc = 0;
    const lo = i + half - (h.length - 1), hi = i + half;
    let p = Math.ceil(Math.max(lo, 0) / R) * R;
    for (; p <= hi; p += R) { const q = p / R; if (q >= L) break; acc += h[i + half - p] * x[q]; }
    y[i] = acc * R;
  }
  return { data: y, fs: sig.fs * R };
}

/** Peak-normalise in place to `peak` (audit/vctk_fixtures.py load(): peak 0.95). */
export function normalise(sig: Signal, peak = 0.95): Signal {
  let m = 0;
  for (let i = 0; i < sig.data.length; i++) { const a = Math.abs(sig.data[i]); if (a > m) m = a; }
  if (m > 0) { const g = peak / m; for (let i = 0; i < sig.data.length; i++) sig.data[i] *= g; }
  return sig;
}
