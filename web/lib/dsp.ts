// Small DSP for the instrument: WAV parsing, a radix-2 FFT, the spectrogram on the evaluation grid
// (1024/256 at 48 kHz, scaled with fs so the time-frequency grid is identical at 24 and 96 kHz),
// a windowed-sinc decimator for the ×2 rendering, and the energy-above-frequency check.
import type { Signal } from "./types";

export function parseWav(buf: ArrayBuffer): Signal {
  const dv = new DataView(buf);
  let p = 12;
  let fmt: { ch: number; fs: number; bits: number } | null = null;
  let data: [number, number] | null = null;
  while (p + 8 <= dv.byteLength) {
    const id = String.fromCharCode(dv.getUint8(p), dv.getUint8(p + 1), dv.getUint8(p + 2), dv.getUint8(p + 3));
    const sz = dv.getUint32(p + 4, true);
    const body = p + 8;
    if (id === "fmt ") fmt = { ch: dv.getUint16(body + 2, true), fs: dv.getUint32(body + 4, true), bits: dv.getUint16(body + 14, true) };
    if (id === "data") { data = [body, sz]; break; }
    p = body + sz + (sz & 1);
  }
  if (!fmt || !data) throw new Error("not a PCM WAV");
  const n = Math.floor(data[1] / (fmt.bits / 8) / fmt.ch);
  const out = new Float32Array(n);
  if (fmt.bits === 16) for (let i = 0; i < n; i++) out[i] = dv.getInt16(data[0] + i * 2 * fmt.ch, true) / 32768;
  else if (fmt.bits === 32) for (let i = 0; i < n; i++) out[i] = dv.getFloat32(data[0] + i * 4 * fmt.ch, true);
  else throw new Error("unsupported bit depth " + fmt.bits);
  return { data: out, fs: fmt.fs };
}

interface Plan { rev: Uint32Array; cos: Float32Array; sin: Float32Array; win: Float32Array; re: Float32Array; im: Float32Array }
const plans = new Map<number, Plan>();
function plan(n: number): Plan {
  const c = plans.get(n);
  if (c) return c;
  const rev = new Uint32Array(n);
  const bits = Math.log2(n) | 0;
  for (let i = 0; i < n; i++) { let r = 0, x = i; for (let b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1; } rev[i] = r; }
  const cos = new Float32Array(n / 2), sin = new Float32Array(n / 2), win = new Float32Array(n);
  for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos((-2 * Math.PI * i) / n); sin[i] = Math.sin((-2 * Math.PI * i) / n); }
  for (let i = 0; i < n; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  const P = { rev, cos, sin, win, re: new Float32Array(n), im: new Float32Array(n) };
  plans.set(n, P);
  return P;
}

/** Magnitude spectrum (n/2+1 bins) of x[off, off+n) under a periodic Hann window. */
export function fftMag(x: Float32Array, off: number, n: number, out: Float32Array): Float32Array {
  const { rev, cos, sin, win, re, im } = plan(n);
  for (let i = 0; i < n; i++) { const j = rev[i]; const k = off + j; re[i] = k >= 0 && k < x.length ? x[k] * win[j] : 0; im[i] = 0; }
  for (let s = 2; s <= n; s <<= 1) {
    const h = s >> 1, step = n / s;
    for (let k = 0; k < n; k += s) for (let j = 0; j < h; j++) {
      const c = cos[j * step], d = sin[j * step], a = k + j, b = a + h;
      const tr = re[b] * c - im[b] * d, ti = re[b] * d + im[b] * c;
      re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
    }
  }
  for (let i = 0; i <= n / 2; i++) out[i] = Math.hypot(re[i], im[i]);
  return out;
}

export interface Spectrogram {
  nfft: number; hop: number; nb: number; nfr: number;
  db: Float32Array;   // nfr * nb, dB, filled up to `done` frames
  done: number;
  hzPerBin: number;
  sig: Signal;
  tmp: Float32Array;
}

export function specOf(sig: Signal): Spectrogram {
  const nfft = Math.max(64, Math.round((1024 * sig.fs) / 48000));
  const hop = nfft >> 2, nb = nfft / 2 + 1;
  const nfr = Math.max(1, 1 + Math.floor((sig.data.length - nfft) / hop));
  return { nfft, hop, nb, nfr, db: new Float32Array(nfr * nb), done: 0, hzPerBin: sig.fs / nfft, sig, tmp: new Float32Array(nb) };
}

/** Fill frames whose window ends before `uptoSample`; returns the spectrogram for chaining. */
export function specFill(sp: Spectrogram, uptoSample: number): Spectrogram {
  const last = Math.min(sp.nfr, Math.floor((uptoSample - sp.nfft) / sp.hop) + 1);
  for (let f = sp.done; f < last; f++) {
    fftMag(sp.sig.data, f * sp.hop, sp.nfft, sp.tmp);
    const o = f * sp.nb;
    for (let b = 0; b < sp.nb; b++) sp.db[o + b] = 20 * Math.log10(sp.tmp[b] + 1e-7);
  }
  if (last > sp.done) sp.done = last;
  return sp;
}

/** 48 kHz -> 24 kHz: 63-tap Hann-windowed sinc at half Nyquist, then take every other sample. */
export function decimate2(x: Float32Array): Float32Array<ArrayBuffer> {
  const T = 63, m = (T - 1) / 2, h = new Float32Array(T);
  let sum = 0;
  for (let i = 0; i < T; i++) {
    const t = i - m;
    const s = t === 0 ? 0.5 : Math.sin(Math.PI * 0.5 * t) / (Math.PI * t);
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (T - 1));
    h[i] = s * w; sum += h[i];
  }
  for (let i = 0; i < T; i++) h[i] /= sum;
  const n = x.length >> 1, y = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    let a = 0; const c = 2 * k;
    for (let i = 0; i < T; i++) { const j = c + i - m; if (j >= 0 && j < x.length) a += h[i] * x[j]; }
    y[k] = a;
  }
  return y;
}

/** Share of energy at or above `hz`, from 4096-sample windows strided across the signal. */
export function energyAbove(sig: Signal, hz: number): number {
  const n = 4096;
  if (sig.data.length < n) return 0;
  const tmp = new Float32Array(n / 2 + 1);
  let hi = 0, tot = 0;
  for (let off = 0; off + n <= sig.data.length; off += n * 4) {
    fftMag(sig.data, off, n, tmp);
    for (let b = 0; b <= n / 2; b++) { const e = tmp[b] * tmp[b]; tot += e; if ((b * sig.fs) / n >= hz) hi += e; }
  }
  return tot > 0 ? hi / tot : 0;
}

export function rgb(hex: string): [number, number, number] {
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}
