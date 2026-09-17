#!/usr/bin/env node
// Validate demo/engine.js against the PyTorch reference vectors from make_testvec.py.
//
//   node demo/tools/validate_engine.mjs [--bin demo/assets/weights/es_erb_l0.1.bin]
//                                       [--manifest demo/assets/weights/manifest.json]
//                                       [--testvec demo/tools/testvec_es_erb_l0.1.json] [--no-synth]
//
// Checks: max |js - torch| at tau = 0 (R = 4 and R = 8) < 1e-4; the seeded tau = 1 draw against torch fed the
// same noise; energy above 24 kHz in the R = 8 output < 0.5 %; chunk = 512 vs 8192 bit-identical; a seeded
// draw run twice bit-identical; different seeds differ; and a timing line.  When lisa_rtm_cache/demo_synth/
// exists the same checks run on the synthetic LISASD (n_dec = 4).  Exit code 1 on any failure.
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const args = process.argv.slice(2);
const opt = (name, dflt) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : dflt; };
const BIN = opt('--bin', path.join(REPO, 'demo/assets/weights/es_erb_l0.1.bin'));
const MANIFEST = opt('--manifest', path.join(BIN, '..', 'manifest.json'));
const TESTVEC = opt('--testvec', path.join(HERE, 'testvec_es_erb_l0.1.json'));
const SYNTH = path.join(REPO, 'lisa_rtm_cache/demo_synth');

const engineMod = await import(pathToFileURL(path.join(HERE, '..', 'engine.js')).href);
const E = engineMod.default || globalThis.LISAEngine;
if (!E || typeof E.run !== 'function') throw new Error('engine.js did not expose LISAEngine');

let failures = 0;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) failures++;
}

// ---- small radix-2 FFT (complex, in place) ----
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}

/** Fraction of power at or above fCut in y sampled at fs: Hann windows of nfft, hop nfft/2, averaged. */
function energyAbove(y, fs, fCut, nfft = 4096) {
  const hop = nfft >> 1, re = new Float64Array(nfft), im = new Float64Array(nfft);
  const win = new Float64Array(nfft);
  for (let i = 0; i < nfft; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / nfft);
  const kCut = Math.ceil(fCut * nfft / fs);
  let above = 0, total = 0, frames = 0;
  for (let s = 0; s + nfft <= y.length; s += hop, frames++) {
    for (let i = 0; i < nfft; i++) { re[i] = y[s + i] * win[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k <= nfft / 2; k++) {
      const p = (re[k] * re[k] + im[k] * im[k]) * (k === 0 || k === nfft / 2 ? 1 : 2);
      total += p;
      if (k >= kCut) above += p;
    }
  }
  return { frac: above / total, frames };
}

function maxAbsDiff(a, b) {
  const n = Math.min(a.length, b.length);
  let m = 0, at = -1;
  for (let i = 0; i < n; i++) { const d = Math.abs(a[i] - b[i]); if (!(d <= m)) { m = d; at = i; } }
  return { max: m, at, lenOk: a.length === b.length };
}

function identical(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i] || Number.isNaN(a[i])) return false;
  return true;
}

function fmt(x) { return Number(x).toExponential(3); }

async function validateArm(binPath, manPath, tvPath, { timing }) {
  console.log(`\n== ${path.basename(binPath)}  (${path.relative(REPO, tvPath)})`);
  const model = await E.load(binPath, manPath);
  const tv = JSON.parse(await readFile(tvPath, 'utf8'));
  const x = Float32Array.from(tv.x12k);
  const L = x.length, R = tv.R || 4;
  console.log(`arm ${model.arm} cls ${model.cls} n_noise ${model.n_noise} n_dec ${model.n_dec} det ${model.det} `
    + `step ${model.step} params ${model.param_count}; input ${L} samples @ ${tv.fs} Hz`);
  check(model.cls === tv.cls && model.n_noise === tv.n_noise && model.n_dec === tv.n_dec, 'manifest matches the test vector');

  // 1. tau = 0, R = 4 vs PyTorch
  const y4 = await E.run(model, x, { R, tau: 0, chunk: 2048 });
  const ref4 = Float32Array.from(tv.y_r4_tau0);
  let d = maxAbsDiff(y4, ref4);
  check(d.lenOk && d.max < 1e-4, `tau=0 R=${R}: max|js - torch| = ${fmt(d.max)} < 1e-4`, `(at sample ${d.at}, N=${y4.length})`);

  // 2. tau = 0, R = 8: vs PyTorch and no energy above 24 kHz
  const y8 = await E.run(model, x, { R: 2 * R, tau: 0, chunk: 2048 });
  const ref8 = Float32Array.from(tv.y_r8_tau0);
  d = maxAbsDiff(y8, ref8);
  check(d.lenOk && d.max < 1e-4, `tau=0 R=${2 * R}: max|js - torch| = ${fmt(d.max)} < 1e-4`, `(N=${y8.length})`);
  const fsOut8 = tv.fs * 2 * R;
  const eJs = energyAbove(y8, fsOut8, tv.fs_out / 2), eRef = energyAbove(ref8, fsOut8, tv.fs_out / 2);
  check(eJs.frac < 0.005, `R=${2 * R}: energy >= ${tv.fs_out / 2000} kHz = ${(100 * eJs.frac).toFixed(4)} % < 0.5 %`,
    `(torch reference ${(100 * eRef.frac).toFixed(4)} %, ${eJs.frames} Hann frames of 4096 at ${fsOut8} Hz)`);

  // 3. tau = 1, seed = 0 vs PyTorch fed the engine's own noise draw
  const y1 = await E.run(model, x, { R, tau: 1, seed: 0, chunk: 2048 });
  const ref1 = Float32Array.from(tv.y_r4_tau1_s0);
  d = maxAbsDiff(y1, ref1);
  check(d.lenOk && d.max < 1e-4, `tau=1 seed=0 R=${R}: max|js - torch(same noise)| = ${fmt(d.max)} < 1e-4`, `(at sample ${d.at})`);
  d = maxAbsDiff(y1, y4);
  check(d.max > 1e-3, `tau=1 differs from tau=0`, `(max diff ${fmt(d.max)})`);

  // 4. chunk size cannot change bits
  const a512 = await E.run(model, x, { R, tau: 1, seed: 3, chunk: 512 });
  const a8192 = await E.run(model, x, { R, tau: 1, seed: 3, chunk: 8192 });
  const a1000 = await E.run(model, x, { R, tau: 1, seed: 3, chunk: 1000 });
  check(identical(a512, a8192), 'chunk=512 and chunk=8192 bit-identical');
  check(identical(a512, a1000), 'chunk=1000 (not a multiple of R or 4) bit-identical');

  // 5. seeded draw reproducible, different seeds differ
  const s7a = await E.run(model, x, { R, tau: 1, seed: 7 });
  const s7b = await E.run(model, x, { R, tau: 1, seed: 7 });
  const s8 = await E.run(model, x, { R, tau: 1, seed: 8 });
  check(identical(s7a, s7b), 'tau=1 seed=7 run twice bit-identical');
  check(!identical(s7a, s8), 'seed=7 and seed=8 differ');

  // 6. onProgress contract
  const calls = [];
  const yp = await E.run(model, x, { R, tau: 0, chunk: 2048, onProgress: (j, N, out) => calls.push([j, N, out.length]) });
  const nChunks = Math.ceil(yp.length / 2048);
  check(calls.length === nChunks && calls[calls.length - 1][0] === yp.length && calls.every(([j, N, n]) => N === yp.length && n === j),
    `onProgress fired ${calls.length}x, last jDone = ${calls.length ? calls[calls.length - 1][0] : '-'} of ${yp.length}`);

  if (!timing) return;
  // 7. timing: tau = 1, R = 4, chunk = 2048 on the input tiled to ~2 s, after a warm-up
  const reps = Math.max(1, Math.round(2 * tv.fs / L));
  const xl = new Float32Array(L * reps);
  for (let r = 0; r < reps; r++) xl.set(x, r * L);
  await E.run(model, xl, { R, tau: 1, seed: 0, chunk: 2048 });
  const t0 = performance.now();
  const yl = await E.run(model, xl, { R, tau: 1, seed: 0, chunk: 2048 });
  const dt = (performance.now() - t0) / 1000;
  const audioSec = yl.length / tv.fs_out;
  console.log(`timing: ${audioSec.toFixed(2)} s of audio in ${dt.toFixed(3)} s (${(yl.length / 2048 | 0)} chunks)`);
  console.log(`audio_seconds_per_compute_second=${(audioSec / dt).toFixed(3)}`);
  check(audioSec / dt >= 0.3, 'throughput >= 0.3 audio-seconds per compute-second');
}

await validateArm(BIN, MANIFEST, TESTVEC, { timing: true });

const synthTv = path.join(SYNTH, 'testvec_synth_lisasd.json');
if (!args.includes('--no-synth') && existsSync(synthTv)) {
  await validateArm(path.join(SYNTH, 'weights/synth_lisasd.bin'), path.join(SYNTH, 'weights/manifest.json'), synthTv, { timing: false });
} else {
  console.log('\n(no lisa_rtm_cache/demo_synth/testvec_synth_lisasd.json: LISASD n_dec > 0 path not exercised)');
}

console.log(failures ? `\n${failures} check(s) FAILED` : '\nall checks passed');
process.exit(failures ? 1 : 0);
