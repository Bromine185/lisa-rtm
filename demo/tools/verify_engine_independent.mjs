#!/usr/bin/env node
/* demo/tools/verify_engine_independent.mjs — adversarial, independent check of demo/engine.js.
 *
 *   node demo/tools/verify_engine_independent.mjs [--fast]
 *
 * Shares no code with demo/tools/validate_engine.mjs: its own .bin reader, its own mulberry32 / Box-Muller,
 * its own naive O(n^2) DFT, its own SNR, its own decimator, its own comparison helpers.
 *
 * It does NOT use demo/assets/weights (a stale step-13500 copy) or demo/tools/testvec_*.json.  It runs the
 * SHIPPED weights, web/public/assets/weights/<arm>.bin, all seven arms, against PyTorch references generated
 * here and now from lisa_rtm_cache/ckpt/final/<arm>.pt through audit/boot.py:
 *
 *   lisa_rtm_cache/verify_adv/prep.py   x12k.f32, arms.json, bincheck.json  (bin == checkpoint, tensor by tensor)
 *   this file                           the noise draws (my mulberry32) -> eps/*.f32, jobs.json
 *   lisa_rtm_cache/verify_adv/refs.py   torch references -> refs/<arm>/<job>.f32, refs.json
 *   this file                           every comparison, every edge case, the timing
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const REPO = '/Users/raghavsharma/projects/lisa-rtm';
const PY = path.join(REPO, 'venv/bin/python');
const WDIR = path.join(REPO, 'web/public/assets/weights');
const VDIR = path.join(REPO, 'lisa_rtm_cache/verify_adv');
const FAST = process.argv.includes('--fast');

const mod = await import(pathToFileURL(path.join(REPO, 'demo/engine.js')).href);
const E = mod.default || globalThis.LISAEngine;
if (!E || typeof E.run !== 'function' || typeof E.load !== 'function') throw new Error('no LISAEngine');

// ------------------------------------------------------------------ reporting
let nFail = 0, nPass = 0;
const problems = [];
function ok(cond, label, detail = '') {
  if (cond) { nPass++; console.log(`  ok    ${label}${detail ? '   ' + detail : ''}`); }
  else { nFail++; problems.push(`${label}${detail ? ' — ' + detail : ''}`); console.log(`  FAIL  ${label}${detail ? '   ' + detail : ''}`); }
  return cond;
}
const ex = (v, n = 3) => Number(v).toExponential(n);

// ------------------------------------------------------------------ my own numerics
/** 10 log10( sum ref^2 / sum (ref-got)^2 ), in dB; +inf when identical. */
function snrDb(ref, got) {
  let s = 0, e = 0;
  const n = Math.min(ref.length, got.length);
  for (let i = 0; i < n; i++) { s += ref[i] * ref[i]; const d = ref[i] - got[i]; e += d * d; }
  if (e === 0) return Infinity;
  return 10 * Math.log10(s / e);
}
function worstAbs(a, b) {
  const n = Math.min(a.length, b.length);
  let m = -1, at = -1, va = 0, vb = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m || Number.isNaN(d)) { m = Number.isNaN(d) ? NaN : d; at = i; va = a[i]; vb = b[i]; if (Number.isNaN(d)) break; }
  }
  return { max: m, at, a: va, b: vb, sameLen: a.length === b.length };
}
function bitSame(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false;
  return true;
}
function allFinite(a) { for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return false; return true; }
function rms(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s / Math.max(1, a.length)); }
function peak(a) { let m = 0; for (let i = 0; i < a.length; i++) { const v = Math.abs(a[i]); if (v > m) m = v; } return m; }

/** Fraction of power at or above fCut.  Naive DFT (no FFT), Hann frames of `n`, up to `maxFrames` of them
 *  spread evenly over the signal.  Deliberately a different algorithm from the validator's radix-2 FFT. */
function powerAbove(y, fs, fCut, n = 2048, maxFrames = 12) {
  const win = new Float64Array(n);
  for (let i = 0; i < n; i++) win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / n));
  const nf = Math.max(1, Math.min(maxFrames, Math.floor(y.length / n)));
  const step = Math.max(n, Math.floor((y.length - n) / Math.max(1, nf - 1)) || n);
  const kCut = Math.ceil((fCut * n) / fs);
  let above = 0, total = 0, frames = 0;
  const buf = new Float64Array(n);
  for (let f = 0; f < nf; f++) {
    const s = Math.min(f * step, y.length - n);
    if (s < 0) break;
    for (let i = 0; i < n; i++) buf[i] = y[s + i] * win[i];
    for (let k = 0; k <= n / 2; k++) {
      let re = 0, im = 0;
      const w0 = (-2 * Math.PI * k) / n;
      for (let i = 0; i < n; i++) { const a = w0 * i; re += buf[i] * Math.cos(a); im += buf[i] * Math.sin(a); }
      const p = (re * re + im * im) * (k === 0 || k === n / 2 ? 1 : 2);
      total += p;
      if (k >= kCut) above += p;
    }
    frames++;
  }
  return { frac: total > 0 ? above / total : 0, frames, n };
}

/** Anti-aliased decimation by 2: Blackman-windowed sinc, cutoff 0.5 Nyquist, then every second sample. */
function decimate2(y) {
  const half = 64, taps = 2 * half + 1, h = new Float64Array(taps);
  let sum = 0;
  for (let i = 0; i < taps; i++) {
    const m = i - half;
    const s = m === 0 ? 0.5 : Math.sin(Math.PI * 0.5 * m) / (Math.PI * m);
    const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (taps - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (taps - 1));
    h[i] = s * w; sum += h[i];
  }
  for (let i = 0; i < taps; i++) h[i] /= sum;
  const out = new Float64Array(y.length >> 1);
  for (let j = 0; j < out.length; j++) {
    const c = 2 * j;
    let acc = 0;
    for (let i = 0; i < taps; i++) { const k = c + i - half; if (k >= 0 && k < y.length) acc += h[i] * y[k]; }
    out[j] = acc;
  }
  return out;
}

// ------------------------------------------------------------------ my own PRNG (engine.js's documented contract)
function rngOf(seed) {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** n tau-scaled standard normals pulled from `r`, Box-Muller on consecutive pairs (odd n drops the sine). */
function normals(r, n, tau) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = 1 - r(), u2 = r();
    const rad = Math.sqrt(-2 * Math.log(u1)), th = 2 * Math.PI * u2;
    out[i] = tau * rad * Math.cos(th);
    if (i + 1 < n) out[i + 1] = tau * rad * Math.sin(th);
  }
  return out;
}
/** The engine's documented draw for (seed, tau, L, R, n_noise, n_dec): eps_enc first, then eps_dec. */
function drawEps(seed, tau, L, R, nNoise, nDec) {
  const r = rngOf(seed);
  const enc = normals(r, nNoise * L, tau);
  const dec = nDec > 0 ? normals(r, L * R * nDec, tau) : null;
  return { enc, dec };
}

// ------------------------------------------------------------------ my own .bin reader
function readBin(buf, meta) {
  const dv = new DataView(buf);
  const T = {};
  for (const l of meta.layers) {
    const a = new Float32Array(l.numel);
    for (let i = 0; i < l.numel; i++) a[i] = dv.getFloat32(l.offset + 4 * i, true);
    T[l.name] = { a, shape: l.shape };
  }
  return T;
}

// ------------------------------------------------------------------ setup
if (!existsSync(path.join(VDIR, 'arms.json'))) {
  console.log('running prep.py ...');
  console.log(execFileSync(PY, [path.join(VDIR, 'prep.py')], { encoding: 'utf8' }).split('\n').slice(-9).join('\n'));
}
const arms = JSON.parse(await readFile(path.join(VDIR, 'arms.json'), 'utf8'));
const binck = JSON.parse(await readFile(path.join(VDIR, 'bincheck.json'), 'utf8'));
const manifest = JSON.parse(await readFile(path.join(WDIR, 'manifest.json'), 'utf8'));
const xBuf = await readFile(path.join(VDIR, 'x12k.f32'));
const x12k = new Float32Array(xBuf.buffer.slice(xBuf.byteOffset, xBuf.byteOffset + xBuf.byteLength));
const L = x12k.length;
const armNames = Object.keys(arms);

console.log(`\n=== 0. the weights under test ===`);
console.log(`     ${WDIR}`);
ok(armNames.length === 7, `seven arms in the shipped manifest`, `(${armNames.join(', ')})`);
for (const a of armNames) {
  ok(binck[a].ok, `${a}: .bin == step-${binck[a].step} checkpoint, tensor by tensor`,
    binck[a].ok ? `${binck[a].bytes} B` : binck[a].why.join('; '));
}
// the validator's own inputs, for the record
const staleBin = path.join(REPO, 'demo/assets/weights/es_erb_l0.1.bin');
if (existsSync(staleBin)) {
  const sm = JSON.parse(await readFile(path.join(REPO, 'demo/assets/weights/manifest.json'), 'utf8'));
  const step = sm.arms['es_erb_l0.1'].step;
  console.log(`     note: validate_engine.mjs reads demo/assets/weights (step ${step}, ${Object.keys(sm.arms).length} arm) —`
    + ` not what the app serves`);
}

// ------------------------------------------------------------------ 1. short inputs for the edge-case refs
const xShort = { 'x_1.f32': 1, 'x_3.f32': 3, 'x_11.f32': 11, 'x_10.f32': 10 };
for (const [f, n] of Object.entries(xShort)) {
  await writeFile(path.join(VDIR, f), Buffer.from(x12k.slice(1000, 1000 + n).buffer.slice(0)));
}
const shortX = {};
for (const [f, n] of Object.entries(xShort)) shortX[f] = x12k.slice(1000, 1000 + n);

// ------------------------------------------------------------------ 2. build the jobs and the noise draws
await mkdir(path.join(VDIR, 'eps'), { recursive: true });
const SEED = 3;
const jobs = { arms: {} };
const epsFor = {};                                   // arm -> job name -> {enc, dec}
for (const arm of armNames) {
  const a = arms[arm];
  const list = [];
  const add = async (name, R, tau, xf, xlen) => {
    const rec = { name, R, x: xf, eps_enc: null, eps_dec: null };
    if (tau > 0) {
      const e = drawEps(SEED, tau, xlen, R, a.n_noise, a.n_dec);
      rec.eps_enc = `eps/${arm}_${name}_enc.f32`;
      await writeFile(path.join(VDIR, rec.eps_enc), Buffer.from(e.enc.buffer.slice(0)));
      if (a.n_dec > 0) {
        rec.eps_dec = `eps/${arm}_${name}_dec.f32`;
        await writeFile(path.join(VDIR, rec.eps_dec), Buffer.from(e.dec.buffer.slice(0)));
      }
      epsFor[`${arm}/${name}`] = e;
    }
    list.push(rec);
  };
  await add('r4_tau0', 4, 0, 'x12k.f32', L);
  await add('r8_tau0', 8, 0, 'x12k.f32', L);
  await add('r1_tau0', 1, 0, 'x12k.f32', L);
  await add('r2_tau0', 2, 0, 'x12k.f32', L);
  await add('r3_tau0', 3, 0, 'x12k.f32', L);
  await add('r4_tau1', 4, 1, 'x12k.f32', L);
  await add('r8_tau1', 8, 1, 'x12k.f32', L);
  await add('r4_tau05', 4, 0.5, 'x12k.f32', L);
  await add('r4_tau2', 4, 2, 'x12k.f32', L);
  for (const f of Object.keys(xShort)) await add(`short_${xShort[f]}_r4`, 4, 0, f, xShort[f]);
  await add('short_11_r4_tau1', 4, 1, 'x_11.f32', 11);
  if (a.n_dec > 0) {                                  // one-hot decoder-noise column probes
    for (let d = 0; d < a.n_dec; d++) {
      const enc = new Float32Array(a.n_noise * L);
      const dec = new Float32Array(L * 4 * a.n_dec);
      for (let j = 0; j < L * 4; j++) dec[j * a.n_dec + d] = 1;
      const rec = { name: `onehot_d${d}`, R: 4, x: 'x12k.f32',
        eps_enc: `eps/${arm}_onehot_d${d}_enc.f32`, eps_dec: `eps/${arm}_onehot_d${d}_dec.f32` };
      await writeFile(path.join(VDIR, rec.eps_enc), Buffer.from(enc.buffer.slice(0)));
      await writeFile(path.join(VDIR, rec.eps_dec), Buffer.from(dec.buffer.slice(0)));
      epsFor[`${arm}/onehot_d${d}`] = { enc, dec };
      list.push(rec);
    }
  }
  jobs.arms[arm] = list;
}
await writeFile(path.join(VDIR, 'jobs.json'), JSON.stringify(jobs, null, 1));

console.log(`\n=== 1. PyTorch references (fresh, from lisa_rtm_cache/ckpt/final/*.pt) ===`);
const refLog = execFileSync(PY, [path.join(VDIR, 'refs.py')], { encoding: 'utf8', maxBuffer: 1 << 28 });
console.log(refLog.split('\n').filter((s) => s.trim() && !s.startsWith('  ') && !/^(python|torch|numpy|in colab|device|cache|seeded|preset|rates|speakers|transport|evaluation|LISA:|OV2|OV3)/.test(s)).join('\n'));
const refsMeta = JSON.parse(await readFile(path.join(VDIR, 'refs.json'), 'utf8'));
console.log(`     CFG ${JSON.stringify(refsMeta.cfg)}`);
ok(refsMeta.cfg.upsample === 4 && refsMeta.cfg.fs_hi === 48000 && refsMeta.cfg.dec_hidden === 144,
  'the torch reference ran at the FULL config', JSON.stringify(refsMeta.cfg));
let worstSub = 0;
for (const r of Object.values(refsMeta.arms)) for (const j of Object.values(r.jobs)) worstSub = Math.max(worstSub, j.subpixel_vs_gather_max);
ok(worstSub < 1e-4, 'torch: _decode_subpixel agrees with _decode_gather on every reference', `max ${ex(worstSub)}`);

async function ref(arm, name) {
  const b = await readFile(path.join(VDIR, 'refs', arm, `${name}.f32`));
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

// ------------------------------------------------------------------ 3. per-arm
const table = [];
for (const arm of armNames) {
  const a = arms[arm];
  const meta = manifest.arms[arm];
  console.log(`\n=== ${arm}   ${a.cls}  n_noise ${a.n_noise}  n_dec ${a.n_dec}  det ${a.det}  ${a.param_count} params ===`);
  const model = await E.load(path.join(WDIR, meta.file), path.join(WDIR, 'manifest.json'));
  const row = { arm, cls: a.cls, n_dec: a.n_dec };

  // -- 3a. structure: my own bin parse vs the engine's slicing of dec.net.0.weight
  {
    const buf = await readFile(path.join(WDIR, meta.file));
    const T = readBin(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), meta);
    const W1 = T['dec.net.0.weight'].a, nin = a.nin, H = a.H, C = a.C, K = 3 * C, nd = a.n_dec;
    ok(model.C === C && model.H === H && model.n_dec === nd && model.n_noise === a.n_noise,
      'engine geometry == checkpoint geometry', `C=${model.C} H=${model.H} n_dec=${model.n_dec}`);
    ok(nin === 1 + 3 * C + nd, `dec.net.0 is ${H} x ${nin} = 1 + 3*${C} + ${nd}`);
    let bc = 0, bz = 0, bd = 0;
    for (let o = 0; o < H; o++) {
      if (!Object.is(model.dec.wc[o], W1[o * nin])) bc++;
      for (let t = 0; t < K; t++) if (!Object.is(model.dec.Wz[o * K + t], W1[o * nin + 1 + t])) bz++;
      for (let d = 0; d < nd; d++) if (!Object.is(model.dec.Wd[o * nd + d], W1[o * nin + 1 + K + d])) bd++;
    }
    ok(bc === 0, 'w_c == column 0 of dec.net.0.weight');
    ok(bz === 0, `W_z == columns 1..${K} (the three latent blocks, in order z_{i-1}, z_i, z_{i+1})`);
    if (nd > 0) {
      ok(bd === 0, `W_d == columns ${1 + K}..${nin - 1} — the decoder-noise columns sit AFTER the latent blocks`);
      const wdRef = await ref(arm, 'W_dec_cols');          // torch's own W1[:, 1+3C:], row-major (H, n_dec)
      ok(bitSame(model.dec.Wd, wdRef), 'engine W_d == torch W1[:, 1+3C:] bit for bit');
      ok(model.dec.Wd.length === H * nd, `eps_dec weight block is ${H} x ${nd}`);
    }
  }

  // -- 3b. tau = 0, R = 4 and R = 8
  const y4 = await E.run(model, x12k, { R: 4, tau: 0, chunk: 2048 });
  const r4 = await ref(arm, 'r4_tau0');
  let d = worstAbs(y4, r4);
  row.n4 = y4.length;
  row.max4 = d.max; row.snr4 = snrDb(r4, y4);
  ok(d.sameLen && y4.length === L * 4, `R=4 output is ${L * 4} samples`, `got ${y4.length}`);
  ok(d.max < 1e-4, `tau=0 R=4 max|js - torch| = ${ex(d.max)} < 1e-4`, `at ${d.at} (js ${d.a}, torch ${d.b})`);
  ok(row.snr4 > 80, `tau=0 R=4 SNR(js vs torch) = ${row.snr4.toFixed(2)} dB > 80 dB`);

  const y8 = await E.run(model, x12k, { R: 8, tau: 0, chunk: 2048 });
  const r8 = await ref(arm, 'r8_tau0');
  d = worstAbs(y8, r8);
  row.max8 = d.max; row.snr8 = snrDb(r8, y8);
  ok(d.sameLen && d.max < 1e-4, `tau=0 R=8 max|js - torch| = ${ex(d.max)} < 1e-4`, `SNR ${row.snr8.toFixed(2)} dB`);

  // -- 3c. energy above 24 kHz at R = 8, my own naive DFT
  const eJs = powerAbove(y8, 96000, 24000);
  const eRef = powerAbove(r8, 96000, 24000);
  row.above24 = eJs.frac;
  ok(eJs.frac < 0.005, `R=8: power >= 24 kHz = ${(100 * eJs.frac).toFixed(4)} % (< 0.5 %)`,
    `torch ${(100 * eRef.frac).toFixed(4)} %, ${eJs.frames} naive-DFT Hann frames of ${eJs.n}`);

  // -- 3d. the seeded noise: the engine's own draw vs my independent mulberry32/Box-Muller
  const e1 = epsFor[`${arm}/r4_tau1`];
  const yInj = await E.run(model, x12k, { R: 4, eps: { enc: e1.enc, dec: e1.dec } });
  const ySeed = await E.run(model, x12k, { R: 4, tau: 1, seed: SEED, chunk: 2048 });
  ok(bitSame(yInj, ySeed), `tau=1 seed=${SEED}: engine's internal draw == my own mulberry32 + Box-Muller draw`,
    bitSame(yInj, ySeed) ? '' : `max diff ${ex(worstAbs(yInj, ySeed).max)}`);
  const rT1 = await ref(arm, 'r4_tau1');
  d = worstAbs(ySeed, rT1);
  row.maxT1 = d.max; row.snrT1 = snrDb(rT1, ySeed);
  ok(d.max < 1e-4 && row.snrT1 > 80, `tau=1 seed=${SEED} R=4 vs torch(same eps): max ${ex(d.max)}, SNR ${row.snrT1.toFixed(2)} dB`);
  const y8t1 = await E.run(model, x12k, { R: 8, tau: 1, seed: SEED, chunk: 2048 });
  d = worstAbs(y8t1, await ref(arm, 'r8_tau1'));
  ok(d.max < 1e-4, `tau=1 seed=${SEED} R=8 vs torch(same eps): max ${ex(d.max)}`,
    a.n_dec ? `(eps_dec is ${L * 8} x ${a.n_dec} here)` : '');
  for (const [nm, tau] of [['r4_tau05', 0.5], ['r4_tau2', 2]]) {
    const yy = await E.run(model, x12k, { R: 4, tau, seed: SEED, chunk: 2048 });
    const dd = worstAbs(yy, await ref(arm, nm));
    ok(dd.max < 1e-4, `tau=${tau} seed=${SEED} vs torch: max ${ex(dd.max)}`);
  }

  // -- 3e. the decoder-noise columns, probed one at a time (LISASD only)
  if (a.n_dec > 0) {
    const deltas = [];
    for (let k = 0; k < a.n_dec; k++) {
      const eh = epsFor[`${arm}/onehot_d${k}`];
      const yh = await E.run(model, x12k, { R: 4, eps: { enc: eh.enc, dec: eh.dec } });
      const rh = await ref(arm, `onehot_d${k}`);
      const dd = worstAbs(yh, rh);
      deltas.push(rms(yh.map((v, i) => v - y4[i])));
      ok(dd.max < 1e-4, `eps_dec = one-hot on column ${k} only: max|js - torch| = ${ex(dd.max)}`,
        `(shifts the output by rms ${deltas[k].toExponential(2)})`);
    }
    ok(deltas.every((v) => v > 0), 'every decoder-noise column moves the output',
      deltas.map((v) => v.toExponential(2)).join(' '));
    row.dec_col_rms = deltas.map((v) => +v.toExponential(3));
    // how much does eps_dec matter at all?  If dropping it entirely stayed inside the 1e-4 acceptance
    // tolerance, none of the tau = 0 / max-abs checks could ever see a broken decoder-noise path.
    const zeroDec = new Float32Array(e1.dec.length);
    const yNoDec = await E.run(model, x12k, { R: 4, eps: { enc: e1.enc, dec: zeroDec } });
    const dd = worstAbs(yInj, yNoDec);
    const rr2 = rms(yInj.map((v, i) => v - yNoDec[i]));
    row.dec_contrib = { max: dd.max, rms: rr2, snr: snrDb(yInj, yNoDec) };
    console.log(`     eps_dec at tau=1 moves the output by max ${ex(dd.max)} / rms ${rr2.toExponential(2)} `
      + `(${snrDb(yInj, yNoDec).toFixed(1)} dB below the draw; signal rms ${rms(yInj).toExponential(2)})`);
    ok(dd.max > 1e-4, 'dropping eps_dec would break the 1e-4 acceptance tolerance (so that tolerance CAN see it)',
      dd.max > 1e-4 ? '' : `max ${ex(dd.max)} < 1e-4 — the tau=0 and max-abs checks are blind to the decoder-noise path`);
  }

  // -- 3f. chunking, determinism, tau sensitivity
  const c100 = await E.run(model, x12k, { R: 4, tau: 1, seed: SEED, chunk: 100 });
  const c2048 = await E.run(model, x12k, { R: 4, tau: 1, seed: SEED, chunk: 2048 });
  const c333 = await E.run(model, x12k, { R: 4, tau: 1, seed: SEED, chunk: 333 });
  ok(bitSame(c100, c2048), 'chunk=100 and chunk=2048 bit-identical');
  ok(bitSame(c333, c2048), 'chunk=333 bit-identical too (not a multiple of 4 or R: partial register blocks everywhere)');
  ok(bitSame(c2048, ySeed), 'chunk=2048 == the default-chunk run');
  const c1 = await E.run(model, x12k.subarray(0, 600), { R: 4, tau: 1, seed: SEED, chunk: 1 });
  const cBig = await E.run(model, x12k.subarray(0, 600), { R: 4, tau: 1, seed: SEED, chunk: 65536 });
  ok(bitSame(c1, cBig), 'chunk=1 and chunk=65536 bit-identical (600-sample input)');
  const t1a = await E.run(model, x12k, { R: 4, tau: 1, seed: SEED });
  const t1b = await E.run(model, x12k, { R: 4, tau: 1, seed: SEED });
  ok(bitSame(t1a, t1b), `tau=1 seed=${SEED} run twice bit-identical`);
  const dT = worstAbs(t1a, y4);
  row.tau1_vs_tau0 = dT.max;
  ok(dT.max > 1e-4, `tau=1 differs from tau=0`, `max |y1 - y0| = ${ex(dT.max)} (rms ${rms(t1a.map((v, i) => v - y4[i])).toExponential(2)})`);
  const s4 = await E.run(model, x12k, { R: 4, tau: 1, seed: SEED + 1 });
  ok(!bitSame(t1a, s4), 'seed=3 and seed=4 differ');
  const y0b = await E.run(model, x12k, { R: 4, tau: 0, seed: 12345 });
  ok(bitSame(y0b, y4), 'tau=0 ignores the seed (every arm is the deterministic LISA at tau=0)');

  // -- 3g. R sweep against torch, and the R=2 aliasing claim
  const rr = {};
  for (const R of [1, 2, 3, 4, 8]) {
    const yy = await E.run(model, x12k, { R, tau: 0, chunk: 2048 });
    const rf = await ref(arm, `r${R}_tau0`);
    const dd = worstAbs(yy, rf);
    rr[R] = { max: dd.max, snr: snrDb(rf, yy), n: yy.length };
    ok(yy.length === L * R && dd.max < 1e-4, `R=${R} tau=0 vs torch: max ${ex(dd.max)}, SNR ${rr[R].snr.toFixed(1)} dB, N=${yy.length}`);
  }
  row.R = rr;
  const y2 = await E.run(model, x12k, { R: 2, tau: 0, chunk: 2048 });
  let same2 = y2.length * 2 === y4.length;
  for (let j = 0; same2 && j < y2.length; j++) same2 = Object.is(y2[j], y4[2 * j]);
  ok(same2, 'R=2 is exactly every second sample of R=4 — a direct R=2 query is naive decimation');
  const dec = decimate2(y4);
  const a24 = powerAbove(y4, 48000, 12000);                       // what R=2 would have to fold away
  const y2f = Float64Array.from(y2);
  const nmin = Math.min(dec.length, y2f.length);
  const aliasSnr = snrDb(dec.subarray(0, nmin), y2f.subarray(0, nmin));
  row.alias = { above12k: a24.frac, snr: aliasSnr };
  ok(aliasSnr < 20, `R=2 direct query DOES alias: ${aliasSnr.toFixed(2)} dB against a properly filtered x2 decimation`,
    `(${(100 * a24.frac).toFixed(2)} % of the R=4 power is above 12 kHz and folds)`);

  // -- 3h. short inputs vs torch (receptive field 11)
  for (const [f, n] of Object.entries(xShort)) {
    const ys = await E.run(model, shortX[f], { R: 4, tau: 0 });
    const rs = await ref(arm, `short_${n}_r4`);
    const dd = worstAbs(ys, rs);
    ok(ys.length === n * 4 && dd.max < 1e-4, `L=${n} (receptive field is 11) vs torch: max ${ex(dd.max)}, N=${ys.length}`);
  }
  {
    const ys = await E.run(model, shortX['x_11.f32'], { R: 4, tau: 1, seed: SEED });
    const dd = worstAbs(ys, await ref(arm, 'short_11_r4_tau1'));
    ok(dd.max < 1e-4, `L=11 tau=1 seed=${SEED} vs torch: max ${ex(dd.max)}`);
  }

  table.push(row);
}

// ------------------------------------------------------------------ 4. things a self-validator would not try
const probeArm = 'es_dec_erb_l0.1';                     // the riskiest: LISASD, n_dec = 4
const pm = await E.load(path.join(WDIR, manifest.arms[probeArm].file), path.join(WDIR, 'manifest.json'));
console.log(`\n=== 4. abuse (${probeArm}, LISASD n_dec=4) ===`);

{ // zero length
  let r = null, err = null;
  try { r = await E.run(pm, new Float32Array(0), { R: 4, tau: 1, seed: 1 }); } catch (e) { err = e; }
  ok(err === null && r && r.length === 0, 'L=0 returns an empty Float32Array, no throw', err ? String(err) : '');
  let r2 = null; err = null;
  try { r2 = await E.run(pm, [], { R: 4, tau: 1 }); } catch (e) { err = e; }
  ok(err === null && r2 && r2.length === 0, 'L=0 as a plain Array too', err ? String(err) : '');
}
{ // non-Float32Array inputs
  const arr = Array.from(x12k.subarray(0, 300));
  const f64 = Float64Array.from(arr);
  const a1 = await E.run(pm, arr, { R: 4, tau: 0 });
  const a2 = await E.run(pm, f64, { R: 4, tau: 0 });
  const a3 = await E.run(pm, x12k.subarray(0, 300), { R: 4, tau: 0 });
  ok(bitSame(a1, a3) && bitSame(a2, a3), 'Array and Float64Array inputs give the same bits as Float32Array');
}
{ // seeds
  const seeds = [0, 2 ** 31 - 1, -1, -12345, 2 ** 31];
  const outs = {};
  let bad = '';
  for (const s of seeds) {
    const p = await E.run(pm, x12k.subarray(0, 1200), { R: 4, tau: 1, seed: s });
    const q = await E.run(pm, x12k.subarray(0, 1200), { R: 4, tau: 1, seed: s });
    if (!bitSame(p, q)) bad += ` seed ${s} not reproducible;`;
    if (!allFinite(p)) bad += ` seed ${s} non-finite;`;
    outs[s] = p;
  }
  ok(bad === '', `seeds 0, 2^31-1, -1, -12345, 2^31 all reproducible and finite`, bad);
  ok(!bitSame(outs[0], outs[2 ** 31 - 1]) && !bitSame(outs[0], outs[-12345]) && !bitSame(outs[-1], outs[-12345]),
    'those seeds give different draws');
  const wrapNeg = await E.run(pm, x12k.subarray(0, 1200), { R: 4, tau: 1, seed: -(2 ** 31) });
  console.log(`     seed 2^31 vs seed -2^31: ${bitSame(outs[2 ** 31], wrapNeg) ? 'the same draw (silent int32 wrap)' : 'different'}`);
  const g = await E.run(pm, x12k.subarray(0, 1200), { R: 4, tau: 1, seed: 0.5 });
  const g0 = await E.run(pm, x12k.subarray(0, 1200), { R: 4, tau: 1, seed: 0 });
  console.log(`     seed 0.5: finite ${allFinite(g)}; ${bitSame(g, g0) ? 'truncates to seed 0 (silent)' : 'its own draw'}`);
}
{ // tau
  for (const tau of [0, 0.5, 2.0]) {
    const p = await E.run(pm, x12k.subarray(0, 1200), { R: 4, tau, seed: 1 });
    ok(allFinite(p), `tau=${tau} finite`, `rms ${rms(p).toExponential(3)} peak ${peak(p).toExponential(3)}`);
  }
  const t0 = await E.run(pm, x12k.subarray(0, 1200), { R: 4, tau: 0, seed: 1 });
  const th = await E.run(pm, x12k.subarray(0, 1200), { R: 4, tau: 0.5, seed: 1 });
  const t2 = await E.run(pm, x12k.subarray(0, 1200), { R: 4, tau: 2, seed: 1 });
  const dh = rms(th.map((v, i) => v - t0[i])), d2 = rms(t2.map((v, i) => v - t0[i]));
  ok(dh < d2, 'the departure from tau=0 grows with tau', `rms delta: 0.5 -> ${dh.toExponential(2)}, 2.0 -> ${d2.toExponential(2)}`);
  let nanTau = null;
  try { nanTau = await E.run(pm, x12k.subarray(0, 120), { R: 4, tau: NaN, seed: 1 }); } catch (e) { nanTau = e; }
  console.log(`     tau=NaN: ${nanTau instanceof Error ? 'throws ' + nanTau.message
    : allFinite(nanTau) ? 'silently finite' : 'NaN propagates into the output (not guarded)'}`);
}
{ // the workspace is sized from `chunk`, not from min(chunk, N)
  let peakBuf = 0;
  const base = process.memoryUsage().arrayBuffers;
  await E.run(pm, x12k.subarray(0, 600), { R: 4, tau: 1, seed: 5, chunk: 262144,
    onProgress: () => { peakBuf = Math.max(peakBuf, process.memoryUsage().arrayBuffers - base); } });
  let peakSmall = 0;
  const base2 = process.memoryUsage().arrayBuffers;
  await E.run(pm, x12k.subarray(0, 600), { R: 4, tau: 1, seed: 5, chunk: 2048,
    onProgress: () => { peakSmall = Math.max(peakSmall, process.memoryUsage().arrayBuffers - base2); } });
  console.log(`     a 2400-sample output with chunk=262144 holds ${(peakBuf / 1e6).toFixed(0)} MB of workspace `
    + `(chunk=2048: ${(peakSmall / 1e6).toFixed(1)} MB) — run() never clamps chunk to N`);
}
{ // R
  for (const R of [0, -1, 0.5]) {
    let err = null;
    try { await E.run(pm, x12k.subarray(0, 120), { R, tau: 0 }); } catch (e) { err = e; }
    ok(err !== null, `R=${R} throws`, err ? err.message : 'returned silently');
  }
  const r25 = await E.run(pm, x12k.subarray(0, 120), { R: 2.5, tau: 0 });
  ok(r25.length === 240, 'R=2.5 is truncated to 2 (|0), not rejected', `N=${r25.length} — documented as "positive integer"`);
  let errN = null;
  try { await E.run(pm, x12k.subarray(0, 120), { R: NaN, tau: 0 }); } catch (e) { errN = e; }
  ok(errN !== null, 'R=NaN throws', errN ? errN.message : 'returned silently');
  const r16 = await E.run(pm, x12k.subarray(0, 300), { R: 16, tau: 0 });
  ok(r16.length === 4800 && allFinite(r16), 'R=16 runs (nothing caps R at 8)');
}
{ // chunk
  const a = await E.run(pm, x12k.subarray(0, 600), { R: 4, tau: 1, seed: 5, chunk: 0 });
  const b = await E.run(pm, x12k.subarray(0, 600), { R: 4, tau: 1, seed: 5, chunk: 7 });
  const c = await E.run(pm, x12k.subarray(0, 600), { R: 4, tau: 1, seed: 5, chunk: -3 });
  ok(bitSame(a, b) && bitSame(a, c), 'chunk 0, 7 and -3 all clamp to a working value and agree');
}
{ // onProgress / abort
  const seen = [];
  const y = await E.run(pm, x12k.subarray(0, 4000), { R: 4, tau: 0, chunk: 999, onProgress: (j, N, o) => seen.push([j, N, o.length]) });
  let mono = seen.length > 0 && seen[seen.length - 1][0] === y.length;
  for (let i = 0; i < seen.length; i++) {
    if (seen[i][1] !== y.length || seen[i][2] !== seen[i][0]) mono = false;
    if (i && seen[i][0] <= seen[i - 1][0]) mono = false;
  }
  ok(mono, `onProgress strictly increasing, last == N`, `${seen.length} calls, N=${y.length}`);
  const ac = new AbortController();
  let aborted = null;
  const p = E.run(pm, x12k, { R: 4, tau: 1, seed: 1, chunk: 256, signal: ac.signal }).catch((e) => { aborted = e; });
  ac.abort();
  await p;
  ok(aborted && aborted.name === 'AbortError', 'AbortSignal rejects with an AbortError', aborted ? aborted.message : 'never rejected');
}
{ // a 30-second input
  const secs = FAST ? 5 : 30;
  const big = new Float32Array(12000 * secs);
  for (let i = 0; i < big.length; i++) big[i] = x12k[i % L];
  if (global.gc) global.gc();
  const before = process.memoryUsage();
  const t0 = performance.now();
  let gaps = 0, last = performance.now(), maxGap = 0;
  const yBig = await E.run(pm, big, { R: 4, tau: 1, seed: 1, chunk: 2048,
    onProgress: () => { const n = performance.now(); maxGap = Math.max(maxGap, n - last); last = n; gaps++; } });
  const dt = (performance.now() - t0) / 1000;
  const after = process.memoryUsage();
  ok(yBig.length === big.length * 4 && allFinite(yBig), `${secs} s input -> ${yBig.length} samples, all finite`,
    `${dt.toFixed(2)} s wall`);
  ok(maxGap < 250, `longest gap between event-loop yields ${maxGap.toFixed(1)} ms (stays responsive)`, `${gaps} chunks`);
  const mb = (after.heapUsed - before.heapUsed + after.arrayBuffers - before.arrayBuffers) / 1e6;
  console.log(`     heap+buffers grew ${mb.toFixed(1)} MB, rss ${(after.rss / 1e6).toFixed(0)} MB `
    + `(out ${(yBig.length * 4 / 1e6).toFixed(1)} MB + eps_dec ${(yBig.length * 4 * 4 / 1e6).toFixed(1)} MB `
    + `+ padded latents ${((big.length + 6) * 32 * 4 / 1e6).toFixed(1)} MB)`);
  ok(after.rss < 2.5e9, 'rss stays under 2.5 GB', `${(after.rss / 1e6).toFixed(0)} MB`);
}

// ------------------------------------------------------------------ 5. timing, honestly
console.log(`\n=== 5. timing on this machine (node ${process.version}, ${process.arch}) ===`);
const timings = [];
for (const arm of ['det', 'es_erb_l0.1', 'es_dec_erb_l0.1']) {
  const m = await E.load(path.join(WDIR, manifest.arms[arm].file), path.join(WDIR, 'manifest.json'));
  const secs = FAST ? 2 : 10;
  const xl = new Float32Array(12000 * secs);
  for (let i = 0; i < xl.length; i++) xl[i] = x12k[i % L];
  await E.run(m, xl.subarray(0, 24000), { R: 4, tau: 1, seed: 0, chunk: 2048 });   // warm-up
  const runs = [];
  for (let k = 0; k < 3; k++) {
    const t0 = performance.now();
    const y = await E.run(m, xl, { R: 4, tau: 1, seed: 0, chunk: 2048 });
    runs.push(y.length / 48000 / ((performance.now() - t0) / 1000));
  }
  const t0 = performance.now();
  await E.run(m, xl, { R: 4, tau: 0, chunk: 2048 });
  const det0 = xl.length * 4 / 48000 / ((performance.now() - t0) / 1000);
  const best = Math.max(...runs), med = runs.sort((p, q) => p - q)[1];
  timings.push({ arm, best, med, det0 });
  console.log(`  ${arm.padEnd(18)} tau=1 ${med.toFixed(3)} x realtime (median of 3; best ${best.toFixed(3)})   tau=0 ${det0.toFixed(3)} x`);
}
ok(timings.every((t) => t.med > 0.3), 'every arm faster than 0.3 x realtime in node');
console.log(`  audio_seconds_per_compute_second = ${timings.map((t) => t.med.toFixed(3)).join(' / ')} (det / LISAS / LISASD)`);

// ------------------------------------------------------------------ 6. the table
console.log(`\n=== per-arm summary ===`);
console.log('arm                  cls      n_dec  max|js-t| R4   SNR R4      max R8       SNR R8     >24kHz@R8   tau1-tau0');
for (const r of table) {
  console.log(`${r.arm.padEnd(20)} ${r.cls.padEnd(8)} ${String(r.n_dec).padEnd(6)} `
    + `${ex(r.max4, 2).padEnd(13)} ${r.snr4.toFixed(2).padStart(7)} dB  ${ex(r.max8, 2).padEnd(12)} `
    + `${r.snr8.toFixed(2).padStart(7)} dB  ${(100 * r.above24).toFixed(4).padStart(8)} %  ${ex(r.tau1_vs_tau0, 2)}`);
}
console.log('\nR sweep, max|js - torch| at tau=0');
console.log('arm                    R=1        R=2        R=3        R=4        R=8      alias SNR(R2 vs filtered x2)');
for (const r of table) {
  console.log(`${r.arm.padEnd(20)} ` + [1, 2, 3, 4, 8].map((R) => ex(r.R[R].max, 1).padEnd(10)).join(' ')
    + ` ${r.alias.snr.toFixed(2)} dB`);
}
for (const r of table) if (r.dec_col_rms) console.log(`decoder-noise column response (rms shift), ${r.arm}: ${r.dec_col_rms.join(', ')}`);

console.log(`\n${nPass} passed, ${nFail} failed`);
if (problems.length) { console.log('\nPROBLEMS'); problems.forEach((p, i) => console.log(` ${i + 1}. ${p}`)); }
process.exit(nFail ? 1 : 0);
