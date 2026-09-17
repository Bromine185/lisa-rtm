/* demo/engine.js — LISA / LISAS / LISASD inference in plain JavaScript.
 *
 *   window.LISAEngine.load(binUrl, manifestUrl) -> Promise<model>
 *   window.LISAEngine.run(model, x12k, {R = 4, tau = 1, seed = 0, chunk = 2048, onProgress}) -> Promise<Float32Array>
 *
 * Mirrors overnight2/c1_model.py (LISAS) and overnight3/e1_model.py (LISASD, _decode_subpixel):
 *   encoder   conv1d stack over the channels [x ; eps_enc (n_noise x L)], PyTorch semantics (cross-correlation,
 *             zero 'same' padding k >> 1), ReLU between layers and none after the last -> latents z (L x C)
 *   decoder   output sample j: q = j / R, i = clamp(floor(q), 0, L-1), c = 2 (q - i) - 1
 *             h1 = relu( W_z . [z_{i-1}, z_i, z_{i+1}] + b1  +  c * w_c  +  eps_dec_j . W_d )
 *             layers 2..4: relu(W h + b) with H = 144;  layer 5: scalar.
 *             z_{-1} = z_0 and z_L = z_{L-1} (index clamping = replicate padding).
 *             The latent block of layer 1 depends only on the cell i, so it is evaluated once per input cell
 *             (u[i] = W_z . [z_{i-1}, z_i, z_{i+1}] + b1, a sub-pixel conv) and shared by the R phases.
 *   noise     one mulberry32 stream per (seed): first eps_enc (n_noise x L, channel-major), then eps_dec
 *             (L*R x n_dec, sample-major), Box-Muller, scaled by tau.  tau = 0 skips the draw entirely, so
 *             every arm at tau = 0 is the deterministic LISA.  eps_enc is drawn before eps_dec, so the same
 *             seed gives the same latents at every R.
 *   chunks    the output is produced in blocks of `chunk` samples with `await setTimeout(0)` between them
 *             (in node too); onProgress(jDone, N, out.subarray(0, jDone)) fires after each block.  Every
 *             output sample's arithmetic is a fixed sequential sum, independent of its chunk, so the result is
 *             bit-identical for any chunk size.
 *   numerics  float32 weights and activations, float64 accumulation (a float32 x float32 product is exact in
 *             float64).  Agrees with PyTorch at tau = 0 to ~1e-6.
 *
 * The hot loop (layers 2-4, 144 x 144 per output sample) is register-blocked 4 outputs x 4 samples: weights
 * are read once per four samples, activations once per four outputs, sixteen accumulators live in registers.
 * Activations are stored interleaved by sample block, [(blk * H + unit) * 4 + lane].
 */
(function (root) {
  'use strict';

  const IS_NODE = typeof window === 'undefined';

  // Between chunks the engine gives the event loop a turn so the page can paint: await setTimeout(0), in node
  // too.  Browsers throttle timers in a hidden tab to once a second, which would make a 10 s clip take four
  // minutes in the background, so a hidden document yields through a MessageChannel task instead (a full
  // event-loop turn, nothing to paint, not throttled).  A visible document always uses the timer.
  let channel = null;
  const yieldNow = () => new Promise((resolve) => {
    if (typeof document !== 'undefined' && document.hidden && typeof MessageChannel !== 'undefined') {
      if (!channel) channel = new MessageChannel();
      channel.port1.onmessage = () => resolve();
      channel.port2.postMessage(null);
    } else {
      setTimeout(resolve, 0);
    }
  });

  // ------------------------------------------------------------------ I/O
  async function readBytes(url) {
    if (IS_NODE) {
      const fs = await import('node:fs/promises');
      const b = await fs.readFile(url);
      return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
    }
    const r = await fetch(url);
    if (!r.ok) throw new Error(`LISAEngine: fetch ${url} -> ${r.status}`);
    return r.arrayBuffer();
  }

  async function readJSON(url) {
    if (IS_NODE) {
      const fs = await import('node:fs/promises');
      return JSON.parse(await fs.readFile(url, 'utf8'));
    }
    const r = await fetch(url);
    if (!r.ok) throw new Error(`LISAEngine: fetch ${url} -> ${r.status}`);
    return r.json();
  }

  function baseName(url) {
    return String(url).split(/[?#]/)[0].split('/').pop();
  }

  function readTensor(dv, layer) {
    const n = layer.numel;
    const out = new Float32Array(n);
    let p = layer.offset;
    for (let i = 0; i < n; i++, p += 4) out[i] = dv.getFloat32(p, true);
    return out;
  }

  /** load(binUrl, manifestUrl): the arm is the .bin's file stem, looked up in manifest.arms. */
  async function load(binUrl, manifestUrl) {
    if (!manifestUrl) manifestUrl = String(binUrl).replace(/[^/]*$/, 'manifest.json');
    const [buf, manifest] = await Promise.all([readBytes(binUrl), readJSON(manifestUrl)]);
    const arms = manifest.arms || {};
    const file = baseName(binUrl);
    let arm = file.replace(/\.bin$/, '');
    if (!arms[arm]) {
      arm = Object.keys(arms).find((k) => arms[k].file === file);
      if (!arm && Object.keys(arms).length === 1) arm = Object.keys(arms)[0];
      if (!arm) throw new Error(`LISAEngine: ${file} is not in ${manifestUrl}`);
    }
    const meta = arms[arm];
    if (buf.byteLength < meta.bytes) throw new Error(`LISAEngine: ${file} is ${buf.byteLength} B, manifest says ${meta.bytes}`);
    const dv = new DataView(buf);
    const T = {};
    for (const layer of meta.layers) T[layer.name] = readTensor(dv, layer);

    // encoder: conv layers in module order enc.0, enc.2, ...
    const enc = [];
    let cin = 1 + meta.n_noise;
    const encIdx = meta.layers.filter((l) => /^enc\.\d+\.weight$/.test(l.name))
      .map((l) => ({ idx: +l.name.split('.')[1], shape: l.shape })).sort((a, b) => a.idx - b.idx);
    for (const { idx, shape } of encIdx) {
      const [cout, ci, k] = shape;
      if (ci !== cin) throw new Error(`LISAEngine: enc.${idx} expects ${ci} channels, got ${cin}`);
      enc.push({ cin, cout, k, pad: k >> 1, W: T[`enc.${idx}.weight`], b: T[`enc.${idx}.bias`] });
      cin = cout;
    }
    const C = cin;

    // decoder: Linear layers in module order dec.net.0, dec.net.2, ...
    const decIdx = meta.layers.filter((l) => /^dec\.net\.\d+\.weight$/.test(l.name))
      .map((l) => ({ idx: +l.name.split('.')[2], shape: l.shape })).sort((a, b) => a.idx - b.idx);
    const W1 = T[`dec.net.${decIdx[0].idx}.weight`], b1 = T[`dec.net.${decIdx[0].idx}.bias`];
    const H = decIdx[0].shape[0], nin = decIdx[0].shape[1];
    const n_dec = nin - 1 - 3 * C;
    if (n_dec < 0 || n_dec !== (meta.n_dec | 0)) throw new Error(`LISAEngine: dec.net.0 is ${H} x ${nin}; C=${C}, manifest n_dec=${meta.n_dec}`);
    const K = 3 * C;
    const wc = new Float32Array(H), Wz = new Float32Array(H * K), Wd = new Float32Array(H * n_dec);
    for (let o = 0; o < H; o++) {
      wc[o] = W1[o * nin];
      for (let t = 0; t < K; t++) Wz[o * K + t] = W1[o * nin + 1 + t];
      for (let d = 0; d < n_dec; d++) Wd[o * n_dec + d] = W1[o * nin + 1 + K + d];
    }
    const hidden = [];
    for (let l = 1; l < decIdx.length - 1; l++) {
      const { idx, shape } = decIdx[l];
      if (shape[0] !== H || shape[1] !== H) throw new Error(`LISAEngine: dec.net.${idx} is ${shape}, expected ${H} x ${H}`);
      hidden.push({ W: T[`dec.net.${idx}.weight`], b: T[`dec.net.${idx}.bias`] });
    }
    const last = decIdx[decIdx.length - 1];
    if (last.shape[0] !== 1 || last.shape[1] !== H) throw new Error(`LISAEngine: dec.net.${last.idx} is ${last.shape}, expected 1 x ${H}`);
    const W5 = T[`dec.net.${last.idx}.weight`], b5 = T[`dec.net.${last.idx}.bias`][0];

    return {
      arm, cls: meta.cls, n_noise: meta.n_noise, n_dec, det: !!meta.det, step: meta.step,
      param_count: meta.param_count, R: meta.R || 4, C, H, meta,
      enc, dec: { wc, Wz, Wd, b1, hidden, W5, b5 },
    };
  }

  // ------------------------------------------------------------------ noise
  function mulberry32(a) {
    a |= 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** out[off .. off+n) <- tau * N(0, 1), Box-Muller on consecutive pairs of the stream. */
  function gaussFill(rng, out, off, n, tau) {
    for (let i = 0; i < n; i += 2) {
      const u1 = 1 - rng(), u2 = rng();                 // u1 in (0, 1]: log never sees 0
      const r = Math.sqrt(-2 * Math.log(u1)), th = 6.283185307179586 * u2;
      out[off + i] = tau * r * Math.cos(th);
      if (i + 1 < n) out[off + i + 1] = tau * r * Math.sin(th);
    }
  }

  // ------------------------------------------------------------------ encoder
  /** out[co, t] = b[co] + sum_{ci,kk} W[co, ci, kk] * P[ci, t + kk], P zero-padded (cin x Lp), optional ReLU. */
  function conv1d(P, cin, Lp, W, b, cout, k, L, out, relu) {
    const ck = cin * k, L4 = L - (L & 3);
    for (let co = 0; co < cout; co++) {
      const wb = co * ck, bo = b[co], ob = co * L;
      for (let t = 0; t < L4; t += 4) {
        let a0 = bo, a1 = bo, a2 = bo, a3 = bo;
        for (let ci = 0; ci < cin; ci++) {
          const pb = ci * Lp + t, wr = wb + ci * k;
          for (let kk = 0; kk < k; kk++) {
            const w = W[wr + kk], p = pb + kk;
            a0 += w * P[p]; a1 += w * P[p + 1]; a2 += w * P[p + 2]; a3 += w * P[p + 3];
          }
        }
        const q = ob + t;
        if (relu) {
          out[q] = a0 > 0 ? a0 : 0; out[q + 1] = a1 > 0 ? a1 : 0; out[q + 2] = a2 > 0 ? a2 : 0; out[q + 3] = a3 > 0 ? a3 : 0;
        } else {
          out[q] = a0; out[q + 1] = a1; out[q + 2] = a2; out[q + 3] = a3;
        }
      }
      for (let t = L4; t < L; t++) {                    // tail: same summation order per sample
        let a = bo;
        for (let ci = 0; ci < cin; ci++) {
          const pb = ci * Lp + t, wr = wb + ci * k;
          for (let kk = 0; kk < k; kk++) a += W[wr + kk] * P[pb + kk];
        }
        out[ob + t] = relu ? (a > 0 ? a : 0) : a;
      }
    }
  }

  /** X: (1 + n_noise) x L channel-major.  Returns zp: replicate-padded latents, row r = z_{r-1}, (L + 6) x C,
   *  the four extra zero rows let the layer-1 kernel run whole 4-cell blocks past the end. */
  async function encode(model, X, L) {
    let inp = X, cin = 1 + model.n_noise;
    const nl = model.enc.length;
    for (let l = 0; l < nl; l++) {
      const { cout, k, pad, W, b } = model.enc[l];
      let P = inp, Lp = L;
      if (pad > 0) {
        Lp = L + 2 * pad;
        P = new Float32Array(cin * Lp);
        for (let ci = 0; ci < cin; ci++) P.set(inp.subarray(ci * L, ci * L + L), ci * Lp + pad);
      }
      const out = new Float32Array(cout * L);
      conv1d(P, cin, Lp, W, b, cout, k, L, out, l < nl - 1);
      inp = out; cin = cout;
      if (l < nl - 1) await yieldNow();
    }
    const C = cin;
    const zp = new Float32Array((L + 6) * C);
    for (let c = 0; c < C; c++) {
      const src = c * L;
      for (let n = 0; n < L; n++) zp[(n + 1) * C + c] = inp[src + n];
      zp[c] = inp[src];
      zp[(L + 1) * C + c] = inp[src + L - 1];
    }
    return zp;
  }

  // ------------------------------------------------------------------ decoder kernels
  /** u[n] = W_z . [z_{n-1}, z_n, z_{n+1}] + b1 for cells i0 .. i0 + nCells (rounded up to 4), U row-major (cell x H).
   *  zp row r holds z_{r-1}, so cell n's three taps are rows n, n+1, n+2 and tap k / channel c of cell n + s sits
   *  at zp[(n + s) * C + k * C + c] = zp[n * C + t + s * C] with t = k * C + c: one flat index per lane. */
  function layer1(zp, C, Wz, b1, H, i0, nCells, U) {
    const K = 3 * C, C2 = 2 * C, C3 = 3 * C;
    for (let blk = 0; blk < nCells; blk += 4) {
      const zb = (i0 + blk) * C, ub = blk * H;
      for (let o = 0; o < H; o += 4) {
        const bA = b1[o], bB = b1[o + 1], bC = b1[o + 2], bD = b1[o + 3];
        let a0 = bA, a1 = bA, a2 = bA, a3 = bA, c0 = bB, c1 = bB, c2 = bB, c3 = bB;
        let d0 = bC, d1 = bC, d2 = bC, d3 = bC, e0 = bD, e1 = bD, e2 = bD, e3 = bD;
        const w0 = o * K, w1 = w0 + K, w2 = w1 + K, w3 = w2 + K;
        for (let t = 0; t < K; t++) {
          const p = zb + t;
          const h0 = zp[p], h1 = zp[p + C], h2 = zp[p + C2], h3 = zp[p + C3];
          const wa = Wz[w0 + t]; a0 += wa * h0; a1 += wa * h1; a2 += wa * h2; a3 += wa * h3;
          const wb = Wz[w1 + t]; c0 += wb * h0; c1 += wb * h1; c2 += wb * h2; c3 += wb * h3;
          const wd = Wz[w2 + t]; d0 += wd * h0; d1 += wd * h1; d2 += wd * h2; d3 += wd * h3;
          const we = Wz[w3 + t]; e0 += we * h0; e1 += we * h1; e2 += we * h2; e3 += we * h3;
        }
        let q = ub + o;
        U[q] = a0; U[q + 1] = c0; U[q + 2] = d0; U[q + 3] = e0; q += H;
        U[q] = a1; U[q + 1] = c1; U[q + 2] = d1; U[q + 3] = e1; q += H;
        U[q] = a2; U[q + 1] = c2; U[q + 2] = d2; U[q + 3] = e2; q += H;
        U[q] = a3; U[q + 1] = c3; U[q + 2] = d3; U[q + 3] = e3;
      }
    }
  }

  /** h1 for samples j0 .. j1: relu(u[i_j] + c_j * w_c (+ eps_dec_j . W_d)), written interleaved by 4-sample block. */
  function gather(U, i0, wc, Wd, nd, epsDec, H, R, L, j0, j1, h) {
    const Lm1 = L - 1;
    for (let j = j0; j < j1; j++) {
      const s = j - j0, q = j / R;
      let i = Math.floor(q);
      if (i < 0) i = 0; else if (i > Lm1) i = Lm1;
      const c = 2 * (q - i) - 1;
      const ub = (i - i0) * H, hb = (s >> 2) * H * 4 + (s & 3);
      if (nd > 0) {
        const eb = j * nd;
        for (let o = 0; o < H; o++) {
          let v = U[ub + o] + c * wc[o];
          const wb = o * nd;
          for (let d = 0; d < nd; d++) v += epsDec[eb + d] * Wd[wb + d];
          h[hb + o * 4] = v > 0 ? v : 0;
        }
      } else {
        for (let o = 0; o < H; o++) {
          const v = U[ub + o] + c * wc[o];
          h[hb + o * 4] = v > 0 ? v : 0;
        }
      }
    }
  }

  /** out = relu(W h + b) over NB blocks of 4 samples; 4 outputs x 4 samples per register block. */
  function dense(W, b, h, out, H, NB) {
    for (let blk = 0; blk < NB; blk++) {
      const hb = blk * H * 4, ob = hb;
      for (let o = 0; o < H; o += 4) {
        const bA = b[o], bB = b[o + 1], bC = b[o + 2], bD = b[o + 3];
        let a0 = bA, a1 = bA, a2 = bA, a3 = bA, c0 = bB, c1 = bB, c2 = bB, c3 = bB;
        let d0 = bC, d1 = bC, d2 = bC, d3 = bC, e0 = bD, e1 = bD, e2 = bD, e3 = bD;
        const w0 = o * H, w1 = w0 + H, w2 = w1 + H, w3 = w2 + H;
        for (let i = 0, p = hb; i < H; i++, p += 4) {
          const h0 = h[p], h1 = h[p + 1], h2 = h[p + 2], h3 = h[p + 3];
          const wa = W[w0 + i]; a0 += wa * h0; a1 += wa * h1; a2 += wa * h2; a3 += wa * h3;
          const wb = W[w1 + i]; c0 += wb * h0; c1 += wb * h1; c2 += wb * h2; c3 += wb * h3;
          const wd = W[w2 + i]; d0 += wd * h0; d1 += wd * h1; d2 += wd * h2; d3 += wd * h3;
          const we = W[w3 + i]; e0 += we * h0; e1 += we * h1; e2 += we * h2; e3 += we * h3;
        }
        const q = ob + o * 4;
        out[q] = a0 > 0 ? a0 : 0; out[q + 1] = a1 > 0 ? a1 : 0; out[q + 2] = a2 > 0 ? a2 : 0; out[q + 3] = a3 > 0 ? a3 : 0;
        out[q + 4] = c0 > 0 ? c0 : 0; out[q + 5] = c1 > 0 ? c1 : 0; out[q + 6] = c2 > 0 ? c2 : 0; out[q + 7] = c3 > 0 ? c3 : 0;
        out[q + 8] = d0 > 0 ? d0 : 0; out[q + 9] = d1 > 0 ? d1 : 0; out[q + 10] = d2 > 0 ? d2 : 0; out[q + 11] = d3 > 0 ? d3 : 0;
        out[q + 12] = e0 > 0 ? e0 : 0; out[q + 13] = e1 > 0 ? e1 : 0; out[q + 14] = e2 > 0 ? e2 : 0; out[q + 15] = e3 > 0 ? e3 : 0;
      }
    }
  }

  /** y_j = W5 . h4_j + b5 for samples j0 .. j1. */
  function readout(W5, b5, h, H, j0, j1, out) {
    const n = j1 - j0;
    for (let s = 0; s < n; s += 4) {
      const hb = (s >> 2) * H * 4;
      let a0 = b5, a1 = b5, a2 = b5, a3 = b5;
      for (let o = 0, p = hb; o < H; o++, p += 4) {
        const w = W5[o];
        a0 += w * h[p]; a1 += w * h[p + 1]; a2 += w * h[p + 2]; a3 += w * h[p + 3];
      }
      const j = j0 + s;
      out[j] = a0;
      if (s + 1 < n) out[j + 1] = a1;
      if (s + 2 < n) out[j + 2] = a2;
      if (s + 3 < n) out[j + 3] = a3;
    }
  }

  function decodeChunk(model, zp, L, R, j0, j1, epsDec, out, ws) {
    const { wc, Wz, Wd, b1, hidden, W5, b5 } = model.dec;
    const H = model.H, C = model.C, nd = model.n_dec;
    const Lm1 = L - 1;
    let i0 = Math.floor(j0 / R), i1 = Math.floor((j1 - 1) / R);
    if (i0 < 0) i0 = 0; else if (i0 > Lm1) i0 = Lm1;
    if (i1 < 0) i1 = 0; else if (i1 > Lm1) i1 = Lm1;
    const nCells = i1 - i0 + 1, NB = (j1 - j0 + 3) >> 2;
    layer1(zp, C, Wz, b1, H, i0, nCells, ws.U);
    gather(ws.U, i0, wc, Wd, nd, epsDec, H, R, L, j0, j1, ws.A);
    let a = ws.A, b = ws.B;
    for (let l = 0; l < hidden.length; l++) {
      dense(hidden[l].W, hidden[l].b, a, b, H, NB);
      const t = a; a = b; b = t;
    }
    readout(W5, b5, a, H, j0, j1, out);
  }

  // ------------------------------------------------------------------ run
  /**
   * run(model, x12k, {R, tau, seed, chunk, onProgress}) -> Promise<Float32Array> of length L * R.
   *   R          output samples per input sample (4 -> 48 kHz from 12 kHz; 8 queries the same latents twice as densely)
   *   tau        noise temperature; 0 is the deterministic LISA
   *   seed       integer; the same seed reproduces the draw
   *   chunk      output samples per block; the engine yields to the event loop after each block
   *   onProgress (jDone, N, out.subarray(0, jDone)) after each block
   *   signal     optional AbortSignal; run() rejects with an AbortError at the next block boundary
   *   eps        optional {enc: Float32Array(n_noise * L, channel-major), dec: Float32Array(L * R * n_dec)}
   *              to inject a noise draw instead of sampling one (already scaled; tau is ignored for it)
   */
  async function run(model, x12k, opts) {
    opts = opts || {};
    const R = opts.R === undefined ? 4 : opts.R | 0;
    const tau = opts.tau === undefined ? 1 : +opts.tau;
    const seed = opts.seed === undefined ? 0 : opts.seed;
    const chunk = Math.max(1, opts.chunk === undefined ? 2048 : opts.chunk | 0);
    const onProgress = typeof opts.onProgress === 'function' ? opts.onProgress : null;
    const signal = opts.signal || null;
    if (!(R >= 1)) throw new Error(`LISAEngine.run: R must be a positive integer, got ${opts.R}`);
    const x = x12k instanceof Float32Array ? x12k : Float32Array.from(x12k);
    const L = x.length, N = L * R, H = model.H, nn = model.n_noise, nd = model.n_dec;
    const out = new Float32Array(N);
    if (L === 0) return out;

    // noise: eps_enc into the encoder input channels 1.., eps_dec per output sample
    const X = new Float32Array((1 + nn) * L);
    X.set(x, 0);
    let epsDec = null;
    if (opts.eps) {
      if (opts.eps.enc) X.set(opts.eps.enc.subarray(0, nn * L), L);
      if (nd > 0) { epsDec = new Float32Array(N * nd); if (opts.eps.dec) epsDec.set(opts.eps.dec.subarray(0, N * nd)); }
    } else if (tau !== 0) {
      const rng = mulberry32(seed);
      gaussFill(rng, X, L, nn * L, tau);
      if (nd > 0) { epsDec = new Float32Array(N * nd); gaussFill(rng, epsDec, 0, N * nd, tau); }
    } else if (nd > 0) {
      epsDec = new Float32Array(N * nd);
    }

    const zp = await encode(model, X, L);
    if (signal && signal.aborted) throw abortError();
    await yieldNow();

    const maxCells = Math.floor((chunk - 1) / R) + 2, cells4 = (maxCells + 3) & ~3, chunk4 = (chunk + 3) & ~3;
    const ws = { U: new Float32Array(cells4 * H), A: new Float32Array(chunk4 * H), B: new Float32Array(chunk4 * H) };
    for (let j0 = 0; j0 < N; j0 += chunk) {
      const j1 = Math.min(j0 + chunk, N);
      decodeChunk(model, zp, L, R, j0, j1, epsDec, out, ws);
      if (onProgress) onProgress(j1, N, out.subarray(0, j1));
      if (j1 < N) {
        await yieldNow();
        if (signal && signal.aborted) throw abortError();
      }
    }
    return out;
  }

  function abortError() {
    const e = new Error('LISAEngine.run aborted');
    e.name = 'AbortError';
    return e;
  }

  const LISAEngine = { load, run, version: 1, _noise: { mulberry32, gaussFill } };
  if (typeof window !== 'undefined') window.LISAEngine = LISAEngine;
  if (root && root !== (typeof window !== 'undefined' ? window : null)) root.LISAEngine = LISAEngine;
  if (typeof module !== 'undefined' && module.exports) module.exports = LISAEngine;
})(typeof globalThis !== 'undefined' ? globalThis : this);
