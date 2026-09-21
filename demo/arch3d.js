/* arch3d.js — the LISA architecture as a 3D diagram.  window.Arch3D
 *
 *   mount(el, {arm, cls, tokens, dims, onTime}) -> handle
 *   handle.setArm(arm, cls); play(); pause(); seek(t01); setInference(p01); dispose()
 *   handle.focus(stage|null); reset(); panBy(dx, dy); zoomBy(f)   -- the camera, eased
 *
 * Controls: drag orbits; shift-drag, middle/right button, two fingers together or a two-finger
 * swipe pan; pinch or ctrl/⌘-wheel zooms; double-click resets. Orbit never re-fits the framing.
 *
 * Needs three.js r128 UMD (window.THREE) loaded first. Nothing else.
 * Flow runs left to right along X; each stage keeps its own time axis along X too;
 * channels and the decoder's width recede along Z. One scalar t in [0,1] drives the
 * whole timeline; everything that depends on t is a pure function of t, so scrubbing
 * is exact and reduced-motion only removes the idle drift, the noise jitter and the
 * packet shimmer (the things driven by the wall clock, not by t).
 */
(function (global) {
'use strict';

var DEFAULT_TOKENS = {
  ground: '#0A0C10', panel: '#11141A', line: '#1F2430', ink: '#E7EBF2', dim: '#7D8798',
  cold: '#6FA3FF', warm: '#FFB347', truth: '#F2F4F7', good: '#58D68D', bad: '#FF6B6B'
};

// The model's real dimensions (SPEC.md). A page may override any of them through opts.dims.
var DEFAULT_DIMS = {
  fsIn: 12000, fsOut: 48000, R: 4,
  encChannels: [16, 32, 64, 32], encKernels: [7, 3, 3, 1], receptiveField: 11, latent: 32,
  decWidth: 144, decLayers: 5, noiseIn: 8, noiseDec: 4,
  params: { LISA: 86881, LISAS: 87777, LISASD: 88353 }
};

var N_CELLS = 48;          // input samples drawn (12 kHz cells); 4x that at 48 kHz
var ANCHOR = 24;           // the cell whose packet we follow through the network
var CYCLE_S = 16;          // seconds for t: 0 -> 1 when playing
var HOLD_S = 2.4;          // pause at t = 1 before looping

var STAGES = [
  [0.00, 0.15], [0.15, 0.45], [0.45, 0.60], [0.60, 0.90], [0.90, 1.00]
];

// ---------- small maths -------------------------------------------------------
function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
function clamp(x, a, b) { return x < a ? a : x > b ? b : x; }
function seg(t, a, b) { return clamp01((t - a) / (b - a)); }
function smooth(a, b, x) { x = clamp01((x - a) / (b - a)); return x * x * (3 - 2 * x); }
function lerp(a, b, f) { return a + (b - a) * f; }
function mulberry32(seed) {
  var a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    var t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randn(rnd) {
  var u = 1 - rnd(), v = rnd();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function rgb(hex) { var c = new THREE.Color(hex); return [c.r, c.g, c.b]; }
function mixc(a, b, f) { return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]; }
function put3(arr, i, a, b, f) {   // arr[i..i+2] = mix(a, b, f)
  arr[i] = a[0] + (b[0] - a[0]) * f;
  arr[i + 1] = a[1] + (b[1] - a[1]) * f;
  arr[i + 2] = a[2] + (b[2] - a[2]) * f;
}
function put3s(arr, i, a, s) {     // arr[i..i+2] = a * s
  arr[i] = a[0] * s; arr[i + 1] = a[1] * s; arr[i + 2] = a[2] * s;
}
function fmtInt(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }

// ---------- the synthetic signal ---------------------------------------------
// A voiced pulse with formant ringing plus a fricative burst, built from sinusoids so the
// 6 kHz split is exact: lf = everything below 6 kHz (what the 12 kHz input keeps), hf = the
// rest (what the model has to put back). hfB is a second draw of the burst's phases — the
// sampler's output is a plausible high band, not the one that was removed.
function synth(nCells, R, seed) {
  var N = nCells * R, fs = 48000, rnd = mulberry32(seed);
  var comps = [];
  var f0 = 168, n0 = 7;
  var formants = [[560, 120, 1.0], [1620, 170, 0.55], [2700, 240, 0.30], [3900, 330, 0.16]];
  for (var h = 1; h * f0 < 20000; h++) {
    var f = h * f0, a = 0;
    for (var k = 0; k < formants.length; k++) {
      var d = (f - formants[k][0]) / (formants[k][1] / 2);
      a += formants[k][2] / Math.sqrt(1 + d * d);
    }
    a *= Math.pow(1 / h, 0.35);
    if (f > 4000) a *= Math.pow(4000 / f, 1.15);
    comps.push({ f: f, a: a, phi: 0.12 * randn(rnd), noise: false });
  }
  for (var m = 0; m < 96; m++) {
    var fn = 2600 + rnd() * 18000;
    var an = 0.065 * (fn < 5000 ? (fn - 2600) / 2400 : 1) * (0.55 + 0.45 * Math.exp(-Math.pow((fn - 9500) / 6500, 2)));
    comps.push({ f: fn, a: an, phi: rnd() * 2 * Math.PI, phi2: rnd() * 2 * Math.PI, noise: true });
  }
  var lf = new Float32Array(N), hf = new Float32Array(N), hfB = new Float32Array(N);
  var c = 0.68 * N, w = 0.30 * N;
  for (var n = 0; n < N; n++) {
    var e = Math.abs(n - c) < w ? 0.5 * (1 + Math.cos(Math.PI * (n - c) / w)) : 0;
    e = 0.05 + 0.95 * e;
    var tt = (n - n0) / fs;
    for (var i = 0; i < comps.length; i++) {
      var q = comps[i];
      var v = q.a * Math.cos(2 * Math.PI * q.f * tt + q.phi);
      if (q.noise) v *= e;
      if (q.f < 6000) lf[n] += v;
      else {
        hf[n] += v;
        hfB[n] += q.noise ? q.a * e * Math.cos(2 * Math.PI * q.f * tt + q.phi2) : v;
      }
    }
  }
  var peak = 1e-6;
  for (n = 0; n < N; n++) peak = Math.max(peak, Math.abs(lf[n] + hf[n]));
  for (n = 0; n < N; n++) { lf[n] /= peak; hf[n] /= peak; hfB[n] /= peak; }
  function envelope(x) {
    var env = new Float32Array(N), mx = 1e-6;
    for (var n = 0; n < N; n++) {
      var s = 0, cnt = 0;
      for (var k = -3; k <= 3; k++) { var j = n + k; if (j >= 0 && j < N) { s += x[j] * x[j]; cnt++; } }
      env[n] = Math.sqrt(s / cnt); mx = Math.max(mx, env[n]);
    }
    for (n = 0; n < N; n++) env[n] = clamp01(Math.pow(env[n] / mx, 0.7));
    return env;
  }
  return { N: N, lf: lf, hf: hf, hfB: hfB, env: envelope(hf), envB: envelope(hfB) };
}

// ---------- the scene -----------------------------------------------------------
function buildDotTexture() {
  var c = document.createElement('canvas'); c.width = c.height = 32;
  var g = c.getContext('2d');
  var grd = g.createRadialGradient(16, 16, 0, 16, 16, 16);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.55, 'rgba(255,255,255,0.85)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd; g.fillRect(0, 0, 32, 32);
  var tex = new THREE.CanvasTexture(c);
  tex.minFilter = THREE.LinearFilter;
  return tex;
}

function dynAttr(n, itemSize) {
  var a = new THREE.BufferAttribute(new Float32Array(n * itemSize), itemSize);
  a.setUsage(THREE.DynamicDrawUsage);
  return a;
}
function makePoints(n, size, dot, blending) {
  var g = new THREE.BufferGeometry();
  g.setAttribute('position', dynAttr(n, 3));
  g.setAttribute('color', dynAttr(n, 3));
  var m = new THREE.PointsMaterial({
    size: size, sizeAttenuation: false, vertexColors: true, map: dot, transparent: true,
    depthWrite: false, alphaTest: 0.02, blending: blending || THREE.NormalBlending
  });
  var p = new THREE.Points(g, m);
  p.frustumCulled = false;
  return { obj: p, pos: g.attributes.position, col: g.attributes.color, mat: m };
}
function makeLine(n, segments, opacity) {
  var g = new THREE.BufferGeometry();
  g.setAttribute('position', dynAttr(n, 3));
  g.setAttribute('color', dynAttr(n, 3));
  var m = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: opacity == null ? 1 : opacity, depthWrite: false });
  var l = segments ? new THREE.LineSegments(g, m) : new THREE.Line(g, m);
  l.frustumCulled = false;
  return { obj: l, pos: g.attributes.position, col: g.attributes.color, mat: m };
}

function injectStyle() {
  if (document.getElementById('a3d-style')) return;
  var s = document.createElement('style');
  s.id = 'a3d-style';
  s.textContent = [
    '.a3d{position:absolute;inset:0;overflow:hidden;background:var(--a3d-ground);user-select:none;-webkit-user-select:none}',
    '.a3d canvas{display:block;width:100%;height:100%;touch-action:pan-y;cursor:grab;outline:none}',
    '.a3d canvas.a3d-drag{cursor:grabbing}',
    '.a3d-ov{position:absolute;inset:0;pointer-events:none;overflow:hidden;',
    ' font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;',
    ' font-size:11px;line-height:1.35;font-variant-numeric:tabular-nums;letter-spacing:.01em}',
    '.a3d-l{position:absolute;left:0;top:0;width:0;height:0;color:var(--a3d-dim);opacity:.62;',
    ' transition:opacity .35s ease,color .35s ease;will-change:transform}',
    '.a3d-l.on{color:var(--a3d-ink);opacity:1}',
    '.a3d-l.warm.on{color:var(--a3d-warm)}',
    '.a3d-l.hid{opacity:0}',
    '.a3d-l .k{position:absolute;left:-2px;top:-2px;width:4px;height:4px;background:currentColor;border-radius:1px}',
    '.a3d-l .b{position:absolute;white-space:nowrap;',
    ' text-shadow:0 0 3px var(--a3d-ground),0 0 6px var(--a3d-ground),0 1px 3px var(--a3d-ground)}',
    '.a3d-l.above .b{left:0;bottom:7px;transform:translateX(-50%)}',
    '.a3d-l.below .b{left:0;top:7px;transform:translateX(-50%)}',
    '.a3d-l.right .b{left:9px;top:-8px}',
    '.a3d-l.left .b{right:9px;top:-8px;text-align:right}',
    '.a3d-l .s{display:block;font-size:10px;color:var(--a3d-dim);margin-top:1px}',
    '.a3d-cap{position:absolute;left:12px;bottom:9px;color:var(--a3d-dim);font-size:10px;letter-spacing:.02em}',
    '.a3d-narrow .a3d-ov{font-size:10px}',
    '.a3d-narrow .a3d-l .s{display:none!important}',
    '.a3d-narrow .a3d-l:not(.on){opacity:0}',
    '@media (prefers-reduced-motion: reduce){.a3d-l{transition:none}}',
    '.a3d-cap b{color:var(--a3d-ink);font-weight:500}',
    '.a3d-cap .w{color:var(--a3d-warm)}'
  ].join('');
  document.head.appendChild(s);
}

function mount(el, opts) {
  if (!global.THREE) throw new Error('Arch3D: three.js (window.THREE) must be loaded first');
  opts = opts || {};
  var tk = Object.assign({}, DEFAULT_TOKENS, opts.tokens || {});
  var D = Object.assign({}, DEFAULT_DIMS, opts.dims || {});
  var R = D.R, N = N_CELLS * R;
  var onTime = typeof opts.onTime === 'function' ? opts.onTime : null;

  var mql = global.matchMedia ? global.matchMedia('(prefers-reduced-motion: reduce)') : null;
  var reducedMotion = !!(mql && mql.matches);

  injectStyle();

  // ---- DOM
  if (getComputedStyle(el).position === 'static') el.style.position = 'relative';
  var wrap = document.createElement('div');
  wrap.className = 'a3d';
  Object.keys(tk).forEach(function (k) { wrap.style.setProperty('--a3d-' + k, tk[k]); });
  var overlay = document.createElement('div');
  overlay.className = 'a3d-ov';
  var caption = document.createElement('div');
  caption.className = 'a3d-cap';
  overlay.appendChild(caption);

  var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'default' });
  renderer.setPixelRatio(Math.min(global.devicePixelRatio || 1, 2));
  renderer.setClearColor(new THREE.Color(tk.ground), 1);
  var canvas = renderer.domElement;
  canvas.setAttribute('aria-label', 'LISA architecture, 3D diagram');
  wrap.appendChild(canvas);
  wrap.appendChild(overlay);
  el.appendChild(wrap);

  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(26, 1, 0.5, 600);

  // ---- colours
  var C = {};
  Object.keys(tk).forEach(function (k) { C[k] = rgb(tk[k]); });
  C.ghost = mixc(C.ground, C.line, 1.25);         // barely-there skeleton
  C.ghost = [Math.min(1, C.ghost[0]), Math.min(1, C.ghost[1]), Math.min(1, C.ghost[2])];
  C.coldDim = mixc(C.ghost, C.cold, 0.32);
  C.coldLit = mixc(C.cold, C.ink, 0.55);
  C.warmDim = mixc(C.ghost, C.warm, 0.35);
  C.slab = mixc(C.panel, C.cold, 0.35);

  // ---- layout (world units; flow along +X, amplitude along Y, channels/width along Z)
  var Lx = {
    inX0: -17, inX1: -10, slabX0: -9, slabGap: 0.34,
    ribX0: -2.5, ribX1: 1.5, ribH: 2.4,
    decX0: 3.0, decPitch: 1.5, grid: 12, nodePitch: 0.2,
    outX0: 10.5, outX1: 17.5, warmLift: 1.9, groundY: -2.8
  };
  // The floor is the diagram's footprint, nothing more: it is what the camera fit measures,
  // so anything drawn on it is guaranteed to be in frame.
  var FLOOR = { x0: Lx.inX0 - 0.8, x1: Lx.outX1 + 0.6, z0: -1.7, z1: 2.6 };
  var dot = buildDotTexture();
  var rnd = mulberry32(7);

  // ---- signal
  var S = synth(N_CELLS, R, 11);
  var x48 = new Float32Array(N), x48o = new Float32Array(N);
  for (var j = 0; j < N; j++) {
    x48[j] = Lx.inX0 + (Lx.inX1 - Lx.inX0) * j / (N - 1);
    x48o[j] = Lx.outX0 + (Lx.outX1 - Lx.outX0) * j / (N - 1);
  }
  var cellX = function (i) { return x48[i * R]; };
  var cellXo = function (i) { return x48o[i * R]; };

  // ---- ground grid (one draw call)
  (function () {
    var verts = [], i;
    for (i = 0; i <= 18; i++) {
      var x = FLOOR.x0 + (FLOOR.x1 - FLOOR.x0) * i / 18;
      verts.push(x, Lx.groundY, FLOOR.z0, x, Lx.groundY, FLOOR.z1);
    }
    for (i = 0; i <= 4; i++) {
      var z = FLOOR.z0 + (FLOOR.z1 - FLOOR.z0) * i / 4;
      verts.push(FLOOR.x0, Lx.groundY, z, FLOOR.x1, Lx.groundY, z);
    }
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verts), 3));
    var m = new THREE.LineBasicMaterial({ color: new THREE.Color(tk.line), transparent: true, opacity: 0.55 });
    scene.add(new THREE.LineSegments(g, m));
  })();

  // ---- stage connectors: hairlines along the flow at y = 0
  var slabs = [];  // {x0,x1,cx,thick,foot,ch,k}
  (function () {
    var x = Lx.slabX0;
    for (var k = 0; k < D.encChannels.length; k++) {
      var thick = D.encChannels[k] / 32, foot = D.encKernels[k] * 0.4;
      slabs.push({ x0: x, x1: x + thick, cx: x + thick / 2, thick: thick, foot: foot, ch: D.encChannels[k], k: D.encKernels[k] });
      x += thick + Lx.slabGap;
    }
  })();
  var slabEnd = slabs[slabs.length - 1].x1;
  var decX = [];
  for (var k = 0; k < D.decLayers; k++) decX.push(Lx.decX0 + k * Lx.decPitch);
  (function () {
    var v = [
      Lx.inX1 + 0.15, 0, 0, slabs[0].x0, 0, 0,
      slabEnd, 0, 0, Lx.ribX0, 0, 0,
      Lx.ribX1, 0, 0, decX[0] - 0.25, 0, 0,
      decX[decX.length - 1] + 0.25, 0, 0, Lx.outX0 - 0.15, 0, 0
    ];
    for (var i = 0; i < slabs.length - 1; i++) v.push(slabs[i].x1, 0, 0, slabs[i + 1].x0, 0, 0);
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(v), 3));
    var m = new THREE.LineBasicMaterial({ color: new THREE.Color(tk.dim), transparent: true, opacity: 0.35 });
    scene.add(new THREE.LineSegments(g, m));
  })();

  // ---- input waveform: points + line, 4 x N_CELLS samples
  var inPts = makePoints(N, 2.8, dot);
  var inLine = makeLine(N, false, 0.9);
  scene.add(inPts.obj); scene.add(inLine.obj);

  // ---- 6-24 kHz at the input: the band decimation throws away. Same height and colour as
  // the output's warm band, so the eye reads one as the answer to the other.
  var inWarm = makeLine(N, false, 0);
  (function () {
    var w = inWarm.pos.array, c = inWarm.col.array;
    for (var j = 0; j < N; j++) {
      w[j * 3] = x48[j]; w[j * 3 + 1] = Lx.warmLift + S.hf[j] * 2.6; w[j * 3 + 2] = 0;
      put3s(c, j * 3, C.warm, 1);
    }
    inWarm.pos.needsUpdate = inWarm.col.needsUpdate = true;
  })();
  scene.add(inWarm.obj);

  // ---- 8 noise channels behind the waveform (LISAS / LISASD)
  var NZ = D.noiseIn * N_CELLS;
  var nzPts = makePoints(NZ, 2.2, dot);
  var nzA = new Float32Array(NZ), nzB = new Float32Array(NZ), nzPh = new Float32Array(NZ);
  for (var q = 0; q < NZ; q++) { nzA[q] = randn(rnd); nzB[q] = randn(rnd); nzPh[q] = rnd() * 6.283; }
  scene.add(nzPts.obj);

  // ---- receptive-field bracket on the input
  var bracket = makeLine(4, false, 1);
  scene.add(bracket.obj);

  // ---- conv slabs: one InstancedMesh + one merged edge set
  var slabMesh, slabEdges;
  (function () {
    var geo = new THREE.BoxGeometry(1, 1, 1);
    var mat = new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true, opacity: 0.26, depthWrite: false });
    slabMesh = new THREE.InstancedMesh(geo, mat, slabs.length);
    slabMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(slabs.length * 3), 3);
    slabMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    var M = new THREE.Matrix4();
    var ev = [];
    slabs.forEach(function (s, i) {
      M.makeScale(s.thick, s.foot, s.foot);
      M.setPosition(s.cx, 0, 0);
      slabMesh.setMatrixAt(i, M);
      var hx = s.thick / 2, hy = s.foot / 2, hz = s.foot / 2, cx = s.cx;
      var P = [[cx - hx, -hy, -hz], [cx + hx, -hy, -hz], [cx + hx, hy, -hz], [cx - hx, hy, -hz],
               [cx - hx, -hy, hz], [cx + hx, -hy, hz], [cx + hx, hy, hz], [cx - hx, hy, hz]];
      var E = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
      E.forEach(function (e) { ev.push(P[e[0]][0], P[e[0]][1], P[e[0]][2], P[e[1]][0], P[e[1]][1], P[e[1]][2]); });
    });
    slabMesh.instanceMatrix.needsUpdate = true;
    slabMesh.renderOrder = 2;
    scene.add(slabMesh);
    slabEdges = makeLine(slabs.length * 24, true, 0.9);
    slabEdges.pos.array.set(ev); slabEdges.pos.needsUpdate = true;
    scene.add(slabEdges.obj);
  })();

  // ---- the packet that travels through the encoder
  var NP = 56;
  var pk = makePoints(NP, 4.5, dot, THREE.AdditiveBlending);
  var pkOff = new Float32Array(NP * 3), pkPh = new Float32Array(NP);
  for (q = 0; q < NP; q++) {
    var r = Math.pow(rnd(), 0.6), th = rnd() * 6.283, ph = Math.acos(2 * rnd() - 1);
    pkOff[q * 3] = r * Math.sin(ph) * Math.cos(th);
    pkOff[q * 3 + 1] = r * Math.sin(ph) * Math.sin(th);
    pkOff[q * 3 + 2] = r * Math.cos(ph);
    pkPh[q] = rnd() * 6.283;
  }
  scene.add(pk.obj);

  // ---- latent ribbon: N_CELLS x latent cells, one InstancedMesh
  var LAT = D.latent, NR = N_CELLS * LAT;
  var ribMesh, ribVal = new Float32Array(NR);
  var ribCW = (Lx.ribX1 - Lx.ribX0) / N_CELLS, ribRH = Lx.ribH / LAT;
  var ribX = function (i) { return Lx.ribX0 + (i + 0.5) * ribCW; };
  (function () {
    var geo = new THREE.BoxGeometry(ribCW * 0.78, ribRH * 0.78, 0.06);
    var mat = new THREE.MeshLambertMaterial({ color: 0xffffff });
    ribMesh = new THREE.InstancedMesh(geo, mat, NR);
    ribMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(NR * 3), 3);
    ribMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    var M = new THREE.Matrix4();
    for (var i = 0; i < N_CELLS; i++) {
      var e = 0; for (var r = 0; r < R; r++) e += Math.abs(S.lf[i * R + r]);
      e /= R;
      for (var d = 0; d < LAT; d++) {
        var idx = i * LAT + d;
        M.makeTranslation(ribX(i), -Lx.ribH / 2 + (d + 0.5) * ribRH, 0);
        ribMesh.setMatrixAt(idx, M);
        var v = 0.8 * Math.sin(0.33 * i + 0.9 * d + 0.012 * i * d) + 1.6 * e * Math.sin(0.5 * d + 1.1) + 0.35 * randn(rnd);
        ribVal[idx] = Math.tanh(v);
      }
    }
    ribMesh.instanceMatrix.needsUpdate = true;
    scene.add(ribMesh);
  })();

  // ---- decoder: decLayers columns of decWidth nodes, one InstancedMesh
  var NW = D.decWidth, ND = D.decLayers * NW, G = Lx.grid;
  var nodeMesh, nodeAct = new Float32Array(ND), nodePos = new Float32Array(ND * 3);
  (function () {
    var geo = new THREE.BoxGeometry(0.085, 0.085, 0.085);
    var mat = new THREE.MeshLambertMaterial({ color: 0xffffff });
    nodeMesh = new THREE.InstancedMesh(geo, mat, ND);
    nodeMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(ND * 3), 3);
    nodeMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    var M = new THREE.Matrix4();
    for (var k = 0; k < D.decLayers; k++) {
      for (var n = 0; n < NW; n++) {
        var idx = k * NW + n, row = Math.floor(n / G), colm = n % G;
        var x = decX[k], y = (row - (G - 1) / 2) * Lx.nodePitch, z = (colm - (G - 1) / 2) * Lx.nodePitch;
        nodePos[idx * 3] = x; nodePos[idx * 3 + 1] = y; nodePos[idx * 3 + 2] = z;
        M.makeTranslation(x, y, z);
        nodeMesh.setMatrixAt(idx, M);
        nodeAct[idx] = clamp01(0.2 + 0.6 * randn(rnd));   // ReLU-ish sparsity
      }
    }
    nodeMesh.instanceMatrix.needsUpdate = true;
    scene.add(nodeMesh);
  })();

  // ---- connections between columns (a sparse sample of the dense weights)
  var PER_GAP = 56, NCON = (D.decLayers - 1) * PER_GAP;
  var con = makeLine(NCON * 2, true, 0.75);
  var conW = new Float32Array(NCON), conMid = new Float32Array(NCON);
  (function () {
    var p = con.pos.array;
    for (var g = 0; g < D.decLayers - 1; g++) {
      for (var i = 0; i < PER_GAP; i++) {
        var c = g * PER_GAP + i;
        var a = g * NW + Math.floor(rnd() * NW), b = (g + 1) * NW + Math.floor(rnd() * NW);
        p[c * 6] = nodePos[a * 3]; p[c * 6 + 1] = nodePos[a * 3 + 1]; p[c * 6 + 2] = nodePos[a * 3 + 2];
        p[c * 6 + 3] = nodePos[b * 3]; p[c * 6 + 4] = nodePos[b * 3 + 1]; p[c * 6 + 5] = nodePos[b * 3 + 2];
        conW[c] = 0.35 + 0.65 * rnd();
        conMid[c] = (decX[g] + decX[g + 1]) / 2;
      }
    }
    con.pos.needsUpdate = true;
  })();
  scene.add(con.obj);

  // ---- inlets into the first column: three latents, the coordinate c, and (LISASD) 4 noise values
  var NIN_LAT = 3 * 4, NIN_C = 4, NIN_NZ = D.noiseDec * 2;
  var NIN = NIN_LAT + NIN_C + NIN_NZ;
  var inlet = makeLine(NIN * 2, true, 0.85);
  var inletTarget = new Int32Array(NIN);
  for (q = 0; q < NIN; q++) inletTarget[q] = Math.floor(rnd() * NW);
  scene.add(inlet.obj);
  var cSlider = { x: decX[0] - 1.0, y0: -2.35, y1: -1.45, z: 0 };
  var cRail = makeLine(2, false, 0.8);
  cRail.pos.array.set([cSlider.x, cSlider.y0, cSlider.z, cSlider.x, cSlider.y1, cSlider.z]);
  cRail.pos.needsUpdate = true;
  scene.add(cRail.obj);
  var knob = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.06, 0.16), new THREE.MeshLambertMaterial({ color: new THREE.Color(tk.ink) }));
  scene.add(knob);
  var dnz = { x: decX[0] - 1.0, y: 2.5, z: 0 };
  var dnzPts = makePoints(D.noiseDec, 3.4, dot);
  var dnzA = new Float32Array(D.noiseDec), dnzB = new Float32Array(D.noiseDec), dnzPh = new Float32Array(D.noiseDec);
  for (q = 0; q < D.noiseDec; q++) { dnzA[q] = randn(rnd); dnzB[q] = randn(rnd); dnzPh[q] = rnd() * 6.283; }
  scene.add(dnzPts.obj);

  // ---- output: 48 kHz points, cold baseband line, warm high band above it
  var outPts = makePoints(N, 2.8, dot);
  var outLine = makeLine(N, false, 0.9);
  var warmLine = makeLine(N, false, 0);
  scene.add(outPts.obj); scene.add(outLine.obj); scene.add(warmLine.obj);
  (function () {
    var w = warmLine.pos.array;
    for (var j = 0; j < N; j++) { w[j * 3] = x48o[j]; w[j * 3 + 1] = Lx.warmLift + S.hfB[j] * 2.6; w[j * 3 + 2] = 0; }
    warmLine.pos.needsUpdate = true;
  })();

  // ---- lights (Lambert on the boxes only; points and lines are unlit)
  scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x0a0c10, 0.95));
  var dl = new THREE.DirectionalLight(0xffffff, 0.55);
  dl.position.set(6, 10, 8);
  scene.add(dl);

  // ---- labels (HTML overlay)
  var labels = [];
  function addLabel(anchor, text, sub, align, stage, cls) {
    var d = document.createElement('div');
    d.className = 'a3d-l ' + (align || 'above') + (cls ? ' ' + cls : '');
    var kdot = document.createElement('span'); kdot.className = 'k';
    var b = document.createElement('span'); b.className = 'b';
    var tt = document.createElement('span'); tt.className = 't'; tt.textContent = text;
    var ss = document.createElement('span'); ss.className = 's'; ss.textContent = sub || '';
    if (!sub) ss.style.display = 'none';
    b.appendChild(tt); b.appendChild(ss); d.appendChild(kdot); d.appendChild(b);
    overlay.appendChild(d);
    var L = { el: d, b: b, t: tt, s: ss, anchor: new THREE.Vector3(anchor[0], anchor[1], anchor[2]), stage: stage, hidden: false,
              align: align || 'above', side: null, dx: 0 };
    labels.push(L);
    return L;
  }
  var inCX = (Lx.inX0 + Lx.inX1) / 2, outCX = (Lx.outX0 + Lx.outX1) / 2;
  var NZ_Y = -0.95, NZ_DZ = 0.28, BR_Y = 1.45;
  var lbIn = addLabel([inCX, Lx.warmLift + 0.85, 0], '48 kHz → 12 kHz', 'everything above 6 kHz is gone', 'above', 0);
  var lbNz = addLabel([Lx.inX0 + 2.6, NZ_Y - 0.2, NZ_DZ * D.noiseIn], 'ε  ' + D.noiseIn + ' Gaussian channels', '+' + fmtInt(D.noiseIn * D.encChannels[0] * D.encKernels[0]) + ' weights', 'below', 1, 'warm');
  var lbConv = addLabel([(slabs[0].x0 + slabEnd) / 2, slabs[0].foot / 2 + 0.4, 0], 'conv1d × ' + D.encChannels.length, 'k ' + D.encKernels.join('·') + '   ch ' + D.encChannels.join('·'), 'above', 1);
  var lbRF = addLabel([cellX(ANCHOR), BR_Y - 0.04, 0], 'RF ' + D.receptiveField + ' samples = ' + (D.receptiveField / D.fsIn * 1000).toFixed(1) + ' ms', '', 'below', 1);
  var lbRib = addLabel([(Lx.ribX0 + Lx.ribX1) / 2, Lx.ribH / 2 + 0.25, 0], 'z  ' + D.latent + ' × ' + (D.fsIn / 1000) + ' kHz', 'one latent per input sample', 'above', 2);
  var lbDec = addLabel([(decX[0] + decX[decX.length - 1]) / 2 + 0.6, 1.35, 0], 'decoder  ' + D.decLayers + ' × Linear(' + D.decWidth + ')', '', 'above', 3);
  var lbC = addLabel([cSlider.x, cSlider.y0 - 0.1, 0], 'c = 2(q − i) − 1', 'c ∈ [−1, 1]', 'below', 3);
  var lbDnz = addLabel([dnz.x, dnz.y + 0.2, 0], 'ε  ' + D.noiseDec + ' per output sample', '+' + fmtInt(D.noiseDec * D.decWidth) + ' weights', 'above', 3, 'warm');
  var lbOut = addLabel([Lx.outX0 + 2.2, Lx.warmLift + 0.55, 0], 'output ' + (D.fsOut / 1000) + ' kHz', '×' + R + ' per input sample', 'above', 4);
  var lbWarm = addLabel([Lx.outX1 + 0.15, Lx.warmLift, 0], '6–24 kHz', 'from the model', 'right', 4, 'warm');
  var lbBase = addLabel([Lx.outX1 + 0.15, 0, 0], '0–6 kHz', 'from the input', 'right', 4);

  // ---- state
  var state = {
    t: 0, p: 1, playing: false, arm: opts.arm || 'lisas', cls: opts.cls || 'LISAS',
    holdUntil: 0, dirty: true, needsRender: true, disposed: false
  };
  // The diagram is 35 units of flow by 6 of amplitude. In a wide panel it is read almost side
  // on; the squarer the panel, the further the camera swings round the flow axis so the run
  // folds into the depth and fills the frame. Both ends are the same scene, same fit.
  var WIDE = { az: 0.22, el: 0.36 }, NARROW = { az: 0.80, el: 0.46 };
  function defaultOrbit(aspect) {
    var f = clamp01((2.2 - aspect) / 1.5);
    return { az: lerp(WIDE.az, NARROW.az, f), el: lerp(WIDE.el, NARROW.el, f) };
  }
  var cam = { az: WIDE.az, el: WIDE.el, zoom: 1, d: 40, target: new THREE.Vector3(0.25, 0, 0), driftFrom: 0, lastTouch: -1e9 };
  var goal = { az: WIDE.az, el: WIDE.el, zoom: 1, d: 40, target: new THREE.Vector3(0.25, 0, 0) };
  var narrow = false, userMoved = false, focusIdx = null;
  var AZ_LIM = 1.35, EL_MIN = 0.03, EL_MAX = 1.25, ZOOM_MIN = 0.3, ZOOM_MAX = 3.5;

  function hasNoiseIn() { return state.cls !== 'LISA'; }
  function hasNoiseDec() { return state.cls === 'LISASD'; }

  function setCaption() {
    var n = D.params[state.cls];
    caption.innerHTML = '<b>' + state.cls + '</b>' +
      (state.arm && state.arm.toUpperCase() !== state.cls ? ' · ' + state.arm : '') +
      (n ? ' · ' + fmtInt(n) + ' parameters' : '') +
      (hasNoiseIn() ? ' · <span class="w">ε</span> ' + D.noiseIn + (hasNoiseDec() ? ' + ' + D.noiseDec : '') : '');
    var inW = 1 + 3 * D.latent + (hasNoiseDec() ? D.noiseDec : 0);
    lbDec.s.textContent = '[c, z[i−1], z[i], z[i+1]' + (hasNoiseDec() ? ', ε' : '') + ']  ' + inW + ' → ' + D.decWidth + ' → 1';
    lbDec.s.style.display = '';
  }
  setCaption();

  // ---- the timeline: everything below is a function of (t, p, cls) plus the wall clock for jitter
  var tmpA = [0, 0, 0];
  function applyTimeline(now) {
    var t = state.t, p = state.p;
    var u0 = seg(t, STAGES[0][0], STAGES[0][1]), u1 = seg(t, STAGES[1][0], STAGES[1][1]),
        u2 = seg(t, STAGES[2][0], STAGES[2][1]), u3 = seg(t, STAGES[3][0], STAGES[3][1]),
        u4 = seg(t, STAGES[4][0], STAGES[4][1]);
    var stage = t < STAGES[1][0] ? 0 : t < STAGES[2][0] ? 1 : t < STAGES[3][0] ? 2 : t < STAGES[4][0] ? 3 : 4;
    var live = reducedMotion ? 0 : now;
    var i, j, k, q, f, x, y, c;

    // -- stage 0: the 48 kHz waveform loses its high band and collapses 4:1
    var hfGain = 1 - smooth(0.18, 0.72, u0);   // the high band goes first ...
    var col = smooth(0.52, 1, u0);             // ... then the samples collapse 4:1
    inWarm.mat.opacity = 0.95 * hfGain;
    inWarm.obj.visible = hfGain > 0.01;
    var brOp = stage === 1 ? smooth(0, 0.12, u1) * (1 - smooth(0.72, 0.95, u1)) : 0;
    var hi0 = ANCHOR - (D.receptiveField - 1) / 2, hi1 = ANCHOR + (D.receptiveField - 1) / 2;
    (function () {
      var P = inPts.pos.array, Cc = inPts.col.array, LC = inLine.col.array;
      for (j = 0; j < N; j++) {
        i = Math.floor(j / R);
        var y48 = S.lf[j] + S.hf[j] * hfGain;
        x = lerp(x48[j], x48[i * R], col);
        y = lerp(y48, S.lf[i * R], col);
        P[j * 3] = x; P[j * 3 + 1] = y; P[j * 3 + 2] = 0;
        var w = S.env[j] * hfGain * (1 - col);
        put3(Cc, j * 3, C.cold, C.warm, w);
        if (brOp > 0 && i >= hi0 && i <= hi1) put3(Cc, j * 3, [Cc[j * 3], Cc[j * 3 + 1], Cc[j * 3 + 2]], C.ink, 0.7 * brOp);
        LC[j * 3] = Cc[j * 3] * 0.8; LC[j * 3 + 1] = Cc[j * 3 + 1] * 0.8; LC[j * 3 + 2] = Cc[j * 3 + 2] * 0.8;
      }
      inLine.pos.array.set(P);
      inPts.pos.needsUpdate = inPts.col.needsUpdate = inLine.pos.needsUpdate = inLine.col.needsUpdate = true;
    })();

    // -- noise channels behind the waveform
    nzPts.obj.visible = hasNoiseIn();
    if (nzPts.obj.visible) {
      var P = nzPts.pos.array, Cc = nzPts.col.array;
      // dim while the input still carries its own high band, lit once the encoder reads them
      var bright = 0.38 + 0.62 * (stage === 1 ? smooth(0, 0.15, u1) : 0);
      for (k = 0; k < D.noiseIn; k++) {
        for (i = 0; i < N_CELLS; i++) {
          q = k * N_CELLS + i;
          var jit = reducedMotion ? nzA[q] : nzA[q] * Math.cos(live * 2.1 + nzPh[q]) + nzB[q] * Math.sin(live * 1.7 + nzPh[q] * 0.7);
          P[q * 3] = cellX(i); P[q * 3 + 1] = NZ_Y + 0.13 * jit; P[q * 3 + 2] = NZ_DZ * (k + 1);
          put3s(Cc, q * 3, C.warm, bright * (0.55 + 0.45 * Math.min(1, Math.abs(jit))));
        }
      }
      nzPts.pos.needsUpdate = nzPts.col.needsUpdate = true;
    }

    // -- the receptive-field bracket
    (function () {
      var xa = cellX(hi0) - ribCW * 0.5, xb = cellX(hi1) + ribCW * 0.5, yb = BR_Y;
      bracket.pos.array.set([xa, yb - 0.14, 0, xa, yb, 0, xb, yb, 0, xb, yb - 0.14, 0]);
      bracket.pos.needsUpdate = true;
      var Cc = bracket.col.array;
      for (q = 0; q < 4; q++) put3s(Cc, q * 3, C.ink, 1);
      bracket.col.needsUpdate = true;
      bracket.mat.opacity = brOp;
      bracket.obj.visible = brOp > 0.001;
      lbRF.hidden = brOp < 0.05;
    })();

    // -- the packet through the encoder (stage 1) and across the ribbon (stage 2)
    var px = 0, pAlpha = 0, spread = 0.12;
    if (stage === 1) {
      px = lerp(cellX(ANCHOR), Lx.ribX0 - 0.3, u1);
      pAlpha = smooth(0, 0.08, u1);
    } else if (stage === 2) {
      px = lerp(Lx.ribX0 - 0.3, Lx.ribX1 + 0.2, u2);
      pAlpha = 1 - smooth(0.75, 1, u2);
    }
    var chFrac = 0;
    for (k = 0; k < slabs.length; k++) {
      var s = slabs[k];
      chFrac += (s.ch / 64) * Math.exp(-Math.pow((px - s.cx) / (s.thick / 2 + 0.3), 2));
    }
    spread = 0.12 + 0.55 * Math.min(1, chFrac);
    (function () {
      pk.obj.visible = pAlpha > 0.001;
      pk.mat.opacity = pAlpha;
      if (!pk.obj.visible) return;
      var py = S.lf[ANCHOR * R] * (1 - smooth(cellX(ANCHOR), Lx.inX1, px));
      var P = pk.pos.array, Cc = pk.col.array;
      var nWarm = hasNoiseIn() ? Math.round(NP * 0.3) : 0;
      for (q = 0; q < NP; q++) {
        var sh = reducedMotion ? 1 : 1 + 0.18 * Math.sin(live * 6 + pkPh[q]);
        P[q * 3] = px + pkOff[q * 3] * spread * 0.6 * sh;
        P[q * 3 + 1] = py + pkOff[q * 3 + 1] * spread * sh;
        P[q * 3 + 2] = pkOff[q * 3 + 2] * spread * sh;
        var isWarm = q >= NP - nWarm;
        put3(Cc, q * 3, isWarm ? C.warm : C.cold, C.ink, isWarm ? 0.15 : 0.35);
      }
      pk.pos.needsUpdate = pk.col.needsUpdate = true;
    })();

    // -- slab lighting
    (function () {
      var IC = slabMesh.instanceColor.array, EC = slabEdges.col.array;
      var settled = stage >= 2 ? 0.28 : 0;
      for (k = 0; k < slabs.length; k++) {
        var s = slabs[k];
        var lit = stage === 1 ? Math.exp(-Math.pow((px - s.cx) / (s.thick / 2 + 0.35), 2)) : settled;
        put3(IC, k * 3, C.slab, C.cold, 0.25 + 0.75 * lit);
        for (q = 0; q < 24; q++) put3(EC, (k * 24 + q) * 3, C.dim, C.cold, 0.15 + 0.85 * lit);
      }
      slabMesh.instanceColor.needsUpdate = true;
      slabEdges.col.needsUpdate = true;
    })();

    // -- the latent ribbon: ghost, then revealed left to right behind the packet
    var hiLat = stage === 3 ? smooth(0, 0.1, u3) : stage === 4 ? 0.35 : 0;
    (function () {
      var IC = ribMesh.instanceColor.array;
      for (i = 0; i < N_CELLS; i++) {
        var rev = stage < 2 ? 0 : stage > 2 ? 1 : smooth(0, 1, (u2 * 1.08 - (i + 0.5) / N_CELLS) / 0.06);
        var nb = (i >= ANCHOR - 1 && i <= ANCHOR + 1) ? hiLat : 0;
        for (var d = 0; d < LAT; d++) {
          q = i * LAT + d;
          var v = (ribVal[q] + 1) / 2;
          tmpA[0] = C.ghost[0] + (C.cold[0] - C.ghost[0]) * (0.08 + 0.92 * v);
          tmpA[1] = C.ghost[1] + (C.cold[1] - C.ghost[1]) * (0.08 + 0.92 * v);
          tmpA[2] = C.ghost[2] + (C.cold[2] - C.ghost[2]) * (0.08 + 0.92 * v);
          put3(IC, q * 3, C.ghost, tmpA, rev);
          if (nb > 0) put3(IC, q * 3, [IC[q * 3], IC[q * 3 + 1], IC[q * 3 + 2]], C.ink, 0.5 * nb * (0.4 + 0.6 * v));
        }
      }
      ribMesh.instanceColor.needsUpdate = true;
    })();

    // -- the decoder: four pulses, one per output sample of the anchor cell
    var pulses = R;
    var pulseIdx = stage === 3 ? Math.min(pulses - 1, Math.floor(u3 * pulses)) : stage > 3 ? pulses - 1 : 0;
    var pf = stage === 3 ? Math.min(1, u3 * pulses - pulseIdx) : stage > 3 ? 1 : 0;
    var pulseX = lerp(decX[0] - 0.9, decX[decX.length - 1] + 0.9, pf);
    var decRev = stage < 3 ? 0 : stage === 3 ? smooth(0, 0.08, u3) : 1;
    var flash = stage === 3 ? Math.exp(-Math.pow(pf / 0.14, 2)) : 0;
    var cVal = 2 * (pulseIdx / R) - 1;
    (function () {
      var IC = nodeMesh.instanceColor.array;
      var base = mixc(C.ghost, C.coldDim, decRev);
      for (q = 0; q < ND; q++) {
        var lit = 0;
        if (stage === 3) lit = nodeAct[q] * Math.exp(-Math.pow((nodePos[q * 3] - pulseX) / 0.75, 2));
        else if (stage === 4) lit = 0.18 * nodeAct[q];
        put3(IC, q * 3, base, C.coldLit, lit);
      }
      nodeMesh.instanceColor.needsUpdate = true;
      var CC = con.col.array;
      var lineBase = mixc(C.ghost, C.coldDim, decRev * 0.8);
      for (c = 0; c < NCON; c++) {
        var a = stage === 3 ? conW[c] * Math.exp(-Math.pow((conMid[c] - pulseX) / 0.9, 2)) : stage === 4 ? 0.12 * conW[c] : 0;
        put3(CC, c * 6, lineBase, C.cold, a);
        put3(CC, c * 6 + 3, lineBase, C.cold, a);
      }
      con.col.needsUpdate = true;
    })();

    // -- inlets into the first column
    (function () {
      var P = inlet.pos.array, Cc = inlet.col.array;
      var q0 = 0, n, tgt;
      var inA = decRev * (0.35 + 0.65 * flash);
      for (var m = -1; m <= 1; m++) {
        var xr = ribX(ANCHOR + m), yr = Lx.ribH / 2 + 0.03;
        for (n = 0; n < 4; n++) {
          tgt = inletTarget[q0];
          P[q0 * 6] = xr; P[q0 * 6 + 1] = yr; P[q0 * 6 + 2] = 0;
          P[q0 * 6 + 3] = nodePos[tgt * 3]; P[q0 * 6 + 4] = nodePos[tgt * 3 + 1]; P[q0 * 6 + 5] = nodePos[tgt * 3 + 2];
          put3(Cc, q0 * 6, C.ghost, C.cold, inA);
          put3(Cc, q0 * 6 + 3, C.ghost, C.cold, inA);
          q0++;
        }
      }
      var ky = lerp(cSlider.y0, cSlider.y1, (cVal + 1) / 2);
      knob.position.set(cSlider.x, ky, cSlider.z);
      knob.material.color.setRGB(
        C.dim[0] + (C.ink[0] - C.dim[0]) * decRev, C.dim[1] + (C.ink[1] - C.dim[1]) * decRev, C.dim[2] + (C.ink[2] - C.dim[2]) * decRev);
      for (n = 0; n < NIN_C; n++) {
        tgt = inletTarget[q0];
        P[q0 * 6] = cSlider.x; P[q0 * 6 + 1] = ky; P[q0 * 6 + 2] = cSlider.z;
        P[q0 * 6 + 3] = nodePos[tgt * 3]; P[q0 * 6 + 4] = nodePos[tgt * 3 + 1]; P[q0 * 6 + 5] = nodePos[tgt * 3 + 2];
        put3(Cc, q0 * 6, C.ghost, C.dim, decRev * (0.4 + 0.6 * flash));
        put3(Cc, q0 * 6 + 3, C.ghost, C.dim, decRev * (0.4 + 0.6 * flash));
        q0++;
      }
      var showDnz = hasNoiseDec();
      dnzPts.obj.visible = showDnz;
      var DP = dnzPts.pos.array, DC = dnzPts.col.array;
      for (n = 0; n < D.noiseDec; n++) {
        var jit = reducedMotion ? dnzA[n] : dnzA[n] * Math.cos(live * 2.3 + dnzPh[n]) + dnzB[n] * Math.sin(live * 1.9 + dnzPh[n] * 0.7);
        var nx = dnz.x + (n - (D.noiseDec - 1) / 2) * 0.18, ny = dnz.y + 0.12 * jit, nzz = dnz.z;
        DP[n * 3] = nx; DP[n * 3 + 1] = ny; DP[n * 3 + 2] = nzz;
        put3s(DC, n * 3, C.warm, 0.6 + 0.4 * decRev);
        for (var e = 0; e < 2; e++) {
          tgt = inletTarget[q0];
          P[q0 * 6] = nx; P[q0 * 6 + 1] = ny; P[q0 * 6 + 2] = nzz;
          P[q0 * 6 + 3] = nodePos[tgt * 3]; P[q0 * 6 + 4] = nodePos[tgt * 3 + 1]; P[q0 * 6 + 5] = nodePos[tgt * 3 + 2];
          var wa = showDnz ? decRev * (0.35 + 0.65 * flash) : 0;
          put3(Cc, q0 * 6, C.ghost, C.warm, wa);
          put3(Cc, q0 * 6 + 3, C.ghost, C.warm, wa);
          if (!showDnz) { P[q0 * 6 + 3] = nx; P[q0 * 6 + 4] = ny; P[q0 * 6 + 5] = nzz; }
          q0++;
        }
      }
      dnzPts.pos.needsUpdate = dnzPts.col.needsUpdate = true;
      inlet.pos.needsUpdate = inlet.col.needsUpdate = true;
      var rc = cRail.col.array;
      put3(rc, 0, C.ghost, C.dim, 0.5 + 0.5 * decRev); put3(rc, 3, C.ghost, C.dim, 0.5 + 0.5 * decRev);
      cRail.col.needsUpdate = true;
      lbC.t.textContent = stage === 3 ? 'c = ' + (cVal < 0 ? '−' : '+') + Math.abs(cVal).toFixed(2) : 'c = 2(q − i) − 1';
      lbDnz.hidden = !showDnz;
      lbNz.hidden = !hasNoiseIn();
    })();

    // -- output: fan out 4:1, the warm band lights, the inference sweep gates it
    (function () {
      var P = outPts.pos.array, Cc = outPts.col.array, BC = outLine.col.array, WC = warmLine.col.array;
      var fanAll = smooth(0, 1, u4);
      var edge = 6 / N, leadW = 5 / N, sweeping = p < 0.999;
      for (j = 0; j < N; j++) {
        i = Math.floor(j / R);
        var r = j - i * R;
        var fan = fanAll;
        if (stage === 3 && i === ANCHOR) fan = Math.max(fan, smooth(0.82, 1.0, u3 * pulses - r));
        else if (stage === 4 && i === ANCHOR) fan = 1;
        x = lerp(cellXo(i), x48o[j], fan);
        y = lerp(S.lf[i * R], S.lf[j] + S.hfB[j], fan);
        P[j * 3] = x; P[j * 3 + 1] = y; P[j * 3 + 2] = 0;
        var sweep = p - j / (N - 1);
        var inf = 0.08 + 0.92 * smooth(-edge, edge, sweep);
        var lead = sweeping ? Math.exp(-Math.pow(sweep / leadW, 2)) : 0;
        var ready = (stage >= 4 ? 1 : 0) * inf;
        var warmW = S.envB[j] * fan;
        tmpA = mixc(C.cold, C.warm, warmW);
        var pb = mixc(C.coldDim, tmpA, fan);
        if (lead > 0.02) pb = mixc(pb, C.ink, 0.6 * lead);
        var gate = stage >= 4 ? inf : 1;
        put3s(Cc, j * 3, pb, gate);
        put3(BC, j * 3, C.ghost, pb, (0.6 + 0.4 * fan) * gate * (0.8 + 0.2 * fan));
        put3s(WC, j * 3, C.warm, Math.min(1.3, ready * (1 + 1.3 * lead)));
      }
      outLine.pos.array.set(P);
      outPts.pos.needsUpdate = outPts.col.needsUpdate = outLine.pos.needsUpdate = outLine.col.needsUpdate = warmLine.col.needsUpdate = true;
      warmLine.mat.opacity = 0.95 * fanAll;
      warmLine.obj.visible = fanAll > 0.001;
    })();

    // -- label emphasis
    for (q = 0; q < labels.length; q++) {
      var L = labels[q];
      var on = L.stage === stage;
      if (L.el.classList.contains('on') !== on) L.el.classList.toggle('on', on);
    }
    return stage;
  }

  // ---- camera --------------------------------------------------------------
  // Two orbits: `goal` is where the camera is asked to be, `cam` is where it is, and every frame
  // `cam` eases toward `goal`. The framing solve runs on a fit (mount, resize, reset, a stage
  // focus), never on a drag, so orbiting turns the scene about a fixed target instead of
  // re-centring it under the cursor. Drag orbits; shift-drag, a middle or right button, two
  // fingers together, or a two-finger swipe pan; pinch or ctrl/⌘-wheel zooms.
  var _v = new THREE.Vector3(), _r = new THREE.Vector3(), _u = new THREE.Vector3();
  var BB = { x0: Lx.inX0 - 0.35, x1: Lx.outX1 + 0.35, y0: cSlider.y0 - 0.2, y1: Lx.warmLift + 1.05,
             z0: -1.35, z1: NZ_DZ * D.noiseIn + 0.1 };
  // one box per stage along the flow, the full height and depth of the diagram
  var STAGE_X = [
    [Lx.inX0 - 0.35, Lx.inX1 + 0.35],
    [Lx.slabX0 - 0.4, slabEnd + 0.4],
    [Lx.ribX0 - 0.35, Lx.ribX1 + 0.35],
    [decX[0] - 1.0, decX[decX.length - 1] + 1.0],
    [Lx.outX0 - 0.35, Lx.outX1 + 0.35]
  ];
  function boxOf(i) {
    if (i == null) return BB;
    return { x0: STAGE_X[i][0], x1: STAGE_X[i][1], y0: BB.y0, y1: BB.y1, z0: BB.z0, z1: BB.z1 };
  }
  function cornersOf(b) {
    var c = [];
    [b.x0, b.x1].forEach(function (x) { [b.y0, b.y1].forEach(function (y) { [b.z0, b.z1].forEach(function (z) {
      c.push(new THREE.Vector3(x, y, z)); }); }); });
    return c;
  }
  function setCam(target, d, az, el) {
    camera.position.set(
      target.x + d * Math.cos(el) * Math.sin(az),
      target.y + d * Math.sin(el),
      target.z + d * Math.cos(el) * Math.cos(az));
    camera.lookAt(target);
    camera.updateMatrixWorld();
  }
  // Solve the distance and target that frame `box` inside the viewport minus paddings at orbit
  // (az, el): scale the distance until the projected corners fit, then walk the target so the box
  // sits in the middle of the padded region. Four rounds converge to a pixel. The right pad on the
  // whole diagram keeps room for the two labels that hang off the output.
  function fitBox(box, az, el, padR) {
    var w = wrap.clientWidth, h = wrap.clientHeight;
    var corners = cornersOf(box);
    var padL = narrow ? 16 : 22, padT = narrow ? 20 : 44, padB = narrow ? 20 : 44;
    if (padR == null) padR = narrow ? 16 : 128;
    var vf = camera.fov * Math.PI / 360;
    var d = Math.max(18 / (Math.tan(vf) * camera.aspect), 4.6 / Math.tan(vf));
    var target = new THREE.Vector3((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, (box.z0 + box.z1) / 2);
    var ext = { minX: 0, maxX: 0, minY: 0, maxY: 0 };
    function measure() {
      ext.minX = ext.minY = 1e9; ext.maxX = ext.maxY = -1e9;
      for (var i = 0; i < corners.length; i++) {
        _v.copy(corners[i]).project(camera);
        var px = (_v.x * 0.5 + 0.5) * w, py = (-_v.y * 0.5 + 0.5) * h;
        if (px < ext.minX) ext.minX = px; if (px > ext.maxX) ext.maxX = px;
        if (py < ext.minY) ext.minY = py; if (py > ext.maxY) ext.maxY = py;
      }
    }
    for (var it = 0; it < 4; it++) {
      setCam(target, d, az, el);
      measure();
      var cx = w / 2, cy = h / 2;
      var sc = Math.max((cx - ext.minX) / (cx - padL), (ext.maxX - cx) / (w - padR - cx),
                        (cy - ext.minY) / (cy - padT), (ext.maxY - cy) / (h - padB - cy));
      d *= sc * 1.01;
      setCam(target, d, az, el);
      measure();
      var offX = (ext.minX + ext.maxX) / 2 - (padL + (w - padR)) / 2;
      var offY = (ext.minY + ext.maxY) / 2 - (padT + (h - padB)) / 2;
      var wpp = 2 * d * Math.tan(vf) / h;
      _r.setFromMatrixColumn(camera.matrixWorld, 0);
      _u.setFromMatrixColumn(camera.matrixWorld, 1);
      target.addScaledVector(_r, offX * wpp).addScaledVector(_u, -offY * wpp);
    }
    return { d: d, target: target };
  }
  // re-solve the framing for the current focus (a stage, or the whole) at the goal orbit
  function refit(immediate) {
    if (!wrap.clientWidth || !wrap.clientHeight) return;
    var padR = focusIdx == null || focusIdx === 4 ? null : (narrow ? 16 : 40);
    var f = fitBox(boxOf(focusIdx), goal.az, goal.el, padR);
    goal.d = f.d; goal.target.copy(f.target);
    if (immediate) snapCamera();
    state.needsRender = true;
  }
  function snapCamera() { cam.az = goal.az; cam.el = goal.el; cam.zoom = goal.zoom; cam.d = goal.d; cam.target.copy(goal.target); }
  // ease `cam` toward `goal`; true while it is still moving
  function easeCamera(dt) {
    var k = 1 - Math.exp(-dt * 9);
    cam.az += (goal.az - cam.az) * k; cam.el += (goal.el - cam.el) * k;
    cam.zoom += (goal.zoom - cam.zoom) * k; cam.d += (goal.d - cam.d) * k;
    cam.target.lerp(goal.target, k);
    return Math.abs(goal.az - cam.az) + Math.abs(goal.el - cam.el) + Math.abs(goal.zoom - cam.zoom) > 1e-4 ||
           Math.abs(goal.d - cam.d) > 1e-3 || cam.target.distanceToSquared(goal.target) > 1e-6;
  }
  // world units per screen pixel in the target's plane
  function worldPerPixel() {
    return 2 * cam.d * cam.zoom * Math.tan(camera.fov * Math.PI / 360) / Math.max(1, wrap.clientHeight);
  }
  // move the target in the camera's plane by a screen offset, kept within reach of the diagram
  function pan(dx, dy) {
    var wpp = worldPerPixel();
    _r.setFromMatrixColumn(camera.matrixWorld, 0);
    _u.setFromMatrixColumn(camera.matrixWorld, 1);
    goal.target.addScaledVector(_r, dx * wpp).addScaledVector(_u, -dy * wpp);
    goal.target.x = clamp(goal.target.x, BB.x0 - 6, BB.x1 + 6);
    goal.target.y = clamp(goal.target.y, BB.y0 - 4, BB.y1 + 4);
    goal.target.z = clamp(goal.target.z, BB.z0 - 4, BB.z1 + 4);
    userMoved = true;
  }
  function placeCamera(now) {
    var az = cam.az, el = cam.el;
    if (!reducedMotion && now - cam.lastTouch > 3.0) {
      var td = now - Math.max(cam.driftFrom, cam.lastTouch + 3.0);
      az += 0.05 * Math.sin(td * 0.11);
      el += 0.014 * Math.sin(td * 0.073);
    }
    setCam(cam.target, cam.d * cam.zoom, az, el);
  }
  // Project every anchor, keep the text inside the viewport, then drop any dim label that
  // would land on top of one already placed. The stage's own labels are never dropped.
  var _rects = [];
  function labelRect(L, x, y) {
    var bw = L.b.offsetWidth, bh = L.b.offsetHeight;
    if (L.align === 'above') return [x - bw / 2 + L.dx, y - 7 - bh, bw, bh];
    if (L.align === 'below') return [x - bw / 2 + L.dx, y + 7, bw, bh];
    if (L.side === 'left') return [x - 9 - bw, y - 8, bw, bh];
    return [x + 9, y - 8, bw, bh];
  }
  function hits(a, b) {
    return a[0] < b[0] + b[2] + 3 && b[0] < a[0] + a[2] + 3 && a[1] < b[1] + b[3] + 2 && b[1] < a[1] + a[3] + 2;
  }
  function clashes(r) {
    for (var k = 0; k < _rects.length; k++) if (hits(r, _rects[k])) return true;
    return false;
  }
  function setDy(L, dy) {
    if (dy === L.dy2) return;
    L.dy2 = dy;
    if (L.align === 'above') L.b.style.marginBottom = dy ? (-dy) + 'px' : '';
    else L.b.style.marginTop = dy ? dy + 'px' : '';
  }
  function placeLabels() {
    var w = wrap.clientWidth, h = wrap.clientHeight, i, L;
    for (i = 0; i < labels.length; i++) {
      L = labels[i];
      L.live = false;
      if (L.hidden) { if (!L.el.classList.contains('hid')) L.el.classList.add('hid'); continue; }
      _v.copy(L.anchor).project(camera);
      if (_v.z > 1 || _v.z < -1) { L.el.style.transform = 'translate3d(-9999px,0,0)'; continue; }
      var x = (_v.x * 0.5 + 0.5) * w, y = (-_v.y * 0.5 + 0.5) * h;
      // An anchor off the canvas gets no label: the clamping below only slides a box back into view,
      // which leaves the text stranded at an edge pointing at nothing (the receptive-field bracket,
      // whose anchor swings past the left edge).  Hide with the same class the stage logic uses --
      // parking el at -9999px does not work here, because the clamp's marginLeft from the previous
      // frame stays on the inner box and drags it back into view.
      if (x < 0 || x > w || y < 0 || y > h) {
        if (!L.el.classList.contains('hid')) L.el.classList.add('hid');
        continue;
      }
      L.el.style.transform = 'translate3d(' + x.toFixed(1) + 'px,' + y.toFixed(1) + 'px,0)';
      // keep the text inside the viewport: slide centred boxes, flip side boxes
      var bw = L.b.offsetWidth, dx = 0;
      if (L.align === 'above' || L.align === 'below') {
        if (x - bw / 2 < 4) dx = 4 - (x - bw / 2);
        else if (x + bw / 2 > w - 4) dx = (w - 4) - (x + bw / 2);
        if (dx !== L.dx) { L.dx = dx; L.b.style.marginLeft = dx ? dx.toFixed(1) + 'px' : ''; }
      } else {
        var side = L.align === 'right' ? (x + 9 + bw > w - 4 ? 'left' : 'right') : (x - 9 - bw < 4 ? 'right' : 'left');
        if (side !== L.side) { L.side = side; L.el.classList.remove('left', 'right'); L.el.classList.add(side); }
      }
      L.live = true;
      L.baseRect = labelRect(L, x, y);
    }
    // the stage's own labels are placed first and never dropped; a label that lands on one
    // already placed is nudged clear of it, and only a dim one that still clashes is dropped
    _rects.length = 0;
    for (var pass = 0; pass < 2; pass++) {
      for (i = 0; i < labels.length; i++) {
        L = labels[i];
        if (!L.live) continue;
        var on = L.el.classList.contains('on');
        if ((pass === 0) !== on) continue;
        var step = L.align === 'above' ? -13 : 13, dy = 0, r = L.baseRect;
        for (var tries = 0; tries < 4 && clashes(r); tries++) {
          dy += step;
          r = [L.baseRect[0], L.baseRect[1] + dy, L.baseRect[2], L.baseRect[3]];
        }
        // a dim label that still clashes is dropped; the stage's own is kept where it ended up
        if (!on && clashes(r)) { if (!L.el.classList.contains('hid')) L.el.classList.add('hid'); continue; }
        setDy(L, dy);
        if (L.el.classList.contains('hid')) L.el.classList.remove('hid');
        _rects.push(r);
      }
    }
  }

  // ---- interaction ---------------------------------------------------------
  var pointers = {}, nPointers = 0, lastX = 0, lastY = 0, pinchD = 0, dragPan = false, lastC = null;
  function nowS() { return performance.now() / 1000; }
  function touched() { cam.lastTouch = nowS(); cam.driftFrom = cam.lastTouch + 3.0; state.needsRender = true; }
  function pinchDist() {
    var ids = Object.keys(pointers); if (ids.length < 2) return 0;
    var a = pointers[ids[0]], b = pointers[ids[1]];
    return Math.hypot(a.x - b.x, a.y - b.y);
  }
  function centroid() {
    var ids = Object.keys(pointers), x = 0, y = 0;
    for (var i = 0; i < ids.length; i++) { x += pointers[ids[i]].x; y += pointers[ids[i]].y; }
    return { x: x / ids.length, y: y / ids.length };
  }
  function onDown(e) {
    pointers[e.pointerId] = { x: e.clientX, y: e.clientY }; nPointers++;
    if (nPointers === 1) {
      lastX = e.clientX; lastY = e.clientY;
      dragPan = e.shiftKey || e.button === 1 || e.button === 2;
      canvas.classList.add('a3d-drag');
    }
    if (nPointers === 2) { pinchD = pinchDist(); lastC = centroid(); }
    try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
    touched();
  }
  function onMove(e) {
    if (!pointers[e.pointerId]) return;
    pointers[e.pointerId].x = e.clientX; pointers[e.pointerId].y = e.clientY;
    if (nPointers >= 2) {
      var d = pinchDist();
      if (pinchD > 0 && d > 0) { goal.zoom = clamp(goal.zoom * pinchD / d, ZOOM_MIN, ZOOM_MAX); pinchD = d; }
      var c = centroid();
      if (lastC) pan(-(c.x - lastC.x), -(c.y - lastC.y));
      lastC = c;
    } else if (dragPan) {
      pan(-(e.clientX - lastX), -(e.clientY - lastY));
      lastX = e.clientX; lastY = e.clientY;
    } else {
      goal.az = clamp(goal.az - (e.clientX - lastX) * 0.005, -AZ_LIM, AZ_LIM);
      goal.el = clamp(goal.el + (e.clientY - lastY) * 0.004, EL_MIN, EL_MAX);
      lastX = e.clientX; lastY = e.clientY;
      userMoved = true;
    }
    touched();
  }
  function onUp(e) {
    if (pointers[e.pointerId]) { delete pointers[e.pointerId]; nPointers = Math.max(0, nPointers - 1); }
    if (nPointers === 0) { canvas.classList.remove('a3d-drag'); dragPan = false; lastC = null; }
    else { var ids = Object.keys(pointers); lastX = pointers[ids[0]].x; lastY = pointers[ids[0]].y; lastC = null; }
    try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
    touched();
  }
  // A trackpad pinch arrives as a ctrl-wheel; a mouse holds ctrl or ⌘. Anything else is a
  // two-finger swipe: horizontal travels along the flow, vertical moves up and down it. The page
  // around the canvas is laid out not to need the wheel, so it is always taken.
  function onWheel(e) {
    e.preventDefault();
    var dx = e.deltaX, dy = e.deltaY;
    if (e.deltaMode === 1) { dx *= 16; dy *= 16; }
    else if (e.deltaMode === 2) { dx *= wrap.clientWidth; dy *= wrap.clientHeight; }
    if (e.ctrlKey || e.metaKey) goal.zoom = clamp(goal.zoom * Math.exp(clamp(dy, -40, 40) * 0.008), ZOOM_MIN, ZOOM_MAX);
    else pan(dx, dy);
    touched();
  }
  function onCtx(e) { e.preventDefault(); }
  function resetView() {
    var o = defaultOrbit(camera.aspect);
    goal.az = o.az; goal.el = o.el; goal.zoom = 1;
    userMoved = false; focusIdx = null;
    refit(false);
    touched();
  }
  function onDbl() { resetView(); }
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointercancel', onUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('dblclick', onDbl);
  canvas.addEventListener('contextmenu', onCtx);
  function onMql() { reducedMotion = !!(mql && mql.matches); state.dirty = true; state.needsRender = true; }
  if (mql) { if (mql.addEventListener) mql.addEventListener('change', onMql); else if (mql.addListener) mql.addListener(onMql); }

  // ---- resize
  function resize() {
    var w = wrap.clientWidth, h = wrap.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    narrow = w < 780 || camera.aspect < 0.85;   // too little room for the dim labels and subtitles
    wrap.classList.toggle('a3d-narrow', narrow);
    if (!userMoved) {
      var o = defaultOrbit(camera.aspect);
      goal.az = o.az; goal.el = o.el;
    }
    refit(true);
  }
  var ro = null;
  if (global.ResizeObserver) { ro = new ResizeObserver(function () { resize(); }); ro.observe(wrap); }
  else global.addEventListener('resize', resize);
  resize();

  // ---- the loop
  var raf = 0, lastNow = nowS();
  function frame() {
    if (state.disposed) return;
    raf = global.requestAnimationFrame(frame);
    var now = nowS(), dt = Math.min(0.1, now - lastNow); lastNow = now;
    if (state.playing) {
      if (state.t >= 1) {
        if (now >= state.holdUntil) { state.t = 0; state.dirty = true; }
      } else {
        state.t = Math.min(1, state.t + dt / CYCLE_S);
        if (state.t >= 1) state.holdUntil = now + HOLD_S;
        state.dirty = true;
      }
      if (onTime) onTime(state.t);
    }
    var moving = easeCamera(dt);
    // wall-clock effects (jitter, shimmer, drift) need a render every frame unless reduced motion
    var animate = !reducedMotion;
    if (state.dirty || animate) { applyTimeline(now); state.dirty = false; state.needsRender = true; }
    if (state.needsRender || animate || moving) {
      placeCamera(now);
      renderer.render(scene, camera);
      placeLabels();
      state.needsRender = false;
    }
  }
  raf = global.requestAnimationFrame(frame);

  // ---- the handle
  var handle = {
    setArm: function (arm, cls) {
      if (arm != null) state.arm = arm;
      if (cls) state.cls = cls;
      setCaption();
      state.dirty = true; state.needsRender = true;
      return handle;
    },
    play: function () {
      if (state.t >= 1) { state.t = 0; }
      state.playing = true; lastNow = nowS(); state.holdUntil = 0;
      return handle;
    },
    pause: function () { state.playing = false; return handle; },
    seek: function (t01) {
      state.t = clamp01(+t01 || 0);
      state.holdUntil = state.t >= 1 ? nowS() + HOLD_S : 0;
      state.dirty = true; state.needsRender = true;
      if (onTime) onTime(state.t);
      return handle;
    },
    setInference: function (p01) {
      state.p = p01 == null ? 1 : clamp01(+p01);
      state.dirty = true; state.needsRender = true;
      return handle;
    },
    // Fly to one stage of the flow (0..4), or back to the whole diagram with null. The orbit is kept;
    // only the framing changes, eased.
    focus: function (i) {
      focusIdx = (i == null || i < 0 || i >= STAGE_X.length) ? null : i | 0;
      goal.zoom = 1;
      refit(false); touched();
      return handle;
    },
    focused: function () { return focusIdx; },
    reset: function () { resetView(); return handle; },
    panBy: function (dx, dy) { pan(dx, dy); touched(); return handle; },
    zoomBy: function (f) { goal.zoom = clamp(goal.zoom * (+f || 1), ZOOM_MIN, ZOOM_MAX); touched(); return handle; },
    dispose: function () {
      if (state.disposed) return;
      state.disposed = true;
      global.cancelAnimationFrame(raf);
      if (ro) ro.disconnect(); else global.removeEventListener('resize', resize);
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('pointercancel', onUp);
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('dblclick', onDbl);
      canvas.removeEventListener('contextmenu', onCtx);
      if (mql) { if (mql.removeEventListener) mql.removeEventListener('change', onMql); else if (mql.removeListener) mql.removeListener(onMql); }
      scene.traverse(function (o) {
        if (o.geometry) o.geometry.dispose();
        if (o.material) { if (o.material.map) o.material.map.dispose(); o.material.dispose(); }
      });
      dot.dispose();
      renderer.dispose();
      if (renderer.forceContextLoss) renderer.forceContextLoss();
      if (wrap.parentNode) wrap.parentNode.removeChild(wrap);
    },
    // extras, harmless to ignore
    getTime: function () { return state.t; },
    isPlaying: function () { return state.playing; },
    reducedMotion: function () { return reducedMotion; },
    stageOf: function (t) { t = clamp01(+t || 0); return t < 0.15 ? 0 : t < 0.45 ? 1 : t < 0.60 ? 2 : t < 0.90 ? 3 : 4; },
    stages: STAGES.slice(),
    // A read-only view of the camera, the canvas and where every label ought to sit. The dev
    // harness checks the overlay against it; nothing in the page needs it.
    debug: function () {
      var w = wrap.clientWidth, h = wrap.clientHeight, v = new THREE.Vector3();
      return {
        fov: camera.fov, aspect: camera.aspect, az: cam.az, el: cam.el, zoom: cam.zoom,
        cssW: w, cssH: h, bufW: canvas.width, bufH: canvas.height, dpr: renderer.getPixelRatio(),
        labels: labels.map(function (L) {
          v.copy(L.anchor).project(camera);
          return { text: L.t.textContent, hidden: L.hidden, on: L.el.classList.contains('on'),
                   hid: L.el.classList.contains('hid'),
                   px: (v.x * 0.5 + 0.5) * w, py: (-v.y * 0.5 + 0.5) * h, tf: L.el.style.transform };
        })
      };
    },
    stats: function () {
      var r = renderer.info.render;
      return { calls: r.calls, triangles: r.triangles, points: r.points, lines: r.lines, geometries: renderer.info.memory.geometries };
    },
    canvas: canvas
  };
  return handle;
}

global.Arch3D = { mount: mount, DEFAULT_TOKENS: DEFAULT_TOKENS, DEFAULT_DIMS: DEFAULT_DIMS, STAGES: STAGES };
})(typeof window !== 'undefined' ? window : this);
