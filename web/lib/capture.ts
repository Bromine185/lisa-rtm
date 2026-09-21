// Getting a 48 kHz mono clip into the page: the laptop microphone, or a dropped file. Either way the
// result is what audit/vctk_fixtures.py load() would give the corpus loader — mono, 48 kHz, peak 0.95 —
// so it can be decimated with the fixture pipeline's own filter and treated like any other speaker.
import { normalise } from "./resample";
import type { Signal } from "./types";

export const FS_CAPTURE = 48000;
export const MAX_SECONDS = 12;

type AC = typeof AudioContext;
function makeContext(): AudioContext {
  const C = (window.AudioContext || (window as unknown as { webkitAudioContext: AC }).webkitAudioContext) as AC;
  try { return new C({ sampleRate: FS_CAPTURE }); } catch { return new C(); }
}

/** Resample an AudioBuffer to 48 kHz mono through the browser's own converter, if it is not already. */
async function to48kMono(buf: AudioBuffer): Promise<Float32Array<ArrayBuffer>> {
  const n = buf.length, ch = buf.numberOfChannels;
  let mono: Float32Array<ArrayBuffer>;
  if (ch === 1) mono = new Float32Array(buf.getChannelData(0));
  else {
    mono = new Float32Array(n);
    for (let c = 0; c < ch; c++) { const d = buf.getChannelData(c); for (let i = 0; i < n; i++) mono[i] += d[i] / ch; }
  }
  if (buf.sampleRate === FS_CAPTURE) return mono;
  const m = Math.round((n * FS_CAPTURE) / buf.sampleRate);
  const off = new OfflineAudioContext(1, m, FS_CAPTURE);
  const b = off.createBuffer(1, n, buf.sampleRate);
  b.copyToChannel(mono, 0);
  const src = off.createBufferSource(); src.buffer = b; src.connect(off.destination); src.start();
  const out = await off.startRendering();
  return new Float32Array(out.getChannelData(0));
}

function fade(x: Float32Array, ms = 8) {
  const n = Math.min(x.length >> 1, Math.round((ms / 1000) * FS_CAPTURE));
  for (let i = 0; i < n; i++) { const g = i / n; x[i] *= g; x[x.length - 1 - i] *= g; }
}

/** Decode any audio file the browser can (wav, mp3, m4a, flac, ogg…) to a 48 kHz mono clip, at most MAX_SECONDS. */
export async function decodeFile(file: File | Blob): Promise<Signal> {
  const ctx = makeContext();
  try {
    const buf = await ctx.decodeAudioData(await file.arrayBuffer());
    let data = await to48kMono(buf);
    if (data.length > MAX_SECONDS * FS_CAPTURE) data = data.slice(0, MAX_SECONDS * FS_CAPTURE);
    fade(data);
    return normalise({ data, fs: FS_CAPTURE });
  } finally { void ctx.close(); }
}

// The worklet processor, inlined so the page needs no extra file: forwards channel 0 of every 128-frame
// block to the main thread as a copy.
const WORKLET = `class Tap extends AudioWorkletProcessor {
  process(inputs) { const c = inputs[0] && inputs[0][0]; if (c) this.port.postMessage(c.slice()); return true; }
}
registerProcessor("lisa-tap", Tap);`;

export interface Recording {
  /** Seconds captured so far. */
  seconds(): number;
  /** RMS of the last block, 0..1. */
  level(): number;
  /** Stop and return the clip (48 kHz mono, peak 0.95). */
  stop(): Promise<Signal>;
  cancel(): void;
}

/** Open the microphone (no processing: no AGC, no noise suppression, no echo cancellation) and start collecting. */
export async function record(): Promise<Recording> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false, sampleRate: FS_CAPTURE },
  });
  const ctx = makeContext();
  if (ctx.state === "suspended") await ctx.resume();
  const src = ctx.createMediaStreamSource(stream);
  const chunks: Float32Array[] = [];
  let total = 0, rms = 0;
  const push = (c: Float32Array) => {
    if (total >= MAX_SECONDS * ctx.sampleRate) return;
    chunks.push(c); total += c.length;
    let s = 0; for (let i = 0; i < c.length; i++) s += c[i] * c[i];
    rms = Math.sqrt(s / c.length);
  };
  let node: AudioNode;
  const sink = ctx.createGain(); sink.gain.value = 0; sink.connect(ctx.destination);   // keeps the graph pulled, silently
  try {
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    await ctx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    const w = new AudioWorkletNode(ctx, "lisa-tap", { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 });
    w.port.onmessage = (e) => push(e.data as Float32Array);
    node = w;
  } catch {
    // no AudioWorklet: the deprecated processor still ships everywhere
    const sp = ctx.createScriptProcessor(2048, 1, 1);
    sp.onaudioprocess = (e) => push(e.inputBuffer.getChannelData(0).slice());
    node = sp;
  }
  src.connect(node).connect(sink);
  const teardown = () => {
    try { src.disconnect(); node.disconnect(); sink.disconnect(); } catch { /* already gone */ }
    stream.getTracks().forEach((t) => t.stop());
    void ctx.close();
  };
  return {
    seconds: () => total / ctx.sampleRate,
    level: () => rms,
    cancel: teardown,
    async stop() {
      teardown();
      const all = new Float32Array(total);
      let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; }
      let data: Float32Array<ArrayBuffer> = all;
      if (ctx.sampleRate !== FS_CAPTURE) {
        const b = new OfflineAudioContext(1, total, ctx.sampleRate).createBuffer(1, total, ctx.sampleRate);
        b.copyToChannel(all, 0);
        data = await to48kMono(b);
      }
      fade(data);
      return normalise({ data, fs: FS_CAPTURE });
    },
  };
}
