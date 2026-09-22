// Phase-locked A/B/C playback: input, output and truth start together on one clock; switching source
// only moves three gains, so the listener hears exactly one thing change — the band above 6 kHz.
import type { Signal } from "./types";

// "a" and "b" are the blind test's two hidden conditions; they ride the same clock as the rest.
export type SourceName = "input" | "output" | "truth" | "a" | "b";

interface Node { src: AudioBufferSourceNode; g: GainNode }

export class ABPlayer {
  private ctx: AudioContext | null = null;
  private nodes: Partial<Record<SourceName, Node>> = {};
  private startedAt = 0;
  private offset = 0;
  playing = false;
  source: SourceName = "output";
  onEnded: (() => void) | null = null;

  private ac(): AudioContext {
    if (!this.ctx) this.ctx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
    return this.ctx;
  }

  private buffer(sig: Signal): AudioBuffer {
    const b = this.ac().createBuffer(1, sig.data.length, sig.fs);
    b.copyToChannel(sig.data, 0);
    return b;
  }

  /** Seconds into the clip, valid while playing. */
  position(): number {
    if (!this.playing || !this.ctx) return 0;
    return this.offset + (this.ctx.currentTime - this.startedAt);
  }

  stop() {
    for (const k of Object.keys(this.nodes) as SourceName[]) {
      try { this.nodes[k]!.src.onended = null; this.nodes[k]!.src.stop(); } catch { /* already stopped */ }
    }
    this.nodes = {};
    this.playing = false;
  }

  start(srcs: Partial<Record<SourceName, Signal>>, offset = 0) {
    const c = this.ac();
    if (c.state === "suspended") void c.resume();
    this.stop();
    const t0 = c.currentTime + 0.05;
    let first: Node | null = null;
    for (const k of Object.keys(srcs) as SourceName[]) {
      const sig = srcs[k];
      if (!sig) continue;
      const src = c.createBufferSource();
      src.buffer = this.buffer(sig);
      const g = c.createGain();
      g.gain.value = k === this.source ? 1 : 0;
      src.connect(g).connect(c.destination);
      src.start(t0, offset);
      this.nodes[k] = { src, g };
      first ??= this.nodes[k]!;
    }
    if (!first) return;
    first.src.onended = () => { if (this.playing) { this.stop(); this.onEnded?.(); } };
    this.startedAt = t0;
    this.offset = offset;
    this.playing = true;
  }

  setSource(k: SourceName) {
    this.source = k;
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    for (const n of Object.keys(this.nodes) as SourceName[]) this.nodes[n]!.g.gain.setTargetAtTime(n === k ? 1 : 0, t, 0.012);
  }

  /** Replace the sources without losing the playhead (used when live inference finishes mid-play). */
  swap(srcs: Partial<Record<SourceName, Signal>>) {
    if (!this.playing) return;
    const pos = this.position();
    this.start(srcs, pos);
  }
}
