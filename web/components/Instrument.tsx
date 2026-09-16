"use client";
import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import s from "@/app/page.module.css";
import { CUT_HZ, FMAX_HZ } from "@/lib/arms";
import { rgb, specFill, type Spectrogram } from "@/lib/dsp";

export type Tint = "band" | "truth";

export interface InstrumentHandle {
  /** Paint the whole spectrogram (fills any unfilled frames first). */
  showAll(sp: Spectrogram, tint: Tint): void;
  /** Size to a spectrogram and blank it — the live sweep starts here. */
  blank(nfr: number): void;
  /** Paint frames [from, to) of a spectrogram already filled that far. */
  paintRange(sp: Spectrogram, tint: Tint, from: number, to: number): void;
  /** Sweep cursor at a fraction of the width, or null. */
  setSweep(frac: number | null): void;
  /** Playhead at a fraction of the width, or null. */
  setPlayhead(frac: number | null): void;
  /** Blank the canvas entirely. */
  clear(): void;
}

const H = 512;

export const Instrument = forwardRef<InstrumentHandle, { empty: string | null }>(function Instrument({ empty }, ref) {
  const specRef = useRef<HTMLCanvasElement>(null);
  const overRef = useRef<HTMLCanvasElement>(null);
  const img = useRef<ImageData | null>(null);
  const sweep = useRef<number | null>(null);
  const play = useRef<number | null>(null);
  const colors = useRef<{ cold: number[]; warm: number[]; truth: number[]; ground: number[] } | null>(null);

  useEffect(() => {
    const cs = getComputedStyle(document.documentElement);
    const t = (k: string) => cs.getPropertyValue("--" + k).trim();
    colors.current = { cold: rgb(t("cold")), warm: rgb(t("warm")), truth: rgb(t("truth")), ground: rgb(t("panel")) };
  }, []);

  function ensure(nfr: number) {
    const c = specRef.current!, o = overRef.current!;
    if (c.width !== nfr || c.height !== H) { c.width = nfr; c.height = H; o.width = nfr; o.height = H; img.current = null; }
    if (!img.current) img.current = c.getContext("2d")!.createImageData(nfr, H);
  }

  function paint(sp: Spectrogram, tint: Tint, from: number, to: number) {
    const C = colors.current; if (!C) return;
    ensure(sp.nfr);
    const d = img.current!.data, W = sp.nfr, floor = -80;
    let peak = -1e9;
    for (let i = 0; i < sp.done * sp.nb; i++) if (sp.db[i] > peak) peak = sp.db[i];
    const ref = isFinite(peak) ? peak : 0;
    for (let f = from; f < to && f < W; f++) {
      for (let y = 0; y < H; y++) {
        const hz = FMAX_HZ * (1 - (y + 0.5) / H);
        const b = Math.round(hz / sp.hzPerBin);
        let a = 0;
        if (b < sp.nb && f < sp.done) { a = (sp.db[f * sp.nb + b] - ref - floor) / -floor; a = a < 0 ? 0 : a > 1 ? 1 : a; a *= a; }
        const col = tint === "truth" ? C.truth : hz >= CUT_HZ ? C.warm : C.cold;
        const o = (y * W + f) * 4;
        d[o] = C.ground[0] + (col[0] - C.ground[0]) * a; d[o + 1] = C.ground[1] + (col[1] - C.ground[1]) * a; d[o + 2] = C.ground[2] + (col[2] - C.ground[2]) * a; d[o + 3] = 255;
      }
    }
    specRef.current!.getContext("2d")!.putImageData(img.current!, 0, 0);
  }

  function overlay() {
    const o = overRef.current; if (!o) return;
    const ctx = o.getContext("2d")!;
    ctx.clearRect(0, 0, o.width, o.height);
    if (sweep.current != null && sweep.current < 1) {
      const x = Math.floor(sweep.current * o.width);
      ctx.fillStyle = "rgba(255,179,71,0.9)"; ctx.fillRect(x, 0, 1, o.height);
      ctx.fillStyle = "rgba(255,179,71,0.07)"; ctx.fillRect(x + 1, 0, o.width - x, o.height);
    }
    if (play.current != null) { const x = Math.floor(play.current * o.width); ctx.fillStyle = "rgba(242,244,247,0.85)"; ctx.fillRect(x, 0, 1, o.height); }
  }

  useImperativeHandle(ref, () => ({
    showAll(sp, tint) { specFill(sp, sp.sig.data.length); paint(sp, tint, 0, sp.nfr); overlay(); },
    blank(nfr) {
      ensure(nfr);
      const C = colors.current!; const d = img.current!.data;
      for (let i = 0; i < d.length; i += 4) { d[i] = C.ground[0]; d[i + 1] = C.ground[1]; d[i + 2] = C.ground[2]; d[i + 3] = 255; }
      specRef.current!.getContext("2d")!.putImageData(img.current!, 0, 0); overlay();
    },
    paintRange(sp, tint, from, to) { paint(sp, tint, from, to); },
    setSweep(f) { sweep.current = f; overlay(); },
    setPlayhead(f) { play.current = f; overlay(); },
    clear() { const c = specRef.current; if (!c) return; const ctx = c.getContext("2d")!; const C = colors.current; ctx.fillStyle = C ? `rgb(${C.ground.join(",")})` : "#11141A"; ctx.fillRect(0, 0, c.width, c.height); img.current = null; sweep.current = null; play.current = null; overlay(); },
  }), []);

  const ticks = [24, 18, 12, 6, 0];
  return (
    <div className={s.specwrap}>
      <canvas ref={specRef} width={600} height={H} aria-label="spectrogram, 0 to 24 kHz" />
      <canvas ref={overRef} width={600} height={H} aria-hidden="true" />
      <div className={s.axis} aria-hidden="true">
        {ticks.map((k) => <span key={k} style={{ top: k === 24 ? "10px" : k === 0 ? "calc(100% - 10px)" : `${100 * (1 - k / 24)}%` }}>{k} kHz</span>)}
      </div>
      <div className={s.cut} style={{ top: `${100 * (1 - CUT_HZ / FMAX_HZ)}%` }} aria-hidden="true"><b>6 kHz</b></div>
      <span className={`${s.band} ${s.hi}`} aria-hidden="true">generated</span>
      <span className={`${s.band} ${s.lo}`} aria-hidden="true">input</span>
      {empty && <div className={s.empty}>{empty}</div>}
    </div>
  );
});
