"use client";
import { useEffect, useRef } from "react";
import type { ArmClass } from "@/lib/arms";

// public/arch3d.js is an imperative Three.js module that expects window.THREE and defines window.Arch3D.
// We hand it the bundled three (one copy in the app), then mount it on a div. Client only.

interface Arch3DHandle {
  setArm(arm: string, cls: ArmClass): void;
  play(): void;
  pause(): void;
  seek(t: number): void;
  setInference(p: number): void;
  dispose(): void;
}
interface Arch3DModule {
  mount(el: HTMLElement, opts: { arm: string; cls: ArmClass; tokens: Record<string, string>; onTime?: (t: number) => void }): Arch3DHandle;
}
declare global {
  interface Window { Arch3D?: Arch3DModule; THREE?: unknown }
}

let loading: Promise<Arch3DModule | null> | null = null;
function getArch(): Promise<Arch3DModule | null> {
  if (typeof window === "undefined") return Promise.resolve(null);
  if (window.Arch3D) return Promise.resolve(window.Arch3D);
  if (loading) return loading;
  loading = (async () => {
    if (!window.THREE) window.THREE = await import("three");
    return new Promise<Arch3DModule | null>((resolve) => {
      const s = document.createElement("script");
      s.src = "/arch3d.js";
      s.async = true;
      s.onload = () => resolve(window.Arch3D ?? null);
      s.onerror = () => resolve(null);
      document.head.appendChild(s);
    });
  })();
  return loading;
}

export function ArchView({ arm, cls, playing, seek, inference, onTime, onMissing }: {
  arm: string; cls: ArmClass; playing: boolean; seek: number | null; inference?: number; onTime?: (t: number) => void; onMissing?: () => void;
}) {
  const el = useRef<HTMLDivElement>(null);
  const handle = useRef<Arch3DHandle | null>(null);
  const onTimeRef = useRef(onTime);
  onTimeRef.current = onTime;

  useEffect(() => {
    let dead = false;
    (async () => {
      const mod = await getArch();
      if (dead || !el.current) return;
      if (!mod) { onMissing?.(); return; }
      const cs = getComputedStyle(document.documentElement);
      const tokens: Record<string, string> = {};
      for (const k of ["ground", "panel", "line", "ink", "dim", "cold", "warm", "truth", "good", "bad"]) tokens[k] = cs.getPropertyValue("--" + k).trim();
      handle.current = mod.mount(el.current, { arm, cls, tokens, onTime: (t) => onTimeRef.current?.(t) });
      if (!playing) handle.current.pause();
    })();
    return () => { dead = true; handle.current?.dispose(); handle.current = null; };
    // mount once per element; arm/cls/playing are pushed through the effects below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { handle.current?.setArm(arm, cls); }, [arm, cls]);
  useEffect(() => { if (!handle.current) return; playing ? handle.current.play() : handle.current.pause(); }, [playing]);
  useEffect(() => { if (seek != null) handle.current?.seek(seek); }, [seek]);
  useEffect(() => { if (inference != null) handle.current?.setInference(inference); }, [inference]);

  return <div ref={el} style={{ position: "absolute", inset: 0 }} aria-label={`${arm} architecture, animated`} role="img" />;
}
