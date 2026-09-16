"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import s from "@/app/page.module.css";
import a from "./arch.module.css";
import { ARM_META, ARM_ORDER, type ArmName } from "@/lib/arms";
import { ArchView } from "./ArchView";

const STAGES: [number, string][] = [[0, "48 kHz → 12 kHz"], [0.15, "conv encoder"], [0.45, "latents"], [0.6, "decoder MLP"], [0.9, "48 kHz output"]];

export function ArchPage({ arm }: { arm: ArmName }) {
  const meta = ARM_META[arm];
  const [playing, setPlaying] = useState(true);
  const [t, setT] = useState(0);
  const [seek, setSeek] = useState<number | null>(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) setPlaying(false);
    const h = (e: KeyboardEvent) => {
      if (e.code === "Space") { e.preventDefault(); setPlaying((p) => !p); }
    };
    window.addEventListener("keydown", h); return () => window.removeEventListener("keydown", h);
  }, []);

  const stage = STAGES.reduce((acc, [at, name]) => (t >= at ? name : acc), STAGES[0][1]);

  return (
    <div className={a.page}>
      <header className={a.bar}>
        <Link href="/" className={a.back}>← instrument</Link>
        <span className={a.title}>{arm}<small>{meta.cls} · {meta.kind} · λ {meta.lam} · {meta.what}</small></span>
        <span className={s.grow} />
        <nav className={a.arms} aria-label="other arms">
          {ARM_ORDER.map((x) => <Link key={x} href={`/architecture/${x}`} aria-current={x === arm ? "page" : undefined}>{x}</Link>)}
        </nav>
        <button type="button" className={a.back} onClick={() => setPlaying((p) => !p)}>{playing ? "pause" : "play"}</button>
      </header>

      <div className={a.mount}>
        <ArchView arm={arm} cls={meta.cls} playing={playing} seek={seek} onTime={setT} onMissing={() => setMissing(true)} />
        {missing && <div className={s.empty}>arch3d.js not loaded</div>}
      </div>

      <div className={a.tl}>
        <span>timeline</span>
        <input type="range" min={0} max={1000} value={Math.round(t * 1000)} aria-label="scrub the timeline"
               onChange={(e) => { const v = +e.target.value / 1000; setT(v); setSeek(v); setPlaying(false); }} />
        <span className="num">{t.toFixed(2)}</span>
      </div>
      <div className={a.stages} aria-hidden="true">
        {STAGES.map(([at, name]) => <span key={at} className={name === stage ? a.on : ""}>{name}</span>)}
      </div>

      <dl className={a.facts}>
        <div><dt>input</dt><dd>12 kHz · 11-sample receptive field · 0.9 ms</dd></div>
        <div><dt>encoder</dt><dd>conv 7 / 3 / 3 / 1 · channels 16 / 32 / 64 / 32 · latent 32</dd></div>
        <div><dt>decoder</dt><dd>[c, z₋₁, z₀, z₊₁{meta.cls === "LISASD" ? ", ε₄" : ""}] → 5 × Linear(144) → 1 · c = 2(q − i) − 1</dd></div>
        <div><dt>noise</dt><dd>{meta.cls === "LISASD" ? "8 channels at the encoder input + 4 per output sample at the decoder" : meta.det ? "8 channels at the encoder input, held at zero" : "8 channels at the encoder input"}</dd></div>
        <div><dt>parameters</dt><dd>{meta.cls === "LISASD" ? "88,353" : "87,777"} · LISA is 86,881 · τ = 0 recovers it exactly</dd></div>
      </dl>
    </div>
  );
}
