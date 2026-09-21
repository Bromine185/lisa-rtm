"use client";
import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import s from "@/app/page.module.css";
import { decodeFile, record, type Recording } from "@/lib/capture";
import { useT } from "@/lib/i18n";
import type { Signal } from "@/lib/types";

export interface VoiceHandle { toggle(): void }

type Mode = "idle" | "recording" | "processing";

export const Voice = forwardRef<VoiceHandle, { onClip: (sig: Signal, kind: "mic" | "file") => void }>(function Voice({ onClip }, ref) {
  const { t } = useT();
  const [mode, setMode] = useState<Mode>("idle");
  const [err, setErr] = useState<string | null>(null);
  const [secs, setSecs] = useState(0);
  const [level, setLevel] = useState(0);
  const [over, setOver] = useState(false);
  const rec = useRef<Recording | null>(null);
  const raf = useRef(0);
  const file = useRef<HTMLInputElement>(null);

  const finish = useCallback(async () => {
    const r = rec.current; if (!r) return;
    rec.current = null; cancelAnimationFrame(raf.current);
    setMode("processing");
    try {
      const sig = await r.stop();
      if (sig.data.length < sig.fs) { setErr(t("v.short")); setMode("idle"); return; }
      onClip(sig, "mic");
    } catch { setErr(t("v.denied")); }
    setMode("idle");
  }, [onClip, t]);

  const start = useCallback(async () => {
    setErr(null);
    try {
      const r = await record();
      rec.current = r; setMode("recording"); setSecs(0);
      const tick = () => {
        if (!rec.current) return;
        setSecs(r.seconds()); setLevel(r.level());
        if (r.seconds() >= 12) { void finish(); return; }
        raf.current = requestAnimationFrame(tick);
      };
      tick();
    } catch { setErr(t("v.denied")); setMode("idle"); }
  }, [finish, t]);

  const toggle = useCallback(() => { if (mode === "recording") void finish(); else if (mode === "idle") void start(); }, [mode, finish, start]);
  useImperativeHandle(ref, () => ({ toggle }), [toggle]);
  useEffect(() => () => { rec.current?.cancel(); cancelAnimationFrame(raf.current); }, []);

  const take = useCallback(async (f: File | undefined) => {
    if (!f) return;
    setErr(null); setMode("processing");
    try { onClip(await decodeFile(f), "file"); } catch { setErr(t("v.badfile")); }
    setMode("idle");
  }, [onClip, t]);

  return (
    <div className={s.voice}>
      <div className={s.voiceRow}>
        <button type="button" className={`${s.rec} ${mode === "recording" ? s.recOn : ""}`} disabled={mode === "processing"} onClick={toggle} aria-pressed={mode === "recording"}>
          <i />{mode === "recording" ? t("v.stop") : t("v.record")} <kbd>V</kbd>
        </button>
        <div className={`${s.drop} ${over ? s.dropOver : ""}`} role="button" tabIndex={0}
             onClick={() => file.current?.click()} onKeyDown={(e) => { if (e.key === "Enter") file.current?.click(); }}
             onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
             onDrop={(e) => { e.preventDefault(); setOver(false); void take(e.dataTransfer.files?.[0]); }}>
          {t("v.drop")}
        </div>
        <input ref={file} type="file" accept="audio/*" hidden onChange={(e) => { void take(e.target.files?.[0]); e.target.value = ""; }} />
      </div>
      {mode === "recording" && (
        <div className={s.meter}><i style={{ width: `${Math.min(100, Math.round(level * 300))}%` }} /><span className="num">{t("v.recording", { s: secs.toFixed(1) })}</span></div>
      )}
      {mode === "processing" && <div className={s.note}>{t("v.processing")}</div>}
      {err && <div className={`${s.note} ${s.bad}`}>{err}</div>}
      <div className={s.note}>{t("v.note")}</div>
    </div>
  );
});
