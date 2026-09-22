"use client";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import s from "@/app/page.module.css";
import a from "./arch.module.css";
import { ARM_META, ARM_ORDER, type ArmName } from "@/lib/arms";
import { useT, type Key, type Lang } from "@/lib/i18n";
import { ArchView, type ArchViewHandle } from "./ArchView";

// where each stage begins on the timeline; the scene's own STAGES, mirrored here for the strip
const STAGE_AT = [0, 0.15, 0.45, 0.6, 0.9];

export function ArchPage({ arm }: { arm: ArmName }) {
  const { t, lang, setLang } = useT();
  const meta = ARM_META[arm];
  const view = useRef<ArchViewHandle>(null);
  const [playing, setPlaying] = useState(true);
  const [tt, setT] = useState(0);
  const [seek, setSeek] = useState<number | null>(null);
  const [missing, setMissing] = useState(false);
  const [focus, setFocus] = useState<number | null>(null);

  // fly the camera to a stage and put the timeline at its start; the same stage again fits the whole
  const focusStage = useCallback((i: number | null) => {
    const next = i != null && i === focus ? null : i;
    setFocus(next);
    view.current?.focus(next);
    if (next != null) { setT(STAGE_AT[next]); setSeek(STAGE_AT[next]); }
  }, [focus]);
  const fitAll = useCallback(() => { setFocus(null); view.current?.reset(); }, []);

  useEffect(() => {
    // the media query is client-only, so the initial state cannot know it; one settle on mount
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) setPlaying(false);
  }, []);
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.matches?.("input, textarea")) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.code === "Space") { e.preventDefault(); setPlaying((p) => !p); }
      else if (e.key >= "1" && e.key <= "5") { e.preventDefault(); focusStage(+e.key - 1); }
      else if (e.key === "0" || e.key === "f" || e.key === "F" || e.key === "Escape") { e.preventDefault(); fitAll(); }
      else if (e.key === "l" || e.key === "L") setLang(lang === "ja" ? "en" : "ja");
    };
    window.addEventListener("keydown", h); return () => window.removeEventListener("keydown", h);
  }, [focusStage, fitAll, lang, setLang]);

  const stage = STAGE_AT.reduce((acc, at, i) => (tt >= at ? i : acc), 0);
  const kind = ({ "plain L1 (the paper)": "kind.plain", "L1 + λ·STFT": "kind.l1stft", "energy score, log-mag": "kind.es.logmag", "energy score, +ERB": "kind.es.erb" } as Record<string, Key>)[meta.kind];

  return (
    <div className={a.page}>
      <header className={a.bar}>
        <Link href="/" className={a.back}>{t("a.back")}</Link>
        <span className={a.title}>{arm}<small>{meta.cls} · {kind ? t(kind) : meta.kind} · λ {meta.lam} · {t(`arm.${arm}` as Key)}</small></span>
        <span className={s.grow} />
        <nav className={a.arms} aria-label={t("a.other")}>
          {ARM_ORDER.map((x) => <Link key={x} href={`/architecture/${x}`} aria-current={x === arm ? "page" : undefined}>{x}</Link>)}
        </nav>
        <button type="button" className={a.back} onClick={fitAll}>{t("a.fit")} <kbd>0</kbd></button>
        <button type="button" className={a.back} onClick={() => setPlaying((p) => !p)}>{playing ? t("pause") : t("play")} <kbd>␣</kbd></button>
        <div className={s.langs} role="radiogroup" aria-label="language">
          {(["en", "ja"] as Lang[]).map((l) => <button key={l} type="button" aria-pressed={lang === l} onClick={() => setLang(l)}>{l === "en" ? "EN" : "日本語"}</button>)}
        </div>
      </header>

      <div className={a.mount}>
        <ArchView ref={view} arm={arm} cls={meta.cls} playing={playing} seek={seek} onTime={setT} onMissing={() => setMissing(true)} />
        {missing && <div className={s.empty}>{t("a.missing")}</div>}
        <div className={a.help} aria-hidden="true">{t("a.help")}</div>
      </div>

      <div className={a.tl}>
        <span>{t("a.timeline")}</span>
        <input type="range" min={0} max={1000} value={Math.round(tt * 1000)} aria-label={t("a.scrub")}
               onChange={(e) => { const v = +e.target.value / 1000; setT(v); setSeek(v); setPlaying(false); }} />
        <span className="num">{tt.toFixed(2)}</span>
      </div>
      <div className={a.stages} role="toolbar" aria-label={t("a.focus")}>
        {STAGE_AT.map((at, i) => (
          <button key={at} type="button" className={`${i === stage ? a.on : ""} ${i === focus ? a.focused : ""}`} aria-pressed={i === focus}
                  onClick={() => focusStage(i)} title={t("a.focus")}>
            <kbd>{i + 1}</kbd>{t(`a.stage.${i}` as Key)}
          </button>
        ))}
      </div>

      <dl className={a.facts}>
        <div><dt>{t("a.f.input")}</dt><dd>{t("a.f.input.v")}</dd></div>
        <div><dt>{t("a.f.encoder")}</dt><dd>{t("a.f.encoder.v")}</dd></div>
        <div><dt>{t("a.f.decoder")}</dt><dd>[c, z₋₁, z₀, z₊₁{meta.cls === "LISASD" ? ", ε₄" : ""}] → 5 × Linear(144) → 1 · c = 2(q − i) − 1</dd></div>
        <div><dt>{t("a.f.noise")}</dt><dd>{t(meta.cls === "LISASD" ? "a.f.noise.sd" : meta.det ? "a.f.noise.det" : "a.f.noise.s")}</dd></div>
        <div><dt>{t("a.f.params")}</dt><dd>{t("a.f.params.v", { n: meta.cls === "LISASD" ? "88,353" : "87,777" })}</dd></div>
      </dl>
    </div>
  );
}
