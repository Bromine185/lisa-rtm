"use client";
// The second page: one speaker, the best arm, three sounds on one clock. The spectrogram follows the
// source being heard. One button on to the full instrument.
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import s from "@/app/page.module.css";
import { DEFAULT_ARM } from "@/lib/arms";
import { ABPlayer, type SourceName } from "@/lib/audio";
import { parseWav, specFill, specOf, type Spectrogram } from "@/lib/dsp";
import { useT, type Lang } from "@/lib/i18n";
import type { AudioManifest, Signal, Speaker } from "@/lib/types";
import { Instrument, type InstrumentHandle } from "./Instrument";

type Src = "input" | "output" | "truth";
async function getWav(u: string): Promise<Signal> {
  const r = await fetch(u); if (!r.ok) throw new Error(u); return parseWav(await r.arrayBuffer());
}
const wavUrl = (p: string) => (p.startsWith("/") ? p : `/assets/audio/${p}`);

export function Listen() {
  const { t, lang, setLang } = useT();
  const inst = useRef<InstrumentHandle>(null);
  const player = useRef<ABPlayer | null>(null);
  const raf = useRef(0);
  const sig = useRef<Record<Src, Signal | null>>({ input: null, output: null, truth: null });
  const spec = useRef<Record<Src, Spectrogram | null>>({ input: null, output: null, truth: null });
  const [spk, setSpk] = useState<Speaker | null>(null);
  const [ready, setReady] = useState(false);
  const [source, setSource] = useState<Src>("output");
  const [playing, setPlaying] = useState(false);
  const [empty, setEmpty] = useState<string | null>("loading");

  const paint = useCallback((k: Src) => {
    const sp = spec.current[k]; if (!sp) return;
    inst.current?.showAll(sp, k === "truth" ? "truth" : "band");
  }, []);

  useEffect(() => {
    let dead = false;
    (async () => {
      const r = await fetch("/assets/audio/manifest.json");
      const am = r.ok ? ((await r.json()) as AudioManifest) : null;
      const sp = am?.speakers?.[0];
      if (dead) return;
      if (!sp) { setEmpty(t("manifest.none")); return; }
      setSpk(sp); setEmpty(t("s.loading", { spk: sp.id }));
      const files = sp.files.arms[DEFAULT_ARM];
      const outPath = files?.logmean16 ?? files?.draw;
      if (!outPath) { setEmpty(t("noout")); return; }
      const [truth, input, naive, out] = await Promise.all([getWav(wavUrl(sp.files.truth)), getWav(wavUrl(sp.files.input)), getWav(wavUrl(sp.files.naive)), getWav(wavUrl(outPath))]);
      if (dead) return;
      sig.current = { input, output: out, truth };
      spec.current = { input: specFill(specOf(naive), naive.data.length), output: specFill(specOf(out), out.data.length), truth: specFill(specOf(truth), truth.data.length) };
      setEmpty(null); setReady(true);
      paint("output");
    })();
    return () => { dead = true; player.current?.stop(); cancelAnimationFrame(raf.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stop = useCallback(() => { player.current?.stop(); setPlaying(false); cancelAnimationFrame(raf.current); inst.current?.setPlayhead(null); }, []);
  const start = useCallback(() => {
    const p = (player.current ??= new ABPlayer());
    const truth = sig.current.truth; if (!truth) return;
    p.source = source;
    p.onEnded = () => { setPlaying(false); inst.current?.setPlayhead(null); };
    p.start({ input: sig.current.input ?? undefined, output: sig.current.output ?? undefined, truth: truth }, 0);
    setPlaying(true);
    const dur = truth.data.length / truth.fs;
    const tick = () => { if (!p.playing) return; inst.current?.setPlayhead(Math.min(1, p.position() / dur)); raf.current = requestAnimationFrame(tick); };
    tick();
  }, [source]);
  const pick = useCallback((k: Src) => { setSource(k); player.current?.setSource(k as SourceName); paint(k); }, [paint]);

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.code === "Space") { e.preventDefault(); if (playing) stop(); else start(); }
      else if (e.key === "1") pick("input"); else if (e.key === "2") pick("output"); else if (e.key === "3") pick("truth");
      else if (e.key === "l" || e.key === "L") setLang(lang === "ja" ? "en" : "ja");
      else if (e.key === "Enter" || e.key === "ArrowRight") { e.preventDefault(); stop(); window.location.assign("/instrument"); }
      else if (e.key === "ArrowLeft" || e.key === "Escape") { e.preventDefault(); stop(); window.location.assign("/"); }
    };
    window.addEventListener("keydown", h); return () => window.removeEventListener("keydown", h);
  }, [playing, stop, start, pick, lang, setLang]);

  const srcs: Src[] = ["input", "output", "truth"];
  return (
    <div className={`${s.app} ${s.land}`}>
      <header className={s.top}>
        <Link href="/" className={s.hbtn}>{t("s.home")}</Link>
        <h1>The Missing Band</h1>
        <span className={s.sub}>{t("s.which", { spk: spk?.id ?? "—", arm: DEFAULT_ARM, ro: t("t.ro.logmean16_pt") })}</span>
        <div className={s.hbtns}>
          <div className={s.langs} role="radiogroup" aria-label="language">
            {(["en", "ja"] as Lang[]).map((l) => <button key={l} type="button" aria-pressed={lang === l} onClick={() => setLang(l)}>{l === "en" ? "EN" : "日本語"}</button>)}
          </div>
        </div>
      </header>
      <main className={s.listenBody}>
        <h2 className={s.landH}>{t("s.h")}</h2>
        <div className={s.listenStage}>
          <Instrument ref={inst} empty={empty} />
          <div className={s.bigRow}>
            <button type="button" className={s.bigPlay} disabled={!ready} aria-label={playing ? t("pause") : t("play")} onClick={() => (playing ? stop() : start())}>
              <svg viewBox="0 0 12 12" aria-hidden="true">{playing ? <path d="M2 1.5h3v9H2zM7 1.5h3v9H7z" /> : <path d="M2 1.5v9l8-4.5z" />}</svg>
              <span>{playing ? t("pause") : t("play")} <kbd>␣</kbd></span>
            </button>
            {srcs.map((k, i) => (
              <button key={k} type="button" className={`${s.src} ${s.bigSrc}`} data-src={k} aria-pressed={source === k} disabled={!ready} onClick={() => pick(k)}>
                <i />{t(`src.${k}`)} <kbd>{i + 1}</kbd>
              </button>
            ))}
          </div>
          <p className={s.prose}>{t("s.p")}</p>
        </div>
        <Link href="/instrument" className={s.giant} onClick={stop}>
          <span className={s.giantMain}>{t("s.next")} <span className={s.giantArrow}>→</span></span>
          <span className={s.giantSub}>{t("s.next.sub")} <kbd>⏎</kbd></span>
        </Link>
      </main>
    </div>
  );
}
