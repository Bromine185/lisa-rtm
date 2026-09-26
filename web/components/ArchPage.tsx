"use client";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import s from "@/app/page.module.css";
import a from "./arch.module.css";
import { ARM_META, ARM_ORDER, DEFAULT_ARM, RELEASED_META, RELEASED_ORDER, isReleased, type ModelId } from "@/lib/arms";
import { useT, type Key, type Lang } from "@/lib/i18n";
import { fmtParams } from "@/lib/tiles";
import type { AudioManifest, ReleasedModelManifest, Results, SotaArch, SotaModel } from "@/lib/types";
import { ArchView, type ArchViewHandle } from "./ArchView";

// where each stage begins on the timeline; the scene's own STAGES, mirrored here for the strip
const STAGE_AT = [0, 0.15, 0.45, 0.6, 0.9];

async function getJSON<T>(u: string): Promise<T | null> {
  try { const r = await fetch(u); return r.ok ? ((await r.json()) as T) : null; } catch { return null; }
}
const fmtInt = (v: number | null | undefined) => (v == null || !isFinite(v) ? "—" : Math.round(v).toLocaleString("en-US"));
// "8 DDIM steps" → 8 for the scene's loop arc; "1 pass" → 1; no leading integer → none
const stepsOf = (str: string) => { const n = parseInt(str, 10); return Number.isFinite(n) ? n : undefined; };

export function ArchPage({ model }: { model: ModelId }) {
  const { t, lang, setLang } = useT();
  const rel = isReleased(model);
  // a released model has no ARM_META; the arm fields below are read only on the arm path
  const arm = rel ? DEFAULT_ARM : model;
  const meta = ARM_META[arm];
  // a released model's facts: results.sota (make_results_sota.py), the audio manifest's released block
  // as the fallback while results.sota is not written; neither exists for an arm
  const [sota, setSota] = useState<SotaModel | null>(null);
  const [man, setMan] = useState<ReleasedModelManifest | null>(null);
  const [pending, setPending] = useState(rel);
  const view = useRef<ArchViewHandle>(null);
  const [playing, setPlaying] = useState(true);
  const [tt, setT] = useState(0);
  const [seek, setSeek] = useState<number | null>(null);
  const [missing, setMissing] = useState(false);
  const [focus, setFocus] = useState<number | null>(null);

  useEffect(() => {
    if (!isReleased(model)) return;
    let dead = false;
    (async () => {
      const [res, am] = await Promise.all([getJSON<Results>("/assets/results.json"), getJSON<AudioManifest>("/assets/audio/manifest.json")]);
      if (dead) return;
      setSota(res?.sota?.models?.[model] ?? null); setMan(am?.released?.models?.[model] ?? null); setPending(false);
    })();
    return () => { dead = true; };
  }, [model]);

  // the facts and blocks: results.sota's copy, else the one demo/tools/add_sota.py folded into the audio
  // manifest (the same sota_models.json entry, block counts from sota/params.json)
  const arch: SotaArch | null = sota?.arch ?? man?.arch ?? null;
  // the block scene has n equal stages, one per block (the arch3d contract); the LISA scene has five
  const blocks = useMemo(() => arch?.blocks ?? [], [arch]);
  const noScene = rel && !blocks.length;
  const stageAt = useMemo(() => (rel ? blocks.map((_, i) => i / blocks.length) : STAGE_AT), [rel, blocks]);

  // fly the camera to a stage and put the timeline at its start; the same stage again fits the whole
  const focusStage = useCallback((i: number | null) => {
    const next = i != null && i === focus ? null : i;
    setFocus(next);
    view.current?.focus(next);
    if (next != null) { setT(stageAt[next]); setSeek(stageAt[next]); }
  }, [focus, stageAt]);
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
      else if (e.key >= "1" && e.key <= "9" && +e.key - 1 < stageAt.length) { e.preventDefault(); focusStage(+e.key - 1); }
      else if (e.key === "0" || e.key === "f" || e.key === "F" || e.key === "Escape") { e.preventDefault(); fitAll(); }
      else if (e.key === "l" || e.key === "L") setLang(lang === "ja" ? "en" : "ja");
    };
    window.addEventListener("keydown", h); return () => window.removeEventListener("keydown", h);
  }, [focusStage, fitAll, lang, setLang, stageAt.length]);

  const stage = stageAt.reduce((acc, at, i) => (tt >= at ? i : acc), 0);
  const relName = rel ? sota?.name ?? man?.name ?? RELEASED_META[model].name : "";
  const kind = ({ "plain L1 (the paper)": "kind.plain", "L1 + λ·STFT": "kind.l1stft", "energy score, log-mag": "kind.es.logmag", "energy score, +ERB": "kind.es.erb" } as Record<string, Key>)[meta.kind];

  return (
    <div className={a.page}>
      <header className={a.bar}>
        <Link href="/instrument" className={a.back}>{t("a.back")}</Link>
        {rel
          ? <span className={a.title}>{relName}<small>{sota?.family ?? man?.family ?? ""}</small></span>
          : <span className={a.title}>{arm}<small>{meta.cls} · {kind ? t(kind) : meta.kind} · λ {meta.lam} · {t(`arm.${arm}` as Key)}</small></span>}
        <span className={s.grow} />
        <nav className={a.arms} aria-label={t("a.other")}>
          {ARM_ORDER.map((x) => <Link key={x} href={`/architecture/${x}`} aria-current={x === model ? "page" : undefined}>{x}</Link>)}
          <span className={a.sep} aria-label={t("a.other.rel")}>·</span>
          {RELEASED_ORDER.map((x) => <Link key={x} href={`/architecture/${x}`} aria-current={x === model ? "page" : undefined}>{RELEASED_META[x].name}</Link>)}
        </nav>
        <button type="button" className={a.back} onClick={fitAll}>{t("a.fit")} <kbd>0</kbd></button>
        <button type="button" className={a.back} onClick={() => setPlaying((p) => !p)}>{playing ? t("pause") : t("play")} <kbd>␣</kbd></button>
        <div className={s.langs} role="radiogroup" aria-label="language">
          {(["en", "ja"] as Lang[]).map((l) => <button key={l} type="button" aria-pressed={lang === l} onClick={() => setLang(l)}>{l === "en" ? "EN" : "日本語"}</button>)}
        </div>
      </header>

      <div className={a.mount}>
        {!rel
          ? <ArchView ref={view} arm={arm} cls={meta.cls} playing={playing} seek={seek} onTime={setT} onMissing={() => setMissing(true)} />
          : !noScene
            ? <ArchView key={model} ref={view} arm={model} cls="LISAS" playing={playing} seek={seek} onTime={setT} onMissing={() => setMissing(true)}
                        blocks={blocks} name={relName} det={sota?.det ?? man?.det ?? RELEASED_META[model].det}
                        steps={arch ? stepsOf(arch.steps) : undefined} params={sota?.params ?? man?.params ?? arch?.params} />
            : !pending && <div className={s.empty}>{t("a.rel.pending")}</div>}
        {missing && <div className={s.empty}>{t("a.missing")}</div>}
        {!noScene && <div className={a.help} aria-hidden="true">{t(rel ? "a.help.rel" : "a.help")}</div>}
      </div>

      <div className={a.tl}>
        <span>{t("a.timeline")}</span>
        <input type="range" min={0} max={1000} value={Math.round(tt * 1000)} aria-label={t("a.scrub")}
               onChange={(e) => { const v = +e.target.value / 1000; setT(v); setSeek(v); setPlaying(false); }} />
        <span className="num">{tt.toFixed(2)}</span>
      </div>
      {!noScene && <div className={a.stages} role="toolbar" aria-label={t("a.focus")}>
        {stageAt.map((at, i) => (
          <button key={at} type="button" className={`${i === stage ? a.on : ""} ${i === focus ? a.focused : ""}`} aria-pressed={i === focus}
                  onClick={() => focusStage(i)} title={t("a.focus")}>
            <kbd>{i + 1}</kbd>{rel ? blocks[i].label : t(`a.stage.${i}` as Key)}
          </button>
        ))}
      </div>}

      {rel ? <RelFacts sota={sota} man={man} arch={arch} /> : <dl className={a.facts}>
        <div><dt>{t("a.f.input")}</dt><dd>{t("a.f.input.v")}</dd></div>
        <div><dt>{t("a.f.encoder")}</dt><dd>{t("a.f.encoder.v")}</dd></div>
        <div><dt>{t("a.f.decoder")}</dt><dd>[c, z₋₁, z₀, z₊₁{meta.cls === "LISASD" ? ", ε₄" : ""}] → 5 × Linear(144) → 1 · c = 2(q − i) − 1</dd></div>
        <div><dt>{t("a.f.noise")}</dt><dd>{t(meta.cls === "LISASD" ? "a.f.noise.sd" : meta.det ? "a.f.noise.det" : "a.f.noise.s")}</dd></div>
        <div><dt>{t("a.f.params")}</dt><dd>{t("a.f.params.v", { n: meta.cls === "LISASD" ? "88,353" : "87,777" })}</dd></div>
      </dl>}
    </div>
  );
}

/** A released model's facts: every string is data from demo/tools/sota_models.json (via results.sota, or
 *  the audio manifest's copy), every count from the checkpoints (sota/count_params.py). The block list says
 *  "pending" once, in the mount, so it is simply left out here when there is none. */
function RelFacts({ sota, man, arch }: { sota: SotaModel | null; man: ReleasedModelManifest | null; arch: SotaArch | null }) {
  const { t } = useT();
  const params = sota?.params ?? man?.params ?? arch?.params;
  const draws = arch?.det_note ?? (man ? (man.det ? t("rel.det") : t("rel.sampler")) : null);
  return (
    <dl className={a.facts}>
      <div><dt>{t("a.f.family")}</dt><dd>{sota?.family ?? man?.family ?? "—"}</dd></div>
      <div><dt>{t("a.f.params")}</dt><dd>{fmtInt(params)}{arch?.params_note ? <> · {arch.params_note}</> : null}</dd></div>
      {arch?.input && <div><dt>{t("a.f.input.rel")}</dt><dd>{arch.input}</dd></div>}
      {draws && <div><dt>{t("a.f.det")}</dt><dd>{draws}</dd></div>}
      {arch?.blocks?.length ? (
        <div className={a.wide}><dt>{t("a.f.blocks")}</dt><dd>
          <ol>{arch.blocks.map((b) => <li key={b.id}><b>{b.label}</b> — {b.detail} — {fmtParams(b.params)}</li>)}</ol>
        </dd></div>
      ) : null}
      {arch?.training && <div><dt>{t("a.f.training")}</dt><dd>{arch.training}</dd></div>}
      <div><dt>{t("a.f.seen")}</dt><dd>{arch?.seen_note ?? man?.seen_note ?? "—"}</dd></div>
      <div><dt>{t("a.f.steps")}</dt><dd>{arch?.steps ?? man?.steps ?? "—"}</dd></div>
      {arch?.context && <div><dt>{t("a.f.context")}</dt><dd>{arch.context}</dd></div>}
      {arch?.paper && <div><dt>{t("a.f.paper")}</dt><dd>{arch.paper}</dd></div>}
      {arch?.code && <div><dt>{t("a.f.code")}</dt><dd>{arch.code}</dd></div>}
    </dl>
  );
}
