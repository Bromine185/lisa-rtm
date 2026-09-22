"use client";
// The first page: the numbers, then the band. Four stat tiles from results.json, the input and the
// best arm's output as spectrograms side by side, their long-term average spectra on one axis with
// the difference above 6 kHz filled, and one button to the listening page.
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import s from "@/app/page.module.css";
import { CUT_HZ, DEFAULT_ARM, FMAX_HZ, REF_ARM } from "@/lib/arms";
import { parseWav, specFill, specOf, type Spectrogram } from "@/lib/dsp";
import { useT, type Lang } from "@/lib/i18n";
import { heroTiles } from "@/lib/tiles";
import type { AudioManifest, Results, Signal, Speaker } from "@/lib/types";
import { Instrument, type InstrumentHandle } from "./Instrument";
import { Tiles } from "./Tiles";

async function getJSON<T>(u: string): Promise<T | null> {
  try { const r = await fetch(u); return r.ok ? ((await r.json()) as T) : null; } catch { return null; }
}
async function getWav(u: string): Promise<Signal> {
  const r = await fetch(u); if (!r.ok) throw new Error(u); return parseWav(await r.arrayBuffer());
}
const wavUrl = (p: string) => (p.startsWith("/") ? p : `/assets/audio/${p}`);

/** Long-term average spectrum: mean power per bin over all frames, in dB. Power, not dB, is averaged,
 *  so the curve is the energy the band carries rather than a mean over silent frames. */
function ltas(sp: Spectrogram): Float32Array {
  const out = new Float32Array(sp.nb);
  for (let f = 0; f < sp.done; f++) for (let b = 0; b < sp.nb; b++) out[b] += Math.pow(10, sp.db[f * sp.nb + b] / 10);
  for (let b = 0; b < sp.nb; b++) out[b] = 10 * Math.log10(out[b] / Math.max(1, sp.done) + 1e-30);
  return out;
}

function Ltas({ curves, spk }: { curves: { input: Float32Array; output: Float32Array; truth: Float32Array; hzPerBin: number } | null; spk: string }) {
  const { t } = useT();
  const W = 1000, H = 260, L = 44, R = 12, T = 14, B = 26;
  const floor = -90;
  if (!curves) return <div className={s.ltas}><div className={s.empty}>{t("loading")}</div></div>;
  let peak = -1e9;
  for (const c of [curves.input, curves.output, curves.truth]) for (let i = 0; i < c.length; i++) if (c[i] > peak) peak = c[i];
  const x = (hz: number) => L + ((W - L - R) * hz) / FMAX_HZ;
  const y = (db: number) => T + (H - T - B) * Math.min(1, Math.max(0, (peak - db) / -floor));
  const path = (c: Float32Array) => {
    let d = "";
    for (let b = 0; b < c.length; b++) { const hz = b * curves.hzPerBin; if (hz > FMAX_HZ) break; d += (d ? " L" : "M") + x(hz).toFixed(1) + " " + y(c[b]).toFixed(1); }
    return d;
  };
  // the region between the input and the output above the cut: what the sampler adds
  const b0 = Math.ceil(CUT_HZ / curves.hzPerBin), b1 = Math.min(curves.output.length - 1, Math.floor(FMAX_HZ / curves.hzPerBin));
  let fill = "";
  for (let b = b0; b <= b1; b++) fill += (fill ? " L" : "M") + x(b * curves.hzPerBin).toFixed(1) + " " + y(curves.output[b]).toFixed(1);
  for (let b = b1; b >= b0; b--) fill += " L" + x(b * curves.hzPerBin).toFixed(1) + " " + y(curves.input[b]).toFixed(1);
  fill += " Z";
  return (
    <div className={s.ltas}>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label={t("l.ltas", { spk })}>
        {[0, -20, -40, -60, -80].map((db) => <line key={db} x1={L} x2={W - R} y1={y(peak + db)} y2={y(peak + db)} className={s.ltasGrid} />)}
        {[0, 6, 12, 18, 24].map((k) => <line key={k} x1={x(k * 1000)} x2={x(k * 1000)} y1={T} y2={H - B} className={k === 6 ? s.ltasCut : s.ltasGrid} />)}
        <path d={fill} className={s.ltasFill} />
        <path d={path(curves.truth)} className={s.ltasTruth} />
        <path d={path(curves.input)} className={s.ltasIn} />
        <path d={path(curves.output)} className={s.ltasOut} />
        {[0, -20, -40, -60, -80].map((db) => <text key={db} x={L - 6} y={y(peak + db) + 3} className={s.ltasTick} textAnchor="end">{db} dB</text>)}
        {[0, 6, 12, 18, 24].map((k) => <text key={k} x={x(k * 1000)} y={H - 8} className={s.ltasTick} textAnchor={k === 24 ? "end" : k === 0 ? "start" : "middle"}>{k} kHz</text>)}
      </svg>
      <div className={s.ltasLegend}>
        <span className={s.c}><i />{t("src.input")}</span>
        <span className={s.w}><i />{t("src.output")}</span>
        <span className={s.t}><i />{t("src.truth")}</span>
        <span className={s.ltasDiffKey}><i />{t("l.ltas.diff")}</span>
      </div>
    </div>
  );
}

export function Landing() {
  const { t, lang, setLang } = useT();
  const inst = useRef<InstrumentHandle>(null);
  const outst = useRef<InstrumentHandle>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [spk, setSpk] = useState<Speaker | null>(null);
  const [curves, setCurves] = useState<{ input: Float32Array; output: Float32Array; truth: Float32Array; hzPerBin: number } | null>(null);
  const [empty, setEmpty] = useState<string | null>("loading");

  useEffect(() => {
    let dead = false;
    (async () => {
      const [am, res] = await Promise.all([getJSON<AudioManifest>("/assets/audio/manifest.json"), getJSON<Results>("/assets/results.json")]);
      if (dead) return;
      setResults(res);
      const sp = am?.speakers?.[0];
      if (!sp) { setEmpty(t("manifest.none")); return; }
      setSpk(sp);
      const files = sp.files.arms[DEFAULT_ARM];
      const outPath = files?.logmean16 ?? files?.draw;
      if (!outPath) { setEmpty(t("noout")); return; }
      const [naive, out, truth] = await Promise.all([getWav(wavUrl(sp.files.naive)), getWav(wavUrl(outPath)), getWav(wavUrl(sp.files.truth))]);
      if (dead) return;
      const spIn = specFill(specOf(naive), naive.data.length), spOut = specFill(specOf(out), out.data.length), spTr = specFill(specOf(truth), truth.data.length);
      setEmpty(null);
      inst.current?.showAll(spIn, "band"); outst.current?.showAll(spOut, "band");
      setCurves({ input: ltas(spIn), output: ltas(spOut), truth: ltas(spTr), hzPerBin: spIn.hzPerBin });
    })();
    return () => { dead = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "l" || e.key === "L") setLang(lang === "ja" ? "en" : "ja");
      else if (e.key === "Enter" || e.key === "ArrowRight" || e.code === "Space") { e.preventDefault(); window.location.assign("/listen"); }
    };
    window.addEventListener("keydown", h); return () => window.removeEventListener("keydown", h);
  }, [lang, setLang]);

  const tiles = heroTiles(results, DEFAULT_ARM, t, true);
  const epochs = /^OV(\d+)/.exec(results?.tag ?? "")?.[1];
  return (
    <div className={`${s.app} ${s.land}`}>
      <header className={s.top}>
        <h1>The Missing Band</h1>
        <span className={s.sub}>{t("sub")}</span>
        <div className={s.hbtns}>
          <div className={s.langs} role="radiogroup" aria-label="language">
            {(["en", "ja"] as Lang[]).map((l) => <button key={l} type="button" aria-pressed={lang === l} onClick={() => setLang(l)}>{l === "en" ? "EN" : "日本語"}</button>)}
          </div>
        </div>
      </header>
      <main className={s.landBody}>
        <div className={`lbl ${s.landKicker}`}>{t("l.kicker", { tag: results?.tag ?? "OV50", n: results?.n_utts ?? 12, M: results?.M ?? 16, arm: DEFAULT_ARM, ref: REF_ARM })}{epochs ? ` · ${t("n.epochs", { e: epochs })}` : ""}</div>
        <Tiles tiles={tiles} big />
        <h2 className={s.landH}>{t("l.spec.h")}</h2>
        <div className={s.landGrid}>
          <div className={s.landSpec}>
            <div className={`lbl ${s.landCap}`}><span className={s.c}><i />{t("l.spec.in")}</span></div>
            <Instrument ref={inst} empty={empty} />
          </div>
          <div className={s.landSpec}>
            <div className={`lbl ${s.landCap}`}><span className={s.w}><i />{t("l.spec.out", { arm: DEFAULT_ARM })}</span></div>
            <Instrument ref={outst} empty={empty} />
          </div>
          <div className={s.landLtas}>
            <div className={`lbl ${s.landCap}`}>{t("l.ltas", { spk: spk?.id ?? "—" })}</div>
            <Ltas curves={curves} spk={spk?.id ?? ""} />
          </div>
        </div>
        <Link href="/listen" className={s.giant}>
          <span className={s.giantMain}>{t("l.next")} <span className={s.giantArrow}>→</span></span>
          <span className={s.giantSub}>{t("l.next.sub")} <kbd>⏎</kbd></span>
        </Link>
      </main>
    </div>
  );
}
