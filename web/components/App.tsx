"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import s from "@/app/page.module.css";
import { ARM_META, ARM_ORDER, DEFAULT_ARM, FS_LO, REF_ARM, type ArmName, type Rate } from "@/lib/arms";
import { ABPlayer, type SourceName } from "@/lib/audio";
import { energyAbove, parseWav, specFill, specOf, type Spectrogram } from "@/lib/dsp";
import { getEngine, type Engine } from "@/lib/engine";
import { useT, type Key, type Lang } from "@/lib/i18n";
import { decimate, upsample } from "@/lib/resample";
import type { AudioManifest, Results, Signal, Speaker, WeightsManifest } from "@/lib/types";
import { Blind, type Tally, type Trial, type Vote } from "./Blind";
import { Instrument, type InstrumentHandle } from "./Instrument";
import { Numbers } from "./Numbers";
import { DrawKnobs, ModelList, RateSeg, ReadoutSeg, SpeakerList, ViewSeg, speakerName } from "./Rails";
import { Tour } from "./Tour";
import { Voice, type VoiceHandle } from "./Voice";

type View = "output" | "truth" | "input";
// The readouts the fixtures carry. `draw` is one sample; `logmean16` is the per-bin mean of
// log|STFT| over 16 draws, which is the condition that wins the perceptual judges.
const READOUTS = ["draw", "mean16", "logmean16", "tau0"] as const;
type Readout = (typeof READOUTS)[number];

// A share of energy as a percentage. Small enough shares would round to 0.00 %, which reads as a claim;
// keep two significant figures instead so the number stays the number.
const pct = (v: number) => { const p = 100 * v; return (p >= 0.01 ? p.toFixed(2) : p.toPrecision(2)) + " %"; };

async function getJSON<T>(u: string): Promise<T | null> {
  try { const r = await fetch(u); return r.ok ? ((await r.json()) as T) : null; } catch { return null; }
}
async function getWav(u: string): Promise<Signal> {
  const r = await fetch(u); if (!r.ok) throw new Error(u); return parseWav(await r.arrayBuffer());
}
const wavUrl = (p: string) => (p.startsWith("/") ? p : `/assets/audio/${p}`);

/** "{a} and {b}" with the named slots set in bold. */
function rich(str: string, bold: Record<string, string>): React.ReactNode[] {
  return str.split(/(\{\w+\})/).map((part, i) => {
    const m = /^\{(\w+)\}$/.exec(part);
    return m && m[1] in bold ? <b key={i}>{bold[m[1]]}</b> : part;
  });
}

const TALLY_KEY = "lisa-rtm.blind";
const zeroTally = (): Tally => ({ sampler: 0, det: 0, tie: 0 });
function readTally(): Tally {
  try { const v = JSON.parse(localStorage.getItem(TALLY_KEY) ?? ""); if (v && typeof v.sampler === "number") return v; } catch { /* none yet */ }
  return zeroTally();
}

interface Base { truth: Signal; input: Signal; naive: Signal }

export function App() {
  const router = useRouter();
  const { t, lang, setLang } = useT();
  const inst = useRef<InstrumentHandle>(null);
  const voice = useRef<VoiceHandle>(null);
  const player = useRef<ABPlayer | null>(null);
  const engine = useRef<Engine | null>(null);
  const models = useRef<Record<string, unknown>>({});
  const gen = useRef(0);
  const raf = useRef(0);

  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [speaker, setSpeaker] = useState<Speaker | null>(null);
  const [rate, setRate] = useState<Rate>(4);
  const [arm, setArm] = useState<ArmName>(DEFAULT_ARM);
  const [readout, setReadout] = useState<Readout>("logmean16");
  const [tau, setTau] = useState(1);
  const [tauCommit, setTauCommit] = useState(0);
  const [seed, setSeed] = useState(0);
  const [view, setView] = useState<View>("output");
  const [results, setResults] = useState<Results | null>(null);
  const [wman, setWman] = useState<WeightsManifest | null>(null);
  const [hasEngine, setHasEngine] = useState(false);
  const [autorun, setAutorun] = useState(true);
  const [source, setSource] = useState<SourceName>("output");
  const [playing, setPlaying] = useState(false);
  const [running, setRunning] = useState(false);
  const [empty, setEmpty] = useState<string | null>("loading");
  const [status, setStatus] = useState<React.ReactNode>(null);
  const [tour, setTour] = useState(false);
  const [blindOn, setBlindOn] = useState(false);
  const [trial, setTrial] = useState<Trial | null>(null);
  const [tally, setTally] = useState<Tally>(zeroTally);

  // signals and spectrograms live in refs: they are large and painted imperatively
  const sig = useRef<{ truth: Signal | null; input: Signal | null; naive: Signal | null; output: Signal | null }>({ truth: null, input: null, naive: null, output: null });
  const spec = useRef<{ truth: Spectrogram | null; input: Spectrogram | null; output: Spectrogram | null }>({ truth: null, input: null, output: null });
  const outputTag = useRef("");
  const outputLive = useRef(false);
  const timing = useRef<{ rtf: number; above24: number | null } | null>(null);
  // the measured share above 24 kHz of a precomputed ×8 file, from the audio manifest; null otherwise
  const x8Above24 = useRef<number | null>(null);
  // clips captured in the page, by speaker id; and outputs computed for the blind test, by condition
  const liveBase = useRef<Record<string, Base>>({});
  const outCache = useRef<Map<string, Signal>>(new Map());
  const blindSig = useRef<{ a: Signal | null; b: Signal | null }>({ a: null, b: null });
  const viewBefore = useRef<View>("output");
  const trialNo = useRef(0);

  const det = ARM_META[arm].det;
  const effTau = det ? 0 : tau;
  const live = !!speaker?.live;
  const effReadout: Readout = live && !det ? "draw" : readout;

  const paintView = useCallback((v: View) => {
    const sp = spec.current[v];
    if (!sp) { setEmpty(v === "output" ? t("noout") : t("loading")); inst.current?.clear(); return; }
    setEmpty(null);
    inst.current?.showAll(sp, v === "truth" ? "truth" : "band");
  }, [t]);

  const showStatus = useCallback((mode?: string, dt?: number, secs?: number) => {
    const spk = speaker ? speakerName(speaker, t("v.name")) : "—";
    if (mode === "running") {
      setStatus(<><span className={s.live}>{t("running")}</span><span>{arm}</span><span>×{rate}</span>
        {dt != null && secs != null && <><span>{rich(t("audio.in", { secs: "{secs}", dt: "{dt}" }), { secs: secs.toFixed(2), dt: dt.toFixed(2) })}</span><span><b>{(secs / dt).toFixed(2)}×</b> {t("realtime")}</span></>}</>);
      return;
    }
    if (mode) { setStatus(<span>{mode}</span>); return; }
    const tm = timing.current;
    // the bench number is measured, not asserted: one pass on the bench CPU, from results.json
    const lat = results?.arms?.[arm]?.latency, env = results?.latency_env;
    const bench = lat?.rtf_one_pass ? 1 / lat.rtf_one_pass : null;
    const x8 = !outputLive.current && rate === 8 ? x8Above24.current : null;
    setStatus(<>
      <span>{spk}</span><span>{arm}</span><span>×{rate} · {((FS_LO * rate) / 1000).toFixed(0)} kHz</span>
      {sig.current.output && <span>{outputTag.current}</span>}
      {tm && outputLive.current && <span><b>{tm.rtf.toFixed(2)}×</b> {t("realtime.here")}{bench != null && <> · <b>{bench.toFixed(0)}×</b> {t("bench.cpu", { cpu: env?.cpu ?? t("bench.the") })}</>}</span>}
      {tm && outputLive.current && tm.above24 != null && <span>{t("above24")}: <b>{pct(tm.above24)}</b></span>}
      {sig.current.output && x8 != null && <span>{t("above24")}: <b>{pct(x8)}</b></span>}
      {!sig.current.output && <span>{t("noout")}</span>}
    </>);
  }, [speaker, arm, rate, results, t]);

  // ---- playback ---------------------------------------------------------------------------------
  const stopAudio = useCallback(() => {
    player.current?.stop(); setPlaying(false); cancelAnimationFrame(raf.current); inst.current?.setPlayhead(null);
  }, []);
  const sources = useCallback((): Partial<Record<SourceName, Signal>> => (
    blindOn
      ? { a: blindSig.current.a ?? undefined, b: blindSig.current.b ?? undefined, input: sig.current.input ?? undefined }
      : { input: sig.current.input ?? undefined, output: sig.current.output ?? undefined, truth: sig.current.truth ?? undefined }
  ), [blindOn]);
  const startAudio = useCallback((offset = 0) => {
    const p = (player.current ??= new ABPlayer());
    if (!sig.current.truth) return;
    p.source = source;
    p.onEnded = () => { setPlaying(false); inst.current?.setPlayhead(null); };
    p.start(sources(), offset);
    setPlaying(true);
    const dur = sig.current.truth.data.length / sig.current.truth.fs;
    const tick = () => { if (!p.playing) return; inst.current?.setPlayhead(Math.min(1, p.position() / dur)); raf.current = requestAnimationFrame(tick); };
    tick();
  }, [source, sources]);
  const pickSource = useCallback((k: SourceName) => { setSource(k); player.current?.setSource(k); }, []);

  // ---- inference --------------------------------------------------------------------------------
  const loadModel = useCallback(async (a: ArmName) => {
    const eng = engine.current; if (!eng) throw new Error("no engine");
    return (models.current[a] ??= await eng.load(`/assets/weights/${a}.bin`, "/assets/weights/manifest.json", a));
  }, []);

  const runInference = useCallback(async () => {
    const eng = engine.current, x = sig.current.input;
    if (!eng || !x) return;
    const g = ++gen.current;
    setRunning(true);
    const R = rate, fsOut = FS_LO * R, N = x.data.length * R;
    const out = new Float32Array(N);
    const liveSig: Signal = { data: out, fs: fsOut };
    const sp = specOf(liveSig);
    spec.current.output = sp; sig.current.output = null; outputLive.current = true;
    if (view === "output") { setEmpty(null); inst.current?.blank(sp.nfr); }
    inst.current?.setSweep(0);
    showStatus("running");
    let model: unknown;
    try { model = await loadModel(arm); }
    catch { if (g === gen.current) { setRunning(false); showStatus(t("weights.unavailable", { arm })); } return; }
    if (g !== gen.current) return;
    const t0 = performance.now(); let lastPaint = 0;
    try {
      await eng.run(model, x.data, { R, tau: effTau, seed, chunk: 2048, onProgress: (j, n, o) => {
        if (g !== gen.current) throw new Error(t("cancelled"));
        out.set(o.subarray(0, j), 0);
        const from = sp.done; specFill(sp, j);
        if (view === "output" && sp.done > from) inst.current?.paintRange(sp, "band", from, sp.done);
        inst.current?.setSweep(j / n);
        const now = performance.now();
        if (now - lastPaint > 120) { lastPaint = now; showStatus("running", (now - t0) / 1000, j / fsOut); }
      } });
    } catch (e) { if (g !== gen.current) return; setRunning(false); showStatus(t("error") + ": " + ((e as Error).message ?? e)); return; }
    if (g !== gen.current) return;
    const dt = (performance.now() - t0) / 1000, secs = N / fsOut;
    sig.current.output = liveSig;
    timing.current = { rtf: secs / dt, above24: rate === 8 ? energyAbove(liveSig, 24000) : null };
    x8Above24.current = null;
    outputTag.current = t("tag.live", { tau: effTau.toFixed(2), seed });
    setRunning(false); inst.current?.setSweep(null);
    paintView(view); showStatus();
    player.current?.swap(sources());
  }, [arm, rate, effTau, seed, view, paintView, showStatus, loadModel, sources, t]);

  const cancelInference = useCallback(() => { gen.current++; setRunning(false); inst.current?.setSweep(null); showStatus(t("stopped")); }, [showStatus, t]);

  // ---- data -------------------------------------------------------------------------------------
  // A precomputed file exists only for the rendered conditions: ×4 at seed 0 with τ ∈ {0, 1} for every
  // readout, and ×8 at seed 0 for one draw (τ = 1; τ = 0 for a deterministic arm, whose one output is
  // its only readout). Anything else (another seed, another τ, a clip recorded here) has to come from
  // the engine, which is the point of it.
  const loadOutput = useCallback(async (sp: Speaker) => {
    const a = sp.files?.arms?.[arm];
    let path: string | undefined, lbl = "";
    x8Above24.current = null;
    if (a && rate === 4 && seed === 0) {
      if (det || effTau === 0) { path = a.tau0 ?? a.draw; lbl = t("lbl.tau0"); }
      else if (effTau === 1) { path = a[readout] ?? a.draw; lbl = t(`lbl.${readout}` as Key); }
    } else if (a?.x8 && rate === 8 && seed === 0 && (det || (effTau === 1 && readout === "draw"))) {
      path = a.x8; lbl = (det ? t("lbl.tau0") : t("lbl.draw")) + " · ×8"; x8Above24.current = a.x8_above24 ?? null;
    }
    if (path) {
      try {
        const o = await getWav(wavUrl(path));
        sig.current.output = o; spec.current.output = specOf(o); outputLive.current = false;
        outputTag.current = t("tag.pre", { lbl });
      } catch { sig.current.output = null; spec.current.output = null; x8Above24.current = null; }
    } else { sig.current.output = null; spec.current.output = null; }
    paintView(view); showStatus();
    // a live clip has no file to fall back on: it always runs
    if ((autorun || sp.live) && engine.current && wman?.arms?.[arm]) void runInference();
  }, [arm, rate, det, effTau, seed, readout, view, autorun, wman, paintView, showStatus, runInference, t]);

  const loadBase = useCallback(async (sp: Speaker): Promise<Base> => {
    if (sp.live) { const b = liveBase.current[sp.id]; if (!b) throw new Error(sp.id); return b; }
    const base = `/assets/audio/${sp.id}/`;
    const [truth, input, naive] = await Promise.all([getWav(base + "truth.wav"), getWav(base + "input.wav"), getWav(base + "naive.wav")]);
    return { truth, input, naive };
  }, []);

  const setBase = useCallback((b: Base) => {
    sig.current.truth = b.truth; sig.current.input = b.input; sig.current.naive = b.naive;
    spec.current.truth = specOf(b.truth); spec.current.input = specOf(b.naive);   // the input on the 48 kHz grid = its sinc upsample
  }, []);

  const loadSpeaker = useCallback(async (sp: Speaker) => {
    stopAudio();
    setSpeaker(sp); setEmpty(t("loading.spk", { id: speakerName(sp, t("v.name")) }));
    sig.current = { truth: null, input: null, naive: null, output: null }; spec.current = { truth: null, input: null, output: null };
    setBase(await loadBase(sp));
    await loadOutput(sp);
  }, [stopAudio, loadOutput, loadBase, setBase, t]);

  // a clip from the microphone or a dropped file becomes a speaker: band-limited with the fixture
  // pipeline's own filter, so its input and naive are what make_fixtures.py would have written
  const onClip = useCallback((truth: Signal, kind: "mic" | "file") => {
    const n = Object.keys(liveBase.current).length + 1;
    const id = n === 1 ? "you" : `you-${n}`;
    const input = decimate(truth, 4);
    const naive = upsample(input, 4, truth.data.length);
    liveBase.current[id] = { truth, input, naive };
    const sp: Speaker = { id, live: kind, seconds: truth.data.length / truth.fs, files: { truth: "", input: "", naive: "", arms: {} } };
    setSpeakers((list) => [...list, sp]);
    void loadSpeaker(sp);
  }, [loadSpeaker]);

  // ---- blind test -------------------------------------------------------------------------------
  // One condition's output for one speaker: the fixture file when there is one, the engine otherwise.
  const getOutput = useCallback(async (sp: Speaker, a: ArmName, ro: string, tauv: number): Promise<Signal> => {
    const key = `${sp.id}|${a}|${ro}|${tauv}`;
    const hit = outCache.current.get(key); if (hit) return hit;
    let out: Signal;
    const f = sp.files?.arms?.[a];
    const path = tauv === 0 ? f?.tau0 ?? f?.draw : f?.[ro as "draw" | "mean16" | "logmean16"];
    if (!sp.live && path) out = await getWav(wavUrl(path));
    else {
      const eng = engine.current, x = sp.live ? liveBase.current[sp.id]?.input : sig.current.input;
      if (!eng || !x) throw new Error("no engine");
      const model = await loadModel(a);
      out = { data: await eng.run(model, x.data, { R: 4, tau: tauv, seed: 0, chunk: 4096 }), fs: FS_LO * 4 };
    }
    outCache.current.set(key, out);
    return out;
  }, [loadModel]);

  const newTrial = useCallback(async (list: Speaker[]) => {
    if (!list.length) return;
    stopAudio();
    // a new generation abandons any inference in flight, so the run button must not stay on "stop"
    const g = ++gen.current; setRunning(false); inst.current?.setSweep(null);
    const sp = list[Math.floor(Math.random() * list.length)];
    const sampler: ArmName = ARM_META[arm].det ? DEFAULT_ARM : arm;
    // the sampler is heard at the readout it wins on when that file exists, else one draw
    const best = results?.arms?.[sampler]?.best_readout?.replace("_pt", "") ?? "draw";
    const ro = !sp.live && sp.files?.arms?.[sampler]?.[best as "draw" | "mean16" | "logmean16"] ? best : "draw";
    const tr: Trial = { n: ++trialNo.current, speaker: sp.id, sampler, ro, det: REF_ARM, swap: Math.random() < 0.5, ready: false, vote: null };
    setTrial(tr);
    setSpeaker(sp); setEmpty(t("loading.spk", { id: speakerName(sp, t("v.name")) }));
    sig.current = { truth: null, input: null, naive: null, output: null }; spec.current = { truth: null, input: null, output: null };
    blindSig.current = { a: null, b: null };
    try {
      setBase(await loadBase(sp));
      if (g !== gen.current) return;
      paintView("input"); showStatus(t("b.preparing"));
      const [os, od] = await Promise.all([getOutput(sp, sampler, ro, 1), getOutput(sp, REF_ARM, "draw", 0)]);
      if (g !== gen.current) return;
      blindSig.current = tr.swap ? { a: od, b: os } : { a: os, b: od };
      setTrial({ ...tr, ready: true });
      setSource("a"); showStatus(t("b.trial", { n: tr.n }));
    } catch (e) { if (g === gen.current) showStatus(t("error") + ": " + ((e as Error).message ?? e)); }
  }, [arm, results, stopAudio, loadBase, setBase, paintView, showStatus, getOutput, t]);

  const startBlind = useCallback(() => {
    if (blindOn) return;
    viewBefore.current = view; setView("input");
    setTally(readTally()); setBlindOn(true);
    void newTrial(speakers);
  }, [blindOn, view, speakers, newTrial]);

  const exitBlind = useCallback(() => {
    if (!blindOn) return;
    gen.current++; setRunning(false); inst.current?.setSweep(null); stopAudio();
    setBlindOn(false); setTrial(null); setSource("output"); setView(viewBefore.current);
    if (speaker) void loadSpeaker(speaker);
  }, [blindOn, stopAudio, speaker, loadSpeaker]);

  const vote = useCallback((v: Vote) => {
    if (!trial || !trial.ready || trial.vote) return;
    setTrial({ ...trial, vote: v });
    const next = { ...tally };
    if (v === "tie") next.tie++; else if ((v === "a") !== trial.swap) next.sampler++; else next.det++;
    setTally(next);
    try { localStorage.setItem(TALLY_KEY, JSON.stringify(next)); } catch { /* private mode */ }
  }, [trial, tally]);

  const resetTally = useCallback(() => { setTally(zeroTally()); try { localStorage.removeItem(TALLY_KEY); } catch { /* none */ } }, []);

  // boot
  useEffect(() => {
    let dead = false;
    (async () => {
      const [am, wm, res, eng] = await Promise.all([
        getJSON<AudioManifest>("/assets/audio/manifest.json"), getJSON<WeightsManifest>("/assets/weights/manifest.json"),
        getJSON<Results>("/assets/results.json"), getEngine(),
      ]);
      if (dead) return;
      engine.current = eng; setHasEngine(!!eng); setResults(res); setWman(wm);
      if (am?.speakers?.length) { setSpeakers(am.speakers); }
      else { setEmpty(t("manifest.none")); setStatus(<span>{t("fixtures.none")}</span>); }
    })();
    return () => { dead = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // first speaker once the manifest is in: the load is the response to the manifest arriving, and it
  // sets state on its way (speaker, the empty label) before its first await
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { if (speakers.length && !speaker) void loadSpeaker(speakers[0]); }, [speakers, speaker, loadSpeaker]);
  // arm / rate / tau-commit / seed → reload the output for the current speaker
  const firstRun = useRef(true);
  useEffect(() => {
    if (firstRun.current) { firstRun.current = false; return; }
    if (blindOn) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (speaker) void loadOutput(speaker);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [arm, rate, tauCommit, seed, readout]);
  useEffect(() => { paintView(view); }, [view, paintView]);
  // the status line is prose assembled from refs, so a language change re-renders it by hand
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { if (!running) showStatus(); }, [lang, running, showStatus]);

  // keyboard
  const step = useCallback((list: readonly string[], cur: string, d: number) => list[(list.indexOf(cur) + d + list.length) % list.length], []);
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.matches?.("input, textarea, [contenteditable]")) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key;
      if (e.code === "Space") { e.preventDefault(); if (playing) stopAudio(); else startAudio(0); return; }
      if (k === "?" || (k === "/" && e.shiftKey)) { e.preventDefault(); setTour((o) => !o); return; }
      if (k === "l" || k === "L") { setLang(lang === "ja" ? "en" : "ja"); return; }
      if (blindOn) {
        if (k === "1") pickSource("a"); else if (k === "2") pickSource("b"); else if (k === "3") pickSource("input");
        else if (k === "a" || k === "A") vote("a"); else if (k === "b" || k === "B") vote("b"); else if (k === "t" || k === "T") vote("tie");
        else if (k === "Enter" && trial?.vote) void newTrial(speakers);
        else if (k === "Escape") exitBlind();
        return;
      }
      if (k === "1") pickSource("input"); else if (k === "2") pickSource("output"); else if (k === "3") pickSource("truth");
      else if (k === "r" || k === "R") { if (running) cancelInference(); else void runInference(); }
      else if ((k === "n" || k === "N") && !det) setSeed((n) => n + 1);
      else if (k === "v" || k === "V") voice.current?.toggle();
      else if (k === "b" || k === "B") startBlind();
      else if (k === "ArrowUp" || k === "ArrowDown") { e.preventDefault(); setArm(step(ARM_ORDER, arm, k === "ArrowDown" ? 1 : -1) as ArmName); }
      else if ((k === "ArrowLeft" || k === "ArrowRight") && speakers.length) {
        e.preventDefault();
        const ids = speakers.map((x) => x.id);
        void loadSpeaker(speakers[ids.indexOf(step(ids, speaker?.id ?? ids[0], k === "ArrowRight" ? 1 : -1))]);
      }
    };
    window.addEventListener("keydown", h); return () => window.removeEventListener("keydown", h);
  }, [playing, stopAudio, startAudio, pickSource, blindOn, vote, trial, newTrial, speakers, exitBlind, running, cancelInference, runInference, det, startBlind, arm, speaker, loadSpeaker, step, lang, setLang]);

  const openArch = useCallback((a: ArmName) => { stopAudio(); router.push(`/architecture/${a}`); }, [router, stopAudio]);

  const srcRow: [SourceName, string][] = blindOn
    ? [["a", "A"], ["b", "B"], ["input", t("src.input")]]
    : [["input", t("src.input")], ["output", t("src.output")], ["truth", t("src.truth")]];
  const keys: [string, Key][] = [["space", "k.space"], ["1 2 3", "k.123"], ["R", "k.r"], ["N", "k.n"], ["↑ ↓", "k.updown"], ["← →", "k.leftright"], ["V", "k.v"], ["B", "k.b"], ["L", "k.l"], ["?", "k.q"]];

  return (
    <div className={s.app}>
      <header className={s.top}>
        <h1>The Missing Band</h1>
        <span className={s.sub}>{t("sub")}</span>
        <div className={s.legend}>
          <span className={s.c}><i />{t("legend.in")}</span>
          <span className={s.w}><i />{t("legend.gen")}</span>
          <span className={s.t}><i />{t("legend.truth")}</span>
        </div>
        <div className={s.hbtns}>
          <button type="button" className={s.hbtn} aria-pressed={blindOn} onClick={() => (blindOn ? exitBlind() : startBlind())}>{t("b.start")} <kbd>B</kbd></button>
          <button type="button" className={s.hbtn} aria-pressed={tour} onClick={() => setTour((o) => !o)}>{t("tour")} <kbd>?</kbd></button>
          <div className={s.langs} role="radiogroup" aria-label="language">
            {(["en", "ja"] as Lang[]).map((l) => <button key={l} type="button" aria-pressed={lang === l} onClick={() => setLang(l)}>{l === "en" ? "EN" : "日本語"}</button>)}
          </div>
        </div>
      </header>

      <main className={s.body}>
        <aside className={s.rail}>
          <div className={s.block}>
            <div className={`lbl ${s.blockhead}`}><span>{t("speaker")}</span><span className={s.hint}>{t("speaker.hint")}</span></div>
            <SpeakerList speakers={speakers} current={speaker?.id ?? null} onPick={(sp) => { if (!blindOn) void loadSpeaker(sp); }} />
          </div>
          <div className={s.block}>
            <div className={`lbl ${s.blockhead}`}><span>{t("voice")}</span><span className={s.hint}>{t("voice.hint")}</span></div>
            <Voice ref={voice} onClip={onClip} />
          </div>
          <div className={s.block}>
            <div className={`lbl ${s.blockhead}`}><span>{t("rate")}</span><span className={s.hint}>{t("rate.hint")}</span></div>
            <RateSeg rate={rate} onPick={setRate} />
          </div>
          <div className={s.block}>
            <div className={`lbl ${s.blockhead}`}><span>{t("model")}</span><span className={s.hint}>{t("model.hint")}</span></div>
            <ModelList current={arm} results={results?.arms ?? null} onPick={setArm} onOpen={openArch} />
          </div>
          <div className={s.block}>
            <div className={`lbl ${s.blockhead}`}><span>{t("readout")}</span><span className={s.hint}>{t("readout.hint")}</span></div>
            <ReadoutSeg readout={det ? "tau0" : effReadout} available={det ? ["tau0"] : live ? ["draw"] : ["draw", "mean16", "logmean16"]} live={live && !det}
                        best={results?.arms?.[arm]?.best_readout?.replace("_pt", "") ?? null}
                        onPick={(r) => setReadout(r as Readout)} />
          </div>
          <div className={s.block}>
            <div className={`lbl ${s.blockhead}`}><span>{t("draw")}</span><span className={s.hint}>{t("draw.hint")}</span></div>
            <DrawKnobs tau={tau} seed={seed} det={det} onTau={setTau} onTauCommit={() => setTauCommit((n) => n + 1)} onSeed={setSeed} onNewDraw={() => setSeed((n) => n + 1)} />
          </div>
        </aside>

        <section className={`${s.stage} ${blindOn ? s.blindStage : ""}`}>
          <div className={s.instrument}>
            <div className={s.toolbar}>
              <ViewSeg view={view} onPick={(v) => { if (!blindOn) setView(v); }} />
              <span className={s.grow} />
              <label className={`lbl ${s.auto}`}><input type="checkbox" checked={autorun} onChange={(e) => setAutorun(e.target.checked)} />{t("autorun")}</label>
              <button type="button" className={`${s.run} ${running ? s.stop : ""}`} disabled={!hasEngine || blindOn} title={hasEngine ? "" : t("engine.none")}
                      onClick={() => (running ? cancelInference() : void runInference())}>{running ? t("stop") : t("run")} <kbd>R</kbd></button>
            </div>
            <Instrument ref={inst} empty={empty} />
            <div className={s.transport}>
              <button type="button" className={s.play} aria-label={playing ? t("pause") : t("play")} onClick={() => (playing ? stopAudio() : startAudio(0))}>
                <svg viewBox="0 0 12 12" aria-hidden="true">{playing ? <path d="M2 1.5h3v9H2zM7 1.5h3v9H7z" /> : <path d="M2 1.5v9l8-4.5z" />}</svg>
              </button>
              <div className={s.sources}>
                {srcRow.map(([k, lbl], i) => (
                  <button key={k} type="button" className={s.src} data-src={k} aria-pressed={source === k} onClick={() => pickSource(k)}><i />{lbl} <kbd>{i + 1}</kbd></button>
                ))}
              </div>
              <div className={s.status}>{status ?? <span>{t("loading")}</span>}</div>
            </div>
            <p className={s.prose}>{blindOn ? t("b.hint") : rich(t("prose"), { input: t("src.input"), output: t("src.output"), truth: t("src.truth"), readout: t("readout").toLowerCase() })}</p>
            <div className={s.keys}>{keys.map(([kb, k]) => <span key={k}><kbd>{kb}</kbd>{t(k)}</span>)}</div>
          </div>
        </section>

        <aside className={`${s.rail} ${s.right}`}>
          {blindOn
            ? <Blind trial={trial} tally={tally} onVote={vote} onNext={() => void newTrial(speakers)} onReset={resetTally} onExit={exitBlind} />
            : <Numbers arm={arm} results={results} onOpen={() => openArch(arm)} />}
        </aside>
      </main>
      {tour && <Tour onClose={() => setTour(false)} />}
    </div>
  );
}
