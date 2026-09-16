"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import s from "@/app/page.module.css";
import { A100_REALTIME, ARM_META, DEFAULT_ARM, FS_LO, type ArmName } from "@/lib/arms";
import { ABPlayer, type SourceName } from "@/lib/audio";
import { decimate2, energyAbove, parseWav, specFill, specOf, type Spectrogram } from "@/lib/dsp";
import { getEngine, type Engine } from "@/lib/engine";
import type { AudioManifest, Results, Signal, Speaker, WeightsManifest } from "@/lib/types";
import { Instrument, type InstrumentHandle } from "./Instrument";
import { Numbers } from "./Numbers";
import { DrawKnobs, ModelList, RateSeg, SpeakerList, ViewSeg } from "./Rails";

type View = "output" | "truth" | "input";
type Rate = 2 | 4 | 8;

async function getJSON<T>(u: string): Promise<T | null> {
  try { const r = await fetch(u); return r.ok ? ((await r.json()) as T) : null; } catch { return null; }
}
async function getWav(u: string): Promise<Signal> {
  const r = await fetch(u); if (!r.ok) throw new Error(u); return parseWav(await r.arrayBuffer());
}

export function App() {
  const router = useRouter();
  const inst = useRef<InstrumentHandle>(null);
  const player = useRef<ABPlayer | null>(null);
  const engine = useRef<Engine | null>(null);
  const models = useRef<Record<string, unknown>>({});
  const gen = useRef(0);
  const raf = useRef(0);

  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [speaker, setSpeaker] = useState<Speaker | null>(null);
  const [rate, setRate] = useState<Rate>(4);
  const [arm, setArm] = useState<ArmName>(DEFAULT_ARM);
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
  const [status, setStatus] = useState<React.ReactNode>(<span>loading</span>);

  // signals and spectrograms live in refs: they are large and painted imperatively
  const sig = useRef<{ truth: Signal | null; input: Signal | null; naive: Signal | null; output: Signal | null }>({ truth: null, input: null, naive: null, output: null });
  const spec = useRef<{ truth: Spectrogram | null; input: Spectrogram | null; output: Spectrogram | null }>({ truth: null, input: null, output: null });
  const outputTag = useRef("");
  const outputLive = useRef(false);
  const timing = useRef<{ rtf: number; above24: number | null } | null>(null);

  const det = ARM_META[arm].det;
  const effTau = det ? 0 : tau;

  const paintView = useCallback((v: View) => {
    const sp = spec.current[v];
    if (!sp) { setEmpty(v === "output" ? "no output yet — run inference" : "loading"); inst.current?.clear(); return; }
    setEmpty(null);
    inst.current?.showAll(sp, v === "truth" ? "truth" : "band");
  }, []);

  const showStatus = useCallback((mode?: string, dt?: number, secs?: number) => {
    const spk = speaker?.id ?? "—";
    if (mode === "running") {
      setStatus(<><span className={s.live}>running</span><span>{arm}</span><span>×{rate}</span>
        {dt != null && secs != null && <><span><b>{secs.toFixed(2)}</b> s audio in <b>{dt.toFixed(2)}</b> s</span><span><b>{(secs / dt).toFixed(2)}×</b> realtime</span></>}</>);
      return;
    }
    if (mode) { setStatus(<span>{mode}</span>); return; }
    const t = timing.current;
    setStatus(<>
      <span>{spk}</span><span>{arm}</span><span>×{rate} · {((FS_LO * rate) / 1000).toFixed(0)} kHz</span>
      {sig.current.output && <span>{outputTag.current}</span>}
      {t && outputLive.current && <span><b>{t.rtf.toFixed(2)}×</b> realtime here · <b>{A100_REALTIME}×</b> on an A100</span>}
      {t && outputLive.current && t.above24 != null && <span>above 24 kHz at ×8: <b>{(100 * t.above24).toFixed(2)} %</b></span>}
      {!sig.current.output && <span>no output yet — run inference</span>}
    </>);
  }, [speaker, arm, rate]);

  // ---- playback ---------------------------------------------------------------------------------
  const stopAudio = useCallback(() => {
    player.current?.stop(); setPlaying(false); cancelAnimationFrame(raf.current); inst.current?.setPlayhead(null);
  }, []);
  const startAudio = useCallback((offset = 0) => {
    const p = (player.current ??= new ABPlayer());
    if (!sig.current.truth) return;
    p.source = source;
    p.onEnded = () => { setPlaying(false); inst.current?.setPlayhead(null); };
    p.start({ input: sig.current.input ?? undefined, output: sig.current.output ?? undefined, truth: sig.current.truth ?? undefined }, offset);
    setPlaying(true);
    const dur = sig.current.truth.data.length / sig.current.truth.fs;
    const tick = () => { if (!p.playing) return; inst.current?.setPlayhead(Math.min(1, p.position() / dur)); raf.current = requestAnimationFrame(tick); };
    tick();
  }, [source]);
  const pickSource = useCallback((k: SourceName) => { setSource(k); player.current?.setSource(k); }, []);

  // ---- inference --------------------------------------------------------------------------------
  const runInference = useCallback(async () => {
    const eng = engine.current, x = sig.current.input;
    if (!eng || !x) return;
    const g = ++gen.current;
    setRunning(true);
    const R = rate === 2 ? 4 : rate, fsOut = FS_LO * R, N = x.data.length * R;
    const out = new Float32Array(N);
    const live: Signal = { data: out, fs: fsOut };
    const sp = specOf(live);
    spec.current.output = sp; sig.current.output = null; outputLive.current = true;
    if (view === "output") { setEmpty(null); inst.current?.blank(sp.nfr); }
    inst.current?.setSweep(0);
    showStatus("running");
    let model: unknown;
    try {
      model = models.current[arm] ??= await eng.load(`/assets/weights/${arm}.bin`, "/assets/weights/manifest.json", arm);
    } catch { if (g === gen.current) { setRunning(false); showStatus(`weights unavailable for ${arm}`); } return; }
    if (g !== gen.current) return;
    const t0 = performance.now(); let lastPaint = 0;
    try {
      await eng.run(model, x.data, { R, tau: effTau, seed, chunk: 2048, onProgress: (j, n, o) => {
        if (g !== gen.current) throw new Error("cancelled");
        out.set(o.subarray(0, j), 0);
        const from = sp.done; specFill(sp, j);
        if (view === "output" && sp.done > from) inst.current?.paintRange(sp, "band", from, sp.done);
        inst.current?.setSweep(j / n);
        const now = performance.now();
        if (now - lastPaint > 120) { lastPaint = now; showStatus("running", (now - t0) / 1000, j / fsOut); }
      } });
    } catch (e) { if (g !== gen.current) return; setRunning(false); showStatus("error: " + ((e as Error).message ?? e)); return; }
    if (g !== gen.current) return;
    const dt = (performance.now() - t0) / 1000, secs = N / fsOut;
    let final: Signal = live;
    if (rate === 2) { final = { data: decimate2(out), fs: 24000 }; spec.current.output = specFill(specOf(final), final.data.length); }
    sig.current.output = final;
    timing.current = { rtf: secs / dt, above24: rate === 8 ? energyAbove(live, 24000) : null };
    outputTag.current = `live · this browser · τ ${effTau.toFixed(2)} · seed ${seed}`;
    setRunning(false); inst.current?.setSweep(null);
    paintView(view); showStatus();
    player.current?.swap({ input: sig.current.input ?? undefined, output: final, truth: sig.current.truth ?? undefined });
  }, [arm, rate, effTau, seed, view, paintView, showStatus]);

  const cancelInference = useCallback(() => { gen.current++; setRunning(false); inst.current?.setSweep(null); showStatus("stopped"); }, [showStatus]);

  // ---- data -------------------------------------------------------------------------------------
  const loadOutput = useCallback(async (sp: Speaker) => {
    const a = sp.files?.arms?.[arm];
    let path: string | null = null;
    if (a && rate === 4) { if (det || effTau === 0) path = a.tau0 || a.draw; else if (effTau === 1 && seed === 0) path = a.draw; }
    if (path) {
      try {
        const o = await getWav(path.startsWith("/") ? path : `/assets/audio/${path}`);
        sig.current.output = o; spec.current.output = specOf(o); outputLive.current = false;
        outputTag.current = `precomputed · A100 · τ ${effTau.toFixed(2)} · seed ${seed}`;
      } catch { sig.current.output = null; spec.current.output = null; }
    } else { sig.current.output = null; spec.current.output = null; }
    paintView(view); showStatus();
    if (autorun && engine.current && wman?.arms?.[arm]) void runInference();
  }, [arm, rate, det, effTau, seed, view, autorun, wman, paintView, showStatus, runInference]);

  const loadSpeaker = useCallback(async (sp: Speaker) => {
    stopAudio();
    setSpeaker(sp); setEmpty(`loading ${sp.id}`);
    sig.current = { truth: null, input: null, naive: null, output: null }; spec.current = { truth: null, input: null, output: null };
    const base = `/assets/audio/${sp.id}/`;
    const [truth, input, naive] = await Promise.all([getWav(base + "truth.wav"), getWav(base + "input.wav"), getWav(base + "naive.wav")]);
    sig.current.truth = truth; sig.current.input = input; sig.current.naive = naive;
    spec.current.truth = specOf(truth); spec.current.input = specOf(naive);   // the input on the 48 kHz grid = its sinc upsample
    await loadOutput(sp);
  }, [stopAudio, loadOutput]);

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
      else { setEmpty("no audio manifest — run demo/tools/make_fixtures.py"); setStatus(<span>no fixtures</span>); }
    })();
    return () => { dead = true; };
  }, []);
  // first speaker once the manifest is in
  useEffect(() => { if (speakers.length && !speaker) void loadSpeaker(speakers[0]); }, [speakers, speaker, loadSpeaker]);
  // arm / rate / tau-commit / seed → reload the output for the current speaker
  const firstRun = useRef(true);
  useEffect(() => {
    if (firstRun.current) { firstRun.current = false; return; }
    if (speaker) void loadOutput(speaker);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [arm, rate, tauCommit, seed]);
  useEffect(() => { paintView(view); }, [view, paintView]);

  // keyboard
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.matches?.("input, textarea")) return;
      if (e.code === "Space") { e.preventDefault(); playing ? stopAudio() : startAudio(0); }
      else if (e.key === "1") pickSource("input"); else if (e.key === "2") pickSource("output"); else if (e.key === "3") pickSource("truth");
    };
    window.addEventListener("keydown", h); return () => window.removeEventListener("keydown", h);
  }, [playing, stopAudio, startAudio, pickSource]);

  const openArch = useCallback((a: ArmName) => { stopAudio(); router.push(`/architecture/${a}`); }, [router, stopAudio]);

  return (
    <div className={s.app}>
      <header className={s.top}>
        <h1>The Missing Band</h1>
        <span className={s.sub}>12 kHz → 48 kHz · one 88k-parameter network · everything above 6 kHz is generated</span>
        <div className={s.legend}>
          <span className={s.c}><i />input band, 0–6 kHz</span>
          <span className={s.w}><i />generated, 6–24 kHz</span>
          <span className={s.t}><i />truth</span>
        </div>
      </header>

      <main className={s.body}>
        <aside className={s.rail}>
          <div className={s.block}>
            <div className={`lbl ${s.blockhead}`}><span>Speaker</span><span className={s.hint}>held out · never trained on</span></div>
            <SpeakerList speakers={speakers} current={speaker?.id ?? null} onPick={(sp) => void loadSpeaker(sp)} />
          </div>
          <div className={s.block}>
            <div className={`lbl ${s.blockhead}`}><span>Output rate</span><span className={s.hint}>same weights, queried differently</span></div>
            <RateSeg rate={rate} onPick={setRate} />
          </div>
          <div className={s.block}>
            <div className={`lbl ${s.blockhead}`}><span>Model</span><span className={s.hint}>one init, identical batches</span></div>
            <ModelList current={arm} results={results?.arms ?? null} onPick={setArm} onOpen={openArch} />
          </div>
          <div className={s.block}>
            <div className={`lbl ${s.blockhead}`}><span>Draw</span><span className={s.hint}>τ = 0 is LISA exactly</span></div>
            <DrawKnobs tau={tau} seed={seed} det={det} onTau={setTau} onTauCommit={() => setTauCommit((n) => n + 1)} onSeed={setSeed} onNewDraw={() => setSeed((n) => n + 1)} />
          </div>
        </aside>

        <section className={s.stage}>
          <div className={s.instrument}>
            <div className={s.toolbar}>
              <ViewSeg view={view} onPick={setView} />
              <span className={s.grow} />
              <label className={`lbl ${s.auto}`}><input type="checkbox" checked={autorun} onChange={(e) => setAutorun(e.target.checked)} />auto-run</label>
              <button type="button" className={`${s.run} ${running ? s.stop : ""}`} disabled={!hasEngine} title={hasEngine ? "" : "engine not loaded"}
                      onClick={() => (running ? cancelInference() : void runInference())}>{running ? "stop" : "run inference"}</button>
            </div>
            <Instrument ref={inst} empty={empty} />
            <div className={s.transport}>
              <button type="button" className={s.play} aria-label={playing ? "pause" : "play"} onClick={() => (playing ? stopAudio() : startAudio(0))}>
                <svg viewBox="0 0 12 12" aria-hidden="true">{playing ? <path d="M2 1.5h3v9H2zM7 1.5h3v9H7z" /> : <path d="M2 1.5v9l8-4.5z" />}</svg>
              </button>
              <div className={s.sources}>
                {(["input", "output", "truth"] as const).map((k, i) => (
                  <button key={k} type="button" className={s.src} data-src={k} aria-pressed={source === k} onClick={() => pickSource(k)}><i />{k} <kbd>{i + 1}</kbd></button>
                ))}
              </div>
              <div className={s.status}>{status}</div>
            </div>
            <p className={s.prose}>Press play, then switch between <b>input</b>, <b>output</b> and <b>truth</b> while it runs — the three are phase-locked, so the only thing that changes is the band above the dashed line.</p>
          </div>
        </section>

        <aside className={`${s.rail} ${s.right}`}>
          <Numbers arm={arm} results={results} onOpen={() => openArch(arm)} />
        </aside>
      </main>
    </div>
  );
}
