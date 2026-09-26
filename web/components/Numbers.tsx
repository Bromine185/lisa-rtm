"use client";
import s from "@/app/page.module.css";
import { ARM_META, DEFAULT_ARM, REF_ARM, RELEASED_META, type ArmName, type ReleasedId } from "@/lib/arms";
import { useT, type Key } from "@/lib/i18n";
import { fmtParams, heroTiles } from "@/lib/tiles";
import { Tiles } from "./Tiles";
import type { GatedDeficit, ReleasedManifest, Results, SotaEntry, SotaExtra, SotaMetrics, SotaResults } from "@/lib/types";

const fmt = (v: number | null | undefined, d = 2) => (v == null || !isFinite(v) ? "—" : v.toFixed(d));
const fmtInt = (v: number | null | undefined) => (v == null || !isFinite(v) ? "—" : Math.round(v).toLocaleString("en-US"));
// milliseconds: two figures under 10 ms, one under 100, whole above
const ms = (v: number | null | undefined) => (v == null || !isFinite(v) ? "—" : (v < 10 ? v.toFixed(2) : v < 100 ? v.toFixed(1) : v.toFixed(0)) + " ms");

function delta(v: number | null | undefined, ref: number | null | undefined, betterLow: boolean): [string, string] {
  if (v == null || ref == null || !isFinite(v) || !isFinite(ref)) return ["", ""];
  const d = v - ref;
  if (Math.abs(d) < 1e-9) return ["", ""];
  const good = betterLow ? d < 0 : d > 0;
  return [(d > 0 ? "+" : "") + fmt(d), good ? s.good : s.bad];
}

// One delta against a named reference, for the paper-split rows where a row can carry two.
interface Delta { d: string; cls: string }
function deltaVs(v: number | null | undefined, ref: number | null | undefined, better: (v: number, ref: number) => boolean, refName: string,
                 t: (k: Key, vars?: Record<string, string | number>) => string, dp = 2): Delta | null {
  if (v == null || ref == null || !isFinite(v) || !isFinite(ref)) return null;
  const d = v - ref;
  if (Math.abs(d) < 1e-9) return null;
  return { d: (d > 0 ? "+" : "") + fmt(d, dp) + " " + t("ps.vs", { ref: refName }), cls: better(v, ref) ? s.good : s.bad };
}
const HIGHER = (v: number, r: number) => v > r;
const LOWER = (v: number, r: number) => v < r;
const NEAR1 = (v: number, r: number) => Math.abs(1 - v) < Math.abs(1 - r);   // spread: 1 = calibrated
const NEAR0 = (v: number, r: number) => Math.abs(v) < Math.abs(r);           // slope: 0 = tracks the utterance

function Row({ k, v, d, cls, ds }: { k: string; v: string; d?: string; cls?: string; ds?: (Delta | null)[] }) {
  return (
    <>
      <span className={s.kk}>{k}</span>
      <span className={s.vv}>{v}{d ? <span className={`${s.d} ${cls ?? ""}`}>{d}</span> : null}
        {ds?.map((x, i) => x && <span key={i} className={`${s.d} ${x.cls}`}>{x.d}</span>)}</span>
    </>
  );
}
const Head = ({ t }: { t: string }) => <span className={s.hh}>{t}</span>;

const RO_LABEL: Record<string, string> = { draw_pt: "1 draw", mean16_pt: "mean16", logmean16_pt: "logmean16" };
const KIND_KEY: Record<string, Key> = { "plain L1 (the paper)": "kind.plain", "L1 + λ·STFT": "kind.l1stft", "energy score, log-mag": "kind.es.logmag", "energy score, +ERB": "kind.es.erb" };

export function Numbers({ arm, released, results, manifest, onOpen }: {
  arm: ArmName; released: ReleasedId | null; results: Results | null; manifest: ReleasedManifest | null; onOpen: () => void;
}) {
  if (released) return <Released id={released} results={results} manifest={manifest} onOpen={onOpen} />;
  return <ArmNumbers arm={arm} results={results} onOpen={onOpen} />;
}

function ArmNumbers({ arm, results, onOpen }: { arm: ArmName; results: Results | null; onOpen: () => void }) {
  const { t } = useT();
  const meta = ARM_META[arm];
  const a = results?.arms?.[arm];
  // every delta is against det (L1 + λ·STFT, τ = 0), not det_paper: det shares the samplers' spectral term
  const det = results?.arms?.[REF_ARM];
  const vs = " vs " + REF_ARM;
  const nv = results?.naive;
  const lat = a?.latency, env = results?.latency_env;
  const speakers = results?.eval_speakers?.length ? results.eval_speakers : null;
  const epochs = /^OV(\d+)/.exec(results?.tag ?? "")?.[1] ?? null;
  // every judge is quoted at the readout that arm actually wins on, and the readout is named
  const roKey = a?.best_readout ?? "draw_pt";
  const ro = a?.readouts?.[roKey];
  const roName = RO_LABEL[roKey] ?? roKey;
  const share = a?.visqol_audio_best_share ?? null;
  const kind = (k: string | undefined) => (k && KIND_KEY[k] ? t(KIND_KEY[k]) : k ?? "");
  const utts = t("n.utts", { n: results?.n_utts ?? 12 }) + (speakers ? " " + (speakers.length === 1 ? t("n.of.one", { spk: speakers[0] }) : t("n.of.many", { k: speakers.length, spk: speakers.join(", ") })) : "");
  const draws = t("n.draws", { M: results?.M ?? 16 });
  const ep = epochs ? " · " + t("n.epochs", { e: epochs }) : "";
  return (
    <>
      <div className={s.card}>
        <h2><span>{arm}</span><small>{meta.cls} · λ {meta.lam} · {kind(meta.kind)}</small></h2>
        {!a ? (
          <>
            <div className={s.kv}><Row k={t("n.pending")} v={t("n.pending.v")} /></div>
            <div className={s.note}>{t("n.pending.note")}</div>
          </>
        ) : (
          <>
            <Tiles tiles={heroTiles(results, arm, t)} />
            <div className={s.kv}>
              <Head t={t("h.disease")} />
              {(() => { const [d, c] = delta(a.deficit_draw, det?.deficit_tau0, false); return <Row k={t("r.deficit.draw")} v={fmt(a.deficit_draw) + " dB"} d={d && d + vs} cls={c} />; })()}
              <Row k={t("r.deficit.tau0")} v={fmt(a.deficit_tau0) + " dB"} />
              {(() => { const [d, c] = delta(a.crps, det?.crps, true); return <Row k="CRPS" v={fmt(a.crps, 3)} d={d && d + vs} cls={c} />; })()}
              {a.coherent != null && <Row k={t("r.coherent")} v={fmt(100 * a.coherent, 1) + " %"} />}
              {a.kappa != null && <Row k={t("r.kappa")} v={fmt(a.kappa, 3)} />}
              {a.pit_end != null && <Row k={t("r.pit")} v={fmt(a.pit_end, 3)} d={t("r.pit.ideal")} />}
              <Head t={t("h.judges")} />
              {(() => { const [d, c] = delta(ro?.lsd, det?.readouts?.draw_pt?.lsd, true); return <Row k={`LSD · ${roName}`} v={fmt(ro?.lsd, 3)} d={d && d + vs} cls={c} />; })()}
              {a.readouts?.draw_pt && roKey !== "draw_pt" && <Row k={t("r.lsd.draw")} v={fmt(a.readouts.draw_pt.lsd, 3)} />}
              {(() => { const [d, c] = delta(ro?.visqol_audio, det?.readouts?.draw_pt?.visqol_audio, false); return <Row k={`${t("r.visqol.audio")} · ${roName}`} v={fmt(ro?.visqol_audio)} d={d && d + vs} cls={c} />; })()}
              {share != null && <Row k={t("r.share")} v={fmt(100 * share, 1) + " %"} d={t("r.share.d")} />}
              {ro?.visqol_speech != null && <Row k={t("r.visqol.speech")} v={fmt(ro.visqol_speech)} />}
              {ro?.pesq != null && <Row k={t("r.pesq")} v={fmt(ro.pesq)} />}
              {lat && (
                <>
                  <Head t={t("h.cost")} />
                  <Row k={t("r.params")} v={fmtInt(lat.params)} />
                  <Row k={t("r.onepass")} v={ms(lat.one_pass_ms_per_s)} d={lat.rtf_one_pass ? `${(1 / lat.rtf_one_pass).toFixed(0)}× ${t("realtime")}` : undefined} />
                  <Row k={t("r.frame")} v={ms(lat.frame_20ms_ms)} />
                  <Row k={t("r.shipped1")} v={ms(lat.shipped_one_ms)} d={env?.utt_seconds != null ? t("r.utt", { s: env.utt_seconds.toFixed(2) }) : undefined} />
                  {!a.det && <Row k={t("r.shipped16")} v={ms(lat.shipped_logmean16_ms)} d={lat.rtf_logmean16 != null ? `RTF ${fmt(lat.rtf_logmean16)}` : undefined} />}
                  {env?.lookahead_ms != null && <Row k={t("r.lookahead")} v={ms(env.lookahead_ms)} />}
                </>
              )}
              <Head t={t("h.snr")} />
              {(() => { const [d, c] = delta(ro?.snr, nv?.snr, false); return <Row k={roName} v={fmt(ro?.snr) + " dB"} d={d && d + " " + t("r.naive.d")} cls={c} />; })()}
              {nv && <Row k={t("r.naive")} v={fmt(nv.snr) + " dB"} />}
              {results?.floor && <Row k={t("r.floor")} v={fmt(results.floor.snr) + " dB"} />}
              {results?.ceiling && <Row k={t("r.ceiling")} v={fmt(results.ceiling.snr) + " dB"} />}
            </div>
            <div className={s.note}>
              {utts} · {draws} · {results?.tag ?? ""}{ep}
              {" · "}{t("n.delta", { ref: REF_ARM, kind: kind(det?.kind) || "L1 + λ·STFT" })}
              {env && <>{" · "}{t("n.cost", { cpu: env.cpu ?? t("n.bench.cpu"), threads: env.threads ?? "—", precision: env.precision ?? t("n.fp32"), torch: env.torch ?? "—", ms: ms(env.passthrough_ms) })}</>}
            </div>
          </>
        )}
      </div>
      <div className={s.card}>
        <h2><span>{t("ps.title")}</span><small>{t("ps.sub")}</small></h2>
        <div className={s.kv}>
          <PaperSplit entry={results?.sota?.ours?.[arm]} sets={results?.sota?.sets} det={meta.det}
                      refs={arm === REF_ARM ? [] : [{ name: REF_ARM, entry: results?.sota?.ours?.[REF_ARM] }]} />
        </div>
      </div>
      <div className={s.card}>
        <h2><span>{t("gate.title")}</span><small>{t("gate.sub")}</small></h2>
        <Gate g={a?.gated ?? null} />
        <div className={s.note}>{t("gate.note")}</div>
      </div>
      <button type="button" className={s.open3d} onClick={onOpen}><span>{t("arch.open")}</span><span>↗</span></button>
      <div className={s.note}>{t("n.foot", { utts, draws, epochs: ep.replace(" · ", ", ") })}</div>
    </>
  );
}

// A reference for the paper-split deltas: our sampler and/or our deterministic model, on the same set.
interface Ref { name: string; entry: SotaEntry | null | undefined }
type MetricKey = keyof SotaMetrics;

/** The paper-split block of a card: wide, core and the slope, every number from results.sota and every
 *  delta against the given references on the same set and key. Missing block → a dash, never a throw.
 *  `det`: a deterministic model has no spread, so that row is left out (as the strip leaves it blank). */
function PaperSplit({ entry, sets, refs, det }: { entry: SotaEntry | null | undefined; sets: SotaResults["sets"] | undefined; refs: Ref[]; det: boolean }) {
  const { t } = useT();
  if (!entry) {
    return (
      <>
        <Row k={t("ps.title")} v={t("n.pending.v")} />
        <span className={s.note} style={{ gridColumn: "1 / -1" }}>{t("ps.pending")}</span>
      </>
    );
  }
  const wide = entry.sets?.wide, core = entry.sets?.core, sl = entry.slope;
  const ds = (set: "wide" | "core", key: MetricKey, better: (v: number, r: number) => boolean, dp = 2) =>
    refs.map((r) => deltaVs(entry.sets?.[set]?.[key], r.entry?.sets?.[set]?.[key], better, r.name, t, dp));
  // a gated ratio has no single good direction (it should be the same in every row), so its delta is uncoloured
  const flat = (set: "wide", key: MetricKey) => ds(set, key, HIGHER).map((x) => x && { ...x, cls: "" });
  const sw = sets?.wide, sc = sets?.core;
  const kw = sw ? (Array.isArray(sw.speakers) ? sw.speakers.length : sw.speakers) : "—";
  const db = (v: number | null | undefined) => fmt(v) + " dB";
  // the draws this entry was scored with on core (a deterministic model has one), not the set's most
  const M = core?.M;
  const coreHead = !sc ? t("ps.set.none") : M === 1 ? t("ps.set.core.one", { n: sc.n_utts }) : t("ps.set.core", { n: sc.n_utts, M: M ?? "—" });
  // a slope delta only against a fit to the same number of utterances
  const slopeDs = refs.map((r) => {
    const o = r.entry?.slope;
    if (!sl || !o) return null;
    if (o.n !== sl.n) return { d: t("ps.slope.nodelta") + " (" + r.name + ")", cls: "" };
    return deltaVs(sl.slope, o.slope, NEAR0, r.name, t, 3);
  });
  return (
    <>
      <Head t={sw ? t("ps.set.wide", { n: sw.n_utts, k: kw }) : t("ps.set.none")} />
      <Row k={t("ps.deficit.band")} v={db(wide?.deficit)} ds={ds("wide", "deficit", HIGHER)} />
      <Row k={t("ps.deficit.bb")} v={db(wide?.deficit_bb)} ds={ds("wide", "deficit_bb", HIGHER)} />
      <Row k={t("gate.loud")} v={db(wide?.loud)} ds={flat("wide", "loud")} />
      <Row k={t("gate.mid")} v={db(wide?.mid)} ds={flat("wide", "mid")} />
      <Row k={t("gate.quiet")} v={db(wide?.quiet)} ds={flat("wide", "quiet")} />
      <Row k={t("ps.lsd")} v={fmt(wide?.lsd, 3)} ds={ds("wide", "lsd", LOWER, 3)} />
      <Row k={t("ps.vsceil")} v={db(wide?.vs_ceiling)} ds={ds("wide", "vs_ceiling", HIGHER)} />
      <Head t={coreHead} />
      <Row k={t("ps.visqol")} v={fmt(core?.visqol_audio)} ds={ds("core", "visqol_audio", HIGHER)} />
      <Row k={t("ps.crpsfair")} v={fmt(core?.crps_fair, 3)} ds={ds("core", "crps_fair", LOWER, 3)} />
      {!det && core?.spread_hb != null && <Row k={t("ps.spread")} v={fmt(core.spread_hb, 3)} ds={ds("core", "spread_hb", NEAR1, 3)} />}
      <Row k={t("ps.gap")} v={fmt(core?.gap, 3)} ds={ds("core", "gap", LOWER, 3)} />
      <Head t={t("ps.slope", { set: sl?.set ?? "—" })} />
      <Row k={sl ? t("ps.slope.r", { r: fmt(sl.r), n: sl.n }) : "—"} v={fmt(sl?.slope, 3)} ds={slopeDs} />
      <span className={s.note} style={{ gridColumn: "1 / -1" }}>{t("ps.slope.note")}</span>
    </>
  );
}

/** The whole card for a released model: its facts from results.sota (the manifest as the fallback for
 *  the name and family), then the paper split with deltas against our sampler and our det. */
function Released({ id, results, manifest, onOpen }: { id: ReleasedId; results: Results | null; manifest: ReleasedManifest | null; onOpen: () => void }) {
  const { t } = useT();
  const sota = results?.sota;
  const m = sota?.models?.[id], mm = manifest?.models?.[id];
  const det = m?.det ?? mm?.det ?? RELEASED_META[id].det;
  const params = m?.params ?? mm?.params;
  const refs: Ref[] = [{ name: DEFAULT_ARM, entry: sota?.ours?.[DEFAULT_ARM] }, { name: REF_ARM, entry: sota?.ours?.[REF_ARM] }];
  return (
    <>
      <div className={s.card}>
        <h2><span>{m?.name ?? mm?.name ?? RELEASED_META[id].name}</span><small>{t(det ? "rel.det" : "rel.sampler")} · {fmtParams(params)}</small></h2>
        <div className={s.kv}>
          <Head t={t("h.model")} />
          <Row k={t("r.family")} v={m?.family ?? mm?.family ?? "—"} />
          <Row k={t("r.params")} v={fmtInt(params)} />
          {(m?.arch?.params_note ?? mm?.arch?.params_note) && <span className={s.note} style={{ gridColumn: "1 / -1" }}>{t("r.params.note")}: {m?.arch?.params_note ?? mm?.arch?.params_note}</span>}
          <Row k={t("r.rtf", { cpu: m?.machine ?? mm?.machine ?? "—" })} v={fmt(m?.rtf_m4 ?? mm?.rtf_m4, 3)} d={m?.rtf_device ?? mm?.rtf_device ?? undefined} />
          <Row k={t("r.steps")} v={m?.arch?.steps ?? mm?.steps ?? "—"} />
          <PaperSplit entry={m} sets={sota?.sets} refs={refs} det={det} />
          {m?.extra && Object.entries(m.extra).map(([k, x]) => <Extra key={k} id={k} x={x} />)}
        </div>
        <div className={s.note}>
          {(m?.arch?.det_note ?? mm?.arch?.det_note) && <>{m?.arch?.det_note ?? mm?.arch?.det_note}<br /></>}
          {(m?.arch?.seen_note ?? mm?.seen_note) && <>{m?.arch?.seen_note ?? mm?.seen_note}<br /></>}
          {t("ps.delta.note", { a: DEFAULT_ARM, b: REF_ARM })}
        </div>
      </div>
      <button type="button" className={s.open3d} onClick={onOpen}><span>{t("arch.open")}</span><span>↗</span></button>
      <div className={s.note}>{t("n.foot.rel")}</div>
    </>
  );
}

/** A variant scored beside a released model: what it changes (prose from make_results_sota.py), then the
 *  same rows with each set named. Rows whose value is absent are left out; nothing here is typed in. */
function Extra({ id, x }: { id: string; x: SotaExtra }) {
  const { t } = useT();
  const c = x.sets?.core, w = x.sets?.wide;
  return (
    <>
      <Head t={`${id} · ${t("ps.extra")}`} />
      <span className={s.note} style={{ gridColumn: "1 / -1" }}>{x.note}</span>
      {c?.spread_hb != null && <Row k={`${t("ps.spread")} · core`} v={fmt(c.spread_hb, 3)} />}
      {w?.deficit != null && <Row k={`${t("ps.deficit.band")} · wide`} v={fmt(w.deficit) + " dB"} />}
      {w?.lsd != null && <Row k={`${t("ps.lsd")} · wide`} v={fmt(w.lsd, 3)} />}
      {c?.lsd != null && <Row k={`${t("ps.lsd")} · core`} v={fmt(c.lsd, 3)} />}
    </>
  );
}

function Gate({ g }: { g: GatedDeficit | null }) {
  const { t } = useT();
  const lo = -24, hi = 6, z = (0 - lo) / (hi - lo);
  const rows: (keyof GatedDeficit)[] = ["loud", "mid", "quiet"];
  return (
    <div className={s.gate}>
      {rows.map((k) => {
        const v = g?.[k] ?? null;
        const p = v == null ? z : Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
        return (
          <div key={k} style={{ display: "contents" }}>
            <span>{t(`gate.${k}`)}</span>
            <div className={s.bar}><i style={{ left: `${Math.min(p, z) * 100}%`, width: `${Math.abs(p - z) * 100}%` }} /><span className={s.z} style={{ left: `${z * 100}%` }} /></div>
            <span className="num">{v == null ? "—" : (v > 0 ? "+" : "") + v.toFixed(2)}</span>
          </div>
        );
      })}
    </div>
  );
}
