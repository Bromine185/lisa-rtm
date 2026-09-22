"use client";
import s from "@/app/page.module.css";
import { ARM_META, REF_ARM, type ArmName } from "@/lib/arms";
import { useT, type Key } from "@/lib/i18n";
import { heroTiles } from "@/lib/tiles";
import { Tiles } from "./Tiles";
import type { GatedDeficit, Results } from "@/lib/types";

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

function Row({ k, v, d, cls }: { k: string; v: string; d?: string; cls?: string }) {
  return (
    <>
      <span className={s.kk}>{k}</span>
      <span className={s.vv}>{v}{d ? <span className={`${s.d} ${cls ?? ""}`}>{d}</span> : null}</span>
    </>
  );
}
const Head = ({ t }: { t: string }) => <span className={s.hh}>{t}</span>;

const RO_LABEL: Record<string, string> = { draw_pt: "1 draw", mean16_pt: "mean16", logmean16_pt: "logmean16" };
const KIND_KEY: Record<string, Key> = { "plain L1 (the paper)": "kind.plain", "L1 + λ·STFT": "kind.l1stft", "energy score, log-mag": "kind.es.logmag", "energy score, +ERB": "kind.es.erb" };

export function Numbers({ arm, results, onOpen }: { arm: ArmName; results: Results | null; onOpen: () => void }) {
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
        <h2><span>{t("gate.title")}</span><small>{t("gate.sub")}</small></h2>
        <Gate g={a?.gated ?? null} />
        <div className={s.note}>{t("gate.note")}</div>
      </div>
      <button type="button" className={s.open3d} onClick={onOpen}><span>{t("arch.open")}</span><span>↗</span></button>
      <div className={s.note}>{t("n.foot", { utts, draws, epochs: ep.replace(" · ", ", ") })}</div>
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
