"use client";
import s from "@/app/page.module.css";
import { ARM_META, type ArmName } from "@/lib/arms";
import type { GatedDeficit, Results } from "@/lib/types";

const fmt = (v: number | null | undefined, d = 2) => (v == null || !isFinite(v) ? "—" : v.toFixed(d));

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

export function Numbers({ arm, results, onOpen }: { arm: ArmName; results: Results | null; onOpen: () => void }) {
  const meta = ARM_META[arm];
  const a = results?.arms?.[arm];
  const det = results?.arms?.det;
  const nv = results?.naive;
  // every judge is quoted at the readout that arm actually wins on, and the readout is named
  const roKey = a?.best_readout ?? "draw_pt";
  const ro = a?.readouts?.[roKey];
  const roName = RO_LABEL[roKey] ?? roKey;
  const share = a?.visqol_audio_best_share ?? null;
  return (
    <>
      <div className={s.card}>
        <h2><span>{arm}</span><small>{meta.cls} · λ {meta.lam} · {meta.kind}</small></h2>
        {!a ? (
          <>
            <div className={s.kv}><Row k="evaluation" v="pending" /></div>
            <div className={s.note}>The evaluation cells have not written results.json yet. The audio and the live inference still work.</div>
          </>
        ) : (
          <>
            <div className={s.kv}>
              <Head t="the disease and the cure" />
              {(() => { const [d, c] = delta(a.deficit_draw, det?.deficit_tau0, false); return <Row k="deficit · 1 draw" v={fmt(a.deficit_draw) + " dB"} d={d && d + " det"} cls={c} />; })()}
              <Row k="deficit · τ = 0" v={fmt(a.deficit_tau0) + " dB"} />
              {(() => { const [d, c] = delta(a.crps, det?.crps, true); return <Row k="CRPS" v={fmt(a.crps, 3)} d={d && d + " det"} cls={c} />; })()}
              {a.coherent != null && <Row k="coherent fraction" v={fmt(100 * a.coherent, 1) + " %"} />}
              {a.kappa != null && <Row k="κ · hallucinated?" v={fmt(a.kappa, 3)} />}
              {a.pit_end != null && <Row k="PIT end bins" v={fmt(a.pit_end, 3)} d="ideal .118" />}
              <Head t="the judges · with passthrough" />
              {(() => { const [d, c] = delta(ro?.lsd, det?.readouts?.draw_pt?.lsd, true); return <Row k={`LSD · ${roName}`} v={fmt(ro?.lsd, 3)} d={d && d + " det"} cls={c} />; })()}
              {a.readouts?.draw_pt && roKey !== "draw_pt" && <Row k="LSD · 1 draw" v={fmt(a.readouts.draw_pt.lsd, 3)} />}
              {(() => { const [d, c] = delta(ro?.visqol_audio, det?.readouts?.draw_pt?.visqol_audio, false); return <Row k={`ViSQOL audio · ${roName}`} v={fmt(ro?.visqol_audio)} d={d && d + " det"} cls={c} />; })()}
              {share != null && <Row k="share of the band" v={fmt(100 * share, 1) + " %"} d="floor→ceiling" />}
              {ro?.visqol_speech != null && <Row k="ViSQOL speech" v={fmt(ro.visqol_speech)} />}
              {ro?.pesq != null && <Row k="PESQ wb" v={fmt(ro.pesq)} />}
              <Head t="SNR · maximised by doing nothing" />
              {(() => { const [d, c] = delta(ro?.snr, nv?.snr, false); return <Row k={roName} v={fmt(ro?.snr) + " dB"} d={d && d + " naive"} cls={c} />; })()}
              {nv && <Row k="naive sinc upsample" v={fmt(nv.snr) + " dB"} />}
              {results?.floor && <Row k="floor · empty high band" v={fmt(results.floor.snr) + " dB"} />}
              {results?.ceiling && <Row k="ceiling · true high band" v={fmt(results.ceiling.snr) + " dB"} />}
            </div>
            <div className={s.note}>{results?.n_utts ?? 12} held-out utterances · {results?.M ?? 16} draws · {results?.tag ?? ""}</div>
          </>
        )}
      </div>
      <div className={s.card}>
        <h2><span>High band by frame loudness</span><small>energy ratio, dB</small></h2>
        <Gate g={a?.gated ?? null} />
        <div className={s.note}>A correct sampler shows the same ratio in every row. A ratio that rises as frames get quieter is energy where the truth has none — that is what a listener hears as hiss.</div>
      </div>
      <button type="button" className={s.open3d} onClick={onOpen}><span>open the architecture</span><span>↗</span></button>
      <div className={s.note}>Numbers are from the evaluation the training notebook wrote, on 12 held-out utterances, 16 draws. Nothing on this page is typed in.</div>
    </>
  );
}

function Gate({ g }: { g: GatedDeficit | null }) {
  const lo = -24, hi = 6, z = (0 - lo) / (hi - lo);
  const rows: [keyof GatedDeficit, string][] = [["loud", "top 25 %"], ["mid", "25–75 %"], ["quiet", "bottom 25 %"]];
  return (
    <div className={s.gate}>
      {rows.map(([k]) => {
        const v = g?.[k] ?? null;
        const p = v == null ? z : Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
        return (
          <div key={k} style={{ display: "contents" }}>
            <span>{k}</span>
            <div className={s.bar}><i style={{ left: `${Math.min(p, z) * 100}%`, width: `${Math.abs(p - z) * 100}%` }} /><span className={s.z} style={{ left: `${z * 100}%` }} /></div>
            <span className="num">{v == null ? "—" : (v > 0 ? "+" : "") + v.toFixed(2)}</span>
          </div>
        );
      })}
    </div>
  );
}
