"use client";
// One compact table: the four released models against our det and our sampler on the paper split.
// Every cell is read from results.sota (make_results_sota.py); the two "ours" rows take params and RTF
// from results.arms[a].latency (fast/bench_latency.py) and name its CPU, and the footnote says so.
import s from "@/app/page.module.css";
import { ARM_META, DEFAULT_ARM, REF_ARM, RELEASED_META, RELEASED_ORDER, type ArmName, type ReleasedId } from "@/lib/arms";
import { useT } from "@/lib/i18n";
import { fmtParams } from "@/lib/tiles";
import type { Results, SotaMetrics, SotaResults } from "@/lib/types";

const fmt = (v: number | null | undefined, d = 2) => (v == null || !isFinite(v) ? "—" : v.toFixed(d));

interface CmpRow {
  id: string;
  name: string;
  ours: boolean;
  det: boolean;
  params: number | null | undefined;
  wide: SotaMetrics | null | undefined;
  core: SotaMetrics | null | undefined;
  rtf: number | null | undefined;
  device: string | null | undefined;
}

export function CompareStrip({ sota, results, compact }: { sota: SotaResults | null; results: Results | null; compact?: boolean }) {
  const { t } = useT();
  const ourCpu = results?.latency_env?.cpu ?? null;
  const rel = (id: ReleasedId): CmpRow => {
    const m = sota?.models?.[id];
    return { id, name: m?.name ?? RELEASED_META[id].name, ours: false, det: m?.det ?? RELEASED_META[id].det, params: m?.params,
      wide: m?.sets?.wide, core: m?.sets?.core, rtf: m?.rtf_m4, device: m?.rtf_device };
  };
  const our = (a: ArmName): CmpRow => {
    const e = sota?.ours?.[a], lat = results?.arms?.[a]?.latency;
    return { id: a, name: a, ours: true, det: ARM_META[a].det, params: lat?.params, wide: e?.sets?.wide, core: e?.sets?.core,
      rtf: lat?.rtf_one_pass, device: ourCpu };
  };
  const rows: CmpRow[] = [...RELEASED_ORDER.map(rel), our(REF_ARM), our(DEFAULT_ARM)];
  const wide = sota?.sets?.wide, core = sota?.sets?.core;
  const kw = wide ? (Array.isArray(wide.speakers) ? wide.speakers.length : wide.speakers) : "—";
  const caption = sota
    ? t("cmp.sets", { nw: wide?.n_utts ?? "—", kw, nc: core?.n_utts ?? "—", M: core?.M ?? "—" })
    : t("cmp.pending");
  // the ceiling line: shown only when the ceiling was measured, and "every model is under it" only when
  // every row's own SNR − ceiling on wide says so (the column beside it lets a reader check)
  const ceil = sota?.ceiling?.snr_12k;
  const under = rows.every((r) => r.wide?.vs_ceiling != null && isFinite(r.wide.vs_ceiling) && r.wide.vs_ceiling < 0);
  const machine = RELEASED_ORDER.map((id) => sota?.models?.[id]?.machine).find((m) => m) ?? "—";
  const table = (
    <>
      <div className={`lbl ${s.cmpCap}`}>{caption}</div>
      <table className={s.cmp}>
        <thead>
          <tr>
            <th>{t("cmp.model")}</th>
            <th>{t("cmp.params")}</th>
            <th>{t("cmp.deficit")}<small>{t("cmp.deficit.sub")}</small></th>
            <th>{t("cmp.vsceil")}<small>{t("cmp.vsceil.sub")}</small></th>
            <th>{t("cmp.visqol")}</th>
            <th>{t("cmp.crps")}</th>
            <th>{t("cmp.spread")}</th>
            <th>{t("cmp.rtf")}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className={r.ours ? s.cmpOurs : undefined}>
              <td>{r.name}{r.ours && <small>{t("cmp.ours")}</small>}</td>
              <td>{fmtParams(r.params)}</td>
              <td>{fmt(r.wide?.deficit, 1)} / {fmt(r.wide?.deficit_bb, 1)} dB</td>
              <td>{fmt(r.wide?.vs_ceiling, 1)}</td>
              <td>{fmt(r.core?.visqol_audio)}</td>
              <td>{fmt(r.core?.crps_fair, 3)}</td>
              <td title={r.det ? t("rel.det") : undefined}>{r.det ? "—" : fmt(r.core?.spread_hb, 3)}</td>
              <td>{fmt(r.rtf, 3)}{r.device && <small>{r.device}</small>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {ceil != null && isFinite(ceil) && (
        <div className={s.note}>{t("cmp.ceil.a")} <b>{fmt(ceil)} dB</b> {t("cmp.ceil.b")}{under ? t("cmp.ceil.under") : ""}</div>
      )}
      <div className={s.note}>{t("cmp.rtf.note", { cpu: machine, ourcpu: ourCpu ?? "—" })}</div>
    </>
  );
  // in the narrow left rail the table folds away behind its heading, so the model list stays readable
  if (compact) {
    return (
      <details className={`${s.cmpWrap} ${s.cmpCompact}`}>
        <summary className={`lbl ${s.cmpCap}`}>{t("cmp.h")}</summary>
        {table}
      </details>
    );
  }
  return <div className={s.cmpWrap}>{table}</div>;
}
