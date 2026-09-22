// The headline numbers as stat tiles: the same four figures on the landing page, the first three at
// the top of the numbers rail. Every value is read from results.json; the deltas are against REF_ARM.
import { REF_ARM, type ArmName } from "./arms";
import type { Key } from "./i18n";
import type { Results } from "./types";

export interface TileSpec { label: string; value: string; delta?: string; good?: boolean | null }
type T = (k: Key, vars?: Record<string, string | number>) => string;

// a real minus sign, as the page prints numbers
const fmt = (v: number | null | undefined, d = 2) => (v == null || !isFinite(v) ? "—" : v.toFixed(d).replace(/^-/, "−"));
const sgn = (d: number, p: number) => (d > 0 ? "+" : "") + d.toFixed(p).replace(/^-/, "−");

export function heroTiles(r: Results | null, arm: ArmName, t: T, withCost = false): TileSpec[] {
  const a = r?.arms?.[arm], det = r?.arms?.[REF_ARM];
  const isRef = arm === REF_ARM;
  const roKey = a?.best_readout ?? "draw_pt";
  const ro = a?.readouts?.[roKey], dro = det?.readouts?.draw_pt;
  const roName = t(`t.ro.${roKey}` as Key);
  const tiles: TileSpec[] = [];
  const d1 = a?.deficit_draw != null && det?.deficit_tau0 != null && !isRef ? a.deficit_draw - det.deficit_tau0 : null;
  tiles.push({ label: t("t.deficit"), value: fmt(a?.deficit_draw) + " dB", delta: d1 != null ? sgn(d1, 2) + " dB " + t("t.vs", { ref: REF_ARM, v: fmt(det?.deficit_tau0) }) : undefined, good: d1 != null ? d1 > 0 : null });
  const d2 = a?.crps != null && det?.crps != null && !isRef ? 100 * (a.crps / det.crps - 1) : null;
  tiles.push({ label: t("t.crps"), value: fmt(a?.crps, 3), delta: d2 != null ? sgn(d2, 0) + " % " + t("t.vs", { ref: REF_ARM, v: fmt(det?.crps, 3) }) : undefined, good: d2 != null ? d2 < 0 : null });
  const d3 = ro?.lsd != null && dro?.lsd != null && !isRef ? ro.lsd - dro.lsd : null;
  tiles.push({ label: t("t.lsd", { ro: roName }), value: fmt(ro?.lsd, 3), delta: d3 != null ? sgn(d3, 3) + " " + t("t.vs", { ref: REF_ARM, v: fmt(dro?.lsd, 3) }) : undefined, good: d3 != null ? d3 < 0 : null });
  if (withCost) {
    const lat = a?.latency, env = r?.latency_env;
    tiles.push({ label: t("t.cost"), value: lat?.one_pass_ms_per_s != null ? fmt(lat.one_pass_ms_per_s, 1) + " ms" : "—",
      delta: lat?.rtf_one_pass ? t("t.rt", { x: (1 / lat.rtf_one_pass).toFixed(0), cpu: env?.cpu ?? "" }) : undefined, good: null });
  }
  return tiles;
}
