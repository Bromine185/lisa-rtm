"use client";
import s from "@/app/page.module.css";
import { ARM_META, ARM_ORDER, FS_LO, RATES, type ArmName, type Rate } from "@/lib/arms";
import { useT, type Key } from "@/lib/i18n";
import type { ArmResult, Speaker } from "@/lib/types";

const fmt = (v: number | null | undefined, d = 2) => (v == null || !isFinite(v) ? "—" : v.toFixed(d));

/** A live clip is named for the person, not for a corpus id; the id stays the React key. */
export function speakerName(sp: Speaker, you: string): string {
  if (!sp.live) return sp.id;
  const n = sp.id.split("-")[1];
  return n ? `${you} ${n}` : you;
}

export function SpeakerList({ speakers, current, onPick }: { speakers: Speaker[]; current: string | null; onPick: (sp: Speaker) => void }) {
  const { t } = useT();
  return (
    <div className={s.list} role="listbox" aria-label={t("speaker")}>
      {speakers.map((sp) => (
        <button key={sp.id} type="button" className={`${s.row} ${sp.live ? s.rowLive : ""}`} aria-pressed={sp.id === current} onClick={() => onPick(sp)}>
          <div>
            <div className={s.name}>{speakerName(sp, t("v.name"))}</div>
            <div className={s.meta}>{sp.live ? t(sp.live === "mic" ? "v.meta" : "v.meta.file") : [sp.gender, sp.accent, sp.region].filter(Boolean).join(" · ")}</div>
          </div>
          <div className={s.stat}><b>{(sp.seconds ?? 0).toFixed(1)}</b> s</div>
        </button>
      ))}
    </div>
  );
}

export function RateSeg({ rate, onPick }: { rate: Rate; onPick: (r: Rate) => void }) {
  const { t } = useT();
  return (
    <div className={s.seg} role="radiogroup" aria-label={t("rate")}>
      {RATES.map((r) => (
        <button key={r} type="button" aria-pressed={rate === r} onClick={() => onPick(r)}>×{r} · {((FS_LO * r) / 1000).toFixed(0)} kHz</button>
      ))}
    </div>
  );
}

export function ReadoutSeg({ readout, available, best, live, onPick }: {
  readout: string; available: string[]; best: string | null; live: boolean; onPick: (r: string) => void;
}) {
  const { t } = useT();
  const label = (r: string) => t(`ro.${r}` as Key), hint = (r: string) => t(`ro.${r}.hint` as Key);
  return (
    <>
      <div className={s.seg} role="radiogroup" aria-label={t("readout")}>
        {available.map((r) => (
          <button key={r} type="button" aria-pressed={readout === r} onClick={() => onPick(r)} title={hint(r)}>
            {label(r)}{best === r ? " ★" : ""}
          </button>
        ))}
      </div>
      <div className={s.note}>{live ? t("ro.live") : hint(readout)}</div>
    </>
  );
}

export function ViewSeg({ view, onPick }: { view: "output" | "truth" | "input"; onPick: (v: "output" | "truth" | "input") => void }) {
  const { t } = useT();
  return (
    <div className={s.seg} role="radiogroup" aria-label="spectrogram">
      {(["output", "truth", "input"] as const).map((v) => (
        <button key={v} type="button" aria-pressed={view === v} onClick={() => onPick(v)}>{t(`view.${v}`)}</button>
      ))}
    </div>
  );
}

export function ModelList({ current, results, onPick, onOpen }: {
  current: ArmName; results: Record<string, ArmResult> | null; onPick: (a: ArmName) => void; onOpen: (a: ArmName) => void;
}) {
  const { t } = useT();
  return (
    <div className={s.list} role="listbox" aria-label={t("model")}>
      {ARM_ORDER.map((arm) => {
        const m = ARM_META[arm];
        const r = results?.[arm];
        return (
          <div key={arm} className={`${s.row} ${s.model}`} aria-pressed={arm === current} role="option" aria-selected={arm === current}
               tabIndex={0} onClick={() => onPick(arm)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPick(arm); } }}>
            <div>
              <div className={s.name}>{arm}</div>
              <div className={s.meta}>{m.cls} · {t(`arm.${arm}` as Key)}</div>
            </div>
            <div className={s.stat}>
              {r ? (<><b>{fmt(r.deficit_draw, 1)}</b> dB<br />{fmt(r.crps, 3)}</>) : <span style={{ color: "var(--dim-2)" }}>—</span>}
            </div>
            <button type="button" className={s.cube} aria-label={t("arch.open.one", { arm })} title={t("arch.open")}
                    onClick={(e) => { e.stopPropagation(); onOpen(arm); }}>
              <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.1" aria-hidden="true">
                <path d="M6 1l4.5 2.5v5L6 11 1.5 8.5v-5z" /><path d="M6 6l4.5-2.5M6 6v5M6 6L1.5 3.5" />
              </svg>
            </button>
          </div>
        );
      })}
    </div>
  );
}

export function DrawKnobs({ tau, seed, det, onTau, onTauCommit, onSeed, onNewDraw }: {
  tau: number; seed: number; det: boolean; onTau: (t: number) => void; onTauCommit: () => void; onSeed: (n: number) => void; onNewDraw: () => void;
}) {
  const { t } = useT();
  return (
    <div className={s.knobs}>
      <label className={s.k} htmlFor="tau">{t("tau")}</label><span />
      <input id="tau" type="range" min={0} max={1.5} step={0.05} value={det ? 0 : tau} disabled={det}
             onChange={(e) => onTau(+e.target.value)} onMouseUp={onTauCommit} onTouchEnd={onTauCommit} onKeyUp={onTauCommit} />
      <span className={s.v}>{(det ? 0 : tau).toFixed(2)}</span>
      <div className={s.seedrow}>
        <label className={s.k} htmlFor="seed">{t("seed")}</label>
        <input id="seed" type="number" min={0} step={1} value={seed} disabled={det} onChange={(e) => onSeed(Math.max(0, Number(e.target.value) | 0))} />
        <button type="button" className={s.mini} disabled={det} onClick={onNewDraw}>{t("newdraw")} <kbd>N</kbd></button>
      </div>
    </div>
  );
}
