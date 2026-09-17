"use client";
import s from "@/app/page.module.css";
import { ARM_META, ARM_ORDER, type ArmName } from "@/lib/arms";
import type { ArmResult, Speaker } from "@/lib/types";

const fmt = (v: number | null | undefined, d = 2) => (v == null || !isFinite(v) ? "—" : v.toFixed(d));

export function SpeakerList({ speakers, current, onPick }: { speakers: Speaker[]; current: string | null; onPick: (sp: Speaker) => void }) {
  return (
    <div className={s.list} role="listbox" aria-label="speaker">
      {speakers.map((sp) => (
        <button key={sp.id} type="button" className={s.row} aria-pressed={sp.id === current} onClick={() => onPick(sp)}>
          <div>
            <div className={s.name}>{sp.id}</div>
            <div className={s.meta}>{[sp.gender, sp.accent, sp.region].filter(Boolean).join(" · ")}</div>
          </div>
          <div className={s.stat}><b>{(sp.seconds ?? 0).toFixed(1)}</b> s</div>
        </button>
      ))}
    </div>
  );
}

export function RateSeg({ rate, onPick }: { rate: 2 | 4 | 8; onPick: (r: 2 | 4 | 8) => void }) {
  return (
    <div className={s.seg} role="radiogroup" aria-label="output rate">
      {([2, 4, 8] as const).map((r) => (
        <button key={r} type="button" aria-pressed={rate === r} onClick={() => onPick(r)}>×{r} · {(12 * r).toFixed(0)} kHz</button>
      ))}
    </div>
  );
}

const READOUT_LABEL: Record<string, string> = { draw: "one draw", mean16: "mean of 16", logmean16: "log-mean of 16", tau0: "τ = 0" };
const READOUT_HINT: Record<string, string> = {
  draw: "a single sample from p(y|x) — what you would ship",
  mean16: "the waveform mean — the SNR-optimal readout, and the muffled one",
  logmean16: "per-bin mean of log|STFT| over 16 draws — LSD's actual minimiser",
  tau0: "noise off: this is the deterministic LISA exactly",
};

export function ReadoutSeg({ readout, available, best, onPick }: {
  readout: string; available: string[]; best: string | null; onPick: (r: string) => void;
}) {
  return (
    <>
      <div className={s.seg} role="radiogroup" aria-label="readout">
        {available.map((r) => (
          <button key={r} type="button" aria-pressed={readout === r} onClick={() => onPick(r)} title={READOUT_HINT[r]}>
            {READOUT_LABEL[r] ?? r}{best === r ? " ★" : ""}
          </button>
        ))}
      </div>
      <div className={s.note}>{READOUT_HINT[readout]}</div>
    </>
  );
}

export function ViewSeg({ view, onPick }: { view: "output" | "truth" | "input"; onPick: (v: "output" | "truth" | "input") => void }) {
  return (
    <div className={s.seg} role="radiogroup" aria-label="what the spectrogram shows">
      {(["output", "truth", "input"] as const).map((v) => (
        <button key={v} type="button" aria-pressed={view === v} onClick={() => onPick(v)}>{v}</button>
      ))}
    </div>
  );
}

export function ModelList({ current, results, onPick, onOpen }: {
  current: ArmName; results: Record<string, ArmResult> | null; onPick: (a: ArmName) => void; onOpen: (a: ArmName) => void;
}) {
  return (
    <div className={s.list} role="listbox" aria-label="model">
      {ARM_ORDER.map((arm) => {
        const m = ARM_META[arm];
        const r = results?.[arm];
        return (
          <div key={arm} className={`${s.row} ${s.model}`} aria-pressed={arm === current} role="option" aria-selected={arm === current}
               tabIndex={0} onClick={() => onPick(arm)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPick(arm); } }}>
            <div>
              <div className={s.name}>{arm}</div>
              <div className={s.meta}>{m.cls} · {m.what}</div>
            </div>
            <div className={s.stat}>
              {r ? (<><b>{fmt(r.deficit_draw, 1)}</b> dB<br />{fmt(r.crps, 3)}</>) : <span style={{ color: "var(--dim-2)" }}>—</span>}
            </div>
            <button type="button" className={s.cube} aria-label={`open the ${arm} architecture`} title="architecture"
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
  return (
    <div className={s.knobs}>
      <label className={s.k} htmlFor="tau">temperature τ</label><span />
      <input id="tau" type="range" min={0} max={1.5} step={0.05} value={det ? 0 : tau} disabled={det}
             onChange={(e) => onTau(+e.target.value)} onMouseUp={onTauCommit} onTouchEnd={onTauCommit} onKeyUp={onTauCommit} />
      <span className={s.v}>{(det ? 0 : tau).toFixed(2)}</span>
      <div className={s.seedrow}>
        <label className={s.k} htmlFor="seed">seed</label>
        <input id="seed" type="number" min={0} step={1} value={seed} disabled={det} onChange={(e) => onSeed(Math.max(0, Number(e.target.value) | 0))} />
        <button type="button" className={s.mini} disabled={det} onClick={onNewDraw}>new draw</button>
      </div>
    </div>
  );
}
