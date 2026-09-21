"use client";
import s from "@/app/page.module.css";
import { useT } from "@/lib/i18n";

export type Vote = "a" | "b" | "tie";
export interface Trial {
  n: number;
  speaker: string;
  sampler: string;       // the arm under test
  ro: string;            // its readout label
  det: string;           // the deterministic reference arm
  swap: boolean;         // true: A is det, B is the sampler
  ready: boolean;        // both outputs are loaded
  vote: Vote | null;
}
export interface Tally { sampler: number; det: number; tie: number }

export function Blind({ trial, tally, onVote, onNext, onReset, onExit }: {
  trial: Trial | null; tally: Tally; onVote: (v: Vote) => void; onNext: () => void; onReset: () => void; onExit: () => void;
}) {
  const { t } = useT();
  const done = trial?.vote != null;
  const aName = trial ? (trial.swap ? trial.det : trial.sampler) : "", bName = trial ? (trial.swap ? trial.sampler : trial.det) : "";
  return (
    <div className={`${s.card} ${s.blind}`}>
      <h2><span>{t("b.title")}</span><small>{trial ? t("b.trial", { n: trial.n }) : ""}</small></h2>
      <div className={s.note}>{t("b.hint")}</div>
      {trial && (
        <>
          <div className={s.bq}>{t("b.q")}</div>
          <div className={s.bvotes}>
            {(["a", "b", "tie"] as const).map((v) => (
              <button key={v} type="button" className={s.bvote} disabled={!trial.ready || done} aria-pressed={trial.vote === v}
                      onClick={() => onVote(v)}>{t(`b.vote.${v}`)}{v !== "tie" && <kbd>{v.toUpperCase()}</kbd>}</button>
            ))}
          </div>
          {!trial.ready && <div className={s.note}>{t("b.preparing")}</div>}
          {done && (
            <div className={s.breveal}>
              <div>{t("b.reveal", { a: aName, b: bName })}</div>
              <button type="button" className={s.run} onClick={onNext}>{t("b.next")} <kbd>⏎</kbd></button>
            </div>
          )}
        </>
      )}
      <div className={s.kv}>
        <span className={s.hh}>{t("b.tally")}</span>
        <span className={s.kk}>{t("b.sampler")}</span><span className={s.vv}>{tally.sampler}</span>
        <span className={s.kk}>{t("b.det")}</span><span className={s.vv}>{tally.det}</span>
        <span className={s.kk}>{t("b.tie")}</span><span className={s.vv}>{tally.tie}</span>
      </div>
      {trial && <div className={s.note}>{t("b.pair", { sampler: trial.sampler, ro: trial.ro, det: trial.det })}</div>}
      <div className={s.brow}>
        <button type="button" className={s.mini} onClick={onReset}>{t("b.reset")}</button>
        <span className={s.grow} />
        <button type="button" className={s.mini} onClick={onExit}>{t("b.exit")} <kbd>Esc</kbd></button>
      </div>
    </div>
  );
}
