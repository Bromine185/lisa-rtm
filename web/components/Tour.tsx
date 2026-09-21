"use client";
import { useEffect, useState } from "react";
import s from "@/app/page.module.css";
import { useT, type Key } from "@/lib/i18n";

const N = 5;

// Mounted only while open, so every opening starts at the first step.
export function Tour({ onClose }: { onClose: () => void }) {
  const { t } = useT();
  const [step, setStep] = useState(0);
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); onClose(); }
      else if (e.key === "ArrowRight" || e.key === "Enter" || e.key === " ") { e.preventDefault(); if (step >= N - 1) onClose(); else setStep(step + 1); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); setStep(Math.max(0, step - 1)); }
      e.stopPropagation();
    };
    // capture phase so the instrument's own keys (space, 1/2/3) do not fire underneath
    window.addEventListener("keydown", h, true);
    return () => window.removeEventListener("keydown", h, true);
  }, [step, onClose]);
  const h = `tour.${step + 1}.h` as Key, p = `tour.${step + 1}.p` as Key;
  return (
    <div className={s.tourBack} onClick={onClose} role="presentation">
      <div className={s.tour} role="dialog" aria-modal="true" aria-labelledby="tour-h" onClick={(e) => e.stopPropagation()}>
        <div className={`lbl ${s.tourKicker}`}>{t("tour.title")}</div>
        <h2 id="tour-h">{t(h)}</h2>
        <p>{t(p)}</p>
        <div className={s.tourNav}>
          <button type="button" className={s.mini} disabled={step === 0} onClick={() => setStep(step - 1)}>{t("tour.prev")}</button>
          <div className={s.tourDots} aria-hidden="true">{Array.from({ length: N }, (_, i) => <i key={i} data-on={i === step || undefined} onClick={() => setStep(i)} />)}</div>
          <button type="button" className={s.run} onClick={() => (step >= N - 1 ? onClose() : setStep(step + 1))}>{step >= N - 1 ? t("tour.close") : t("tour.next")}</button>
        </div>
        <div className={s.note}>{t("tour.esc")}</div>
      </div>
    </div>
  );
}
