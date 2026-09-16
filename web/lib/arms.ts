// The seven arms of the OV3_fast run: one initialisation, identical batches, one objective each.
// Everything numeric on the page comes from public/assets/results.json; this file only names things.

export type ArmClass = "LISAS" | "LISASD";

export interface ArmMeta {
  cls: ArmClass;
  kind: string;
  lam: string;
  what: string;
  det: boolean;
}

export const ARM_ORDER = [
  "det",
  "es_marg",
  "es_marg_l0.1",
  "es_split_l0.1",
  "es_erb_l0.1",
  "es_dec_l0.1",
  "es_dec_erb_l0.1",
] as const;

export type ArmName = (typeof ARM_ORDER)[number];

export const ARM_META: Record<ArmName, ArmMeta> = {
  det:               { cls: "LISAS",  kind: "L1 + λ·STFT",          lam: "1e-2", what: "the paper’s loss, τ = 0",        det: true },
  es_marg:           { cls: "LISAS",  kind: "energy score, log-mag", lam: "1e-2", what: "the 8 Sep sampler",             det: false },
  "es_marg_l0.1":    { cls: "LISAS",  kind: "energy score, log-mag", lam: "1e-1", what: "spectral weight ×10",           det: false },
  "es_split_l0.1":   { cls: "LISAS",  kind: "energy score, split",   lam: "1e-1", what: "no waveform term above 6 kHz",  det: false },
  "es_erb_l0.1":     { cls: "LISAS",  kind: "energy score, +ERB",    lam: "1e-1", what: "scores log ERB band energies",  det: false },
  "es_dec_l0.1":     { cls: "LISASD", kind: "energy score, log-mag", lam: "1e-1", what: "noise at the decoder",          det: false },
  "es_dec_erb_l0.1": { cls: "LISASD", kind: "energy score, +ERB",    lam: "1e-1", what: "both",                          det: false },
};

export const DEFAULT_ARM: ArmName = "es_erb_l0.1";

export function isArm(s: string): s is ArmName {
  return (ARM_ORDER as readonly string[]).includes(s);
}

// Fixed by the task, not by taste.
export const FS_LO = 12000;
export const CUT_HZ = 6000;
export const FMAX_HZ = 24000;
export const A100_REALTIME = 72; // audio-seconds per compute-second, measured on the training run
