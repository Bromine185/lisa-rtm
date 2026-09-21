// The eight arms of the OV50 run (50 epochs): one initialisation, identical batches, one objective each.
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
  "det_paper",
  "det",
  "es_marg",
  "es_dec_l0.01",
  "es_erb_l0.001",
  "es_erb_l0.01",
  "es_erb_l0.1",
  "es_dec_erb_l0.1",
] as const;

export type ArmName = (typeof ARM_ORDER)[number];

export const ARM_META: Record<ArmName, ArmMeta> = {
  det_paper:         { cls: "LISAS",  kind: "plain L1 (the paper)",  lam: "0",    what: "LISA as published: plain L1, τ = 0",                            det: true },
  det:               { cls: "LISAS",  kind: "L1 + λ·STFT",           lam: "1e-2", what: "L1 + λ·STFT, τ = 0",                                            det: true },
  es_marg:           { cls: "LISAS",  kind: "energy score, log-mag", lam: "1e-2", what: "energy score, log-mag",                                         det: false },
  "es_dec_l0.01":    { cls: "LISASD", kind: "energy score, log-mag", lam: "1e-2", what: "noise at the decoder",                                          det: false },
  "es_erb_l0.001":   { cls: "LISAS",  kind: "energy score, +ERB",    lam: "1e-3", what: "+ERB, weight too small: collapses toward a point predictor",    det: false },
  "es_erb_l0.01":    { cls: "LISAS",  kind: "energy score, +ERB",    lam: "1e-2", what: "+ERB: scores log ERB band energies",                            det: false },
  "es_erb_l0.1":     { cls: "LISAS",  kind: "energy score, +ERB",    lam: "1e-1", what: "+ERB, spectral weight ×10",                                     det: false },
  "es_dec_erb_l0.1": { cls: "LISASD", kind: "energy score, +ERB",    lam: "1e-1", what: "both: decoder noise + ERB",                                     det: false },
};

// Best on CRPS, PIT, logmean16 LSD and audio ViSQOL in OV50.
export const DEFAULT_ARM: ArmName = "es_dec_erb_l0.1";

// The deterministic arm every delta on the page is taken against. det_paper is the paper's model and
// sits in the list as the reference line; det is the same class with the spectral term the samplers share.
export const REF_ARM: ArmName = "det";

export function isArm(s: string): s is ArmName {
  return (ARM_ORDER as readonly string[]).includes(s);
}

// Fixed by the task, not by taste.
export const FS_LO = 12000;
export const CUT_HZ = 6000;
export const FMAX_HZ = 24000;

// The output rates the page offers. ×4 is what the model was trained and evaluated at; ×8 is the same
// weights queried at twice the density, which the decoder's continuous coordinate allows and training
// never asked for. There is no ×2: it was never trained, never evaluated, and only ever ×4 decimated.
export const RATES = [4, 8] as const;
export type Rate = (typeof RATES)[number];
