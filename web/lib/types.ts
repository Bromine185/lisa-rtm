// Shapes of the JSON the Python side writes into public/assets. Nothing here is typed in by hand.

export interface Signal {
  data: Float32Array<ArrayBuffer>;
  fs: number;
}

export type ReadoutName = "draw" | "tau0" | "mean16" | "logmean16";

// One arm's precomputed files for a speaker: a wav per readout at ×4 (48 kHz), plus the ×8 render
// (96 kHz, one draw at τ = 1, seed 0; τ = 0 for a deterministic arm) and the measured share of that
// file's energy above 24 kHz, so the page never claims ×8 is clean without a number.
export type ArmFiles = Partial<Record<ReadoutName, string>> & { x8?: string; x8_above24?: number };

export interface SpeakerFiles {
  truth: string;
  input: string;
  naive: string;
  arms: Record<string, ArmFiles>;
  released?: Record<string, ReleasedFiles>;
}

// ---- audio manifest: the released models' precomputed files (demo/tools/add_sota.py) ------------

// One released model's files for a speaker: 48 kHz, 16-bit, the truth's length, gain-aligned on the
// band below 5.5 kHz (the applied gain is recorded). draw2 exists only for the two samplers.
export interface ReleasedFiles {
  draw: string;
  draw2?: string;
  gain: number;
}

// How the training-set list was arrived at: the paper's own split, or an assumption the page must say.
export type SeenBasis = "stated" | "assumed";

export interface ReleasedModelManifest {
  name: string;
  family: string;
  params: number;          // sota/count_params.py, from the checkpoint
  det: boolean;
  draws: 1 | 2;
  seen: string[];          // speaker ids in the model's training set
  seen_basis?: SeenBasis;  // absent in an older manifest: read as "stated"
  seen_note: string;
  steps: string;           // e.g. "8 DDIM steps"
  rtf_m4: number | null;   // compute seconds per second of audio: sota/rtf.json (bench_rtf.py, idle, warm) else the batch run's _run.json
  rtf_device: string | null;
  machine?: string | null; // the CPU that run was on
  arch?: SotaArch;         // the architecture facts, blocks with measured counts (demo/tools/add_sota.py)
}

export interface ReleasedManifest {
  models: Record<string, ReleasedModelManifest>;
  input?: string;
  gain?: string;
  source?: string;
}

export interface Speaker {
  id: string;
  gender?: string;
  age?: string | number;
  accent?: string;
  region?: string;
  utterance?: string;
  seconds?: number;
  text?: string;
  files: SpeakerFiles;
  // A clip captured in the page (microphone or a dropped file): its signals live in memory, it has no
  // precomputed outputs, and nothing about it is in results.json. Every number shown for it is measured live.
  live?: "mic" | "file";
}

export interface AudioManifest {
  speakers: Speaker[];
  fs_hi: number;
  fs_lo: number;
  x8?: { fs: number; R: number; tau: number; seed: number; note?: string };
  released?: ReleasedManifest;
}

export interface GatedDeficit {
  loud: number | null;
  mid: number | null;
  quiet: number | null;
}

// One arm's wall time from fast/bench_latency.py, CPU, median of repeated runs. Milliseconds unless named.
export interface ArmLatency {
  params: number | null;
  one_pass_ms_per_s: number | null;   // one pass over 1 s of 12 kHz input to 48 kHz output
  rtf_one_pass: number | null;        // compute time / audio time for that pass
  frame_20ms_ms: number | null;       // one 20 ms frame
  utt_ms?: number | null;             // one pass over the bench utterance (latency_env.utt_seconds)
  shipped_one_ms: number | null;      // the eval's pipeline: one output or draw + baseband passthrough
  shipped_logmean16_ms: number | null; // 16 draws + STFT log-mean + passthrough; null for a deterministic arm
  rtf_logmean16: number | null;
}

export interface LatencyEnv {
  cpu: string | null;
  torch: string | null;
  threads: number | null;
  utt_seconds: number | null;
  passthrough_ms: number | null;
  lookahead_ms: number | null;
  precision?: string | null;
}

export interface ArmResult {
  cls: string;
  kind: string;
  lam: number;
  det: boolean;
  snr_draw: number | null;
  snr_mean16?: number | null;
  lsd_draw: number | null;
  lsd_logmean16?: number | null;
  hb_lsd?: number | null;
  deficit_draw: number | null;
  deficit_tau0: number | null;
  deficit_mean16?: number | null;
  crps: number | null;
  sliced_crps?: number | null;
  kappa?: number | null;
  coherent?: number | null;
  pit_end?: number | null;
  snr_gap?: number | null;
  visqol_speech?: number | null;
  visqol_audio?: number | null;
  pesq?: number | null;
  gated?: GatedDeficit | null;
  readouts?: Record<string, Perceptual>;
  best_readout?: string | null;
  visqol_audio_share?: number | null;
  visqol_audio_best?: number | null;
  visqol_audio_best_share?: number | null;
  latency?: ArmLatency | null;
}

export interface Perceptual {
  snr: number | null; lsd: number | null; hb_lsd: number | null;
  visqol_audio: number | null; nsim_audio: number | null;
  visqol_speech: number | null; nsim_speech: number | null; pesq: number | null;
}

export interface Results {
  tag: string;
  n_utts: number;
  M: number;
  eval_speakers?: string[];
  arms: Record<string, ArmResult>;
  naive?: { snr: number | null; lsd: number | null; visqol_audio?: number | null; visqol_speech?: number | null };
  floor?: { snr: number | null; lsd: number | null; visqol_audio?: number | null };
  ceiling?: { snr: number | null; lsd: number | null; visqol_audio?: number | null };
  visqol_audio_range?: { floor: number; ceiling: number };
  latency_env?: LatencyEnv | null;
  sota?: SotaResults | null;
}

// ---- results.sota: the released models and our arms on the paper split (make_results_sota.py) -----

export type SotaSetName = "wide" | "core" | "ourtest";

// One condition on one set. Every key may be null (the page prints a dash); a set that was not scored
// is absent from `sets` altogether.
export interface SotaMetrics {
  lsd: number | null;
  lsd_hf: number | null;
  lsd_lf: number | null;
  deficit: number | null;       // high-band level vs truth, per band, dB
  deficit_bb: number | null;    // pooled over the band, dB
  loud: number | null;
  mid: number | null;
  quiet: number | null;
  snr: number | null;
  vs_ceiling: number | null;    // SNR minus the empty-band ceiling, dB
  coh: number | null;
  kappa: number | null;
  visqol_audio: number | null;
  nsim_audio: number | null;
  visqol_speech: number | null;
  pesq: number | null;
  crps: number | null;
  crps_fair: number | null;
  sliced_crps: number | null;
  corr_err: number | null;
  gap: number | null;
  gap_hb: number | null;
  spread: number | null;
  spread_hb: number | null;     // of a calibrated ensemble; samplers only
  pit_lo: number | null;
  pit_hi: number | null;
  M: number | null;
}

export interface SotaSet {
  n_utts: number;
  seconds: number;
  speakers: string[] | number;  // the ids, or a count, as score_<set>.json._meta has it
  M: number | null;
  input: string;
  note: string;
  gain?: string | number | null;
  lsd_basis?: string | number | null;
}

// Per-utterance slope of the deficit against the truth's high-band share, fit on one set (`wide`).
export interface SotaSlope {
  slope: number;   // dB of deficit per dB of high-band share; 0 = tracks the utterance
  r: number;
  n: number;
  set?: SotaSetName;
}

export interface SotaEntry {
  sets: Partial<Record<SotaSetName, SotaMetrics | null>>;
  slope: SotaSlope | null;
}

// A block of a released model's architecture, from demo/tools/sota_models.json; `params` is counted from
// the checkpoint by sota/count_params.py. Zero is real (a front end, a post-process, the noise): the scene
// draws it as a thin plate.
export interface SotaBlock {
  id: string;
  label: string;
  detail: string;
  params: number;
}

export interface SotaArch {
  name: string;
  paper: string;
  code: string;
  family: string;
  params: number;
  params_note: string;
  det: boolean;
  det_note: string;
  input: string;
  blocks: SotaBlock[];
  training: string;
  seen: string[];
  seen_basis?: SeenBasis;
  seen_note: string;
  steps: string;
  context: string;
}

// A variant of a released model scored beside it (FLowHigh with its trained prior restored, AP-BWE on its
// own input filter): what it changes, in prose, and its metrics per set.
export interface SotaExtra {
  note: string;
  sets: Partial<Record<SotaSetName, SotaMetrics | null>>;
}

export interface SotaModel extends SotaEntry {
  name: string;
  family: string;
  params: number;
  det: boolean;
  rtf_m4: number | null;
  rtf_device: string | null;
  machine?: string | null;
  arch: SotaArch;
  extra?: Record<string, SotaExtra> | null;
}

// The SNR a perfect low band with an empty high band reaches on `wide`, per input rate.
export interface SotaCeiling {
  snr_12k?: number | null;
  snr_8k?: number | null;
  snr_16k?: number | null;
  snr_24k?: number | null;
  note?: string;
}

export interface SotaAvg {
  note: string;
  rows: number[][];
  logmean16?: number[];
  det?: number[];
}

export interface SotaResults {
  sets: Partial<Record<SotaSetName, SotaSet>>;
  ceiling: SotaCeiling | null;
  models: Record<string, SotaModel>;   // keyed by ReleasedId
  ours: Record<string, SotaEntry>;     // keyed by ArmName, measured on the same utterances
  avg: SotaAvg | null;
  sources?: { scores?: string[]; cost?: string; arch?: string };
}

export interface WeightsManifest {
  arms: Record<string, { cls: string; n_noise: number; n_dec: number; det: boolean; step: number; param_count: number; bin: string }>;
}
