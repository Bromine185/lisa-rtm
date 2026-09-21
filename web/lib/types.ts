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
}

export interface AudioManifest {
  speakers: Speaker[];
  fs_hi: number;
  fs_lo: number;
  x8?: { fs: number; R: number; tau: number; seed: number; note?: string };
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
}

export interface WeightsManifest {
  arms: Record<string, { cls: string; n_noise: number; n_dec: number; det: boolean; step: number; param_count: number; bin: string }>;
}
