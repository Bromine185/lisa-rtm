// Shapes of the JSON the Python side writes into public/assets. Nothing here is typed in by hand.

export interface Signal {
  data: Float32Array<ArrayBuffer>;
  fs: number;
}

export type ReadoutName = "draw" | "tau0" | "mean16" | "logmean16";

export interface SpeakerFiles {
  truth: string;
  input: string;
  naive: string;
  arms: Record<string, Partial<Record<ReadoutName, string>>>;
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
}

export interface GatedDeficit {
  loud: number | null;
  mid: number | null;
  quiet: number | null;
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
  visqol_speech?: number | null;
  visqol_audio?: number | null;
  pesq?: number | null;
  gated?: GatedDeficit | null;
  readouts?: Record<string, Perceptual>;
  best_readout?: string | null;
  visqol_audio_share?: number | null;
  visqol_audio_best?: number | null;
  visqol_audio_best_share?: number | null;
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
  arms: Record<string, ArmResult>;
  naive?: { snr: number | null; lsd: number | null; visqol_audio?: number | null; visqol_speech?: number | null };
  floor?: { snr: number | null; lsd: number | null; visqol_audio?: number | null };
  ceiling?: { snr: number | null; lsd: number | null; visqol_audio?: number | null };
  visqol_audio_range?: { floor: number; ceiling: number };
}

export interface WeightsManifest {
  arms: Record<string, { cls: string; n_noise: number; n_dec: number; det: boolean; step: number; param_count: number; bin: string }>;
}
