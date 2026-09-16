// Adapter around the plain-JS inference engine (public/engine.js), which defines window.LISAEngine.
// Loaded lazily on the client; the page works without it (precomputed audio only).

export interface RunOptions {
  R?: number;
  tau?: number;
  seed?: number;
  chunk?: number;
  onProgress?: (jDone: number, N: number, out: Float32Array<ArrayBuffer>) => void;
}

export interface Engine {
  load(binUrl: string, manifestUrl: string, arm?: string): Promise<unknown>;
  run(model: unknown, x12k: Float32Array<ArrayBuffer>, opts: RunOptions): Promise<Float32Array<ArrayBuffer>>;
}

declare global {
  interface Window { LISAEngine?: Engine }
}

let loading: Promise<Engine | null> | null = null;

export function getEngine(): Promise<Engine | null> {
  if (typeof window === "undefined") return Promise.resolve(null);
  if (window.LISAEngine) return Promise.resolve(window.LISAEngine);
  if (loading) return loading;
  loading = new Promise((resolve) => {
    const s = document.createElement("script");
    s.src = "/engine.js";
    s.async = true;
    s.onload = () => resolve(window.LISAEngine ?? null);
    s.onerror = () => resolve(null);
    document.head.appendChild(s);
  });
  return loading;
}
