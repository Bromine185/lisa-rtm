# The Missing Band — demo spec

One page. A listener picks a speaker, an upsample rate and a model, watches the model run on the
12 kHz input in the browser, and hears the result against the input and the truth. Clicking a model
opens a 3D view of the architecture with the audio travelling through it. Minimalist, dark, dense,
monospace numbers — the lineage is the user's `3d-night` project (Three.js scene as hero, JetBrains
Mono, real numbers everywhere, nothing decorative).

The thesis the page carries: everything above 6 kHz is absent from the input. The deterministic
model leaves it empty. The sampler puts it back. The visual language is built on that — the
baseband is **cold**, the generated band is **warm**, the truth is **white**.

## Tokens

```
--ground   #0A0C10      page background
--panel    #11141A      rails and cards
--line     #1F2430      hairlines, grid
--ink      #E7EBF2      primary text
--dim      #7D8798      secondary text, axis labels
--cold     #6FA3FF      the baseband, 0–6 kHz, what the input contains
--warm     #FFB347      the high band, 6–24 kHz, what the model generates
--truth    #F2F4F7      the 48 kHz reference
--good     #58D68D      a delta in the right direction   (semantic, used sparingly)
--bad      #FF6B6B      a delta in the wrong direction   (semantic, used sparingly)
```

Single committed dark theme (an instrument panel). Paint every background and colour explicitly.

Type: `JetBrains Mono` for every number, label, axis and control; `IBM Plex Sans` for the few lines
of prose. Both from Google Fonts with real fallback stacks. Tabular numerals everywhere digits align.

## Layout

Desktop: three columns. Left rail 260 px — speaker, rate, model list. Centre — the instrument: a
spectrogram canvas (0–24 kHz, a hairline at 6 kHz) on which inference visibly sweeps left to right,
the warm band appearing above the cold one; under it the transport (input / output / truth, A/B
crossfade, a realtime-factor readout). Right rail 300 px — the selected model's numbers from the
evaluation JSON: deficit, CRPS, LSD, ViSQOL, SNR beside naive, the gated deficit (loud / mid / quiet
frames). Phone: the rails stack above and below the instrument.

Clicking a model in the list swaps the centre for the 3D architecture scene (with a way back).

## The real numbers the page must carry

- Input 12 kHz. Output 48 kHz by default; the decoder is continuous in its coordinate, so ×2, ×4 and
  ×8 are the same weights queried differently (×2 is rendered by querying ×4 and decimating — the
  decoder has no idea what rate it is asked for, and querying below 48 kHz aliases).
- Encoder: four conv1d layers, kernels 7 / 3 / 3 / 1, channels 16 / 32 / 64 / 32, ReLU between,
  `same` padding. Receptive field 11 input samples = 0.9 ms. Latent dim 32.
- Decoder: input `[c, z_{i−1}, z_i, z_{i+1}]` (97 wide; 101 for LISASD with 4 noise channels per
  output sample), five Linear layers of width 144 with ReLU between, scalar out.
  c = 2(q − i) − 1 with q = j/R, i = floor(q).
- LISAS: 8 Gaussian channels concatenated to the waveform at the encoder input (+896 weights,
  87,777 parameters). LISASD: plus 4 Gaussian channels per output sample at the decoder (88,353).
- τ = 0 makes every arm the deterministic LISA exactly.
- Latency measured elsewhere: 1.5 ms per second of audio on an A100, 16.5 ms on an Apple M4 CPU.
  The browser number is whatever the engine measures live — print it honestly.

## Modules and their contracts

`demo/engine.js` — `window.LISAEngine`
```
load(binUrl, manifestUrl) -> Promise<model>
run(model, x12k /* Float32Array */, {R = 4, tau = 1, seed = 0, chunk = 2048,
    onProgress(jDone, N, outSoFar /* Float32Array view */)}) -> Promise<Float32Array>
```
Yields to the event loop between chunks so the page can paint. Seeded Gaussian noise (mulberry32 +
Box–Muller) so a draw is reproducible. Must match PyTorch at τ = 0 to 1e-4 max abs error, and at
R = 8 must invent no energy above 24 kHz (a measured property of the trained decoder).

`demo/arch3d.js` — `window.Arch3D`
```
mount(el, {arm, cls /* 'LISAS' | 'LISASD' */, tokens}) -> handle
handle.setArm(arm, cls); handle.play(); handle.pause(); handle.seek(t01);
handle.setInference(p01);  // optional: light the output as the engine progresses
handle.dispose()
```
Timeline: 48 kHz waveform → decimation to 12 kHz (samples collapse, the warm band fades out) →
samples enter the conv stack (four slabs sized by channel count and kernel width) → latents as a
ribbon → the coordinate and the three neighbouring latents feed the decoder columns (five layers,
width 144) → output at 48 kHz with the warm band lit. Noise inlets drawn for LISAS (8 channels at
the input) and LISASD (4 per output sample at the decoder). Hand-rolled orbit, slow idle drift,
respects `prefers-reduced-motion`.

`demo/assets/audio/<speaker>/{truth,input,naive,<arm>}.wav` — 16-bit PCM; truth/naive/arms at
48 kHz, input at 12 kHz. `demo/assets/audio/manifest.json` carries speaker id, gender, accent,
region, utterance id, duration, text if available, and the file list.

`demo/assets/weights/<arm>.bin` + `demo/assets/weights/manifest.json` — Float32 little-endian.

## Packaging

A Next.js app at `web/` (App Router, TypeScript, CSS modules, next/font), deployed on Vercel from this
repo with root directory `web` — the same shape as the user's `3d-night`. The two plain-JS modules
are served from `web/public/` and mounted from client components; assets live in `web/public/assets/`.
Every number on screen comes from a JSON the evaluation wrote, never typed in.
