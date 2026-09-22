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
of prose; `Noto Sans JP` behind both for the kana and kanji they lack. All from Google Fonts with real
fallback stacks (Hiragino, Yu Gothic, Meiryo for Japanese). Tabular numerals everywhere digits align.

## Languages

The page reads in English or Japanese; `L` or the header toggle switches, the choice is remembered in
localStorage and mirrored onto `<html lang>`. One dictionary, `web/lib/i18n.tsx`, every string keyed.
Technical tokens are never translated — arm names, CRPS, LSD, ViSQOL, PESQ, τ, λ, kHz, `logmean16` —
only the prose around them. Nothing numeric passes through the dictionary. The 3D scene's own labels
are formulae and stay as they are.

## Three steps

The site opens on the numbers, not the controls. `/` (`web/components/Landing.tsx`): four stat tiles
from results.json for the arm the instrument opens on — high-band deficit, CRPS, LSD at its best
readout, cost per second of audio — each with its delta against `det`; the input and that arm's
log-mean-of-16 output as spectrograms side by side; their long-term average spectra (mean power per
bin, in dB) on one axis with the truth behind and the region between input and output above 6 kHz
filled, which is what the sampler adds; then one large button. `/listen` (`Listen.tsx`): one speaker,
that arm, truth / input / output on one clock with the spectrogram following the source, three large
source buttons, one large button on. `/instrument`: the page below, unchanged. `⏎` and `→` step
forward, `←` back; the numbers rail on the instrument opens with the same three tiles.

## Layout

Desktop: three columns. Left rail 260 px — speaker, your voice, rate, model list, readout, draw.
Centre — the instrument: a
spectrogram canvas (0–24 kHz, a hairline at 6 kHz) on which inference visibly sweeps left to right,
the warm band appearing above the cold one; under it the transport (input / output / truth, A/B
crossfade, a realtime-factor readout). Right rail 300 px — the selected model's numbers from the
evaluation JSON: deficit, CRPS, LSD, ViSQOL, SNR beside naive, the gated deficit (loud / mid / quiet
frames). Phone: the rails stack above and below the instrument.

Clicking a model in the list swaps the centre for the 3D architecture scene (with a way back).

Header: the legend, then blind test (`B`), tour (`?`) and the language toggle (`L`). The tour is five
short cards — the problem, the fix, listen, the three readouts, your voice and the blind test — opened
on demand, never on load; `Esc` closes. Under the prose sits the key map, so a presenter never has to
remember it: `space` play, `1 2 3` sources, `R` run, `N` new draw, `↑ ↓` model, `← →` speaker, `V`
record, `B` blind test, `L` language, `?` tour.

## Your voice

`V`, or the record button, opens the laptop microphone (no AGC, no noise suppression, no echo
cancellation — the clip must be what the room sounds like) and collects at 48 kHz for up to 12 s;
a dropped file (anything the browser decodes) takes the same path. The clip becomes a speaker in the
list, marked `live`, and is treated exactly like a corpus fixture: mono, peak 0.95, then
`input` = resample_poly(y, 1, 4) and `naive` = resample_poly(input, 4, 1) — the filter is a port of
scipy's (81-tap windowed sinc, Kaiser β = 5, cutoff at 6 kHz, unity DC gain, zero phase;
`web/lib/resample.ts`), so the band limit is the corpus's band limit, not an approximation of it.
There are no precomputed outputs for it: every output is the engine, every number in the status line
is measured on this browser, and the readout is one draw (the statistics need sixteen). Nothing in
results.json refers to it and the page says so. Training was English only (VCTK); a Japanese sentence
through the network is an out-of-distribution test the room can hear for itself — which is the point.

## Blind test

`B` turns the page into a listening test on the one laptop. A trial: a speaker drawn at random from
the list (recorded clips included), the current sampler arm at the readout it wins on when that file
exists, else one draw, against `det` at τ = 0 — the deterministic model with the same spectral term.
A and B are shuffled; the transport plays A / B / input on the one clock; the spectrogram is forced to
the input view and dimmed, because the two differ only above 6 kHz and the eye must not vote. Vote
`A`, `B` or `T` (can't tell); the reveal names both arms; the tally (sampler / deterministic / tie)
persists in localStorage until reset; `⏎` draws the next trial, `Esc` leaves. The tally is a count of
votes in a room, not an evaluation, and the page presents it as nothing more.

## The real numbers the page must carry

- Input 12 kHz. Output 48 kHz by default; the decoder is continuous in its coordinate, so ×4 and ×8
  are the same weights queried differently. The demo offers ×4 and ×8 only: the model was trained and
  evaluated at ×4, ×8 is the same weights at twice the density (the decoder has no idea what rate it is
  asked for), and there is no ×2 — it was never trained or evaluated, and querying below 48 kHz aliases.
- Encoder: four conv1d layers, kernels 7 / 3 / 3 / 1, channels 16 / 32 / 64 / 32, ReLU between,
  `same` padding. Receptive field 11 input samples = 0.9 ms. Latent dim 32.
- Decoder: input `[c, z_{i−1}, z_i, z_{i+1}]` (97 wide; 101 for LISASD with 4 noise channels per
  output sample), five Linear layers of width 144 with ReLU between, scalar out.
  c = 2(q − i) − 1 with q = j/R, i = floor(q).
- LISAS: 8 Gaussian channels concatenated to the waveform at the encoder input (+896 weights,
  87,777 parameters). LISASD: plus 4 Gaussian channels per output sample at the decoder (88,353).
- τ = 0 makes every arm the deterministic LISA exactly.
- Latency measured by `fast/bench_latency.py` (OV50, fp32 eager, batch 1, Apple M4 CPU, 8 threads):
  one pass over 1 s of audio is 12.7 ms for a LISAS arm and 16.0 ms for a LISASD arm (the decoder
  noise costs 22 % of wall time for 1 % of the MACs); the shipped pipeline on a 3.5 s utterance is
  88–100 ms for one output or draw with passthrough and 0.9–1.1 s for logmean16. Kind and lambda do
  not move it. The page reads these from results.json (`latency`, `latency_env`); the browser number
  is whatever the engine measures live — print it honestly.

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
handle.focus(stage | null); handle.reset(); handle.panBy(dx, dy); handle.zoomBy(f)   // the camera, eased
handle.dispose()
```
Camera: two orbits, the goal and the eased actual, so every move settles instead of snapping. Drag
orbits about a fixed target — the framing is solved on a fit (mount, resize, reset, a stage focus),
never on a drag. A two-finger swipe pans (horizontal travels along the flow), as do shift-drag, a
middle or right button, or two fingers together; pinch or ctrl/⌘-wheel zooms; double-click resets.
`focus(i)` frames one of the five stages; the page binds it to the stage strip and to `1`–`5`, `0`
fits the whole. The architecture page is laid out to fit the viewport — header, scene, timeline, stage
strip, facts — so the wheel belongs to the scene and never to the page.
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
