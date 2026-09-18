# lisa-rtm — regression to the mean in LISA, and the sampler that ends it

**`sampler/lisa_rtm_sampler.ipynb` is the deliverable.** Open it in Colab, pick a GPU runtime, Run all.

[LISA](https://arxiv.org/abs/2111.00195) (ICASSP 2022) is an audio super-resolution model: a conv
encoder makes local latent codes from a low-rate waveform, a 5-layer ReLU MLP decodes an amplitude
at any continuous time coordinate, ~89k parameters. It is deterministic, trained on a deterministic
loss, on a one-to-many problem. So the loss minimiser is a conditional barycentre, and the barycentre
of a random-phase high band has no energy. The output is muffled. That is not a bug in the model. It
is what the objective asks for.

## The sampler

The fix is an objective, not an architecture. `LISAS` concatenates eight Gaussian channels to the
waveform at the encoder input and trains the same network under the **energy score**, two draws per
step:

```
ES(P, y) = E d(Y, y) - 1/2 E d(Y, Y')            estimate: 1/2 [d(y, y1) + d(y, y2)] - 1/2 d(y1, y2)
d(a, b)  = L1(waveform) + 1e-2 * L1(log|STFT|)  over three scales
```

The score is strictly proper, so the network converges to `p(y|x)` rather than to its mean. With the
noise at zero the network is LISA exactly (+896 weights, 87,777 parameters). One forward pass is one
calibrated sample. No discriminator, no diffusion, no second network.

What the 8 September run found (`notes/2026-09-08-network-as-transport-map.md`), held-out speakers,
every number naming its set:

- CRPS on high-band log-magnitude falls 43 % against the deterministic model on the Hub set and
  32 % on the DataShare set, at zero inference cost.
- The ensemble mean is a better point predictor than the point-trained model: SNR 18.90 vs 18.86 dB
  (Hub), 18.78 vs 18.35 (DataShare).
- The band above 6 kHz is incoherent with the input for every model trained here: coherent fraction
  ≤ 2 %, so no deterministic LISA of this class can beat naive upsampling by more than 0.3 dB. The
  deterministic model's high-band energy is hallucinated (κ = 0.06–0.12); the sampler's ensemble-mean
  coherence is a predictability spectrum.
- Real time: 1.5 ms per second of audio on an A100, 16.5 ms on an Apple M4 CPU, 0.5 ms look-ahead.
- The same move on a graph separates objective-induced over-smoothing (the conditional mean has no
  high-graph-frequency energy) from architecture-induced over-smoothing (deep propagation is a
  low-pass); one sampler gives both numbers.

The recipe itself is not new: noise at the input plus the energy score is DISCO Nets (2016) and the
spectral energy distance of Gritsenko et al. (2020). What is new here is the bandwidth-extension
target, the stochastic local implicit decoder, and the diagnostics: the coherent fraction, the
predictability spectrum, the deterministic SNR bound, and the calibration test
`SNR(mean) - SNR(one draw) = 10 log10(2 / (1 + 1/M))`.

## The earlier programme: post-hoc transport maps (superseded)

`lisa_rtm.ipynb` reimplements LISA, measures the high-band deficit, and tries to correct it after
training with a ladder of optimal transport maps fitted to low-order statistics of the high-band
log-magnitude distribution:

| rung | map | statistics |
|---|---|---|
| `T0` | identity | none — the baseline |
| `T1` | per-bin quantile map, exact 1-D OT | full marginal per bin |
| `T2` | Bures–Wasserstein, `m_t + A(l - m_s)` | mean + covariance |
| `T3` | affine-in-covariates location-scale | mean + covariance, level-dependent |
| `T4` | `(1-λ)·Id + λ·T`, McCann interpolation | the continuum between any rung and identity |
| `S` | noise through a conditional transport map | the only rung that produces *samples* |

It did not work well enough to keep. On an honest source `T1` moved HB-LSD by 1.7 % and `S` moved
CRPS by 21 %; the learned sampler moved CRPS by 43 % and dominates every rung. A map on a point
prediction is still a point prediction. The notebook is kept for three things that survive:

1. **LSD structurally rewards regression to the log-domain mean.** A perfect sampler scores ~2×
   worse in expected squared error than the conditional-mean predictor. That is one reason
   LISA (deterministic, 89k) reports LSD 0.81 against WSRGlow (a flow, 229M) at 1.01 *in LISA's own
   table*; the other is that their LSD code logs `|X|⁴` and pins 16.6 % of the spectrogram at its
   epsilon floor (`notes/2026-09-17-lisa-reported-numbers-audit.md` §3.3). Band-energy ratio measures
   the disease, CRPS measures the cure, LSD is reported but never trusted alone — and never compared
   across papers, because no two of them define it the same way.
2. **Gates before data.** §5 validates the transport code against closed-form optimal transport
   before anything is downloaded (`T1` to 1e-12, Bures identity `A·Σs·A = Σt` to 1e-10); §9 measures
   the deficit on held-out speakers and stops if there is no headroom.
3. **Adversarial controls in the results table**: shaped noise (1970s noise-filling BWE), a
   deliberately mis-specified map, and a frame shuffle, plus HB-LSD in a second STFT basis. They fired
   on the first clean run and caught LSD rewarding generic energy inflation.

## Files

```
sampler/             the sampler on its own: LISAS trained under the energy score
  lisa_rtm_sampler.ipynb     self-contained Colab notebook (SMOKE / QUICK / FULL presets)
  build_sampler_notebook.py  regenerates it -- edit here, not the JSON
  validate_sampler.py        runs every cell on CPU with a synthetic corpus, no network
lisa_rtm.ipynb       the original notebook: LISA reimplementation + the transport ladder
build_notebook.py    regenerates it deterministically -- edit here, not the JSON
validate.py          runs every code cell end-to-end on synthetic audio, no network
overnight2/          the 8 Sep run: LISAS, losses, trainer, evaluation, graph toy, ViSQOL
overnight/           the 4 Sep λ-frontier run on the full Hub corpus
audit/               protocol checks, no GPU: the task's SNR ceiling, LISA's reported numbers,
                     the anti-aliased/aliased protocol fork, whether the high band gates with the
                     speech, whether the decoder is scale-free
paper/               bwe-information-ceiling.md -- the two-page write-up of the ceiling argument,
                     every number reproduced by three CPU scripts in audit/
web/                 "The Missing Band": the Next.js demo. Pick a speaker, watch the 88k network run
                     in the browser, hear the readouts, open a 3D view of any arm
demo/                the demo's sources: SPEC.md, the plain-JS engine and 3D scene that web/public
                     serves, and the Python tools that render its audio, weights and numbers
notes/               the research notes, in date order
```

## The demo

```bash
cd web && npm install && npm run dev        # http://localhost:3000
```

**Deploying it.** The app is a subdirectory of a research repo, so Vercel needs one setting that is
not in any file: **Project Settings → Build and Deployment → Root Directory = `web`**. Everything
else auto-detects. Root Directory is a project setting, not a `vercel.json` key, so it cannot be
committed — and changing it does **not** rebuild what is already live. A production deployment made
before that setting existed was built from the repo root, found no app, and 404s on every route; the
dashboard flags this as *"Configuration Settings in the current Production deployment differ from
your current Project Settings."* Push a commit to get a fresh build rather than redeploying the old
one, which can carry its own Production Overrides forward.

`web/public/assets` is committed on purpose — 168 WAVs, seven weight files, and the evaluation JSON,
about 60 MB. The build needs no Python and no GPU; `npm ci && next build` from a clean clone is the
whole of it.

## Local development

```bash
python3.11 -m venv venv && ./venv/bin/pip install numpy scipy matplotlib soundfile requests tqdm torch ipython
python3 sampler/build_sampler_notebook.py      # -> sampler/lisa_rtm_sampler.ipynb
./venv/bin/python sampler/validate_sampler.py  # every cell on CPU, ~1 min, synthetic corpus
python3 build_notebook.py                      # -> lisa_rtm.ipynb
./venv/bin/python validate.py                  # every cell, ~2 min, VCTK replaced by synthetic speech
```

Both validators are smoke tests of the code path, not scientific results: they substitute a synthetic
harmonic-plus-fricative corpus for VCTK so the whole pipeline runs without the network. Real numbers
come from running the notebooks on Colab against real VCTK.
