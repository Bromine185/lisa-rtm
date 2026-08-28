# lisa-rtm — regression to the mean in LISA, and transport maps against it

**`lisa_rtm.ipynb` is the deliverable.** Upload it to Colab, pick a GPU runtime, Run all.

[LISA](https://arxiv.org/abs/2111.00195) (ICASSP 2022) is an audio super-resolution model: a conv
encoder makes local latent codes from a low-rate waveform, a 5-layer ReLU MLP decodes an amplitude
at any continuous time coordinate, ~89k parameters. It is deterministic, trained on a deterministic
loss, on a one-to-many problem — so the loss minimiser is roughly the conditional mean, and the
conditional mean of a random-phase high band has suppressed energy. The output is muffled.

The notebook reimplements LISA, measures that deficit, and corrects it with a ladder of optimal
transport maps fitted to low-order statistics of the high-band log-magnitude distribution:

| rung | map | statistics |
|---|---|---|
| `T0` | identity | none — the baseline |
| `T1` | per-bin quantile map, exact 1-D OT | full marginal per bin |
| `T2` | Bures–Wasserstein, `m_t + A(l - m_s)` | mean + covariance |
| `T3` | affine-in-covariates location-scale | mean + covariance, level-dependent |
| `T4` | `(1-λ)·Id + λ·T`, McCann interpolation | the continuum between any rung and identity |
| `S` | noise through a conditional transport map | the only rung that produces *samples* |

## What the notebook is built around

1. **A deterministic map on a point prediction is still a point prediction.** `T1`–`T4` make one
   output sharper; they cannot represent `p(y|x)` or be calibrated. §12 adds the stochastic rung,
   which is where the generative claim lives.
2. **A global map matches the *pooled* marginal**, which is a mixture over frame types — so it
   inflates silence into hiss and under-corrects fricatives. §10's `T3` conditions on covariates
   computed from the input only.
3. **LSD structurally rewards regression to the log-domain mean.** A perfect sampler scores ~2×
   worse in expected squared error than the conditional-mean predictor. This is very likely why
   LISA (deterministic, 89k) reports LSD 0.81 against WSRGlow (a flow, 229M) at 1.01 *in LISA's own
   table*. Beating a generative model on LSD is evidence of the disease, not health. So band-energy
   ratio measures the disease, CRPS measures the cure, and LSD is reported but never trusted alone.

## Two gates

§5 validates the transport code against closed-form optimal transport **before any data is
downloaded or any model is trained** — the exact map between log-Rayleigh distributions is a pure
shift, so `T1` is checked to 1e-12, and Bures gets the algebraic identity `A·Σs·A = Σt` to 1e-10.

§9 measures the actual high-band energy deficit on held-out speakers. If it is under ~1–2 dB, the
premise is wrong for this model and the honest move is to write that down and stop.

## Controls

Three adversarial controls ship with the results table, not as an appendix: **shaped noise** (1970s
noise-filling BWE — if the ladder cannot beat this, the OT machinery bought vocabulary rather than
value), a **mis-specified map** (if a deliberately wrong map still improves HB-LSD, then LSD is
rewarding generic energy inflation and gets demoted), and a **frame shuffle** (pooled marginals are
permutation-invariant, so a metric suite that does not score this much worse cannot see conditional
structure at all). Plus HB-LSD recomputed in a second STFT basis, and a latency benchmark.

## Files

```
lisa_rtm.ipynb       the deliverable
build_notebook.py    regenerates the notebook deterministically -- edit here, not the JSON
validate.py          runs every code cell end-to-end on synthetic audio, no network
```

## Local development

```bash
python3.11 -m venv venv && ./venv/bin/pip install numpy scipy matplotlib soundfile requests tqdm torch ipython
python3 build_notebook.py     # -> lisa_rtm.ipynb
./venv/bin/python validate.py # every cell, ~2 min, VCTK replaced by synthetic speech
```

`validate.py` is a smoke test of the code path, not a scientific result: it substitutes a synthetic
harmonic-plus-fricative corpus for VCTK so the whole pipeline runs without the network. Real numbers
come from running the notebook on Colab against real VCTK.
