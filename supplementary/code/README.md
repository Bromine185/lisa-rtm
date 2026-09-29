# Code

A pruned snapshot of the research code that produced every number in the paper. Relative paths are
kept as they were, because the model and metric definitions are notebook cells that the scripts execute
in place (`audit/boot.py`, `fast/train_arm.py`); the files were not refactored, so what runs here is
what ran. Personal paths were replaced by one environment variable, `BWE_CACHE` (default `./cache`).
Comments refer to internal lab notes (`notes/...`) that are not included.

## Map

| Path | What it is |
|---|---|
| `notebook.ipynb`, `build_notebook.py` | STFT, log-magnitude, LSD, SNR, third-octave bands, CRPS, PIT, spread-skill, the LISA reimplementation. `build_notebook.py` generates the notebook; edit there. |
| `overnight2/c1_model.py` | `LISAS`: LISA plus 8 Gaussian channels at the encoder input; waveform and log-magnitude distances; the two-draw energy-score loss |
| `overnight3/e1_model.py` | `LISASD` (4 more Gaussian channels per output sample at the decoder), the 32-band ERB distance, `logmag_ensemble_readout` |
| `overnight3/e2b_fast.py` | `ArmStack`: the training step (energy-score estimator, distances, per-arm clipping) |
| `fast/run_contract.py` | The eight arms (kind, λ, class), the step count, the learning-rate schedule, and the fixed batch-stream labels that give every arm identical batches |
| `fast/stage_corpus.py`, `fast/train_arm.py`, `fast/bootstrap.sh`, `fast/run_8gpu.sh` | Corpus staging (VCTK 0.92 mic1, 37.3 h), one arm per GPU, resumable |
| `fast/convert_ckpt.py` | Training checkpoints to the evaluation format |
| `sota/make_inputs.py`, `sota/fetch_wide.py` | The three test sets and their 12 kHz inputs (Chebyshev I, the FLowHigh / NU-Wave 2 protocol) |
| `sota/run_models.py` | The four released models through their own inference code, M seeded draws |
| `sota/run_ours.py` | Our arms on the same inputs |
| `sota/score.py` | The single scorer behind both tables: levels, LSD, SNR against the ceiling, coherence, CRPS (both estimators), spread, PIT, perceptual |
| `sota/mean_of_M.py` | **Added for this supplement.** The mean-of-M table (see below) |
| `sota/ceiling.py` | The empty-band SNR ceiling at 8/12/16/24 kHz input |
| `sota/count_params.py`, `sota/bench_rtf.py` | Parameter counts, real-time factors |
| `sota/figs.py` | The figures in `../figures/` |
| `demo/tools/make_results_sota.py`, `demo/tools/sota_models.json` | Collects the score files into `../results/results.json`; architecture facts for the released models |
| `sota/run_all.sh` | **Added for this supplement.** The evaluation, end to end, in the order it was run |
| `patches/flowhigh_inference_restore_prior.diff` | One-line change to FLowHigh's `inference.py` that keeps the trained prior (`flowhigh_std1`) |
| `validate.py`, `fast/test_run_contract.py`, `fast/test_train_arm.py` | CPU tests on synthetic audio, no network |

## Environments

- **Project** (training, our inference, inputs, figures): Python 3.11–3.12, `torch` (2.13 was used; any
  2.x with `torch.stft` works for CPU tests), `numpy`, `scipy`, `soundfile`, `matplotlib`, `pyarrow`,
  `huggingface_hub`. Training ran with one arm per NVIDIA RTX PRO 6000 (Blackwell) GPU, 104,950 steps at
  about 105 ms per step, which is about 3 GPU-hours per arm.
- **Released models** (`sota/run_models.py`): torch 2.2.1, numpy 1.23, librosa 0.9.2, plus each model's own
  repository and checkpoint under `$BWE_CACHE/sota/`:
  - FLowHigh `FLowHigh_indep_adaptive_400k.pt` + BigVGAN `g_48_00850000`;
  - NU-Wave 2 (loaded without pytorch-lightning);
  - AP-BWE 48 kHz;
  - AudioSR `speech`.
- **Scoring** (`sota/score.py`): the project environment plus `visqol-python[lattice]`, `pesq`,
  `torchaudio`, `torchmetrics`.

## Check it runs (CPU, no data, about 3 minutes)

```bash
python fast/test_run_contract.py      # 31 checks: arms, schedule, batch-stream pairing
python fast/test_train_arm.py         # 18 checks: resume is exact; an energy-score arm and a decoder-noise arm train
```

Both pass on this snapshot, run on a fresh CPU-only environment (torch 2.14, numpy 2.4, scipy 1.17).

The scorer and `sota/mean_of_M.py` were also run end to end on a synthetic set, where each high band is a
fresh draw of shaped noise at the true level, so a calibrated ensemble exists by construction. The scorer
returned:
- gap 2.50 dB (target 10 log10(2/(1+1/8)) = 2.50);
- spread 0.999 (target 1);
- PIT tails 0.109 and 0.125 (target 1/9 = 0.111).

The waveform mean of M draws lost 3.01, 6.02 and 9.03 dB at M = 2, 4 and 8, against a predicted
10 log10 M = 3.01, 6.02 and 9.03.

## Reproduce the paper

1. **Train.**
   ```bash
   fast/bootstrap.sh
   python fast/train_arm.py --arm <arm> --corpus <dir> --out <dir>
   ```
   Run this once per arm in `fast/run_contract.ARMS`. Then run `fast/convert_ckpt.py`.
2. **Evaluate.** `sota/run_all.sh`, with `V`, `P`, `Q` pointing at the three environments and `CKPT` at
   the converted checkpoints.
3. **Tables.** `python ../results/make_tables.py` rebuilds every table in the paper and the appendix from
   `../results/results.json`.

## Anonymisation and the changes made for this supplement

- Personal paths became `BWE_CACHE`.
- The notebook was renamed `notebook.ipynb`, and cache and Colab folder names were neutralised.
- An `import os` was added where the new path needed one.
- The original ad hoc chain scripts were replaced by `sota/run_all.sh`.
- `sota/mean_of_M.py` is new.
- No numerical code was changed.
