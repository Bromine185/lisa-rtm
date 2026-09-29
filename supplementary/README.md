# Supplementary material

**The Missing Band Is a Distribution: Bandwidth Extension as a Sampling Problem** (anonymous submission)

Start with `technical_appendix.pdf` (16 pages). The appendix sets out:
- the protocol, the model and the training;
- exact metric definitions and the derivations;
- the audit of the four released models;
- every metric for every model on every test set;
- notes on the main text, and the limitations.

To hear the difference, open `audio/` (FLAC; `audio/README.md` lists what each file is).

## Contents

| Path | What it is |
|---|---|
| `technical_appendix.pdf` | The technical appendix |
| `appendix/` | Its LaTeX source; `build.sh` rebuilds the tables and the PDF |
| `results/results.json` | Every number, in one file. `sota` holds the evaluation behind both tables of the paper |
| `results/make_tables.py` | Rebuilds every table (CSV and LaTeX) from `results.json`, and checks each number printed in the paper against it |
| `results/number_check.md` | That check: 61 of 67 printed numbers match exactly. The six differences are listed, with the values to print |
| `results/tables/` | The generated tables |
| `results/training/` | Per-arm training histories (every logged train and validation point) and trainer logs, for all eight arms |
| `figures/` | High-band level per band, rank histograms, spread, the SNR ceiling, quality against cost, training curves |
| `audio/` | Three utterances from speakers none of our arms, FLowHigh, NU-Wave 2 or AP-BWE saw in training (AudioSR may have seen them). Each has input, truth, both deterministic baselines, three draws of our sampler, the mean of 16 draws, and all four released models (two draws each for the stochastic ones) |
| `code/` | The code that produced the numbers: training, the released-model harness, the scorer, and CPU tests that pass on this snapshot. See `code/README.md` |

## Checking the numbers without a GPU

```bash
python3 results/make_tables.py        # standard library only; rewrites results/tables/ and results/number_check.md
cd code && python fast/test_run_contract.py && python fast/test_train_arm.py    # needs torch (CPU), numpy, scipy
```

## Data and licences

- **VCTK 0.92** (CC BY 4.0) is not redistributed, apart from the three short test utterances in `audio/`.
  The scripts fetch it: training through the Hugging Face copy, test utterances from the Edinburgh
  DataShare archive.
- **Released models.** FLowHigh (MIT), NU-Wave 2 (BSD), AP-BWE (MIT) and AudioSR (MIT) were run from their
  official code and checkpoints, which are not included.
- **Our code** is released under CC0.

`build_zip.sh` packages this folder as one zip under the 20 MB limit.
