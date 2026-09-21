# OV50 results

Copied out of `/Users/raghavsharma/lisa-results` by `fast/publish_results.py` on 2026-09-21. Read them with `notes/2026-09-21-eval-results.md`.

| file | what |
|---|---|
| `manifest.json` | corpus fingerprint, g3 digests, test speakers |
| `calibration.json` | calibration probe from the run |
| `history_OV50_det.json` | det: every logged train and val point |
| `history_OV50_det_paper.json` | det_paper: every logged train and val point |
| `history_OV50_es_dec_erb_l0.1.json` | es_dec_erb_l0.1: every logged train and val point |
| `history_OV50_es_dec_l0.01.json` | es_dec_l0.01: every logged train and val point |
| `history_OV50_es_erb_l0.001.json` | es_erb_l0.001: every logged train and val point |
| `history_OV50_es_erb_l0.01.json` | es_erb_l0.01: every logged train and val point |
| `history_OV50_es_erb_l0.1.json` | es_erb_l0.1: every logged train and val point |
| `history_OV50_es_marg.json` | es_marg: every logged train and val point |
| `det.log` | det: trainer log |
| `det_paper.log` | det_paper: trainer log |
| `es_dec_erb_l0.1.log` | es_dec_erb_l0.1: trainer log |
| `es_dec_l0.01.log` | es_dec_l0.01: trainer log |
| `es_erb_l0.001.log` | es_erb_l0.001: trainer log |
| `es_erb_l0.01.log` | es_erb_l0.01: trainer log |
| `es_erb_l0.1.log` | es_erb_l0.1: trainer log |
| `es_marg.log` | es_marg: trainer log |
| `results_OV50.json` | EVAL12: every metric, every arm, every condition |
| `table_OV50.md` | EVAL12 table |
| `val_wave_OV50.json` | val_wave verification: seeds, AMP, bands, residuals |
| `val_wave_decompose_OV50.json` | val_wave split into single-draw L1 and spread, per seed |
| `visqol_OV50.json` | ViSQOL / PESQ per condition |
| `visqol_table_OV50.md` | ViSQOL table |
| `latency_OV50.json` | inference latency per arm, fp32 eager on the eval laptop |
| `DATASHEET_OV50.md` | MASTER: every comparative metric, one row per arm |
| `ov3_spectrum_OV50.png` | energy ratio per band |
| `ov3_calibration_OV50.png` | PIT histograms and spread-skill |
| `train_val_OV50.png` | per-arm training and validation curves |
| `cross_arm_OV50.png` | val_wave, val_spec and spread on shared axes |
| `endgame_OV50.png` | the last 12k steps, linear axes |

## Not here

- **`<arm>.pt`** -- the eight resume checkpoints (9.9 MB) and their converted forms
  (4.4 MB). `fast/publish_results.py --checkpoints` copies the converted ones. The
  resume checkpoints are currently single-copy: the instance is terminated.
- **`ov3/audio/`** -- 48 MB of wav, rebuildable in about three minutes with
  `fast/run_e4_eval.py`.
- **`train_index.json`** -- 6.9 MB corpus index, a cache;
  `fast/val_wave_check.py` rebuilds it and gates it against `manifest.json`.

## Reproducing

```bash
python fast/convert_ckpt.py   --src ~/lisa-results
python fast/run_e4_eval.py    --src ~/lisa-results
python fast/val_wave_check.py --src ~/lisa-results --seeds 1234,1,2,3 --amp
python fast/run_e5_visqol.py  --src ~/lisa-results   # needs a separate install
python fast/publish_results.py --src ~/lisa-results
```
