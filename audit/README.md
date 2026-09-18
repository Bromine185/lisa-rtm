# audit — protocol checks that need no GPU and no training

Scripts that produce every number in
[`notes/2026-09-17-lisa-reported-numbers-audit.md`](../notes/2026-09-17-lisa-reported-numbers-audit.md)
and [`notes/2026-09-17-snr-ceiling-gating-and-scale.md`](../notes/2026-09-17-snr-ceiling-gating-and-scale.md).
They exist because three overnight runs were spent trying to reproduce a published number that is
above the task's information ceiling. Reading the evaluation code would have cost minutes.

Run them from the repo root, with the venv the root README builds — every path here resolves from
the script's own location, so a clone anywhere works.

```bash
venv/bin/python audit/lisa_paper_protocol.py                 # no checkpoint needed
venv/bin/python audit/snr_scale.py                           # no checkpoint needed
venv/bin/python audit/two_test_sets.py                       # no checkpoint needed
venv/bin/python audit/band_gating.py   <arm.pt> [utt.flac]
venv/bin/python audit/scale_freedom.py <arm.pt> [utt.flac]
venv/bin/python audit/lisa_in_their_code.py --lisa <clone of ml-postech/LISA>
```

`two_test_sets.py` additionally needs `huggingface_hub` and `pyarrow` (the Hub route's own
dependencies); `lisa_in_their_code.py` needs `torchaudio` and a clone of their repo. Everything else
runs on the README's venv.

| file | what it answers |
|---|---|
| `vctk_fixtures.py` | pulls a few VCTK 0.92 utterances out of the 11 GB archive over HTTP ranges, into `lisa_rtm_cache/audit/` (override with `AUDIT_DATA`) |
| `boot.py` | execs the notebook's model/metric cells plus `overnight2/c1_model.py` and `overnight3/e1_model.py` into one namespace, CPU, `FULL` config |
| `lisa_paper_protocol.py` | reimplements torchaudio 0.6.0's `kaldi.resample_waveform` and LISA's `calc_snr` / `compute_log_distortion` from source, then measures the protocol's ceiling, the variance of its single-batch evaluation, and how much of its LSD is pinned at the epsilon floor |
| `snr_scale.py` | SNR as a function of the coherently recovered fraction of the high band; shows naive upsampling *is* the zero-recovery optimum |
| `band_gating.py` | per-third-octave shape, the deficit split by frame loudness (the gating failure), and what a low-pass costs |
| `scale_freedom.py` | whether a decoder trained only at ×4 generalises off its four trained coordinate phases |
| `two_test_sets.py` | the Hub and DataShare acquisition routes, paired by FLAC basename: are they the same audio, the same utterances, or only the same label? Model-free, because the disagreement already shows up in naive upsampling (19.48 dB vs 19.20 dB) |
| `lisa_in_their_code.py` | §9 of the reported-numbers note: the same ceiling measurement, but calling ml-postech/LISA's own `calc_snr` and `compute_log_distortion` and torchaudio's kaldi resampler instead of our transcriptions of them |

The two protocol scripts depend on nothing in this repo's model code, by design: if our own
implementation were wrong, the ceiling argument should still stand. `lisa_in_their_code.py` goes one
step further and depends on nothing of ours at all for the metrics: it imports theirs. Agreement
between it and `lisa_paper_protocol.py` is what retires the reimplementation caveat in §7 of the
note; disagreement is a bug in ours, and the note has to be corrected.
