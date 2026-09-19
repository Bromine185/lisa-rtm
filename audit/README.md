# audit — protocol checks that need no GPU and no training

The scripts that produce every number in
[`notes/2026-09-17-lisa-reported-numbers-audit.md`](../notes/2026-09-17-lisa-reported-numbers-audit.md),
[`notes/2026-09-17-snr-ceiling-gating-and-scale.md`](../notes/2026-09-17-snr-ceiling-gating-and-scale.md)
and [`paper/bwe-information-ceiling.md`](../paper/bwe-information-ceiling.md).
They exist because three overnight runs were spent trying to reproduce a published number that is
above the task's information ceiling. Reading the evaluation code would have cost minutes.

```bash
venv/bin/python audit/lisa_paper_protocol.py                 # no checkpoint needed
venv/bin/python audit/snr_scale.py                           # no checkpoint needed
venv/bin/python audit/protocol_fork.py                       # no checkpoint needed
venv/bin/python audit/band_gating.py   <arm.pt> [utt.flac]
venv/bin/python audit/scale_freedom.py <arm.pt> [utt.flac]
```

| file | what it answers |
|---|---|
| `vctk_fixtures.py` | pulls a few VCTK 0.92 utterances out of the 11 GB archive over HTTP ranges, into `lisa_rtm_cache/audit/` (override with `AUDIT_DATA`) |
| `boot.py` | execs the notebook's model/metric cells plus `overnight2/c1_model.py` and `overnight3/e1_model.py` into one namespace, CPU, `FULL` config |
| `lisa_paper_protocol.py` | reimplements torchaudio 0.6.0's `kaldi.resample_waveform` and LISA's `calc_snr` / `compute_log_distortion` from source, then measures the protocol's ceiling, the variance of its single-batch evaluation, and how much of its LSD is pinned at the epsilon floor |
| `snr_scale.py` | SNR as a function of the coherently recovered fraction of the high band; shows naive upsampling *is* the zero-recovery optimum |
| `protocol_fork.py` | what dropping the anti-alias filter does: 2.15 % of the band above 6 kHz survives an anti-aliased decimation against 100.1 % of an aliased one. Fits the same per-bin linear unfolder under both protocols, over all eight rotations of a 4/4 speaker split, because one split spans 1 % to 21 % |
| `band_gating.py` | per-third-octave shape, the deficit split by frame loudness (the gating failure), and what a low-pass costs |
| `scale_freedom.py` | whether a decoder trained only at ×4 generalises off its four trained coordinate phases |

The three no-checkpoint scripts depend on nothing in this repo's model code, by design: if our own
implementation were wrong, the ceiling argument should still stand. They need only numpy, scipy,
soundfile and requests, and they fetch their own ~15 MB of VCTK over HTTP ranges on first run.
