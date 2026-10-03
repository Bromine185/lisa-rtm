#!/bin/bash
# The whole inference run, strictly one model at a time (AudioSR alone peaks near 15 GB on a 16 GB Mac).
# core: 32 utterances x 8 draws for the calibration test; wide: 240 utterances x 1 draw for levels and SNR.
set -u
cd "$(dirname "$0")/.."
P=/Users/raghavsharma/projects/lisa-rtm/lisa_rtm_cache/sota/sotaenv/bin/python
V=/Users/raghavsharma/projects/lisa-rtm/venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
for set in core wide; do
  seeds=$([ $set = core ] && echo 8 || echo 1)
  $P sota/run_models.py --model apbwe    --set $set --seeds 1 --device cpu
  $V sota/run_ours.py   --set $set --seeds $seeds
  $P sota/run_models.py --model flowhigh --set $set --seeds $seeds --device cpu
  $P sota/run_models.py --model audiosr  --set $set --seeds $seeds --device mps
  $P sota/run_models.py --model nuwave2  --set $set --seeds $seeds --device cpu
done
echo CHAIN DONE
