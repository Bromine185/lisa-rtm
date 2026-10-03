#!/bin/bash
cd "$(dirname "$0")/.."; P=/Users/raghavsharma/projects/lisa-rtm/lisa_rtm_cache/sota/sotaenv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
$P sota/run_models.py --model nuwave2 --set demo --seeds 2 --device cpu
$P sota/run_models.py --model audiosr --set demo --seeds 2 --device mps
echo CHAIN6B DONE
