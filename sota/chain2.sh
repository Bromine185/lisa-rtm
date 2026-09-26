#!/bin/bash
# After chain.sh: FLowHigh with the prior its training used (std_1 = std_2 = 1).
cd "$(dirname "$0")/.."
while pgrep -f sota/chain.sh > /dev/null; do sleep 60; done
P=/Users/raghavsharma/projects/lisa-rtm/lisa_rtm_cache/sota/sotaenv/bin/python
$P sota/run_models.py --model flowhigh_std1 --set core --seeds 8 --device cpu
$P sota/run_models.py --model flowhigh_std1 --set wide --seeds 1 --device cpu
echo CHAIN2 DONE
