#!/bin/bash
# The eight OV50 arms through the SOTA harness: the dissection of the deficit on the paper split, not p236 only.
set -u; cd "$(dirname "$0")/.."
V=/Users/raghavsharma/projects/lisa-rtm/venv/bin/python; Q=/Users/raghavsharma/projects/lisa-rtm/lisa_rtm_cache/visqolenv/bin/python
ARMS="det_paper det es_marg es_erb_l0.001 es_erb_l0.01 es_erb_l0.1 es_dec_l0.01 es_dec_erb_l0.1"
$V sota/run_ours.py --set wide --seeds 1 --arms $ARMS
$V sota/run_ours.py --set core --seeds 8 --arms $ARMS
$V sota/run_ours.py --set ourtest --seeds 16 --arms $ARMS
M=""; for a in $ARMS; do M="$M ours_$a"; done
$Q sota/score.py --set wide --workers 4 --no-perceptual --models flowhigh nuwave2 audiosr apbwe $M
$Q sota/score.py --set core --workers 4 --no-perceptual --models flowhigh nuwave2 audiosr apbwe $M
$Q sota/score.py --set ourtest --workers 3 --no-perceptual --models flowhigh apbwe $M
echo CHAIN4 DONE
