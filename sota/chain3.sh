#!/bin/bash
# p236-p238 (our held-out speakers, EVAL12's) through the same harness: Chebyshev input and the repo's poly input.
set -u; cd "$(dirname "$0")/.."
V=/Users/raghavsharma/projects/lisa-rtm/venv/bin/python; P=/Users/raghavsharma/projects/lisa-rtm/lisa_rtm_cache/sota/sotaenv/bin/python
Q=/Users/raghavsharma/projects/lisa-rtm/lisa_rtm_cache/visqolenv/bin/python
$V sota/make_inputs.py --set ourtest --src audit --speakers p236 p237 p238 --per-speaker 11
$V sota/make_inputs.py --set ourtest_poly --src audit --speakers p236 p237 p238 --per-speaker 11 --resampler poly
for s in ourtest ourtest_poly; do $V sota/run_ours.py --set $s --seeds 16; done
$P sota/run_models.py --model apbwe --set ourtest --device cpu
$P sota/run_models.py --model flowhigh --set ourtest --seeds 1 --device cpu
$Q sota/score.py --set ourtest --workers 3 --models flowhigh apbwe ours_es_dec_erb_l0.1 ours_det
$Q sota/score.py --set ourtest_poly --workers 3 --models ours_es_dec_erb_l0.1 ours_det
echo CHAIN3 DONE
