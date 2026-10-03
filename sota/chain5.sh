#!/bin/bash
# After chain4: rescore wide and core WITH the perceptual metrics for every model, so the JSONs behind the
# paper's tables hold ViSQOL/PESQ for the eight OV50 arms too (chain4 scored them --no-perceptual).
cd "$(dirname "$0")/.."
while ! grep -q "CHAIN4 DONE" /Users/raghavsharma/projects/lisa-rtm/lisa_rtm_cache/sota/chain4.log; do sleep 30; done
Q=/Users/raghavsharma/projects/lisa-rtm/lisa_rtm_cache/visqolenv/bin/python
M="ours_det_paper ours_det ours_es_marg ours_es_erb_l0.001 ours_es_erb_l0.01 ours_es_erb_l0.1 ours_es_dec_l0.01 ours_es_dec_erb_l0.1"
$Q sota/score.py --set core --workers 4 --models flowhigh flowhigh_std1 nuwave2 audiosr apbwe apbwe_sinc $M
$Q sota/score.py --set wide --workers 4 --models flowhigh flowhigh_std1 nuwave2 audiosr apbwe apbwe_sinc $M
$Q sota/score.py --set ourtest --workers 3 --models flowhigh apbwe $M
echo CHAIN5 DONE
