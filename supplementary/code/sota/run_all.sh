#!/bin/bash
# The whole evaluation behind Tables 1-2 of the paper, in the order it was run, one model at a time
# (AudioSR alone peaks near 15 GB). Replaces the original ad hoc chain scripts; the commands are theirs.
#
# Three Python environments (see code/README.md):
#   V  project env   (torch, numpy, scipy, soundfile)             -- our arms, inputs, ceiling, figures
#   P  sota env      (torch 2.2.1, numpy 1.23, librosa 0.9.2)     -- the four released models, each in its own repo
#   Q  scoring env   (project env + visqol-python[lattice], pesq, torchaudio, torchmetrics)
# Released code and checkpoints go under $BWE_CACHE/sota (FLowHigh_code, fh_ckpt, nuwave2, nuwave2_ckpt.ckpt,
# AP-BWE, AudioSR via pip); our converted checkpoints under $CKPT (fast/convert_ckpt.py output).
set -euo pipefail
cd "$(dirname "$0")/.."
export BWE_CACHE="${BWE_CACHE:-$PWD/cache}" PYTORCH_ENABLE_MPS_FALLBACK=1
V="${V:-python}"; P="${P:-python}"; Q="${Q:-python}"; CKPT="${CKPT:-$HOME/results/ckpt/OV50}"
DEV_AUDIOSR="${DEV_AUDIOSR:-cpu}"      # mps on Apple silicon, cuda where available
ARMS="det_paper det es_marg es_erb_l0.001 es_erb_l0.01 es_erb_l0.1 es_dec_l0.01 es_dec_erb_l0.1"
OURS=""; for a in $ARMS; do OURS="$OURS ours_$a"; done

# 1. test inputs: VCTK 0.92 mic1, the NU-Wave 2 / FLowHigh test speakers, Chebyshev input at 12 kHz
$V sota/fetch_wide.py --per-speaker 30
$V sota/make_inputs.py --set wide --per-speaker 30                                   # 240 utterances
$V sota/make_inputs.py --set core --per-speaker 4                                    #  32 utterances
$V sota/make_inputs.py --set ourtest --src audit --speakers p236 p237 p238 --per-speaker 11

# 2. the empty-band SNR ceiling at 8/12/16/24 kHz input
$V sota/ceiling.py --set wide

# 3. draws: 8 per utterance on core (calibration), 1 on wide (levels), 16 of ours on ourtest
for set in core wide; do
  seeds=$([ $set = core ] && echo 8 || echo 1)
  $P sota/run_models.py --model apbwe         --set $set --seeds 1 --device cpu
  $P sota/run_models.py --model apbwe_sinc    --set $set --seeds 1 --device cpu
  $P sota/run_models.py --model flowhigh      --set $set --seeds $seeds --device cpu
  $P sota/run_models.py --model flowhigh_std1 --set $set --seeds $seeds --device cpu
  $P sota/run_models.py --model audiosr       --set $set --seeds $seeds --device "$DEV_AUDIOSR"
  $P sota/run_models.py --model nuwave2       --set $set --seeds $seeds --device cpu
  $V sota/run_ours.py --set $set --seeds $seeds --arms $ARMS --ckpt-dir "$CKPT"
done
$P sota/run_models.py --model apbwe    --set ourtest --device cpu
$P sota/run_models.py --model flowhigh --set ourtest --seeds 1 --device cpu
$V sota/run_ours.py --set ourtest --seeds 16 --arms $ARMS --ckpt-dir "$CKPT"

# 4. one scorer for every model, with the perceptual metrics
$Q sota/score.py --set core    --workers 4 --models flowhigh flowhigh_std1 nuwave2 audiosr apbwe apbwe_sinc $OURS
$Q sota/score.py --set wide    --workers 4 --models flowhigh flowhigh_std1 nuwave2 audiosr apbwe apbwe_sinc $OURS
$Q sota/score.py --set ourtest --workers 3 --models flowhigh apbwe $OURS

# 5. the mean-of-M table (the paper's "averaging rebuilds the deficit"); needs >= 16 draws for M = 16
$V sota/mean_of_M.py --set ourtest --model ours_es_dec_erb_l0.1 --Ms 1 2 4 8 16

# 6. collect into one results file, then figures
$V demo/tools/make_results_sota.py --results ../results/results.json
$V sota/figs.py
echo "run_all: done"
