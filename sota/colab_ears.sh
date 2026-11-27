#!/bin/bash
# The EARS evaluation on a Colab GPU runtime, in stages. Each stage is safe to re-run (outputs are skipped).
#
#   !git clone -b <branch> https://github.com/Bromine185/lisa-rtm.git /content/lisa-rtm
#   !bash /content/lisa-rtm/sota/colab_ears.sh setup     # envs, released code and checkpoints   (~15 min)
#   !bash /content/lisa-rtm/sota/colab_ears.sh gate      # reproduce the Mac on 32 VCTK utterances (stops on mismatch)
#   !bash /content/lisa-rtm/sota/colab_ears.sh inputs    # EARS test speakers -> 96 utterances
#   !bash /content/lisa-rtm/sota/colab_ears.sh run       # 8 draws per utterance, every model
#   !bash /content/lisa-rtm/sota/colab_ears.sh score     # scorer, bootstrap, slopes, draw agreement
#   !bash /content/lisa-rtm/sota/colab_ears.sh pack      # results (JSON only) -> Drive
#
# Needs Drive mounted at /content/drive (the six Table 2 checkpoints in MyDrive/lisa_rtm/ckpt_OV50/<arm>.pt; results go to
# MyDrive/lisa_rtm/ears_cr/). Environments: P = released models (Python 3.11, torch 2.2.1, the Mac's sotaenv);
# Q = Colab's own Python for our arms and the scorer.
set -euo pipefail
REPO=/content/lisa-rtm
export BWE_CACHE=/content/cache PYTHONUNBUFFERED=1
P=/content/sotaenv/bin/python; Q="env CUDA_VISIBLE_DEVICES= python3"   # Q on CPU: the scorer forks workers, and our arms run on CPU as on the Mac
DRIVE=/content/drive/MyDrive/lisa_rtm; CKPT=$DRIVE/ckpt_OV50; OUTD=$DRIVE/ears_cr
SOTA=$BWE_CACHE/sota; LOG=$BWE_CACHE/logs; mkdir -p "$SOTA" "$LOG"
ARMS="det_paper det es_marg es_erb_l0.01 es_erb_l0.001 es_dec_erb_l0.1"   # the six arms of Table 2
OURS=""; for a in $ARMS; do OURS="$OURS ours_$a"; done
SPK="p102 p103 p104 p105 p106 p107"
cd "$REPO"

setup() {
  pip -q install uv gdown
  # P: the released models, pinned as on the Mac (sotaenv: Python 3.11, torch 2.2.1, numpy 1.23, librosa 0.9.2)
  [ -x $P ] || uv venv -q -p 3.11 /content/sotaenv
  uv pip install -q -p $P torch==2.2.1 torchaudio==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
  uv pip install -q -p $P numpy==1.23.5 scipy==1.10.1 librosa==0.9.2 soundfile omegaconf einops==0.8.0 \
      torchdiffeq==0.2.3 torchode==1.0.0 beartype==0.17.2 gateloop_transformer==0.2.4 accelerate==0.28.0 \
      torchinfo==1.8.0 matplotlib tqdm audiosr==0.0.7 transformers==4.30.2 "huggingface_hub<0.26" "setuptools<70"
  # Q: our arms and the scorer, on Colab's Python
  pip -q install soundfile pyarrow "visqol-python[lattice]" pesq torchmetrics
  # released code
  cd "$SOTA"
  [ -d FLowHigh_code ] || git clone -q https://github.com/jjunak-yun/FLowHigh_code.git
  [ -d nuwave2 ] || git clone -q https://github.com/maum-ai/nuwave2.git
  [ -d AP-BWE ] || git clone -q https://github.com/yxlu-0102/AP-BWE.git
  # checkpoints (public Drive links from each README)
  [ -f fh_ckpt/FLowHigh_indep_adaptive_400k.pt ] || { gdown -q --folder https://drive.google.com/drive/folders/1clsJ3bFTZSCLOb4I0wk0l6FL9Yh33Vof -O fh_dl
      mkdir -p fh_ckpt; find fh_dl -name 'FLowHigh_indep_adaptive_400k.pt' -exec mv {} fh_ckpt/ \; ; }
  [ -f nuwave2_ckpt.ckpt ] || gdown -q 11t0cQYx6ZadKQjmfGnqxUUH2UEk5Yzk7 -O nuwave2_ckpt.ckpt
  [ -f apbwe_ckpt/12kto48k/g_12kto48k ] || { gdown -q --folder https://drive.google.com/drive/folders/1IIYTf2zbJWzelu4IftKD6ooHloJ8mnZF -O apbwe_dl
      d=$(dirname "$(find apbwe_dl -name 'g_12kto48k' | head -1)"); mkdir -p apbwe_ckpt; cp -r "$d" apbwe_ckpt/12kto48k; }
  # the AudioSR weights download on first use (haoheliu/audiosr_speech on the Hub)
  ls -la fh_ckpt nuwave2_ckpt.ckpt apbwe_ckpt/12kto48k FLowHigh_code/vocoder/BIGVGAN/checkpoint
  sz=$(stat -c %s FLowHigh_code/vocoder/BIGVGAN/checkpoint/g_48_00850000)
  [ "$sz" -gt 1000000 ] || { echo "BigVGAN checkpoint is a $sz-byte pointer, not weights"; exit 1; }
  ls "$CKPT"/*.pt >/dev/null || { echo "our checkpoints missing in $CKPT"; exit 1; }
  nvidia-smi --query-gpu=name,memory.total --format=csv
  echo "setup: done"
}

gate() {
  # Edinburgh DataShare refuses Colab's addresses (HTTP 403), so the Mac's own 32 inputs come through Drive;
  # gate_check inputs still verifies them. Without the bundle, fetch from DataShare as the Mac did.
  if [ -f "$DRIVE/core_inputs.tar.gz" ]; then
    mkdir -p "$BWE_CACHE/sota/work" && tar -xzf "$DRIVE/core_inputs.tar.gz" -C "$BWE_CACHE/sota/work"
  else
    $Q sota/fetch_wide.py --per-speaker 30
    $Q sota/make_inputs.py --set core --per-speaker 4
  fi
  $Q sota/gate_check.py inputs
  $P sota/run_models.py --model apbwe --set core --seeds 1 --device cpu    # CPU, as on the Mac: on CUDA its per-utterance deficit drifts 0.02 dB
  $P sota/run_models.py --model flowhigh --set core --seeds 1 --device cuda
  $Q sota/score.py --set core --workers 4 --no-perceptual --models apbwe flowhigh
  $Q sota/gate_check.py models 2>&1 | tee "$LOG/gate.log"
}

inputs() {
  $Q sota/fetch_ears.py --per-speaker 16
  $Q sota/make_inputs.py --set ears --src ears --speakers $SPK --per-speaker 16
}

run() {
  $P sota/run_models.py --model apbwe         --set ears --seeds 1 --device cpu 2>&1 | tail -3
  for m in flowhigh flowhigh_std1 nuwave2 audiosr; do
    $P sota/run_models.py --model $m --set ears --seeds 8 --device cuda 2>&1 | tail -3
  done
  $Q sota/run_ours.py --set ears --seeds 8 --arms $ARMS --ckpt-dir "$CKPT" 2>&1 | tail -10
}

score() {
  $Q sota/score.py --set ears --workers "$(nproc)" --models flowhigh flowhigh_std1 nuwave2 audiosr apbwe $OURS \
      2>&1 | tee "$LOG/score_ears.log" | tail -60
  $Q sota/bootstrap.py --set ears 2>&1 | grep -v Warning | tail -20
  $Q sota/level_checks.py --set ears
  $Q sota/draw_agreement.py --set ears --models flowhigh flowhigh_std1 nuwave2 audiosr ours_es_dec_erb_l0.1
}

pack() {
  mkdir -p "$OUTD"; W=$BWE_CACHE/sota/work
  cp $W/ears/score_ears.json $W/ears/score_ears.md $W/ears/bootstrap_ears.json $W/ears/level_checks_ears.json \
     $W/ears/draw_agreement_ears.json $W/ears/utts.json $BWE_CACHE/ears_seg/manifest.json "$OUTD"/
  cp $W/core/score_core.json "$OUTD"/gate_score_core.json; cp "$LOG"/*.log "$OUTD"/
  for d in $W/ears/out/*/; do cp "$d/_run.json" "$OUTD/run_$(basename $d).json" 2>/dev/null || true; done
  ls -la "$OUTD"
}

"$@"
