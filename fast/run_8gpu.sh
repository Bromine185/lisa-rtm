#!/usr/bin/env bash
# One 8-GPU node, eight arms, one process per GPU. Paste-and-go.
#
#   ./fast/run_8gpu.sh            # calibrate, stage, launch all eight
#   ./fast/run_8gpu.sh --status   # how far along, and what it is costing
#   ./fast/run_8gpu.sh --stop     # kill training (does NOT terminate the rental)
#
# WHY THIS IS SIMPLER THAN EIGHT SEPARATE RENTALS. The original plan put one arm on each of eight
# VMs, which meant staging the 37.3 h corpus eight times and a cross-machine barrier to prove all
# eight builds were identical. On one node with one filesystem, every process reads the SAME
# corpus files, so the G3 corpus digest is identical by construction and the barrier collapses to
# "stage once, then start". It costs $14.32/hr against $59.52 for separate rentals -- about $7 more
# for the whole run -- and removes eight download paths, eight failure points and the barrier.
#
# The arms still share one batch stream: every process calls stream(BATCH_TAG) with the same fixed
# constant from fast/run_contract.py, never a per-arm tag. That is what keeps the comparison paired.
set -euo pipefail

ROOT="${ROOT:-$HOME/lisa}"
CORPUS="${CORPUS:-$ROOT/corpus}"
RUN="${RUN:-$ROOT/run}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARMS=(det_paper det es_marg es_erb_l0.001 es_erb_l0.01 es_erb_l0.1 es_dec_l0.01 es_dec_erb_l0.1)
RATE_PER_HR=14.32

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

status() {
  echo "=== arms ==="
  for i in "${!ARMS[@]}"; do
    a="${ARMS[$i]}"; f="$RUN/$a.log"
    if [ -f "$f" ]; then
      last=$(grep -E "step [0-9]+/" "$f" | tail -1 || true)
      printf "  gpu%d %-18s %s\n" "$i" "$a" "${last:-starting...}"
    else
      printf "  gpu%d %-18s (no log yet)\n" "$i" "$a"
    fi
  done
  echo "=== gpus ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
  if [ -f "$RUN/.started" ]; then
    s=$(cat "$RUN/.started"); n=$(date +%s); h=$(echo "scale=3; ($n-$s)/3600" | bc)
    echo "=== cost ===";  echo "  running ${h} h  ->  \$$(echo "scale=2; $h*$RATE_PER_HR" | bc)"
  fi
}

case "${1:-}" in
  --status) status; exit 0;;
  --stop) pkill -f "fast/train_arm.py" && log "training stopped (rental still running -- terminate it in the console)"; exit 0;;
esac

mkdir -p "$ROOT" "$RUN"
cd "$REPO"

# ---- 1. torch, matched to the driver ------------------------------------------------------------
log "driver:"; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
# A Google Deep Learning VM image ships torch preinstalled -- typically a cu129 build whose arch
# list stops short of sm_120. "torch imports and sees a GPU" is therefore NOT the test: the test is
# whether it carries an sm_12x cubin, because release wheels ship SASS with no PTX fallback and a
# cu126/cu129 build imports fine, reports the GPU, then dies at the first kernel launch with
# "no kernel image is available".
have_sm12x() {
  python3 -c "
import sys
try:
    import torch
    ok = torch.cuda.is_available() and any(a.startswith('sm_12') for a in torch.cuda.get_arch_list())
except Exception:
    ok = False
sys.exit(0 if ok else 1)
" 2>/dev/null
}

if have_sm12x; then
  log "preinstalled torch already carries sm_12x kernels -- keeping it"
else
  python3 -c "import torch;print('  preinstalled:',torch.__version__,torch.cuda.get_arch_list())" 2>/dev/null \
    || log "  no usable torch present"
  WHEEL="$(python3 fast/calibrate.py --wheel-only || true)"
  case "$WHEEL" in
    torch==*) log "installing $WHEEL";;
    *) log "FATAL: no usable wheel for this driver ($WHEEL)"; exit 1;;
  esac
  python3 -m pip install -q --upgrade pip
  # shellcheck disable=SC2086
  python3 -m pip install -q --force-reinstall $WHEEL
  python3 -m pip install -q numpy scipy soundfile huggingface_hub pyarrow matplotlib
  have_sm12x || { log "FATAL: still no sm_12x cubin after installing $WHEEL"; exit 1; }
fi
python3 -c "
import torch
print(f'  torch {torch.__version__} cuda {torch.version.cuda} cc {torch.cuda.get_device_capability()} gpus {torch.cuda.device_count()}')
print(f'  arch {torch.cuda.get_arch_list()}')"

# ---- 2. calibrate on GPU 0. Cheap, and it is the only measurement of this card that exists. ------
if [ ! -f "$RUN/calibration.json" ]; then
  log "calibrating (~3 min) -- nothing here has ever run on sm_120"
  CUDA_VISIBLE_DEVICES=0 python3 fast/calibrate.py --json "$RUN/calibration.json" || \
    log "calibration reported a failure (continuing -- read $RUN/calibration.json)"
  python3 -c "
import json;d=json.load(open('$RUN/calibration.json'))
v=d.get('verdict',{});print('  VERDICT:',v.get('status','?'))
print('  factor',v.get('blackwell_factor'),' est step',v.get('est_step_ms_8arm'),'ms')" || true
fi

# ---- 3. stage the corpus ONCE. All eight processes read these same files. ------------------------
if [ ! -f "$CORPUS/manifest.json" ]; then
  log "staging corpus (~15 min, 384 vCPU)"
  python3 fast/stage_corpus.py --out "$CORPUS" --workers 64
fi
python3 -c "
import json;m=json.load(open('$CORPUS/manifest.json'))
print(f\"  corpus {m['n']} utts {m['hours']:.4f} h  g3 {m['g3']['corpus'][:16]}... / {m['g3']['batches'][:16]}...\")"

# ---- 4. eight arms, one per GPU -----------------------------------------------------------------
date +%s > "$RUN/.started"
for i in "${!ARMS[@]}"; do
  a="${ARMS[$i]}"
  if pgrep -f "train_arm.py --arm $a\b" >/dev/null 2>&1; then log "$a already running"; continue; fi
  log "gpu$i <- $a"
  CUDA_VISIBLE_DEVICES=$i nohup python3 fast/train_arm.py \
      --arm "$a" --corpus "$CORPUS" --out "$RUN" >> "$RUN/$a.log" 2>&1 &
  sleep 2
done

log "all eight launched. monitor with:"
echo "    ./fast/run_8gpu.sh --status"
echo "    tail -f $RUN/es_erb_l0.1.log"
echo
log "training is resumable: if anything dies, re-run this script and it continues from the last checkpoint"
log "REMEMBER to terminate the rental in the console when done -- it bills \$$RATE_PER_HR/hr regardless"
