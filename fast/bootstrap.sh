#!/usr/bin/env bash
# Runs ON a freshly rented instance. Installs the right wheel, stages the corpus, publishes the G3
# digest, then waits at a barrier until the launcher says every instance agrees.
#
#   ./fast/bootstrap.sh --arm es_erb_l0.1 [--calibrate-only] [--skip-stage]
#
# WHY THE BARRIER. Eight VMs each build their own corpus from their own download with no shared
# filesystem. Identical batch indices into different corpora are different data, so each instance
# writes its digest to $RUN/g3.json and then BLOCKS on $RUN/GO, which only appears after the
# launcher has compared all eight. An instance that trains before the comparison is an instance
# whose results cannot be trusted, so waiting is the safe default and the timeout aborts rather
# than proceeding.
set -euo pipefail

ARM=""; CALIBRATE_ONLY=0; SKIP_STAGE=0
RUN="${RUN:-/data/run}"; CORPUS="${CORPUS:-/data/corpus}"; REPO="${REPO:-$HOME/lisa-rtm}"
BARRIER_TIMEOUT="${BARRIER_TIMEOUT:-3600}"
while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM="$2"; shift 2;;
    --calibrate-only) CALIBRATE_ONLY=1; shift;;
    --skip-stage) SKIP_STAGE=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
cd "$REPO"
mkdir -p "$RUN"

# ---- 1. the wheel. One source of truth: calibrate.py's own driver table, which imports nothing. ----
log "driver check"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader || {
  log "FATAL: no nvidia-smi"; exit 1; }
WHEEL="$(python3 fast/calibrate.py --wheel-only || true)"
case "$WHEEL" in
  torch==*) log "wheel: $WHEEL";;
  *) log "FATAL: no usable wheel for this driver ($WHEEL)"; exit 1;;
esac

if ! python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  log "installing torch"
  python3 -m pip install --quiet --upgrade pip
  # shellcheck disable=SC2086
  python3 -m pip install --quiet $WHEEL
  python3 -m pip install --quiet numpy scipy soundfile huggingface_hub pyarrow matplotlib
fi

# Release wheels ship SASS only -- no PTX, no JIT fallback -- so a missing sm_12x cubin is fatal
# rather than merely slow. Fail here, before staging burns 20 minutes of instance time.
python3 - <<'PY' || { echo "FATAL: no sm_12x cubin in this wheel"; exit 1; }
import sys, torch
al = torch.cuda.get_arch_list()
cc = torch.cuda.get_device_capability()
print(f"torch {torch.__version__} cuda {torch.version.cuda} cc {cc} arch_list {al}")
sys.exit(0 if any(a.startswith("sm_12") for a in al) else 1)
PY

# ---- 2. calibrate: cheap, and it decides whether the long run is affordable at all ----
log "calibrating"
python3 fast/calibrate.py --json "$RUN/calibration.json" || log "calibration reported a failure (kept going)"
[ "$CALIBRATE_ONLY" = "1" ] && { log "calibrate-only: done"; exit 0; }

[ -n "$ARM" ] || { log "FATAL: --arm is required unless --calibrate-only"; exit 2; }

# ---- 3. stage the corpus and publish the digest ----
if [ "$SKIP_STAGE" = "0" ] && [ ! -f "$CORPUS/manifest.json" ]; then
  log "staging corpus to $CORPUS"
  python3 fast/stage_corpus.py --out "$CORPUS" --workers "$(nproc)"
fi
python3 - <<PY > "$RUN/g3.json"
import json, pathlib
m = json.loads(pathlib.Path("$CORPUS/manifest.json").read_text())
print(json.dumps({"arm": "$ARM", "g3": m["g3"], "hours": m["hours"], "n": m["n"]}))
PY
log "published g3: $(cat "$RUN/g3.json")"

# ---- 4. barrier: do not train until the launcher confirms all instances agree ----
log "waiting for $RUN/GO (timeout ${BARRIER_TIMEOUT}s)"
waited=0
while [ ! -f "$RUN/GO" ]; do
  if [ -f "$RUN/ABORT" ]; then log "FATAL: launcher signalled ABORT"; cat "$RUN/ABORT"; exit 1; fi
  if [ "$waited" -ge "$BARRIER_TIMEOUT" ]; then
    log "FATAL: barrier timed out after ${waited}s -- refusing to train unverified"; exit 1
  fi
  sleep 10; waited=$((waited + 10))
done
log "barrier released after ${waited}s"

# ---- 5. train. Resumable, so a restart of this script continues rather than starting over. ----
log "training $ARM"
exec python3 fast/train_arm.py --arm "$ARM" --corpus "$CORPUS" --out "$RUN"
