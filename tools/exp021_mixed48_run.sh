#!/usr/bin/env bash
# EXP-021 mixed48 quant runner.
#
# SAFETY CONTRACT: unlike the older autokick/run_gptq_qwen38.sh scripts, this runner does
# NOT stop or restart prod (vllm-qwen27b.service) itself. That is the overnight launcher's
# job. This script's only safety net is a hard abort if it finds prod's engine workers still
# running -- it must never fight the live service for VRAM.
#
# Usage:
#   ./exp021_mixed48_run.sh            launch the real run (nohup, backgrounded, logged)
#   ./exp021_mixed48_run.sh --dry-run  config-only sanity check (foreground, no GPU, seconds)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUANT_PY="$HERE/exp021_mixed48_quant.py"
QUANT_ENV=/home/kevin/quant-env
LOG=/home/kevin/Desktop/qwythos-quant/exp021_mixed48.log
PIDFILE=/tmp/exp021_mixed48.pid
MODELS=/home/kevin/Desktop/models
OUT="$MODELS/Qwen3.8-27B-GPTQ-mixed48"

log(){ echo "$(date -u +%H:%M:%SZ) $*"; }

if [[ ! -f "$QUANT_PY" ]]; then
  log "ABORT: quant script not found at $QUANT_PY"
  exit 1
fi
if [[ ! -f "$QUANT_ENV/bin/activate" ]]; then
  log "ABORT: quant-env not found at $QUANT_ENV"
  exit 1
fi

# shellcheck disable=SC1091
source "$QUANT_ENV/bin/activate"

if [[ "${1:-}" == "--dry-run" ]]; then
  log "dry-run: config-only sanity, no GPU, exits in seconds"
  exec python "$QUANT_PY" --dry-run
fi

# ---------- safety: prod must already be stopped by the overnight launcher ----------
if pgrep -f 'VLLM::' >/dev/null 2>&1; then
  log "ABORT: VLLM:: worker processes are running -- prod is still up."
  log "        The overnight launcher is responsible for stopping prod (sudo systemctl"
  log "        stop vllm-qwen27b) and confirming both GPUs are idle BEFORE invoking this"
  log "        runner. This script will not fight prod for VRAM."
  pgrep -af 'VLLM::' 2>/dev/null
  exit 1
fi
log "safety check passed: no VLLM:: processes running"

mkdir -p "$(dirname "$LOG")"

# ---------- duration estimate ----------
# Provenance: the pure-int4 baseline run for this exact source model measured 148 min
# quantize() wall time + ~15 min load/tokenize/save overhead = ~2h29m total, on this box,
# with N=512/SEQ=1024/batch_size=1/offload_to_disk=True (2026-08-14 run; log at
# /home/kevin/Desktop/qwythos-quant/gptq38.log, timestamps 15:14:17Z -> 17:43:13Z).
# mixed48 runs the identical calibration loop over the identical 64 layers -- the only
# difference is 64 of ~1200 tensors pack at int8 instead of int4, and a handful more are
# skipped entirely (cheaper than quantizing them). That's noise against the Hessian-capture
# cost that dominates GPTQ wall time. Expect the same ballpark, call it 2.5-4h with margin.
log "estimated duration: ~2.5-4h"
log "  (baseline: 148 min quantize() + ~15 min overhead for the pure-int4 recipe on"
log "   2026-08-14, this exact source model, same N/SEQ/batch_size/offload settings)"

log "launching EXP-021 mixed48 quant (nohup, backgrounded) -- log: $LOG"
nohup python "$QUANT_PY" >>"$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
log "launched pid=$PID (pidfile: $PIDFILE)"
log "follow progress:  tail -f $LOG"
log "on completion, output lands at: $OUT"
log "NEXT (after completion): bench vs current champion (Qwen3.8-27B-GPTQ-Int4) before"
log "  swapping prod -- do not point serve script at $OUT unsupervised."
