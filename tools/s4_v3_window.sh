#!/usr/bin/env bash
# EXP-039 v3 window driver — arms A/B/C per docs/exp039-v3-gpu-merge.md §8.
# A = v2 CPU-list merge (async OFF, the proven baseline)
# B = v3 GPU merge   (async AUTO-ON, the target)
# C = v3 GPU merge   (async forced OFF — isolates merge overhead from async gain)
#
# Maintenance-window only: stops prod (+watchdog), restores both on exit (trap).
set -uo pipefail

WT=/home/kevin/Desktop/.ftree-s4drafter
PY=/home/kevin/Desktop/vLLM-2080Ti-Definitive/.venv/bin/python
MODEL=/home/kevin/Desktop/models/Qwen3.8-27B-GPTQ-Int4
SHARE=/home/kevin/.local/share/vllm-qwen27b
OUT=/tmp/s4v3-window
mkdir -p "$OUT"

restore_prod() {
  echo "[window] restoring prod ..."
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  sleep 8
  sudo systemctl start vllm-qwen27b.service
  for i in $(seq 1 90); do
    curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1 && break; sleep 10
  done
  sudo systemctl start vllm-qwen27b-watchdog.timer
  echo "[window] prod restored: $(curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null && echo HEALTHY || echo NOT-HEALTHY)"
}
trap restore_prod EXIT

echo "[window] taking engine ownership (watchdog off, prod down)"
sudo systemctl stop vllm-qwen27b-watchdog.timer
sudo systemctl stop vllm-qwen27b.service
sleep 5

launch_arm() {  # $1=arm-name  $2=VLLM_S4_GPU_MERGE  $3=extra-flag ("" or --no-async-scheduling)
  local arm=$1 gpumerge=$2 extra=${3:-}
  echo "[arm $arm] launching (gpu_merge=$gpumerge extra='$extra')"
  env VLLM_S4_SCOPED_DRAFTER=1 VLLM_S4_GPU_MERGE=$gpumerge \
      VLLM_MTP_DRAFT_CAP=2 VLLM_S4_K_SCOPED=16 VLLM_S4_G=12 VLLM_S4_MIN_UNIQ=1 \
      VLLM_QWOPUS_MTP_BF16_DRAFT=1 VLLM_SUFFIX_OVERLAY=0 VLLM_S4_LOG_EVERY=200 \
      PYTHONPATH=$WT \
    "$PY" -m vllm.entrypoints.openai.api_server \
      --host 127.0.0.1 --port 8001 \
      --model "$MODEL" --served-model-name qwen-local \
      --dtype half --tensor-parallel-size 2 \
      --generation-config "$SHARE/gencfg" \
      --quantization gptq_marlin \
      --gpu-memory-utilization 0.75 \
      --speculative-config '{"method":"mtp","num_speculative_tokens":16}' \
      --max-model-len 65536 --max-num-seqs 4 --max-num-batched-tokens 3968 \
      --mamba-cache-mode align \
      --enable-prefix-caching \
      --language-model-only --skip-mm-profiling \
      --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}' \
      $extra \
      > "$OUT/engine_$arm.log" 2>&1 &
  ENGINE_PID=$!
  for i in $(seq 1 120); do
    curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1 && return 0
    kill -0 $ENGINE_PID 2>/dev/null || { echo "[arm $arm] ENGINE DIED at boot:"; tail -25 "$OUT/engine_$arm.log"; return 1; }
    sleep 10
  done
  echo "[arm $arm] HEALTH TIMEOUT"; tail -25 "$OUT/engine_$arm.log"; return 1
}

kill_arm() {
  kill "$ENGINE_PID" 2>/dev/null || true
  for i in $(seq 1 30); do kill -0 "$ENGINE_PID" 2>/dev/null || break; sleep 2; done
  kill -9 "$ENGINE_PID" 2>/dev/null || true
  sleep 8
}

check_async() {  # $1=arm  $2=expected substring in engine log
  if grep -q "$2" "$OUT/engine_$1.log"; then
    echo "[arm $1] async check OK: '$2'"
  else
    echo "[arm $1] WARN: expected '$2' not found in engine log"
    grep -i "async scheduling" "$OUT/engine_$1.log" | head -3
  fi
}

run_bench() {  # $1=arm  $2=label
  "$PY" "$WT/tools/s4_replay_bench.py" --base-url http://127.0.0.1:8001 --model qwen-local \
      --label "$2" --reps 3 --warmup 1 --out "$OUT/s4_$1.json" 2>&1 | tail -12
}

declare -A ARM_MERGE=( [A]=0 [B]=1 [C]=1 )
declare -A ARM_EXTRA=( [A]="" [B]="" [C]="--no-async-scheduling" )
declare -A ARM_LABEL=( [A]=A_v2_asyncoff [B]=B_v3_asyncon [C]=C_v3_asyncoff )
declare -A ARM_EXPECT=( [A]="Async scheduling disabled" [B]="Asynchronous scheduling is enabled" [C]="Async" )

for arm in A B C; do
  if launch_arm "$arm" "${ARM_MERGE[$arm]}" "${ARM_EXTRA[$arm]}"; then
    check_async "$arm" "${ARM_EXPECT[$arm]}"
    grep -i "scoped-reemission drafter ENABLED" "$OUT/engine_$arm.log" | head -1
    run_bench "$arm" "${ARM_LABEL[$arm]}"
  else
    echo "[arm $arm] SKIPPED (boot failure)"
  fi
  kill_arm
done

echo "=== COMPARE A vs B ==="
[ -f "$OUT/s4_A.json" ] && [ -f "$OUT/s4_B.json" ] && \
  "$PY" "$WT/tools/s4_replay_bench.py" --compare "$OUT/s4_A.json" "$OUT/s4_B.json"
echo "=== COMPARE C vs B (async recovery) ==="
[ -f "$OUT/s4_C.json" ] && [ -f "$OUT/s4_B.json" ] && \
  "$PY" "$WT/tools/s4_replay_bench.py" --compare "$OUT/s4_C.json" "$OUT/s4_B.json"
echo "=== WINDOW DONE (prod restore via trap) ==="
