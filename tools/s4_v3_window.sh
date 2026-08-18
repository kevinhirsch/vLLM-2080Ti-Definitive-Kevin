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
# fresh window: stale arm results from a previous run must never reach compare
rm -f "$OUT"/s4_*.json "$OUT"/engine_*.log
# CRITICAL: python -m puts CWD at sys.path[0], AHEAD of PYTHONPATH. Run from
# the worktree so the arm engines load THIS branch's vllm no matter where the
# caller's shell happens to sit (window r2 silently ran another worktree's code
# because the caller's cwd held a different vllm checkout).
cd "$WT"

# torch cpp_extension JIT (sampler/GDN warmup in the default-PIECEWISE boot)
# shells out to `ninja`, which lives only in the venv bin — put it on PATH.
export PATH="/home/kevin/Desktop/vLLM-2080Ti-Definitive/.venv/bin:$PATH"

ENGINE_PID=""
restore_prod() {
  echo "[window] restoring prod ..."
  # kill only OUR arm engine (a global pkill here would murder prod itself
  # when the trap fires after a partial restore or an outside restart)
  [ -n "$ENGINE_PID" ] && kill "$ENGINE_PID" 2>/dev/null
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

launch_arm() {  # $1=arm-name  $2=VLLM_S4_GPU_MERGE  $3=extra-flag  $4=scoped (default 1)
  local arm=$1 gpumerge=$2 extra=${3:-} scoped=${4:-1}
  echo "[arm $arm] launching (scoped=$scoped gpu_merge=$gpumerge extra='$extra')"
  env VLLM_S4_SCOPED_DRAFTER=$scoped VLLM_S4_GPU_MERGE=$gpumerge \
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
  # wait for the VRAM to actually come back before the next arm boots
  # (window r4: arm A's corpse held ~11 GiB and starved arms B/C at startup)
  for i in $(seq 1 30); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    [ "${free_mib:-0}" -ge 19000 ] && break
    sleep 5
  done
  echo "[window] GPU free after kill: ${free_mib:-?} MiB"
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

declare -A ARM_MERGE=( [Z]=0 [A]=0 [B]=1 [C]=1 )
declare -A ARM_SCOPED=( [Z]=0 [A]=1 [B]=1 [C]=1 )
declare -A ARM_EXTRA=( [Z]="" [A]="" [B]="" [C]="--no-async-scheduling" )
declare -A ARM_LABEL=( [Z]=Z_s4off_baseline [A]=A_v2_asyncoff [B]=B_v3_asyncon [C]=C_v3_asyncoff )
declare -A ARM_EXPECT=( [Z]="Async" [A]="Async scheduling disabled" [B]="Asynchronous scheduling is enabled" [C]="Async" )

# Arm Z first: S4 fully OFF on the same minimal boot — discriminates "the
# minimal boot + spec16 crashes at HEAD regardless of S4" from "the v2 S4
# path itself is the crasher" (window r4: arm A died on
# 'assert num_required_blocks > len(req_blocks)' at first spec-decode traffic).
for arm in Z A B C; do
  if launch_arm "$arm" "${ARM_MERGE[$arm]}" "${ARM_EXTRA[$arm]}" "${ARM_SCOPED[$arm]}"; then
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
