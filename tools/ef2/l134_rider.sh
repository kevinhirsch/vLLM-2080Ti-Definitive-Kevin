#!/usr/bin/env bash
# EF2 / L134 GPU rider: runs tools/ef2/l134_probe.py on ONE quiet GPU in both trees,
# plain and under compute-sanitizer memcheck. Needs the engine STOPPED (or ~2 GiB free
# on $GPU), ~10-15 min. No engine boot and no override changes, so it is safe to attach
# to any window's engine-stopped phase. It refuses (rc 3) unless gpuok-style free memory
# holds, and it never runs the 'poison' part without the sanitizer.
#   usage: l134_rider.sh [GPU]      outputs: ~/projects/lanes/ef2/l134/<ts>/
set -u
GPU=${1:-1}
OUT=/home/kevin/projects/lanes/ef2/l134/$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
EF2=/home/kevin/Desktop/wt-ef2
P=$EF2/tools/ef2/l134_probe.py
LEG_PY=/home/kevin/Desktop/vLLM-2080Ti-Definitive/.venv/bin/python   # FlashInfer 0.6.8 (legacy tree)
INT_PY=/home/kevin/Desktop/wt-integrate/.venv/bin/python              # FlashInfer 0.6.18 (integrate tree)
SAN12=/usr/local/cuda-12.8/bin/compute-sanitizer                      # torch cu128
SAN13=/usr/local/cuda-13.2/bin/compute-sanitizer                      # torch cu130
log(){ echo "$(date '+%F %T') $*" | tee -a "$OUT/rider.log"; }
free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[ "${free:-0}" -ge 2000 ] || { log "REFUSE gpu$GPU free=${free}MiB < 2000 (engine must be stopped)"; exit 3; }
XID0=$(journalctl -k --no-pager | grep -c "NVRM: Xid")
log "start gpu=$GPU free=${free}MiB xid=$XID0"
export CUDA_VISIBLE_DEVICES=$GPU
SANOPT="--tool memcheck --leak-check no --print-limit 40 --error-exitcode 9 --report-api-errors no"
run(){ # run LABEL PY SAN|- ENVSET... -- ARGS
  local lab=$1 py=$2 san=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local t0=$(date +%s)
  if [ "$san" = "-" ]; then
    timeout -k 20 600 env "${envs[@]}" "$py" "$P" "$@" > "$OUT/$lab.jsonl" 2> "$OUT/$lab.err"
  else
    timeout -k 20 900 env "${envs[@]}" PYTORCH_NO_CUDA_MEMORY_CACHING=1 "$san" $SANOPT "$py" "$P" "$@" > "$OUT/$lab.jsonl" 2> "$OUT/$lab.err"
  fi
  local rc=$?
  log "$lab rc=$rc ($(( $(date +%s) - t0 ))s) last=$(tail -1 "$OUT/$lab.jsonl" | cut -c1-160) sanitizer_errors=$(grep -c '========= Invalid\|========= Program hit' "$OUT/$lab.jsonl" "$OUT/$lab.err" 2>/dev/null | awk -F: '{s+=$2} END{print s}')"
}
LEG_ENV=(PYTHONNOUSERSITE=1)
INT_ENV=(PYTHONNOUSERSITE=1 FLASHINFER_ENABLE_AOT=1 FLASHINFER_WORKSPACE_BASE=$EF2 PYTHONPATH=$EF2 TRITON_CACHE_DIR=$EF2/.triton-cache)
# 1. race proof (plain, both trees: both venvs exercise the same plan() DMA semantics)
run legacy_race   "$LEG_PY" - "${LEG_ENV[@]}" -- --part race
run integ_race    "$INT_PY" - "${INT_ENV[@]}" -- --part race
# 2. correctness + shapes (plain)
run legacy_shapes "$LEG_PY" - "${LEG_ENV[@]}" -- --part shapes --sweep 24
run integ_shapes  "$INT_PY" - "${INT_ENV[@]}" -- --part shapes --sweep 24
run integ_dequant "$INT_PY" - "${INT_ENV[@]}" -- --part dequant
# 3. memcheck: fault shapes per FlashInfer version, the dequant kernel, and the race consequence
run legacy_shapes_memcheck "$LEG_PY" "$SAN12" "${LEG_ENV[@]}" -- --part shapes --sweep 6
run integ_shapes_memcheck  "$INT_PY" "$SAN13" "${INT_ENV[@]}" -- --part shapes --sweep 6
run integ_dequant_memcheck "$INT_PY" "$SAN13" "${INT_ENV[@]}" -- --part dequant
run legacy_poison_memcheck "$LEG_PY" "$SAN12" "${LEG_ENV[@]}" -- --part poison
log "done xid now $(journalctl -k --no-pager | grep -c 'NVRM: Xid') (start $XID0); outputs $OUT"
