#!/usr/bin/env bash
# EF2 / L134 GPU rider: runs tools/ef2/l134_probe.py on ONE quiet GPU in both trees,
# plain and under compute-sanitizer memcheck. Needs the engine STOPPED (or ~2 GiB free
# on $GPU), ~10-15 min. No engine boot and no override changes, so it is safe to attach
# to any window's engine-stopped phase. It refuses (rc 3) unless gpuok-style free memory
# holds, and it never runs the 'poison' part without the sanitizer.
#   usage: l134_rider.sh [GPU]   env: EF2_BUDGET_S (default 1140), EF2_POISON=1 (sanitizer-only poison leg)
#   outputs: ~/projects/lanes/ef2/l134/<ts>/
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
    timeout -k 10 "${LEG_T:-300}" env "${envs[@]}" "$py" "$P" "$@" > "$OUT/$lab.jsonl" 2> "$OUT/$lab.err"
  else
    timeout -k 10 "${LEG_T:-600}" env "${envs[@]}" PYTORCH_NO_CUDA_MEMORY_CACHING=1 "$san" $SANOPT "$py" "$P" "$@" > "$OUT/$lab.jsonl" 2> "$OUT/$lab.err"
  fi
  local rc=$?
  log "$lab rc=$rc ($(( $(date +%s) - t0 ))s) last=$(tail -1 "$OUT/$lab.jsonl" | cut -c1-160) sanitizer_errors=$(grep -c '========= Invalid\|========= Program hit' "$OUT/$lab.jsonl" "$OUT/$lab.err" 2>/dev/null | awk -F: '{s+=$2} END{print s}')"
}
LEG_ENV=(PYTHONNOUSERSITE=1)
INT_ENV=(PYTHONNOUSERSITE=1 FLASHINFER_ENABLE_AOT=1 FLASHINFER_WORKSPACE_BASE=$EF2 PYTHONPATH=$EF2 TRITON_CACHE_DIR=$EF2/.triton-cache)
# Wall budget (EF2_BUDGET_S, default 1140 s = 19 min): legs run in priority order; a leg is skipped when the
# remaining budget is below its own bound, so the whole rider stays inside a 20-min slot.
BUDGET=${EF2_BUDGET_S:-1140}; T0=$(date +%s)
left(){ echo $(( BUDGET - ($(date +%s) - T0) )); }
leg(){ # leg BOUND_S LABEL ...run args
  local b=$1; shift
  if [ "$(left)" -lt "$b" ]; then log "$1 SKIPPED (budget: $(left)s left < ${b}s)"; return 99; fi
  LEG_T=$(( b - 15 )) run "$@"
}
san_errs(){ grep -c '========= Invalid\|========= Program hit' "$OUT/$1.jsonl" "$OUT/$1.err" 2>/dev/null | awk -F: '{s+=$2} END{print s+0}'; }
# Xid safety: nothing that can touch memory out of bounds runs WITHOUT the sanitizer.
#  - race legs never launch a kernel on the corrupted wrapper (they only read its indptr back);
#  - legacy fault shapes run under memcheck FIRST; the plain legacy shape run (correctness) only if memcheck was clean;
#  - the poison leg (sanitizer only) runs only with EF2_POISON=1 (default off: memcheck checks each access before it is
#    issued, but a between-window Xid gate should not depend on that).
leg 120 legacy_race   "$LEG_PY" - "${LEG_ENV[@]}" -- --part race
leg 120 integ_race    "$INT_PY" - "${INT_ENV[@]}" -- --part race
leg 420 legacy_shapes_memcheck "$LEG_PY" "$SAN12" "${LEG_ENV[@]}" -- --part shapes --sweep 4
leg 420 integ_shapes_memcheck  "$INT_PY" "$SAN13" "${INT_ENV[@]}" -- --part shapes --sweep 4
leg 240 integ_dequant_memcheck "$INT_PY" "$SAN13" "${INT_ENV[@]}" -- --part dequant
leg 180 integ_shapes  "$INT_PY" - "${INT_ENV[@]}" -- --part shapes --sweep 24
if [ -s "$OUT/legacy_shapes_memcheck.jsonl" ] && [ "$(san_errs legacy_shapes_memcheck)" = 0 ] && grep -q '"ok": true' "$OUT/legacy_shapes_memcheck.jsonl"; then
  leg 180 legacy_shapes "$LEG_PY" - "${LEG_ENV[@]}" -- --part shapes --sweep 24
else
  log "legacy_shapes (plain) SKIPPED: legacy memcheck not clean or not run -> never run a suspected-OOB kernel without the sanitizer"
fi
[ "${EF2_POISON:-0}" = 1 ] && leg 240 legacy_poison_memcheck "$LEG_PY" "$SAN12" "${LEG_ENV[@]}" -- --part poison
log "done in $(( $(date +%s) - T0 ))s; xid now $(journalctl -k --no-pager | grep -c 'NVRM: Xid') (start $XID0); outputs $OUT"
