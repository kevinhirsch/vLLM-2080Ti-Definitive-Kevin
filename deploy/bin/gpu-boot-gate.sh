#!/usr/bin/env bash
# gpu-boot-gate.sh -- sourced by the engine serve scripts right before they launch vLLM (leads L110 + L133 + L138, lane A6).
#
# Why: an engine boot profiles FREE GPU memory. Any foreign CUDA process alive at that moment -- a lane bench, or the previous
# engine's workers still tearing down -- either shrinks the KV pool for the whole engine generation (966K -> 922K -> 900K tokens on
# 2026-10-03) or fails the boot outright ('No available memory for the cache blocks', which under Restart=always became the
# 01:19-01:24 crash loop). windowctl and release.py already gate THEIR boots (gpuguard.py); the boots they do not own -- the
# watchdog's recovery, a manual `systemctl restart`, the unit's own Restart=always -- had no gate at all.
#
# At ExecStart this unit has no engine process of its own yet, so ANY compute app on the cards is foreign (or a dying predecessor).
# The gate waits a bounded time for them to leave, then boots anyway and says exactly what was still there: it must never
# turn a recovery into an outage.
#
#   V02_GPU_WAIT_S   seconds to wait (default 90; 0 = off)      GPU_GATE_POLL_S   poll period (default 3)
#   GPU_GATE_SMI     nvidia-smi binary (tests)   GPU_GATE_QUERY_TIMEOUT_S  per-query bound (default 10)
gpu_boot_gate() {
  local wait=${V02_GPU_WAIT_S:-90} poll=${GPU_GATE_POLL_S:-3} smi=${GPU_GATE_SMI:-nvidia-smi} apps="" waited=0 t0 now
  t0=${EPOCHREALTIME/./}                       # microseconds (bash >= 5); SECONDS has whole-second granularity and made a 1 s bound fire early
  [ "$wait" -gt 0 ] 2>/dev/null || return 0
  command -v "$smi" >/dev/null 2>&1 || return 0
  while :; do
    apps=$(timeout "${GPU_GATE_QUERY_TIMEOUT_S:-10}" "$smi" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')
    now=${EPOCHREALTIME/./}
    if [ -z "$apps" ]; then
      [ "$waited" = 1 ] && echo "gpu-boot-gate: GPUs free after $(( (now - t0) / 1000000 ))s" >&2
      return 0
    fi
    [ $(( now - t0 )) -ge $(( wait * 1000000 )) ] && break
    waited=1
    sleep "$poll"
  done
  echo "gpu-boot-gate: WARNING compute apps still on the GPUs after ${wait}s -- booting anyway; the KV pool will be smaller, or the boot may fail with 'No available memory for the cache blocks':" >&2
  printf '%s\n' "$apps" | while IFS= read -r l; do
    pid=${l%%,*}
    echo "  [pid, name, MiB] $l  unit=$(ps -o unit= -p "$pid" 2>/dev/null | tr -d ' ')" >&2
  done
  return 0
}
