#!/usr/bin/env bash
# warmup-after-start.sh -- run after every vllm-qwen27b start (systemd ExecStartPost drop-in).
#
# WHY (2026-09-05 evidence): the same 13 Triton kernels JIT-compile on FIRST USE after every engine
# restart (fused_sigmoid_gating_delta_rule_update_kernel, _tq_decode_stage1, _fwd_kernel_stage2,
# _zero_kv_blocks_kernel, eagle_* ...), and a compile landing on live traffic looks like a 10-30 s
# stall (13:01 probe timeout). This script pays those compiles BEFORE traffic: it waits for :8001
# health, then issues a small, representative set of requests straight to the engine.
#
# SAFETY: never fails the unit (always exit 0); skips itself while a frontier-queue window is running
# (done/*.running marker) so arm measurements are not contaminated; hard time cap; X-Client: warmup.
set -u
ENGINE=http://127.0.0.1:8001
QDIR=/home/kevin/.local/share/vllm-qwen27b/frontier-queue
LOG=/home/kevin/.local/share/vllm-qwen27b/warmup.log
CAP_SECS=${WARMUP_CAP_SECS:-420}
say(){ echo "$(date -Is) $*" >> "$LOG"; }

if compgen -G "$QDIR/done/*.running" > /dev/null 2>&1; then
  say "skip: a frontier-queue window is running (its own probes warm the engine)"; exit 0
fi

# 1) wait for health (the API takes minutes to come up after a cold start)
t0=$(date +%s)
# AU 2026-10-03: stop waiting the moment the engine process is gone. systemd keeps the unit "activating" while this
# ExecStartPost hook runs, so a boot that died in 20 s used to hold the unit (and delay ExecStopPost + Restart=) for the
# full cap: 10-03 03:01-03:30 four failed boots each took 7 min 16 s instead of ~25 s. MAINPID is set by systemd here.
main_alive(){
  local p="${MAINPID:-}"
  [ -z "$p" ] && p=$(systemctl show -p MainPID --value vllm-qwen27b 2>/dev/null)
  [ -z "$p" ] && return 0            # unknown: keep the old behaviour (bounded by CAP_SECS)
  [ "$p" = "0" ] && return 1
  kill -0 "$p" 2>/dev/null
}
until curl -sf -m 3 "$ENGINE/health" > /dev/null 2>&1; do
  main_alive || { say "engine main process exited during boot; not waiting for health"; exit 0; }
  sleep 5
  [ $(( $(date +%s) - t0 )) -ge "$CAP_SECS" ] && { say "gave up waiting for health after ${CAP_SECS}s"; exit 0; }
done
say "engine healthy after $(( $(date +%s) - t0 ))s; warming"

post(){ # $1=label $2=json-body ; logs elapsed + status, never fails
  local s e code
  s=$(date +%s.%N)
  code=$(curl -s -m 240 -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' -H 'X-Client: warmup' \
        "$ENGINE/v1/completions" -d "$2")
  e=$(date +%s.%N)
  say "  $1: http $code in $(python3 -c "print(round($e-$s,1))")s"
}

# 2) tiny decode (spec-decode + decode kernels)
post tiny '{"model":"qwen-local","prompt":"Say OK.","max_tokens":8,"temperature":0}'

# 3) [LANE EF 2026-10-02] replace the synthetic 24K prefill/continuation (junk that evicted real prefixes from the
# ~195-page prefix pool) with the REAL client prefixes: newest flight-recorded Halo/pi/Hermes bodies, Halo last and
# twice (second pass = full prefix hit = continuation-prefill kernels). Cache ends warm for the dominant client.
/usr/bin/python3 /home/kevin/.local/share/vllm-qwen27b/rewarm-prefix.py --engine "$ENGINE" --cap 200 || true

# 4) four concurrent ~2K prompts (batched prefill + concurrent decode shapes)
for i in 1 2 3 4; do
  B=$(python3 -c "import json; print(json.dumps({'model':'qwen-local','prompt':('Warmup batch item $i. ' * 230) + ' Reply with one word.','max_tokens':16,'temperature':0}))")
  post "batch-$i" "$B" &
done
wait

# 5) one longer generation (steady-state decode + MTP acceptance path)
post gen-256 '{"model":"qwen-local","prompt":"List the numbers from 1 to 120 separated by spaces.","max_tokens":256,"temperature":0}'

say "done in $(( $(date +%s) - t0 ))s total"
exit 0
