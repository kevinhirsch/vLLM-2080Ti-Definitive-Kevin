#!/usr/bin/env bash
# GW2 window (L105): validate the read-only prefix-cache probe (VLLM_FORK_PREFIX_PROBE=1) on the real engine.
#   GW2E arm: window-start override + V02_ROOT=wt-gw2e + VLLM_FORK_PREFIX_PROBE=1 -> probe_validate.py
#             (render parity, cached exactness vs usage, latency idle/under a cold prefill, decode impact),
#             evalkit tool_call smoke, journal error grep.
#   RESTORE:  window-start override verbatim + restart + health. Nothing here changes production defaults.
# wt-gw2e = integrate-v0.2.2-post3 bf46537a45 + CR2 326846fdc2 (flags off) + GW2 probe (flag on in this arm only).
# Run under: cd /home/kevin/Desktop/vLLM-2080Ti-Definitive && .venv/bin/python deploy/bin/gateway-offline.py run \
#   --reason "GW2 L105 engine prefix-cache probe validation" --by GW2 --ttl 1800 --wait-s 90 -- \
#   bash /home/kevin/Desktop/wt-gw2e/tools/gw2/gw2_win.sh > /home/kevin/projects/lanes/gw2/win.out 2>&1
set -u
OUT=/home/kevin/projects/lanes/gw2/win; mkdir -p $OUT
G=/home/kevin/Desktop/wt-gw2e; SD=/home/kevin/.local/share/vllm-qwen27b; O=$SD/v02.override.env
log(){ echo "$(date '+%F %T') $*" | tee -a $OUT/win.log; }
health(){ curl -s -m 3 -o /dev/null -w "%{http_code}" localhost:8001/health; }
XID0=$(journalctl -k --no-pager | grep -c "NVRM: Xid")
cp -a $O $OUT/override.at_start.env; log "window start; xid=$XID0; override $(wc -c < $O) bytes; gw2e HEAD $(git -C $G rev-parse --short HEAD)"
BOOT_SINCE=""
boot(){ # boot LABEL KV...
  local lab=$1; shift; cp -a $OUT/override.at_start.env $O; for kv in "$@"; do echo "export $kv" >> $O; done
  log "boot $lab override: $(tr '\n' ';' < $O)"
  for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
  [ -x /home/kevin/projects/lanes/windows/gpu_foreign.sh ] && { t0=$(date +%s); while f=$(/home/kevin/projects/lanes/windows/gpu_foreign.sh); [ -n "$f" ] && [ $(( $(date +%s) - t0 )) -lt 600 ]; do sleep 10; done; }
  BOOT_SINCE=$(date '+%F %T')
  python3 $SD/engine-actuator.py restart --by GW2 --reason "GW2 arm $lab" --no-drain --foreground > $OUT/boot_$lab.log 2>&1 &
  sleep 20; local T0=$(date +%s)
  until [ "$(health)" = 200 ]; do
    for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
    [ $(( $(date +%s) - T0 )) -ge 900 ] && { log "BOOT FAILED $lab: $(journalctl -u vllm-qwen27b --since "$BOOT_SINCE" --no-pager | grep -iE 'error|Traceback|memory' | tail -4 | cut -c1-260 | tr '\n' '|')"; return 1; }
    sleep 4
  done
  for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
  log "booted $lab: $(journalctl -u vllm-qwen27b --since "$BOOT_SINCE" --no-pager | grep -E 'Model loading took|GPU KV cache size' | sed 's/.*INFO//' | cut -c1-160 | sort -u | tr '\n' '|')"
  [ $# -gt 0 ] && [ -x /home/kevin/projects/lanes/windows/envcheck.sh ] && log "$lab api_server env: $(/home/kevin/projects/lanes/windows/envcheck.sh $(for kv in "$@"; do echo "${kv%%=*}"; done))"
  return 0
}
errs(){ journalctl -u vllm-qwen27b --since "$1" --no-pager | grep -cE 'Traceback|CUDA error|illegal memory|fork probe failed'; }
if boot gw2e "V02_ROOT=$G" "VLLM_FORK_PREFIX_PROBE=1"; then
  log "gw2e endpoint: $(curl -s -m 10 localhost:8001/v1/fork/prefix_cache_probe -H 'Content-Type: application/json' -d '{"prompt_token_ids":[1,2,3]}')"
  T=$(date '+%F %T')
  log "gw2e validate: $(timeout 1200 $G/.venv/bin/python $G/tools/gw2/probe_validate.py --out $OUT --label gw2e > $OUT/validate.out 2>&1; grep -E 'VERDICT|latency|decode' $OUT/validate.out | tr '\n' ' ' | cut -c1-900)"
  log "gw2e evalkit tool_call: $(cd /home/kevin/Desktop/qwen38-evalkit && timeout 900 python3 run_eval.py --tag gw2e-probe --categories tool_call > $OUT/evalkit_gw2e.out 2>&1; tail -3 $OUT/evalkit_gw2e.out | tr '\n' ' ' | cut -c1-400)"
  log "gw2e journal errors since boot: $(errs "$BOOT_SINCE")  xid now $(journalctl -k --no-pager | grep -c 'NVRM: Xid')"
fi
# RESTORE
boot restore || { sleep 30; boot restore2; }
cmp -s $O $OUT/override.at_start.env && log "override restored verbatim"
log "window end: health $(health); xid now $(journalctl -k --no-pager | grep -c 'NVRM: Xid') (start $XID0)"
