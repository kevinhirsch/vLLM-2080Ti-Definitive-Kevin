#!/usr/bin/env bash
# CR2 window (L101): prefix-aware short-first A/B. Scheduler-only change, no kernels.
#   0. save the current override env (restored verbatim at the end, whatever happens)
#   1. BASE arm: probe the RUNNING engine as-is (no restart)
#   2. CR2 arm: saved override + V02_ROOT=wt-cr2 + VLLM_SCHED_SHORT_FIRST_PREFIX_AWARE=1 -> probe, error count
#   3. restore the saved override + engine-actuator restart, health check
# PASS: cr2 warm_p50_s < 0.5 x base warm_p50_s, markers_ok == trials in both arms,
#       cr2 cold_p50_s <= base cold_p50_s + 4 s (one yielded step), no Traceback/CUDA error in the cr2 boot.
# Run under: python3 /home/kevin/Desktop/wt-integrate/deploy/bin/gateway-offline.py run --reason "CR2 prefix-aware short-first A/B" --by CR2 --ttl 1800 --wait-s 90 -- bash /home/kevin/Desktop/wt-cr2/tools/cr2/cr2_win.sh
set -u
L=/home/kevin/projects/lanes/cr2; C=/home/kevin/Desktop/wt-cr2
SD=/home/kevin/.local/share/vllm-qwen27b; O=$SD/v02.override.env
mkdir -p $L/win; cp $O $L/win/override.saved; echo "cr2 window start $(date)"; cat $L/win/override.saved
boot() {  # boot LABEL EXTRA_LINES...   (override = saved override + extra export lines)
  local lab=$1; shift; cp $L/win/override.saved $O; for kv in "$@"; do echo "export $kv" >> $O; done
  for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
  python3 $SD/engine-actuator.py restart --by CR2 --reason "CR2 arm $lab" --no-drain --foreground > $L/win/boot_$lab.log 2>&1 &
  sleep 20; local t0=$(date +%s)
  until curl -s -m 2 -o /dev/null -w "%{http_code}" localhost:8001/health | grep -q 200; do
    for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
    [ $(( $(date +%s) - t0 )) -ge 900 ] && { echo "BOOT FAILED $lab"; journalctl -u vllm-qwen27b --since "-10 min" --no-pager | grep -iE "error|Traceback|assert" | tail -8 | cut -c1-250; return 1; }
    sleep 4
  done
  for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
  echo "booted $lab $(date +%T)"
}
# 1. BASE arm on the running engine
echo "== base probe $(date +%T)"; python3 $C/tools/cr2/warm_overtake_probe.py --label base --out $L/win 2>&1 | tail -8
# 2. CR2 arm
T2=$(date '+%Y-%m-%d %H:%M:%S')
boot cr2 "V02_ROOT=$C" "VLLM_SCHED_SHORT_FIRST_PREFIX_AWARE=1" && {
  echo "== cr2 probe $(date +%T)"; python3 $C/tools/cr2/warm_overtake_probe.py --label cr2 --out $L/win 2>&1 | tail -8
  echo "cr2 errors in journal: $(journalctl -u vllm-qwen27b --since "$T2" --no-pager | grep -cE 'Traceback|CUDA error|illegal memory')"
}
# 3. restore
echo "== restore $(date +%T)"; cp $L/win/override.saved $O
python3 $SD/engine-actuator.py restart --by CR2 --reason "CR2 window restore" --no-drain --foreground > $L/win/boot_restore.log 2>&1 &
sleep 20; t0=$(date +%s); until curl -s -m 2 -o /dev/null -w "%{http_code}" localhost:8001/health | grep -q 200; do [ $(( $(date +%s) - t0 )) -ge 900 ] && break; sleep 4; done
cmp -s $O $L/win/override.saved && echo "override restored verbatim"
echo "restored health $(curl -s -m3 -o /dev/null -w '%{http_code}' localhost:8001/health) $(date)"
