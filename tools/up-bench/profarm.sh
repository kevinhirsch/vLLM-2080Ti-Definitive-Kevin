#!/usr/bin/env bash
# profarm.sh LABEL ARGSFILE MODELDIR SERVED [extra override lines...]  -- switch the engine unit to an upstream profile command, bench reference lane + natural text
L=$1; AF=$2; MD=$3; SN=$4; shift 4
B=/home/kevin/.local/share/vllm-qwen27b; D=/home/kevin/projects/lanes/up-integrate
O=$B/v02.override.env; : > $O; echo "export V02_PROFILE_ARGS=$AF" >> $O; for kv in "$@"; do echo "export $kv" >> $O; done
echo serve-profile-v02.sh > $B/active-serve
for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
python3 $B/engine-actuator.py restart --by claude-up --reason "UP profile arm $L" --no-drain --foreground > $D/prof_$L.restart 2>&1 &
sleep 25
T0=$(date +%s)
until curl -s -m 2 -o /dev/null -w "%{http_code}" localhost:8001/health | grep -q 200; do
  for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
  if ! systemctl is-active --quiet vllm-qwen27b && [ "$(systemctl is-active vllm-qwen27b)" != activating ]; then echo "UNIT NOT ACTIVE"; fi
  if [ $(( $(date +%s) - T0 )) -gt 900 ]; then echo "BOOT TIMEOUT"; exit 1; fi
  sleep 5
done
for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
echo "healthy after $(( $(date +%s) - T0 + 25 ))s"; journalctl -u vllm-qwen27b --since "-8 min" --no-pager | grep -E "GPU KV cache size" | tail -1 | sed 's/.*INFO/KV/' | cut -c1-120
sleep 5
$D/refbench.sh $MD $SN 8001 $L 2>&1 | sed 's/"http_status.*//' | tee $D/prof_$L.ref
python3 $D/quick.py | tee $D/prof_$L.quick
