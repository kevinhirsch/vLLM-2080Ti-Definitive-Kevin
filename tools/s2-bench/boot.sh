#!/usr/bin/env bash
# boot.sh LABEL 'VAR=val' ...   restart engine with override env, wait healthy (no suite)
L=$1; shift
O=/home/kevin/.local/share/vllm-qwen27b/v02.override.env
: > $O; for kv in "$@"; do echo "export $kv" >> $O; done
echo serve-hauhaucs-v02.sh > /home/kevin/.local/share/vllm-qwen27b/active-serve; cat $O
for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
python3 /home/kevin/.local/share/vllm-qwen27b/engine-actuator.py restart --by claude-s2 --reason "S2 arm $L" --no-drain --foreground > /home/kevin/projects/lanes/s2-speed/boot_$L.log 2>&1 &
sleep 25
until curl -s -m 2 -o /dev/null -w "%{http_code}" localhost:8001/health | grep -q 200; do for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done; sleep 4; done
for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
sleep 2
journalctl -u vllm-qwen27b --since "-8 min" --no-pager | grep "GPU KV cache size" | tail -1 | sed 's/.*INFO/KV/' | cut -c1-120
