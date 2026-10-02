#!/usr/bin/env bash
# arm.sh LABEL 'VAR=val ...'  -- write override env, restart engine (no drain), wait healthy (skip warmup hook), run suite
L=$1; shift
O=/home/kevin/.local/share/vllm-qwen27b/v02.override.env
: > $O; for kv in "$@"; do echo "export $kv" >> $O; done
echo serve-hauhaucs-v02.sh > /home/kevin/.local/share/vllm-qwen27b/active-serve; cat $O
for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
python3 /home/kevin/.local/share/vllm-qwen27b/engine-actuator.py restart --by claude-up --reason "UP arm $L" --no-drain --foreground > /home/kevin/projects/lanes/up-integrate/arm_$L.restart 2>&1 &
sleep 20
until curl -s -m 2 -o /dev/null -w "%{http_code}" localhost:8001/health | grep -q 200; do sleep 4; done
for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
sleep 3
journalctl -u vllm-qwen27b --since "-6 min" --no-pager | grep "GPU KV cache size" | tail -1 | sed 's/.*INFO/KV/' | cut -c1-120
python3 /home/kevin/projects/lanes/up-integrate/quick.py
