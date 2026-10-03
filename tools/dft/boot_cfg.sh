#!/usr/bin/env bash
# boot_cfg.sh LABEL -- restart the engine through engine-actuator with whatever v02.override.env currently says (DOES NOT rewrite it), wait healthy.
# Same wait/warmup-kill logic as s2-speed/boot2.sh. exit 0 healthy, 1 gave up after BOOT_TIMEOUT (caller restores).
L=$1
echo serve-hauhaucs-v02.sh > /home/kevin/.local/share/vllm-qwen27b/active-serve
for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
python3 /home/kevin/.local/share/vllm-qwen27b/engine-actuator.py restart --by DFT --reason "DFT MTP drafter fine-tune: $L" --no-drain --foreground > /home/kevin/projects/lanes/dft/results/boot_$L.log 2>&1 &
sleep 20; T0=$(date +%s)
until curl -s -m 2 -o /dev/null -w "%{http_code}" localhost:8001/health | grep -q 200; do
  for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
  [ $(( $(date +%s) - T0 )) -ge ${BOOT_TIMEOUT:-600} ] && { echo "BOOT FAILED $L"; journalctl -u vllm-qwen27b --since "-12 min" --no-pager | grep -iE "error|Traceback|assert" | tail -5 | cut -c1-250; exit 1; }
  sleep 4
done
for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
sleep 2; exit 0
