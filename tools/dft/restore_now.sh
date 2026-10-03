#!/usr/bin/env bash
# restore exact baseline (empty override) and reboot engine
L=/home/kevin/projects/lanes/dft; SV=/home/kevin/.local/share/vllm-qwen27b
B=$(ls -t $L/override.baseline.* | head -1)
cp -p $B $SV/v02.override.env; rm -f $L/override.saved
BOOT_TIMEOUT=900 $L/boot_cfg.sh restore_final
echo "health $(curl -s -m3 -o /dev/null -w '%{http_code}' localhost:8001/health) override_bytes=$(wc -c < $SV/v02.override.env)"
