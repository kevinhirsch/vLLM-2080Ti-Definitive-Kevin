#!/usr/bin/env bash
# Stop the vLLM server (frees ~19.6 GB VRAM per GPU).
# Launched from the "vLLM - Stop" desktop shortcut (runs in konsole).
set -u
SERVICE=vllm-qwen27b

echo "Stopping ${SERVICE} — KDE will ask for your password (polkit)."
echo "Note: it stays enabled, so it will start again on the next boot."
echo "To also disable boot-start:  sudo systemctl disable ${SERVICE}"
echo
# LV 2026-10-03: the liveness authority restarts an engine that is stopped with nobody owning it. A manual stop owns it for
# STOP_HOURS (default 8) -- after that the engine comes back by itself. 'vLLM - Restart' releases the hold at once.
ACT=/home/kevin/.local/share/vllm-qwen27b/engine-actuator.py
HOURS="${STOP_HOURS:-8}"
if [ -f "$ACT" ]; then
  /usr/bin/python3 "$ACT" hold acquire --kind engine --by kevin-desktop --reason "manual stop (desktop shortcut)" \
    --ttl $(( HOURS * 3600 )) >/dev/null && echo "Engine held for ${HOURS} h (the watchdog will not restart it before then)."
fi
if systemctl stop "$SERVICE"; then
    echo "Stopped. GPU memory now:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
else
    echo "!! Stop failed (auth cancelled or systemd error)."
    systemctl status "$SERVICE" --no-pager -l | head -10
fi
echo
read -rp "Press Enter to close..."
