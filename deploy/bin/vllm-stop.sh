#!/usr/bin/env bash
# Stop the vLLM server (frees ~19.6 GB VRAM per GPU).
# Launched from the "vLLM - Stop" desktop shortcut (runs in konsole).
set -u
SERVICE=vllm-qwen27b

echo "Stopping ${SERVICE} — KDE will ask for your password (polkit)."
echo "Note: it stays enabled, so it will start again on the next boot."
echo "To also disable boot-start:  sudo systemctl disable ${SERVICE}"
echo
if systemctl stop "$SERVICE"; then
    echo "Stopped. GPU memory now:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
else
    echo "!! Stop failed (auth cancelled or systemd error)."
    systemctl status "$SERVICE" --no-pager -l | head -10
fi
echo
read -rp "Press Enter to close..."
