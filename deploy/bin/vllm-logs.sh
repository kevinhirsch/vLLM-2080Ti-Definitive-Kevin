#!/usr/bin/env bash
# Live log view for the vLLM server. Ctrl+C to stop following, then Enter to close.
set -u
SERVICE=vllm-qwen27b

systemctl status "$SERVICE" --no-pager -l | head -12
echo
echo "--- following live logs (Ctrl+C to stop) ---"
trap '' INT   # let Ctrl+C kill journalctl but not the window
journalctl -u "$SERVICE" -n 50 -f
trap - INT
echo
read -rp "Press Enter to close..."
