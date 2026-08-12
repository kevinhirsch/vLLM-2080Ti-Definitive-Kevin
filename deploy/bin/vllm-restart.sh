#!/usr/bin/env bash
# Restart the vLLM server and wait for the API to come back up.
# Launched from the "vLLM - Restart" desktop shortcut (runs in konsole).
set -u
SERVICE=vllm-qwen27b

echo "==============================================="
echo " Restarting ${SERVICE} — KDE will ask for your"
echo " password (polkit). Model load takes ~1-2 min."
echo "==============================================="
systemctl reset-failed "$SERVICE" 2>/dev/null
if ! systemctl restart "$SERVICE"; then
    echo
    echo "!! Restart was not dispatched (auth cancelled or systemd error)."
    systemctl status "$SERVICE" --no-pager -l | head -15
    echo
    read -rp "Press Enter to close..."
    exit 1
fi

echo -n "Restart dispatched. Waiting for health"
code=000
for i in $(seq 1 60); do
    code=$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health || true)
    if [ "$code" = "200" ]; then
        echo
        echo "OK: vLLM is UP after ~$((i * 5))s"
        echo "    Endpoint: http://10.0.1.225:8000/v1   Model: qwen3.6:27b"
        break
    fi
    echo -n "."
    sleep 5
done

if [ "$code" != "200" ]; then
    echo
    echo "!! Still not healthy after 5 min. Last 30 log lines:"
    journalctl -u "$SERVICE" -n 30 --no-pager
    echo
    echo "Run the 'vLLM - Troubleshoot' shortcut for a full diagnostic."
fi
echo
read -rp "Press Enter to close..."
