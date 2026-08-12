#!/usr/bin/env bash
# install.sh — restore Kevin's 2080 Ti vLLM operational layer onto a host.
# Run from inside deploy/:  ./install.sh
# Assumes the weicj engine is already built at ~/Desktop/vLLM-2080Ti-Definitive/.venv
# (see the repo root build.sh). Idempotent; NEVER overwrites an existing shim.env.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L="$HOME/.local/share/vllm-qwen27b"
SYS=/etc/systemd/system
D="$HOME/Desktop"

echo "==> [1/5] runtime scripts + templates -> $L"
mkdir -p "$L/gencfg"
cp -v "$HERE"/bin/*                          "$L"/
cp -v "$HERE"/templates/*.jinja              "$L"/            2>/dev/null || true
cp -v "$HERE"/templates/gencfg/*             "$L"/gencfg/     2>/dev/null || true
cp -rv "$HERE"/templates/froggeric-templates "$L"/            2>/dev/null || true
cp -v "$HERE"/env/vllm-qwen27b.env           "$L"/
chmod +x "$L"/*.sh "$L"/*.py 2>/dev/null || true

echo "==> [2/5] shim.env (holds the DeepSeek key — never overwritten)"
if [ -f "$L/shim.env" ]; then
  echo "    kept existing $L/shim.env"
else
  cp -v "$HERE/env/shim.env.example" "$L/shim.env"; chmod 600 "$L/shim.env"
  echo "    !! EDIT $L/shim.env and set SHIM_REMOTE_KEY to your DeepSeek key"
fi

echo "==> [3/5] desktop helpers -> $D (optional)"
cp -v "$HERE"/desktop/swap-model.sh "$HERE"/desktop/*.desktop "$D"/ 2>/dev/null || true

echo "==> [4/5] systemd units -> $SYS (sudo)"
sudo cp -v "$HERE"/systemd/*.service "$HERE"/systemd/*.timer "$SYS"/
sudo mkdir -p "$SYS/vllm-qwen27b.service.d"
sudo cp -v "$HERE"/systemd/vllm-qwen27b.service.d/oom.conf "$SYS/vllm-qwen27b.service.d/"
sudo systemctl daemon-reload

echo "==> [5/5] enable services (reboot-persistent)"
sudo systemctl enable vllm-qwen27b vllm-keepalive-shim vllm-qwen27b-watchdog.timer

cat <<EOF

✅ Installed. Before first start:
  • Engine venv must exist:  ~/Desktop/vLLM-2080Ti-Definitive/.venv  (build via repo-root build.sh)
  • DeepSeek key set in:     $L/shim.env   (chmod 600)

Start:   sudo systemctl start vllm-qwen27b && sudo systemctl start vllm-keepalive-shim
Verify:  curl -s localhost:8000/v1/models
Dashboard: http://<host>:8000/gateway/dashboard   (live routing/requests/GPU)

Current config = 4 local lanes (enforce-eager, max-num-seqs 4, shim budget 4, queue-first).
EOF
