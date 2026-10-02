#!/usr/bin/env bash
# ============================================================================
# PHASE 2 of 2 — CUDA + model + build vLLM + persistent service + verify.
# Run AFTER phase1-driver.sh AND a reboot, on HNET00 / 10.0.1.225, as kevin.
# Needs sudo. The build (STAGE 3b) compiles CUDA kernels and is SLOW (tens of
# minutes) — expected. Safe to re-run; it skips finished steps.
# ============================================================================
set -euo pipefail
log(){ echo -e "\n\033[1;33m== $* ==\033[0m"; }
die(){ echo -e "\n\033[1;31mABORT: $*\033[0m"; exit 1; }

REPO_DIR="$HOME/Desktop/vLLM-2080Ti-Definitive"
MODEL_DIR="$HOME/Desktop/models/Qwen3.6-27B-GPTQ-Int4"
SVC_DEST="$HOME/.local/share/vllm-qwen27b"
UNIT_SRC="/home/kevin/Desktop/right_thing/vllm-2080ti-handoff/vllm-qwen27b-configs/vllm-qwen27b.service"

export CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13

# ---------------------------------------------------------------------------
log "SANITY  driver must be live (did you reboot after Phase 1?)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — run phase1 then REBOOT first."
GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '2080 Ti' || true)
[ "$GPUS" -eq 2 ] || die "Expected 2x RTX 2080 Ti visible, saw $GPUS. Fix driver before continuing."
echo ">> OK: two 2080 Ti visible."

# ---------------------------------------------------------------------------
log "STAGE 2  CUDA 12.8 toolkit (ubuntu2404 repo works on 26.04)"
if [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
  DISTRO=ubuntu2404
  curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb" -o /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-12-8
fi
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
"$CUDA_HOME/bin/nvcc" --version | tail -2

# ---------------------------------------------------------------------------
log "STAGE 3  Recover the 19GB model locally (skip re-download if found)"
recover_from(){  # $1 = mounted source root that contains .../Qwen3.6-27B*GPTQ-Int4/config.json
  local hit; hit=$(find "$1" -maxdepth 7 -type f -name config.json 2>/dev/null \
      | grep -iE 'Qwen3\.6-27B.*GPTQ-Int4' | head -1 || true)
  [ -n "$hit" ] || return 1
  echo ">> found model at $(dirname "$hit") — copying..."
  mkdir -p "$MODEL_DIR"; rsync -a --info=progress2 "$(dirname "$hit")/" "$MODEL_DIR/"
}
if [ -f "$MODEL_DIR/config.json" ]; then
  echo ">> model already present, skipping recovery."
else
  # Try the Master WD drive (sdb1, NTFS, label 'Master') — the migration staging disk.
  sudo mkdir -p /mnt/master
  if sudo mount -t ntfs-3g -o ro /dev/sdb1 /mnt/master 2>/dev/null; then
    recover_from /mnt/master || echo ">> not on Master."
    sudo umount /mnt/master || true
  fi
  # Fallback: the old Ubuntu 24.04 root (nvme LVM ubuntu-vg). New root is ext4,
  # so there is NO vg name-clash — just activate + mount read-only.
  if [ ! -f "$MODEL_DIR/config.json" ]; then
    sudo vgchange -ay ubuntu-vg >/dev/null 2>&1 || true
    sudo mkdir -p /mnt/old2404
    if sudo mount -o ro /dev/ubuntu-vg/ubuntu-lv /mnt/old2404 2>/dev/null; then
      recover_from /mnt/old2404/home/kevin/Desktop/models || recover_from /mnt/old2404 || echo ">> not on old nvme."
      sudo umount /mnt/old2404 || true
    fi
  fi
fi

# ---------------------------------------------------------------------------
log "STAGE 3b  Build the fork (compiles CUDA kernels — SLOW, tens of minutes)"
[ -d "$REPO_DIR/.git" ] || git clone https://github.com/weicj/vLLM-2080Ti-Definitive.git "$REPO_DIR"
cd "$REPO_DIR"
# Stop any crash-looping service so it can't import a half-built tree mid-build.
sudo systemctl stop vllm-qwen27b 2>/dev/null || true
# Robust "is it really built?" check: import the NATIVE extension from a neutral
# CWD (so a bare source-dir shadow can't produce a false positive). Plain
# `import vllm` passes even with zero kernels compiled — do NOT trust it.
if ( cd /tmp && "$REPO_DIR/.venv/bin/python" -c 'import vllm._C' ) >/dev/null 2>&1; then
  echo ">> vllm native extension (vllm._C) present, skipping build."
else
  echo ">> vllm native extension missing — building (compiles CUDA kernels)..."
  ASSUME_YES=1 CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13 ./build.sh
fi

# ---------------------------------------------------------------------------
log "STAGE 4  Model weights (only if disk-recovery above didn't find them)"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  mkdir -p "$HOME/Desktop/models"
  "$REPO_DIR/.venv/bin/hf" download \
    llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4 \
    --local-dir "$MODEL_DIR"
fi
[ -f "$MODEL_DIR/config.json" ] || die "model still missing after recovery+download."

log "Quick import check (expect vllm/torch versions + device_count 2)"
"$REPO_DIR/.venv/bin/python" -c 'import vllm,torch;print("vllm",vllm.__version__,"torch",torch.__version__,"gpus",torch.cuda.device_count())'

# ---------------------------------------------------------------------------
log "STAGE 5  Install boot-persistent systemd service (user-space files pre-staged)"
# ~/.local/share/vllm-qwen27b/{serve-tqk8v4-fg.sh,vllm-qwen27b.env} are already in place.
ls -l "$SVC_DEST"/serve-tqk8v4-fg.sh "$SVC_DEST"/vllm-qwen27b.env
sudo cp -v "$UNIT_SRC" /etc/systemd/system/vllm-qwen27b.service
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-qwen27b.service

# ---------------------------------------------------------------------------
log "STAGE 6  Verify (must match the 24.04 baseline before trusting)"
echo ">> waiting for :8000/health (model load + Triton JIT can take a few min)..."
for i in $(seq 1 120); do
  curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo "  healthy after ~$((i*5))s"; break; }
  sleep 5
  [ "$i" -eq 120 ] && die "server not healthy in 10min — check: sudo journalctl -u vllm-qwen27b -e"
done

echo ">> served models (expect the two names incl. qwen3.6:27b):"
curl -fsS http://127.0.0.1:8000/v1/models | "$REPO_DIR/.venv/bin/python" -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])'

echo ">> coherent chat (enable_thinking:false → direct answer):"
curl -fsS http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"qwen3.6:27b","chat_template_kwargs":{"enable_thinking":false},
  "messages":[{"role":"user","content":"Reply with exactly: HNET00 online."}]}' \
  | "$REPO_DIR/.venv/bin/python" -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])'

echo ">> reachable from the LAN (Hermes @ 10.0.1.18 needs this):"
curl -fsS http://10.0.1.225:8000/v1/models >/dev/null && echo "  OK: 10.0.1.225:8000 answers on the LAN."

cat <<'EOF'

============================================================================
DONE. Left to confirm by hand (matches the handoff's acceptance bar):
  * Benchmark greedy decode — expect ~81 tok/s warm (first call ~41 cold JIT).
      Use COHERENT input, not vllm bench --random (it understates MTP badly).
  * Tool-call smoke test (qwen3_xml parser; keep enable_thinking:false).
  * On the Hermes box (http://10.0.1.18:9119) confirm the provider base_url is
      http://10.0.1.225:8000/v1  and model  qwen3.6:27b  — then a live chat.
  * ROTATE the Hermes admin password (was exposed in chat).
Manage the server:  sudo systemctl {status,restart,stop} vllm-qwen27b
============================================================================
EOF
