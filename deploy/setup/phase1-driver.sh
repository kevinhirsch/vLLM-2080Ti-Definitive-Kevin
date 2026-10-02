#!/usr/bin/env bash
# ============================================================================
# PHASE 1 of 2 — toolchain + NVIDIA driver, then REBOOT.
# Run on the fresh Ubuntu/Kubuntu 26.04 box (HNET00 / 10.0.1.225), as kevin.
# Needs sudo. Ends by asking you to reboot; then run phase2-build-serve.sh.
#
# Tailored to THIS machine (verified 2026-07-03): dual RTX 2080 Ti (SM75),
# Intel iGPU drives the desktop, Secure Boot off, root = ext4 on /dev/sda2.
# ============================================================================
set -euo pipefail
log(){ echo -e "\n\033[1;33m== $* ==\033[0m"; }

log "STAGE 0  Base packages + build toolchain (gcc-13 pin for CUDA 12.8 on GCC-15 host)"
sudo apt-get update
sudo apt-get install -y build-essential git curl ca-certificates \
    gcc-13 g++-13 python3-venv pkg-config \
    ntfs-3g lvm2                                  # ntfs-3g: mount Master; lvm2: mount old 24.04

log "STAGE 5a (early)  Persist vm.overcommit_memory=1 (needed for checkpoint mmap)"
echo 'vm.overcommit_memory = 1' | sudo tee /etc/sysctl.d/99-vllm-overcommit.conf >/dev/null
sudo sysctl --system >/dev/null

log "STAGE 1  NVIDIA driver for the dual RTX 2080 Ti (Turing/SM75)"
sudo apt-get install -y ubuntu-drivers-common
# ubuntu-drivers picks the recommended driver; Turing is long-supported.
sudo ubuntu-drivers install || sudo apt-get install -y nvidia-driver-595

cat <<'EOF'

============================================================================
PHASE 1 DONE.  The NVIDIA driver was just installed — you MUST reboot before
CUDA / nvidia-smi will work.

    sudo reboot

After it comes back up, confirm the GPUs are visible:

    nvidia-smi        # expect TWO GeForce RTX 2080 Ti

then run:

    ~/Desktop/vllm-setup/phase2-build-serve.sh
============================================================================
EOF
