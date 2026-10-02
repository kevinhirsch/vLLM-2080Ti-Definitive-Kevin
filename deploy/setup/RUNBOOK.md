# Bring the vLLM box back up on Kubuntu 26.04 — runbook

Prepared 2026-07-03 by picking up from `right_thing/vllm-2080ti-handoff/HANDOFF.md`.
This box **is** the migration target: `HNET00` / `10.0.1.225`, dual RTX 2080 Ti,
now booted on the fresh **Ubuntu/Kubuntu 26.04** (`/dev/sda2`, label `kubuntu_2604`).
The old 24.04 is intact on the nvme LVM as fallback.

## Already done for you (no privileges needed)
- Cloned the fork → `~/Desktop/vLLM-2080Ti-Definitive`
- Staged the systemd service's user-space files → `~/.local/share/vllm-qwen27b/`
  (`serve-tqk8v4-fg.sh` + `vllm-qwen27b.env`, verbatim from the handoff, paths already correct)
- Verified the winning launch profile exists in the repo

## Why the rest needs you
Every remaining step needs **root** (apt, NVIDIA driver, CUDA, mounting the Master/old
drives, systemd) and the driver install forces a **reboot** in the middle. This automation
session has no terminal for `sudo` and can't survive a reboot, so run the two scripts below
in your own terminal. They're tailored to this exact machine.

## Run it (≈ two commands + a reboot + a long build)

```bash
# 1) toolchain + NVIDIA driver
~/Desktop/vllm-setup/phase1-driver.sh

# 2) reboot so the driver loads
sudo reboot

# 3) after login, confirm both GPUs are visible
nvidia-smi          # expect TWO GeForce RTX 2080 Ti

# 4) CUDA + recover model + build + persistent service + verify
#    (the build compiles CUDA kernels — tens of minutes; re-runnable)
~/Desktop/vllm-setup/phase2-build-serve.sh
```

Phase 2 auto-recovers the 19 GB model from disk (tries the **Master** drive `/dev/sdb1`
first, then the old 24.04 nvme — no LVM name-clash since this root is ext4) and only
falls back to re-downloading if neither has it.

## Acceptance bar (from the handoff — verify before trusting the migration)
1. `nvidia-smi` shows both 2080 Ti; `torch.cuda.device_count()` == 2.
2. Server healthy on `http://10.0.1.225:8000/v1`; `/v1/models` lists
   `qwen27b-int4-tqk8v4-two250K-mtp3-text-only-cu128` **and** alias `qwen3.6:27b`.
3. Greedy decode ≈ **81 tok/s** warm (first call ≈41 cold — Triton JIT). Benchmark with
   **coherent** input, never `vllm bench --random` (collapses MTP acceptance).
4. Tool-call smoke test passes (qwen3_xml parser; keep `enable_thinking:false`).
5. Reachable from the LAN; on Hermes (`http://10.0.1.18:9119`) the provider is
   `base_url=http://10.0.1.225:8000/v1`, `model=qwen3.6:27b` → a live chat works.
6. **Rotate the Hermes admin password** (was exposed in chat).

## If the systemd service won't start on 26.04
The wrapper argv/env were captured on 24.04. If the unit fails, launch once to
regenerate the correct argv for this build, then re-capture:

```bash
cd ~/Desktop/vLLM-2080Ti-Definitive
SERVICE_SCOPE=lan ./launcher.sh --non-interactive \
  --profile profiles/qwen27b/fast/int4/tqk8v4-two250K-mtp3-text-only.env
# once it serves on 0.0.0.0:8000, capture the live argv from /proc/<pid>/cmdline
# and env from /proc/<pid>/environ into ~/.local/share/vllm-qwen27b/, then restart the unit.
```

## Rollback
If the 26.04 rebuild misbehaves, reboot and pick **Ubuntu 24.04** in GRUB — untouched
and working. Only 24.04 change was the visible GRUB menu (`/etc/default/grub.bak-*`).
