# Hermes dedicated engine (:8002) — install-day kit
Prereq: third modded 2080 Ti 22GB installed (riser, x4 slot OK), PSU cable, power caps.
1. `nvidia-smi -pl 220` on all 3 GPUs (persist via systemd oneshot or nvidia-persistenced hook).
2. Copy `serve-hermes-8002.sh` to `~/.local/share/vllm-qwen27b/`, chmod +x. Confirm `HERMES_GPU_INDEX` matches the new card's index in nvidia-smi.
3. Install + enable `vllm-hermes-8002.service` (systemctl daemon-reload; enable --now).
4. Gateway routing: add per-client upstream map to keepalive-shim.py — Hermes IPs (10.0.1.10, 10.0.1.250) -> http://127.0.0.1:8002, default -> :8001. (Small change: choose LOCAL base per request before relay; keep DeepSeek as shared overflow.)
5. Verify: curl :8002/health; Hermes turn via :8000 shows served_by the :8002 model id.
Expected: ~45-55 tok/s solo for Hermes, NVLink pair exclusive to interactive/pi at 84-106 tok/s.
