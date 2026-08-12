# deploy/ — Kevin's operational layer for the 2080 Ti vLLM stack

Everything that makes **this box** actually serve: systemd units, the capacity-routing
gateway, serve scripts, watchdog, env, chat templates, and docs. The rest of this repo is
the vLLM engine fork (from `github.com/weicj/vLLM-2080Ti-Definitive`); this directory is the
deployment/config layer that lives outside the source tree on the running host.

> **Host:** HNET00 / `10.0.1.225`, 2× RTX 2080 Ti (Turing SM75, NVLink), 44 GB VRAM.
> **Engine pinned to:** weicj/vLLM-2080Ti-Definitive `v0.1.12`, commit `65c727f`, CUDA 12.8 / torch cu128, TP=2.
> Paths in these files are absolute (`/home/kevin/...`) — this is a **snapshot backup of a live host**, not a portable installer.

## Architecture

```
clients (pi @10.0.1.12, 2× Hermes @10.0.1.10, Applicant, AZ)
        │  OpenAI-compatible
        ▼
:8000  vllm-keepalive-shim.service  ── capacity-routing gateway (bin/keepalive-shim.py)
        │   local-first · 4 lanes · queue-first · DeepSeek overflow · OOM self-heal
        ▼
:8001  vllm-qwen27b.service         ── vLLM (bin/serve-tqk8v4-fg.sh), Qwen3.6-27B-GPTQ-Int4
        ▲
        └── vllm-qwen27b-watchdog.timer (60s) → bin/vllm-watchdog.sh (wedge-restart, model-agnostic)
```

## ⭐ Concurrency config (2026-08-12) — 4 local lanes

The box now serves **4 concurrent requests locally** (was 1), so all three autonomous
harnesses fit with a spare and DeepSeek is a true last resort. Reliability was chosen over
single-stream speed.

- `serve-tqk8v4-fg.sh`: **`--enforce-eager`** (drops cudagraphs → +1.7 GB headroom: 683→2383 MiB/GPU)
  and **`--max-num-seqs 4`** (grounded in weicj's toolbox "validated up to max_num_seqs=4").
- `env/shim.env`: **`SHIM_LOCAL_BUDGET=4`** + **`SHIM_LOCAL_WAIT_SECS=15`** (queue-first: wait for a
  local slot before overflowing).
- Load-tested long-gen (4×moderate, 4×big-prefill, 1 big-ctx + 3 moderate): all local, 0 OOM,
  NRestarts=0, peak free ≥1497 MiB.
- Trade: single-stream **71 → 42 tok/s** (enforce-eager). Revert = restore a cudagraph serve-script
  variant from `archive/` and set `SHIM_LOCAL_BUDGET=2`.
- Concurrent prefills take a Triton fallback (the FlashQLA multi-prefill fast-path isn't in the
  cu128 build) — stable, just not the fused kernel.

## Layout

| Path | What |
|---|---|
| `systemd/` | Unit files → `/etc/systemd/system/` (`vllm-qwen27b`, `vllm-keepalive-shim`, watchdog `.service`+`.timer`, `oom.conf` drop-in) |
| `bin/` | Scripts → `~/.local/share/vllm-qwen27b/`: `serve-tqk8v4-fg.sh` (vLLM launch), `keepalive-shim.py` (gateway), `model-router.py` (retired predecessor), `vllm-watchdog.sh`, ops helpers |
| `env/` | `vllm-qwen27b.env` (CUDA/vLLM env, no secrets) and **`shim.env.example`** (copy → `shim.env`, add your DeepSeek key, `chmod 600`) |
| `templates/` | Active `chat_template-froggeric-v21.3.jinja`, `gencfg/generation_config.json`, and the full froggeric template history |
| `desktop/` | `swap-model.sh` (Qwen⇄GLM), KDE `.desktop` shortcuts, `qwen38-release-detector.sh` |
| `docs/` | Server guide, benchmarks, the overnight chaos/hardening report, AZ optimization notes |
| `setup/` | Original build/serve runbook + phase scripts (`fix-cuda128-glibc.sh`, `phase1-driver.sh`, `phase2-build-serve.sh`) |
| `archive/` | Historical `serve-tqk8v4-fg.sh` variants (qwen36, qwythos, fable, pre-router, direct8000) |

## Restore onto a host

1. Build the engine per the repo root (weicj build), into `~/Desktop/vLLM-2080Ti-Definitive/.venv`.
2. `cp bin/* ~/.local/share/vllm-qwen27b/` ; `cp -r templates/* ~/.local/share/vllm-qwen27b/` (adjust paths).
3. `cp env/vllm-qwen27b.env ~/.local/share/vllm-qwen27b/` ; `cp env/shim.env.example ~/.local/share/vllm-qwen27b/shim.env`, add your DeepSeek key, `chmod 600 shim.env`.
4. `sudo cp -r systemd/* /etc/systemd/system/` ; `sudo systemctl daemon-reload`.
5. `sudo systemctl enable --now vllm-qwen27b vllm-keepalive-shim vllm-qwen27b-watchdog.timer`.
6. Verify: `curl -s localhost:8000/v1/models` and a test chat.

## ⚠️ Secrets

The only secret in the live stack is the DeepSeek key in `shim.env` (`SHIM_REMOTE_KEY`). It is
**not** committed — only `shim.env.example` with a placeholder. Never commit the real `shim.env`.
