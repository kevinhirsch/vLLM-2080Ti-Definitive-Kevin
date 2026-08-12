# vLLM Server Guide — HNET00 (Qwen3.6-27B on 2× RTX 2080 Ti)

Last updated: 2026-07-17. This box serves an OpenAI-compatible API on the LAN.

| What | Value |
|---|---|
| Endpoint | `http://10.0.1.225:8000/v1` (listens on `0.0.0.0:8000`) |
| Model names | `qwen3.6:27b` (alias) and `qwen27b-int4-tqk8v4-two250K-mtp3-text-only-cu128` |
| Model weights | `~/Desktop/models/Qwen3.6-27B-GPTQ-Int4` |
| systemd service | `vllm-qwen27b.service` (enabled, starts on boot) |
| vLLM install | `~/Desktop/vLLM-2080Ti-Definitive` (custom SM75 fork, venv inside) |
| Expected perf | 64–86 tok/s decode (MTP-acceptance dependent), ~1600 tok/s prefill, ~19.6 GB VRAM per GPU |
| Main client | Hermes box at `10.0.1.18` |

## Desktop shortcuts

- **vLLM - Restart** — restarts the service, waits until `/health` returns 200 (~1–2 min model load). KDE asks for your password.
- **vLLM - Stop** — stops the server to free VRAM. It will still start on next boot unless you also `sudo systemctl disable vllm-qwen27b`.
- **vLLM - Logs** — live journal follow (Ctrl+C to stop).
- **vLLM - Troubleshoot** — full diagnostic: service state, health, a real inference test, GPU/power check, disk, recent errors, and the known-issues cheat sheet.
- **vLLM - Edit Config** — opens the three config files (below) in Kate.
- **vLLM - Guide** — opens this file.

## The three config files

1. **`~/.local/share/vllm-qwen27b/vllm-qwen27b.env`** — environment variables (CUDA paths, TurboQuant/FlashInfer tuning, the gcc-13 pin). Editable as your user. **After editing: restart the service.**
2. **`~/.local/share/vllm-qwen27b/serve-tqk8v4-fg.sh`** — the vLLM command line (all `--flags`). Editable as your user. **After editing: restart the service.**
3. **`/etc/systemd/system/vllm-qwen27b.service`** — the unit itself (restart policy, ordering). Root-owned: edit with `sudoedit /etc/systemd/system/vllm-qwen27b.service`, then `sudo systemctl daemon-reload` **before** restarting.

## Config options you might actually change

All of these live in `serve-tqk8v4-fg.sh` unless noted. This config was carefully validated — change one thing at a time and re-run the Troubleshoot shortcut afterwards.

### Capacity / memory
- `--gpu-memory-utilization 0.92` — fraction of each GPU vLLM claims. Lower (e.g. `0.85`) if you need VRAM for something else; raising it risks OOM at startup.
- `--max-model-len 256000` — max context window. Lowering (e.g. `65536`) frees KV-cache memory and speeds startup; raise only if you have headroom.
- `--max-num-seqs 2` — concurrent requests. This rig is tuned for 1–2 heavy clients; raising it splits KV cache and slows each stream.
- `--max-num-batched-tokens 2560` — chunked-prefill batch size. Bigger = faster prefill but more latency spikes for the other stream.
- `--kv-cache-dtype turboquant_k8v4` — quantized KV cache (the "two250K" trick that makes 256K context fit). Don't change unless you also change `--max-model-len` drastically.

### Speed
- `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` — MTP speculative decoding ("mtp3"). The main source of the 64–86 tok/s spread: technical/coherent text accepts more speculated tokens and runs faster. Set to 2 or remove it entirely if you suspect spec-decode bugs.
- `--compilation-config '{...FULL_AND_PIECEWISE...}'` — CUDA-graph capture. Leave alone.
- `--enable-prefix-caching` — reuses KV for repeated prompt prefixes (big win for chat clients like Hermes).
- GPU power cap: **260 W** with persistence mode, applied at boot by `nvidia-powerprep.service` (staged in `~/Desktop/right_thing/vllm-2080ti-handoff/vllm-qwen27b-configs/`). GPU0 pins the default 250 W cap without it. Manual re-apply: `sudo nvidia-smi -pm 1 && sudo nvidia-smi -pl 260`.

### API behavior
- `--served-model-name <name1> <name2>` — the model IDs clients may request. Add another alias here if a client insists on a different name.
- `--reasoning-parser qwen3` / `--tool-call-parser qwen3_xml` / `--enable-auto-tool-choice` — Qwen3 thinking + tool-calling support. Required for Hermes tool calls.
- `--host 0.0.0.0 --port 8000` — bind address/port. If you change the port, update Hermes too.
- Disable thinking per-request from the client with `"chat_template_kwargs": {"enable_thinking": false}`.

### Env file (`vllm-qwen27b.env`) — handle with care
- `CC/CXX/CUDAHOSTCXX/NVCC_CCBIN → gcc-13/g++-13` — **do not remove.** On Ubuntu 26.04 the default gcc is 15, which CUDA 12.8's runtime nvcc JIT (FlashInfer) rejects; without this pin the server crash-loops.
- `VLLM_TURBOQUANT_*`, `VLLM_SM75_SPEC_SYNC_MODE=safe`, `TORCH_CUDA_ARCH_LIST=7.5` — SM75/TurboQuant tuning from the validated setup. Leave alone unless following the fork's docs.
- `VLLM_ENFORCE_STRICT_TOOL_CALLING=1` — strict tool-call schema validation.

## Common operations (terminal)

```bash
systemctl status vllm-qwen27b                 # state + recent log lines
sudo systemctl restart vllm-qwen27b           # restart (what the shortcut does)
sudo systemctl stop vllm-qwen27b              # stop, frees VRAM
journalctl -u vllm-qwen27b -f                 # follow logs
journalctl -u vllm-qwen27b -b --no-pager      # everything since boot
curl -s http://127.0.0.1:8000/health          # 200 = up
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
nvidia-smi                                    # VRAM/power/temps
```

Quick inference test:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.6:27b",
  "messages": [{"role": "user", "content": "Say hi"}],
  "max_tokens": 20,
  "chat_template_kwargs": {"enable_thinking": false}
}' | python3 -m json.tool
```

## Troubleshooting

**Start with the "vLLM - Troubleshoot" shortcut** — it checks all of the below automatically.

| Symptom | Likely cause → fix |
|---|---|
| Crash-loop, log says `unsupported GNU version` | gcc-13 pin missing from the env file (26.04 defaults to gcc-15, CUDA 12.8 nvcc rejects it). Restore the four `*=/usr/bin/g*-13` lines, then `sudo systemctl reset-failed vllm-qwen27b && sudo systemctl restart vllm-qwen27b`. |
| Crash-loop, `Failed to infer device type` | Usually a downstream symptom of the gcc issue above — scroll up in `journalctl -u vllm-qwen27b -b` to find the first real error. |
| Rebuild of the fork fails at cmake CUDA probe (`exception specification is incompatible`, `cospi`/`sinpi`/`rsqrt`) | glibc 2.43 vs CUDA 12.8 headers. Run `sudo ~/Desktop/vllm-setup/fix-cuda128-glibc.sh` (idempotent, backs up the header), then rebuild. |
| Slow decode / GPU0 throttling | Power cap fell back to 250 W. Check `nvidia-smi --query-gpu=power.limit --format=csv`; fix with `sudo nvidia-smi -pm 1 && sudo nvidia-smi -pl 260`; make sure `nvidia-powerprep.service` is enabled. |
| CUDA OOM at startup | Something else is holding VRAM (`nvidia-smi` to see), or config was changed. Free the VRAM or lower `--gpu-memory-utilization` / `--max-model-len`. |
| Crash under real traffic (not the health check), log says `Workspace is locked but allocation from 'turboquant_attn.py:...:_continuation_prefill' requires X MB, current size is Y MB` | The TurboQuant continuation-prefill scratch buffer locks its size after CUDA-graph capture and can't grow. A large enough real prompt (long context/tool schemas — client health checks with tiny prompts won't trigger this) exceeds it and kills both TP workers. **Fix:** raise `VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS` in the env file (bumped 65536→131072 on 2026-07-17 after this hit Agent Zero's traffic) and restart. VRAM headroom is generous (~2GB free/GPU), so doubling it again is safe if it recurs. |
| Port 8000 up but Hermes can't connect | Check from Hermes: `curl http://10.0.1.225:8000/health`. Verify LAN/firewall and that Hermes' provider uses model `qwen3.6:27b`. |
| Restart limit hit (`start-limit-hit`) | 5 failed starts in 10 min trips the limiter. `sudo systemctl reset-failed vllm-qwen27b` then restart (the Restart shortcut does this for you). |
| Node acting up after driver/kernel updates | 26.04 needs: gcc-13 toolchain pin, the CUDA header patch (only for rebuilds), and the runtime env pin. The full rebuild runbook is `~/Desktop/vllm-setup/RUNBOOK.md` (phase1 = driver, phase2 = build+serve). |

Model load takes **~1–2 minutes** — `/health` failing right after a restart is normal; the Restart shortcut waits for it.

## Deeper documentation

- **Hardware/deployment spec (the source of truth):** `~/Desktop/right_thing/vllm-2080ti-handoff/HANDOFF.md`
- **Rebuild runbook (fresh 26.04 install):** `~/Desktop/vllm-setup/RUNBOOK.md` + `phase1-driver.sh` / `phase2-build-serve.sh` / `fix-cuda128-glibc.sh`
- **Setup narrative/history:** `~/Desktop/right_thing/vllm-2080ti-handoff/vllm-2080ti-setup.md`
- **The fork:** https://github.com/weicj/vLLM-2080Ti-Definitive
- **Upstream vLLM docs:** https://docs.vllm.ai — engine args reference: https://docs.vllm.ai/en/latest/serving/engine_args.html
- **OpenAI-compatible API reference:** https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
