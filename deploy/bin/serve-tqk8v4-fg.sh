#!/usr/bin/env bash
# Auto-generated from the validated live TQK8V4-fast server (ground-truth argv).
# Env is supplied by systemd EnvironmentFile (vllm-qwen27b.env).
# 2026-07-17: --chat-template overrides the model's built-in Qwen 3.6 template with
#   froggeric's patched v21.3 (chat_template-froggeric-v21.3.jinja) — fixes tool-call
#   formatting / reasoning-hallucination bugs. ROLLBACK: delete the two --chat-template
#   lines below and restart; the model's own chat_template.jinja takes over again.
set -euo pipefail
export VLLM_SUFFIX_OVERLAY=0
export VLLM_MTP_DRAFT_CAP=3
export VLLM_SUFFIX_COVER_MIN=4
export VLLM_SUFFIX_OVERLAY_MIN=2

# 2026-07-17 chaos-tested envelope (2x2080Ti, util 0.88, WORKSPACE_RESERVE 262144):
#   - gpu-memory-utilization 0.88 (0.92 OOM'd under 2-concurrent: 19 MiB free).
#   - This box = ONE big request (up to ~256K ctx, single-concurrency) OR TWO moderate
#     (<=~30K ctx each). NOT two long-context at once (a single 200K req peaks 97.6% VRAM).
#   - Long context is prefill-bound + slow (~850 tok/s prefill). max-num-seqs 2 queues extras.
ARGS=(
  /home/kevin/Desktop/vLLM-2080Ti-Definitive/.venv/bin/python
  -m
  vllm.entrypoints.openai.api_server
  --host
  0.0.0.0
  --port
  8001
  --model
  /home/kevin/Desktop/models/Qwen3.8-27B-GPTQ-Int4
  --served-model-name
  qwen-local
  qwen3.6:27b
  qwen3.8-27b-gptq-int4
  qwen27b-int4-tqk8v4-two250K-mtp3-text-only-cu128
  --dtype
  half
  --tensor-parallel-size
  2
  --generation-config
  /home/kevin/.local/share/vllm-qwen27b/gencfg
  --gpu-memory-utilization
  0.82
  --quantization
  gptq_marlin
  --compilation-config
  '{"cudagraph_mode":"PIECEWISE"}'
  --max-model-len
  256000
  --enable-chunked-prefill
  --max-num-seqs
  8
  --max-num-batched-tokens
  2560
  --kv-cache-dtype
  turboquant_k8v4
  --mamba-cache-mode
  align
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --language-model-only
  --skip-mm-profiling
  --chat-template
  /home/kevin/.local/share/vllm-qwen27b/chat_template-froggeric-v22-official.jinja
  --reasoning-parser
  qwen3
  --tool-call-parser
  qwen3_xml
  --enable-auto-tool-choice
  --additional-config
  '{"gdn_prefill_backend":"flashqla_legacy"}'
)

exec "${ARGS[@]}"
