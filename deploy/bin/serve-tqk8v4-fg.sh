#!/usr/bin/env bash
# EXP-036 MTP draft-load A/B launcher (maintenance window only)
# Auto-generated from the validated live TQK8V4-fast server (ground-truth argv).
# Env is supplied by systemd EnvironmentFile (vllm-qwen27b.env).
# 2026-07-17: --chat-template overrides the model's built-in Qwen 3.6 template with
#   froggeric's patched v21.3 (chat_template-froggeric-v21.3.jinja) — fixes tool-call
#   formatting / reasoning-hallucination bugs. ROLLBACK: delete the two --chat-template
#   lines below and restart; the model's own chat_template.jinja takes over again.
set -euo pipefail
# 2026-08-24 EXP-048 r6: K=3, ALL EXP-036 header exports removed (cap!=2 crashes/deadlocks [M-7]; QWOPUS ignore_torch_compile deadlocks K=3 chained forwards; SUFFIX trio unproven-at-K3 and arm H ran without them). Arm-H-exact env. Rollback: cp the newest BANKED-k2-pre-k3-* over this file and restart.

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
  0.84
  --quantization
  gptq_marlin
  --compilation-config
  '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}'
  --speculative-config
  '{"method":"mtp","num_speculative_tokens":3}'
  --max-model-len
  524288
  --hf-overrides
  '{"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144,"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25,"rope_theta":10000000}}'
  --enable-chunked-prefill
  --max-num-seqs
  16
  --max-num-batched-tokens
  3584
  --scheduling-policy
  priority
  --kv-cache-dtype
  turboquant_k3v4_nc
  --mamba-cache-mode
  align
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --limit-mm-per-prompt
  '{"image": 4}'
  --mm-processor-kwargs
  '{"max_pixels": 1000000}'
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

# 2026-09-05: optional extra CLI flags from the EnvironmentFile (VLLM_SERVE_EXTRA_ARGS), so a
# frontier window can A/B a CLI-only setting by editing vllm-qwen27b.env instead of this file
# (first use: window 92, VLLM_SERVE_EXTRA_ARGS=--no-async-scheduling). Unset/empty = argv
# byte-identical to before this hook. Word-split on purpose: flags only, never paths with spaces.
if [ -n "${VLLM_SERVE_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  ARGS+=( ${VLLM_SERVE_EXTRA_ARGS} )
fi

exec "${ARGS[@]}"
