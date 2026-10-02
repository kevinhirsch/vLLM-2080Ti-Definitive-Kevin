#!/usr/bin/env bash
# serve-hauhaucs.sh -- serve HauhauCS's "Aggressive" abliterated Qwen3.8-27B
# (twolven's vLLM build: HauhauCS's exact weights, dequantized from the Q8_K_P
# GGUF and re-quantized with llm-compressor to compressed-tensors W4A16, g128,
# ASYMMETRIC; MTP/NextN head + vision tower + lm_head kept in BF16).
#
# Kevin 2026-09-10: "vLLM, as almost identical a way as we are currently using
# it." This is serve-abliterated.sh with exactly TWO changes: the model path and
# the served names (a distinct estate-hauhaucs alias, PLUS every incumbent alias
# so the runner/estate keep working unchanged -- the 2026-09-10 alias-bug lesson).
# Everything else (TP=2, 524k yarn, TurboQuant KV, MTP-3, cudagraphs, template,
# parsers, gdn backend) is byte-identical, so a bake-off measures the WEIGHTS,
# not the config.
#
# Known unknown: this quant is asymmetric (zero-points). Our fork runs
# gptq_marlin on SM75; whether that path accepts zero-points is decided by the
# load itself -- if it fails, switch-model.sh abliterated reverts.
# Candidate caveat (twolven): avoid greedy decoding (temperature 0) -- loops
# without repetition_penalty 1.05. Our gencfg default is temp 0.6, so normal
# clients are fine; evalkit inherits gencfg too.
set -euo pipefail
ARGS=(
  /home/kevin/Desktop/vLLM-2080Ti-Definitive/.venv/bin/python
  -m vllm.entrypoints.openai.api_server
  --host 0.0.0.0 --port 8001
  --model /home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven
  --served-model-name
    estate-hauhaucs
    qwen3.8-27b-hauhaucs-aggressive
    estate-abliterated
    qwen3.8-27b-abliterated
    qwen-local
    qwen3.8-27b-uncensored
    qwen3.6:27b
    qwen3.8-27b-gptq-int4
    qwen27b-int4-tqk8v4-two250K-mtp3-text-only-cu128
  --dtype half
  --no-async-scheduling
  --tensor-parallel-size 2
  --generation-config /home/kevin/.local/share/vllm-qwen27b/gencfg
  --gpu-memory-utilization 0.84
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}'
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  --max-model-len 524288
  --hf-overrides '{"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144,"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25,"rope_theta":10000000}}'
  --enable-chunked-prefill
  --max-num-seqs 16
  --max-num-batched-tokens ${VLLM_MNBT:-3584}
  --scheduling-policy priority
  --kv-cache-dtype turboquant_k3v4_nc
  --mamba-cache-mode align
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --limit-mm-per-prompt '{"image": 4}'
  --mm-processor-kwargs '{"max_pixels": 1000000}'
  --chat-template /home/kevin/.local/share/vllm-qwen27b/chat_template-froggeric-v22-official.jinja
  --reasoning-parser qwen3
  --tool-call-parser qwen3_xml
  --enable-auto-tool-choice
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'
)
if [ -n "${VLLM_SERVE_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  ARGS+=( ${VLLM_SERVE_EXTRA_ARGS} )
fi
exec "${ARGS[@]}"
