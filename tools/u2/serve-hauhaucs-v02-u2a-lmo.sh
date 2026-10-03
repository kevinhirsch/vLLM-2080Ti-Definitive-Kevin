#!/usr/bin/env bash
# Lane U2 arm 1a (UNTESTED, default off): text-only serving to free VRAM for prefix-cache pages.
# --language-model-only skips building the vision tower (0.858 GiB bf16 total in model.safetensors, ~0.43-0.86 GiB/GPU depending on TP sharding)
# and the encoder-cache profiling pass (log: "Encoder cache ... profiled with 3 image items"). Image/video requests then 400.
# Evidence images are unused: vllm:mm_cache_queries_total == 0 on the 20:06 boot; 0/40 flightrec bodies contain image_url.
# NOT proven for the whole week (gateway telemetry does not record content types). Rollback: use serve-hauhaucs-v02.sh.
# Expect KV pool 6.74 GiB -> ~7.2-7.6 GiB/GPU (+7-13% pages). S2 must verify with the boot log "Available KV cache memory".
export VLLM_SERVE_EXTRA_ARGS="--language-model-only ${VLLM_SERVE_EXTRA_ARGS:-}"
exec /home/kevin/.local/share/vllm-qwen27b/serve-hauhaucs-v02.sh "$@"
