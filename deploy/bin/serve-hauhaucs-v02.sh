#!/usr/bin/env bash
# serve-hauhaucs-v02.sh -- the SAME HauhauCS W4A16 / TP=2 / 524K / TurboQuant k3v4_nc / MTP-3 service as
# serve-hauhaucs.sh, but on the integrated weicj v0.2.2-post3 (vLLM 0.29.1rc0, torch 2.13 cu130, py3.12) build
# living in $V02_ROOT. Selected via the pointer file ~/.local/share/vllm-qwen27b/active-serve.
# Rollback = write "serve-hauhaucs.sh" into that pointer and restart through engine-actuator.py (the 0.1.x tree
# at ~/Desktop/vLLM-2080Ti-Definitive and its .venv are untouched).
# Differences from serve-hauhaucs.sh (everything else byte-identical on purpose):
#   * python/venv/CUDA 13/GCC 15/extension caches come from the v0.2 tree (env overrides below);
#   * we cd into the v0.2 tree (python -m would otherwise import the OLD tree from the unit's WorkingDirectory);
#   * speculative-config gains disable_eagle_block_drop=true: upstream #53388 supersedes our
#     VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK retention patch (keeps the trailing aligned Mamba block under MTP).
set -euo pipefail
# experiment overrides (V02_* variables), written by the UP lane; absent = production defaults
[ -f /home/kevin/.local/share/vllm-qwen27b/v02.override.env ] && . /home/kevin/.local/share/vllm-qwen27b/v02.override.env
# lane S3/U2b (2026-10-02, Kevin approved 23:03): DEFAULT STACK = text-only (--language-model-only: no vision tower, image/video requests 400) + int4 lm_head +
# int4 MTP block (load-time RTN+MSE-clip quantization cached in <model>-u2cache; fidelity top-1 93.2% / KL 0.009 vs bf16 over 3,636 estate rows).
# Measured vs the bf16/vision build: weights 9.6 -> 7.97 GiB/rank, KV pool 711,996 -> 919,122 tokens (+29%), natural decode 64-68 -> 81-84 tok/s,
# 12-body estate pass 18.3-18.6 s (no run-2 cliff), evalkit 60/60, 483K needle correct.  ROLLBACK: export V02_STACK=0 in v02.override.env + engine-actuator restart.
if [ "${V02_STACK:-1}" = "1" ]; then
  export VLLM_U2_INT4_HEAD=${VLLM_U2_INT4_HEAD:-1} VLLM_U2_INT4_MTP=${VLLM_U2_INT4_MTP:-1}
  case " ${VLLM_SERVE_EXTRA_ARGS:-} " in *" --language-model-only "*) ;; *) VLLM_SERVE_EXTRA_ARGS="--language-model-only ${VLLM_SERVE_EXTRA_ARGS:-}"; export VLLM_SERVE_EXTRA_ARGS;; esac
fi
V02_ROOT=${V02_ROOT:-/home/kevin/Desktop/wt-integrate}
cd "$V02_ROOT"
export PATH="/home/kevin/.local/share/shim-gcc15:$V02_ROOT/.venv/bin:/usr/local/cuda-13/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/snap/bin"
export CUDA_HOME=/usr/local/cuda-13 CUDA_PATH=/usr/local/cuda-13
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15 NVCC_CCBIN=/usr/bin/g++-15
export TORCH_EXTENSIONS_DIR="$V02_ROOT/.deps/FlashQLA-SM70-SM75/.torch_extensions_vllm_flashqla_legacy"
export TORCHINDUCTOR_CACHE_DIR="$V02_ROOT/torchinductor-cache"
export FLASHQLA_ROOT="$V02_ROOT/.deps/FlashQLA-SM70-SM75"
export PYTHONPATH="$V02_ROOT:$FLASHQLA_ROOT" PYTHONSAFEPATH=1 PYTHONUNBUFFERED=1
export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1} FLASHINFER_WORKSPACE_BASE="$V02_ROOT" TRITON_CACHE_DIR="$V02_ROOT/triton-cache"
# retired 0.1.x-only knobs: make sure a stale env file cannot leak them into the new tree
# Model Runner V2 is the 0.2.x default but is 5-6x slower on TurboQuant+hybrid GDN+MTP here (10-15 vs 64-68 tok/s natural text, 2026-10-02); this script defaults to the V1 runner (V02_RUNNER=v2 to override).
if [ "${V02_RUNNER:-v1}" = "v1" ]; then export VLLM_USE_V2_MODEL_RUNNER=0; fi
# lane S2 (2026-10-02): grouped sm_75 TurboQuant decode attention (vllm/v1/attention/ops/tq_gqa_ext). The stock Triton TQ decode kernel
# dequantizes each cached token once per query head x speculative row (24x redundant) and was 81% of GPU time at 12 streams x 28K context.
# VLLM_TQ_GQA_CUDA=0 restores the stock kernel (instant rollback via v02.override.env). Prebuilt into $V02_ROOT/.deps/tq_gqa_build.
export VLLM_TQ_GQA_CUDA=${VLLM_TQ_GQA_CUDA:-1}
export VLLM_TQ_GQA_BUILD_DIR=${VLLM_TQ_GQA_BUILD_DIR:-$V02_ROOT/.deps/tq_gqa_build}
CC_DEFAULT='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64],"max_cudagraph_capture_size":64}'
unset VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK VLLM_PREFIX_CACHE_USE_RETAINED_MTP_BLOCK VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS || true
SPEC_DEFAULT='{"method":"mtp","num_speculative_tokens":3,"disable_eagle_block_drop":true}'
ARGS=(
  "$V02_ROOT/.venv/bin/python"
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
  ${V02_ASYNC_FLAG:---no-async-scheduling}
  --tensor-parallel-size 2
  --generation-config /home/kevin/.local/share/vllm-qwen27b/gencfg
  --gpu-memory-utilization ${VLLM_GPU_UTIL:-0.84}
  --compilation-config "${V02_COMPILATION_CONFIG:-$CC_DEFAULT}"
  --speculative-config "${V02_SPEC:-$SPEC_DEFAULT}"
  --max-model-len ${V02_MAXLEN:-524288}
  --hf-overrides '{"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144,"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25,"rope_theta":10000000}}'
  --enable-chunked-prefill
  --max-num-seqs ${V02_MAXSEQS:-16}
  --max-num-batched-tokens ${VLLM_MNBT:-3584}
  --scheduling-policy priority
  --kv-cache-dtype ${V02_KV_DTYPE:-turboquant_k3v4_nc}
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
# Optional upstream #243 SSD prefix-KV persistence (experimental, opt-in): V02_SSD_KV_DIR=/path V02_SSD_KV_CPU_BYTES=N
if [ -n "${V02_SSD_KV_DIR:-}" ]; then
  export PYTHONHASHSEED=0
  FP=$("$V02_ROOT/.venv/bin/python" "$V02_ROOT/tools/checkpoint_fingerprint.py" /home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven "")
  ARGS+=( --kv-transfer-config "$(python3 -c 'import json,os,sys;print(json.dumps({"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":int(sys.argv[2]),"spec_name":"TieringOffloadingSpec","secondary_tiers":[{"type":"fs","root_dir":os.path.join(sys.argv[1],"checkpoint-"+sys.argv[3])}]}},separators=(",",":")))' "$V02_SSD_KV_DIR" "${V02_SSD_KV_CPU_BYTES:-8589934592}" "$FP")" )
fi
if [ -n "${VLLM_SERVE_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  ARGS+=( ${VLLM_SERVE_EXTRA_ARGS} )
fi
exec "${ARGS[@]}"
