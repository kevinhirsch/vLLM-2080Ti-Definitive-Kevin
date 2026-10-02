#!/usr/bin/env bash
# serve-profile-v02.sh -- run an upstream profile command (saved from `DRY_RUN=1 launcher.sh ...`) under the engine unit.
# Pointer: echo serve-profile-v02.sh > ~/.local/share/vllm-qwen27b/active-serve ; args file: V02_PROFILE_ARGS (override env).
set -euo pipefail
[ -f /home/kevin/.local/share/vllm-qwen27b/v02.override.env ] && . /home/kevin/.local/share/vllm-qwen27b/v02.override.env
V02_ROOT=${V02_ROOT:-/home/kevin/Desktop/wt-integrate}
cd "$V02_ROOT"
export PATH="/home/kevin/.local/share/shim-gcc15:$V02_ROOT/.venv/bin:/usr/local/cuda-13/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/snap/bin"
export CUDA_HOME=/usr/local/cuda-13 CUDA_PATH=/usr/local/cuda-13 CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15 NVCC_CCBIN=/usr/bin/g++-15
export TORCH_EXTENSIONS_DIR="$V02_ROOT/.deps/FlashQLA-SM70-SM75/.torch_extensions_vllm_flashqla_legacy"
export TORCHINDUCTOR_CACHE_DIR="$V02_ROOT/torchinductor-cache"
export FLASHQLA_ROOT="$V02_ROOT/.deps/FlashQLA-SM70-SM75"
export PYTHONPATH="$V02_ROOT:$FLASHQLA_ROOT" PYTHONSAFEPATH=1 PYTHONUNBUFFERED=1
export FLASHINFER_ENABLE_AOT=1 FLASHINFER_WORKSPACE_BASE="$V02_ROOT" TRITON_CACHE_DIR="$V02_ROOT/triton-cache"
export VLLM_SM75_SPEC_SYNC_MODE=${V02_SPEC_SYNC:-safe} VLLM_DISABLE_TILELANG=1 VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=${V02_FULL_GRAPH:-1} VLLM_ENFORCE_STRICT_TOOL_CALLING=0
export VLLM_TURBOQUANT_USE_FLASHINFER_PREFILL=1 VLLM_TURBOQUANT_FLASHINFER_BACKEND=fa2 VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE=1 VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH=1
unset VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK VLLM_PREFIX_CACHE_USE_RETAINED_MTP_BLOCK VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE VLLM_TURBOQUANT_CONTINUATION_BOUNDS_CHECK VLLM_TURBOQUANT_STAGE1_QTILE VLLM_FLASHQLA_VARLEN_LOOP VLLM_ROPE_MAX_POSITION || true
if [ "${V02_RUNNER:-v2}" = "v1" ]; then export VLLM_USE_V2_MODEL_RUNNER=0; fi
eval "exec \"$V02_ROOT/.venv/bin/python\" $(cat "${V02_PROFILE_ARGS:?V02_PROFILE_ARGS not set}")"
