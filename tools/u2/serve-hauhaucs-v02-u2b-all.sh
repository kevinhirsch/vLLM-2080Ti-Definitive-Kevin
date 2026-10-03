#!/usr/bin/env bash
# Lane U2b arm 'all' (int4 lm_head + MTP block + embeddings, and --language-model-only). Default-off env switches in the integrate branch; this wrapper only sets them and execs the LIVE serve script
# unmodified (never edit serve-hauhaucs-v02.sh).  Pre-built int4 tensors are read from <model>-u2cache (tools/u2/prebuild_headquant.py); a missing
# entry is quantized on the GPU at load (~seconds).  Rollback: point active-serve back at serve-hauhaucs-v02.sh and engine-actuator restart.
# Gate before keeping: tools/u2/fidelity.py numbers (see build log), boot log 'Available KV cache memory', estate pass, evalkit, needles 131000 + 524K.
export VLLM_U2_INT4_HEAD=1
export VLLM_U2_INT4_MTP=1
export VLLM_U2_INT4_EMBED=1
export VLLM_SERVE_EXTRA_ARGS="--language-model-only ${VLLM_SERVE_EXTRA_ARGS:-}"
exec /home/kevin/.local/share/vllm-qwen27b/serve-hauhaucs-v02.sh "$@"
