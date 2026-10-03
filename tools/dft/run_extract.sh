#!/usr/bin/env bash
# run_extract.sh MANIFEST OUTDIR [extract_hidden.py args]  -- same environment as the production serve script, no U2 arms, then extract_hidden.py
set -euo pipefail
S=/home/kevin/.local/share/vllm-qwen27b/serve-hauhaucs-v02.sh
V02_ROOT=/home/kevin/Desktop/wt-integrate
eval "$(sed -n '/^V02_ROOT=/,/^CC_DEFAULT=/p' $S | grep -v '^CC_DEFAULT=' | grep -v '^\[ -f')"
unset VLLM_U2_INT4_HEAD VLLM_U2_INT4_MTP VLLM_U2_INT4_EMBED VLLM_SERVE_EXTRA_ARGS || true
export PYTHONPATH="/home/kevin/projects/lanes/dft:$PYTHONPATH"
cd "$V02_ROOT"
exec "$V02_ROOT/.venv/bin/python" /home/kevin/projects/lanes/dft/extract_hidden.py "$@"
