#!/usr/bin/env bash
# dft_cfg.sh on VARIANT_DIR | off   -- switch the production serve script to the tuned drafter via ~/.local/share/vllm-qwen27b/v02.override.env
# (the serve script sources it; `--model` given twice -> last wins, verified with the vLLM CLI parser).  `off` restores the saved override file
# byte-for-byte (instant rollback; then engine-actuator restart).  The original model dir and serve-hauhaucs-v02.sh are never edited.
set -euo pipefail
O=${DFT_OVERRIDE:-/home/kevin/.local/share/vllm-qwen27b/v02.override.env}
B=${DFT_SAVED:-/home/kevin/projects/lanes/dft/override.saved}
case "${1:-}" in
  on)
    D=${2:?variant dir}; [ -f "$D/model-mtp.safetensors" ] || { echo "no variant at $D"; exit 2; }
    [ -f "$B" ] || cp -p "$O" "$B"            # keep the FIRST saved original
    prev=$(grep -h '^export VLLM_SERVE_EXTRA_ARGS=' "$B" | tail -1 | sed 's/^export VLLM_SERVE_EXTRA_ARGS=//; s/^"//; s/"$//' || true)
    { grep -v '^export VLLM_SERVE_EXTRA_ARGS=' "$B" || true; echo "export VLLM_SERVE_EXTRA_ARGS=\"${prev:+$prev }--model $D\""; } > "$O"
    cat "$O" ;;
  off)
    [ -f "$B" ] || { echo "nothing saved"; exit 0; }
    cp -p "$B" "$O"; rm -f "$B"; cat "$O" ;;
  *) echo "usage: $0 on VARIANT_DIR | off"; exit 2 ;;
esac
