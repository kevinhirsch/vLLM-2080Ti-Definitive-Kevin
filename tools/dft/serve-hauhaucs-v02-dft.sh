#!/usr/bin/env bash
# Lane DFT serve variant: the LIVE serve script, unmodified, but with the tuned MTP-drafter model dir. ENV-GATED: DFT_MODEL_DIR unset => exactly the production boot.
# (For engine-actuator boots use dft_cfg.sh on/off, which routes the same --model through v02.override.env.)  Rollback: unset DFT_MODEL_DIR / dft_cfg.sh off.
if [ -n "${DFT_MODEL_DIR:-}" ]; then
  [ -f "$DFT_MODEL_DIR/model-mtp.safetensors" ] || { echo "DFT_MODEL_DIR has no model-mtp.safetensors" >&2; exit 2; }
  export VLLM_SERVE_EXTRA_ARGS="${VLLM_SERVE_EXTRA_ARGS:-} --model $DFT_MODEL_DIR"
fi
exec /home/kevin/.local/share/vllm-qwen27b/serve-hauhaucs-v02.sh "$@"
