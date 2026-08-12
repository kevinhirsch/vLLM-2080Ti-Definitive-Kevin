#!/usr/bin/env bash
# swap-model — flip the 2x2080Ti (44GB VRAM) between the vLLM daily driver
# (Qwen3.6, systemd) and GLM-4.5-Air (llama.cpp, on-demand). They can't coexist
# in 44GB, so this STOPS whatever is live before starting the requested model.
# Whatever is active answers OpenAI-style on http://0.0.0.0:8000, so Hermes/AZ
# transparently use it (llama-server serves its one model regardless of the
# requested model name).
#
#   swap-model qwen     -> Qwen3.6-27B  (vLLM, ~86 tok/s, the daily driver)
#   swap-model glm      -> GLM-4.5-Air  (llama.cpp IQ2_M, fast ~12 tok/s)
#   swap-model glm-q4   -> GLM-4.5-Air  (llama.cpp Q4_K_M, best quality, slower)
#   swap-model status   -> what's live on :8000
#   swap-model stop     -> stop everything
#
# Boot default = Qwen3.6 (systemd-enabled). A reboot returns to Qwen.
set -uo pipefail

PORT=8000
VLLM_SVC=vllm-qwen27b
LLAMA_SERVER=/home/kevin/llama.cpp/build/bin/llama-server
GLM_IQ2=/home/kevin/Desktop/models/GLM-4.5-Air-UD-IQ2_M.gguf                                  # NVMe (Samsung /mnt/storage removed 2026-08-07)
GLM_Q4=/run/media/kevin/Master/models-archive/GLM-4.5-Air-GGUF/Q4_K_M/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf  # WD cold archive (slow)
PIDFILE=/home/kevin/.local/share/vllm-qwen27b/glm-server.pid
LOG=/home/kevin/.local/share/vllm-qwen27b/glm-server.log

health(){ curl -fsS -m2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }
wait_health(){ for i in $(seq 1 70); do health && { echo "  ✓ live on :$PORT"; return 0; }; sleep 3; done; echo "  ✗ did not come up — check log"; return 1; }
wait_vram(){ for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-9999}" -lt 1200 ] 2>/dev/null && return 0; sleep 2; done; }

glm_running(){ [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; }
qwen_running(){ systemctl is-active --quiet "$VLLM_SVC"; }

stop_glm(){ if glm_running; then echo "· stopping GLM (llama-server)"; kill "$(cat "$PIDFILE")" 2>/dev/null; sleep 2; kill -9 "$(cat "$PIDFILE")" 2>/dev/null; fi; rm -f "$PIDFILE"; }
stop_qwen(){ if qwen_running; then echo "· stopping Qwen3.6 (vLLM)"; sudo systemctl stop "$VLLM_SVC"; fi; }
stop_all(){ stop_glm; stop_qwen; wait_vram; }

start_qwen(){ echo "→ starting Qwen3.6 (vLLM)"; sudo systemctl start "$VLLM_SVC"; wait_health; }
start_glm(){  # $1=gguf  $2=ngl  $3=label
  echo "→ starting GLM-4.5-Air (llama.cpp $3, ngl=$2)"
  nohup "$LLAMA_SERVER" -m "$1" --host 0.0.0.0 --port "$PORT" -ngl "$2" -c 8192 -t 8 \
    --no-warmup --alias glm-4.5-air > "$LOG" 2>&1 &
  echo $! > "$PIDFILE"; wait_health
}

case "${1:-status}" in
  qwen|qwen3.6)      stop_all; start_qwen ;;
  glm|glm-air)       stop_all; start_glm "$GLM_IQ2" 40 "IQ2_M" ;;
  glm-q4|glm-quality) stop_all; start_glm "$GLM_Q4" 24 "Q4_K_M" ;;
  stop)              stop_all; echo "all stopped." ;;
  status)
    echo "== swap-model status =="
    if   qwen_running; then echo "ACTIVE: Qwen3.6 (vLLM systemd)"
    elif glm_running;  then echo "ACTIVE: GLM-4.5-Air (llama-server pid $(cat "$PIDFILE"))"
    else echo "ACTIVE: none"; fi
    if health; then curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
      | python3 -c "import sys,json; print('  serving:', [m['id'] for m in json.load(sys.stdin).get('data',[])])" 2>/dev/null || echo "  :$PORT up"
    else echo "  :$PORT not responding"; fi
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu /' ;;
  *) echo "usage: swap-model <qwen|glm|glm-q4|stop|status>"; exit 1 ;;
esac
