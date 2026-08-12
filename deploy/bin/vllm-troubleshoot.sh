#!/usr/bin/env bash
# Full diagnostic for the vLLM server (vllm-qwen27b).
# Launched from the "vLLM - Troubleshoot" desktop shortcut (runs in konsole).
set -u
SERVICE=vllm-qwen27b
API=http://127.0.0.1:8000

hr() { echo; echo "=== $1 ==="; }

hr "1/8 Service state"
systemctl is-active "$SERVICE" && echo "(active)" || echo "(NOT active)"
systemctl show "$SERVICE" -p ActiveState,SubState,NRestarts,ExecMainStartTimestamp --no-pager

hr "2/8 API health"
code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$API/health" || true)
echo "GET /health -> HTTP $code $( [ "$code" = 200 ] && echo '(OK)' || echo '(FAIL)')"
echo "Models served:"
curl -s -m 3 "$API/v1/models" | python3 -c 'import json,sys; [print("  -", m["id"]) for m in json.load(sys.stdin)["data"]]' 2>/dev/null \
    || echo "  (could not list models)"

hr "3/8 Live inference test (short completion)"
resp=$(curl -s -m 60 "$API/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.6:27b","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}' || true)
echo "$resp" | python3 -c 'import json,sys; print("  Model replied:", json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>/dev/null \
    || { echo "  Inference test FAILED. Raw response:"; echo "  $resp" | head -5; }

hr "4/8 Port 8000 listening"
ss -tln | grep -q ':8000 ' && ss -tln | grep ':8000 ' || echo "NOT listening on 8000"

hr "5/8 GPUs (expect: 2x 2080 Ti, persistence Enabled, cap 260 W, ~19.6 GB used each when up)"
nvidia-smi --query-gpu=index,name,persistence_mode,power.limit,power.draw,temperature.gpu,memory.used,memory.total --format=csv
systemctl is-active nvidia-powerprep >/dev/null 2>&1 \
    && echo "nvidia-powerprep.service: $(systemctl is-active nvidia-powerprep)" \
    || echo "nvidia-powerprep.service: NOT active (power cap may fall back to 250 W after reboot)"

hr "6/8 Disk space (model dir needs headroom for caches)"
df -h / /home/kevin/Desktop/models 2>/dev/null | awk 'NR==1 || !seen[$0]++'

hr "7/8 Recent errors in the journal (last 200 lines)"
journalctl -u "$SERVICE" -n 200 --no-pager -q | grep -iE 'error|traceback|CUDA (error|out of memory)|failed|unsupported GNU|workspace is locked' | tail -15 \
    || echo "(none found)"

hr "8/8 Known 26.04 gotchas — check these if the server crash-loops"
cat <<'EOF'
  a) "unsupported GNU version" in logs -> runtime nvcc JIT grabbed gcc-15.
     Fix: gcc-13 pin must be in vllm-qwen27b.env (CC/CXX/CUDAHOSTCXX/NVCC_CCBIN
     = .../gcc-13 | g++-13). It is there by default -- check it wasn't removed.
  b) Rebuilding the fork fails at the cmake CUDA probe ("exception specification
     is incompatible", cospi/sinpi/rsqrt) -> glibc 2.43 vs CUDA 12.8 headers.
     Fix: sudo ~/Desktop/vllm-setup/fix-cuda128-glibc.sh   (idempotent)
  c) GPU0 throttling / perf regression -> power cap fell back to 250 W.
     Fix: sudo nvidia-smi -pm 1 && sudo nvidia-smi -pl 260
     (persisted across boots by nvidia-powerprep.service -- see 5/8 above)
  d) Crash-loop with "Failed to infer device type" -> usually (a); check the
     first error above it in:  journalctl -u vllm-qwen27b -b --no-pager | less
  e) "Workspace is locked but allocation from '...continuation_prefill'
     requires X MB, current size is Y MB" -> TurboQuant continuation-prefill
     scratch buffer locked too small for a real (long) prompt. Only shows up
     under real traffic, not tiny health-check prompts.
     Fix: raise VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS in
     vllm-qwen27b.env (was 65536, bumped to 131072 on 2026-07-17) and restart.
  Full guide: ~/Desktop/vLLM-Server-Guide.md  ("vLLM - Guide" shortcut)
EOF
echo
read -rp "Press Enter to close..."
