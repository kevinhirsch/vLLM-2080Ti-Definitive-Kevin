#!/usr/bin/env bash
# Lane K7 engine-off window: GPTQ re-quant of the abliterated W4A16 model to rotated symmetric per-channel int4 (GPU1),
# with the CPU-or-GPU fidelity gate of the GPTQ weights trailing it layer by layer on GPU0.  Needs the engine STOPPED
# (both GPUs ~free).  ~45-60 min.  Resumable: re-running continues GPTQ from the last finished layer.
# Output (new dir, originals only read): /home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A4rot128-gptq-k7
# Restore: nothing to restore (no engine/config change); just restart the engine/stack default after the window.
set -euo pipefail
OUT=/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A4rot128-gptq-k7
PY=/home/kevin/Desktop/wt-integrate/.venv/bin/python
LOG=/home/kevin/projects/lanes/k7
VARS=${K7_VARS:-k7_pc,k7_gpc,k7_gagpc,k7_gagpcin}
free_g=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "$free_g" -ge 20 ] || { echo "K7 window: only ${free_g} GB free on / (need >= 20)"; exit 2; }
if curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null; then echo "K7 window: engine is UP - stop it first"; exit 2; fi
cd /home/kevin/Desktop/wt-k7
CUDA_VISIBLE_DEVICES=1 "$PY" tools/k7/gptq_requant.py --device cuda --out "$OUT" > "$LOG/gptq_window.log" 2>&1 &
GP=$!
sleep 60
cd /home/kevin/Desktop/wt-lp
CUDA_VISIBLE_DEVICES=0 K7_GPTQ_DIR="$OUT" K7_STATS="$LOG/vg_stats_gptq.json" \
  VG_PLUGINS=/home/kevin/Desktop/wt-k7/tools/k7/k7_variants.py \
  "$PY" tools/lp/variant_gate.py --device cuda --gpu-need-mib 3000 --variants "$VARS" --windows 6 \
  --out "$LOG/vg_gptq.json" > "$LOG/vg_gptq.log" 2>&1 &
VP=$!
wait $GP; echo "gptq rc=$?"
wait $VP; echo "gate rc=$?"
grep -E "k7_.*top1" "$LOG/vg_gptq.log" | cut -c1-400 || true
