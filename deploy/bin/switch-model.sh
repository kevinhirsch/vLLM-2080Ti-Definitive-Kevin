#!/usr/bin/env bash
# switch-model.sh — swap the served local model (e.g. Qwen3.6-27B <-> Qwen3.8-27B) safely.
#
#   ./switch-model.sh list              # show available models + what's live now
#   ./switch-model.sh 3.6               # switch to Qwen3.6-27B-GPTQ-Int4 (current champion)
#   ./switch-model.sh 3.8               # switch to Qwen3.8-27B-AWQ-Int4 (once quantized)
#   ./switch-model.sh /path/to/model    # switch to an explicit path
#   ./switch-model.sh rollback          # restore the previous serve script + restart
#
# WHY THIS EXISTS: clients hardcode the model id "qwen3.6:27b" (pi models.json, both Hermes
# boxes). A naive --model swap changes the served id and breaks every client. This script
# ALWAYS serves a stable alias set so switching is invisible downstream:
#     qwen-local      <- stable, preferred id going forward
#     qwen3.6:27b     <- legacy alias, keeps existing clients working unchanged
#     <real-model-id> <- the actual checkpoint name, for clarity in logs/metrics
#
# The gateway (:8000) and watchdog are already model-agnostic (they auto-discover from
# /v1/models), so nothing else needs touching.
set -uo pipefail
SERVE=/home/kevin/.local/share/vllm-qwen27b/serve-tqk8v4-fg.sh
MODELS=/home/kevin/Desktop/models
BAK_DIR=/home/kevin/.local/share/vllm-qwen27b
STAMP=$(date -u +%Y%m%d-%H%M%S)

die(){ echo "ERROR: $*" >&2; exit 1; }
current_model(){ grep -A1 -- '--model' "$SERVE" | tail -1 | tr -d ' '; }

case "${1:-list}" in
  list)
    echo "LIVE NOW:"
    echo "  model:  $(current_model)"
    echo "  served: $(curl -fsS -m5 http://localhost:8001/v1/models 2>/dev/null | python3 -c 'import sys,json;print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null || echo '(engine down)')"
    echo "  engine: $(systemctl is-active vllm-qwen27b)"
    echo
    echo "AVAILABLE:"
    for d in "$MODELS"/*/; do
      [ -f "$d/config.json" ] || continue
      fmt=$(python3 - "$d" <<'PY'
import json,sys,os
c=json.load(open(os.path.join(sys.argv[1],"config.json")))
q=c.get("quantization_config") or {}
m=(q.get("quant_method") or q.get("method") or "").lower()
d=str(c.get("torch_dtype") or c.get("dtype") or "").lower()
if any(k in m for k in ("awq","gptq","compressed")): print("Int4 (SERVABLE)")
# FP8 WEIGHTS run on SM75 via the fork's FP8-weight route — upstream v0.1.15
# validates Qwen3.8-27B-FP8 on this exact dual-2080Ti TP=2 rig. Only NVFP4
# stays Blackwell-only. (The old blanket "BLOCKED on SM75" label was wrong and
# also hid the FP8 lane in the release detector.)
elif "nvfp4" in m: print("NVFP4 (BLOCKED on SM75)")
elif "fp8" in m or "modelopt" in m: print("FP8-weight (SERVABLE, v0.1.15 route)")
else: print(f"{d or 'bf16'} (needs quantizing)")
PY
)
      printf "  %-46s %-10s %s\n" "$(basename "$d")" "$(du -sh "$d" 2>/dev/null|cut -f1)" "$fmt"
    done
    exit 0 ;;
  rollback)
    LAST=$(ls -t "$BAK_DIR"/serve-tqk8v4-fg.sh.bak-switch-* 2>/dev/null | head -1)
    [ -z "$LAST" ] && die "no switch backup found"
    cp -a "$LAST" "$SERVE"; echo "restored $LAST"
    sudo systemctl restart vllm-qwen27b; TARGET_DESC="rolled back" ;;
  3.6) NEW="$MODELS/Qwen3.6-27B-GPTQ-Int4" ;;
  3.8) NEW=$(ls -dt "$MODELS"/Qwen3.8-27B*Int4* "$MODELS"/Qwen3.8-27B*AWQ* 2>/dev/null | head -1) ;;
  *)   NEW="$1" ;;
esac

if [ "${1:-}" != "rollback" ]; then
  [ -z "${NEW:-}" ] && die "no matching model found (has 3.8 been quantized yet? run: $0 list)"
  [ -f "$NEW/config.json" ] || die "not a model dir: $NEW"

  FMT=$(python3 - "$NEW" <<'PY'
import json,sys,os
c=json.load(open(os.path.join(sys.argv[1],"config.json")))
q=c.get("quantization_config") or {}
m=(q.get("quant_method") or q.get("method") or "").lower()
if "gptq" in m or "compressed" in m: print("gptq_marlin")
elif "awq" in m: print("awq_marlin")
elif "fp8" in m or "modelopt" in m or "nvfp4" in m: print("BLOCKED")
else: print("NONE")
PY
)
  [ "$FMT" = "BLOCKED" ] && die "$NEW is FP8/NVFP4 — requires compute capability >=89; these are SM75 cards"
  [ "$FMT" = "NONE" ]    && die "$NEW is unquantized (BF16) — quantize first: ~/quant-env/bin/python ~/Desktop/qwythos-quant/quantize_qwen38.py awq"

  REAL_ID=$(basename "$NEW" | tr 'A-Z' 'a-z')
  cp -a "$SERVE" "$BAK_DIR/serve-tqk8v4-fg.sh.bak-switch-$STAMP"
  echo "backup: serve-tqk8v4-fg.sh.bak-switch-$STAMP"

  python3 - "$SERVE" "$NEW" "$REAL_ID" "$FMT" <<'PY'
import sys,re
serve,new,real,fmt=sys.argv[1:5]
s=open(serve).read()
# --model <path>
s=re.sub(r'(--model\n)(\s*)\S+\n', lambda m: f"{m.group(1)}{m.group(2)}{new}\n", s, count=1)
# --served-model-name -> stable aliases PLUS every id already being served.
# A swap must only ever ADD names, never remove them: dropping an id silently breaks
# whichever client happened to pin it (e.g. the long qwen27b-int4-tqk8v4-... id).
# NOTE: the value lines must be matched with a (?!--) guard. Without it the group is
# greedy across newlines and swallows the ENTIRE remainder of the serve script — the
# original version of this regex would have replaced every following flag with the three
# alias lines. Caught by dry-run 2026-08-14; do not "simplify" this pattern.
VAL=r'(?:[ \t]+(?!--)\S+\n)+'
cur=re.search(r'--served-model-name\n('+VAL+r')', s)
old=[l.strip() for l in cur.group(1).splitlines() if l.strip()] if cur else []
# Carry forward the STABLE aliases only, and drop any other checkpoint's real id.
# Blindly keeping every previous name made the engine advertise `qwen3.8-27b-gptq-int4`
# while actually serving 3.6 -- a client pinning that id would silently get the wrong
# model, which is worse than a clean 404. A checkpoint id names one specific checkpoint;
# only the aliases are allowed to follow the swap.
CHECKPOINT_ID=re.compile(r'^qwen[0-9.]+-27b-.*int4$', re.I)
stable=[n for n in old if not CHECKPOINT_ID.match(n)]
names=[]
for n in ["qwen-local","qwen3.6:27b",real]+stable:
    if n not in names: names.append(n)
s=re.sub(r'(--served-model-name\n)'+VAL,
         lambda m: m.group(1)+"".join(f"  {n}\n" for n in names), s, count=1)
# quantization flag matched to the checkpoint
if '--quantization' in s:
    s=re.sub(r'(--quantization\n)(\s*)\S+\n', lambda m: f"{m.group(1)}{m.group(2)}{fmt}\n", s, count=1)
else:
    s=s.replace("  --max-model-len\n", f"  --quantization\n  {fmt}\n  --max-model-len\n",1)
open(serve,'w').write(s)
print(f"  serve script -> model={new} quant={fmt} aliases={','.join(names)}")
PY

  bash -n "$SERVE" || { cp -a "$BAK_DIR/serve-tqk8v4-fg.sh.bak-switch-$STAMP" "$SERVE"; die "edited script failed syntax check — reverted"; }
  TARGET_DESC="$NEW"
  echo "restarting engine (cold start ~4-5 min if the compile cache is cold)..."
  sudo systemctl restart vllm-qwen27b
fi

# ---------- verify, auto-rollback on failure ----------
for i in $(seq 1 100); do
  sleep 6
  if curl -fsS -m4 http://localhost:8001/health >/dev/null 2>&1; then
    echo "engine healthy after ~$((i*6))s"
    SERVED=$(curl -fsS -m5 http://localhost:8001/v1/models | python3 -c 'import sys,json;print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))')
    echo "  served ids: $SERVED"
    GEN=$(curl -fsS -m60 http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"qwen3.6:27b","messages":[{"role":"user","content":"reply with OK"}],"max_tokens":5}' \
      | python3 -c 'import sys,json;print(json.load(sys.stdin).get("model","?"))' 2>/dev/null)
    echo "  legacy-id smoke test -> served by: $GEN"
    echo "SWITCH COMPLETE: $TARGET_DESC"
    echo "NEXT: bench it -> python3 /tmp/claude-1000/-home-kevin-Desktop/94211903-d24f-4c4e-9538-16e427c7ea91/scratchpad/specbench/bench.py <label>"
    exit 0
  fi
  [ "$(systemctl is-active vllm-qwen27b)" = "failed" ] && break
done

echo "ENGINE DID NOT COME UP — auto-rolling back"
LAST=$(ls -t "$BAK_DIR"/serve-tqk8v4-fg.sh.bak-switch-* 2>/dev/null | head -1)
[ -n "$LAST" ] && cp -a "$LAST" "$SERVE"
sudo systemctl restart vllm-qwen27b
echo "rolled back to previous model; check: journalctl -u vllm-qwen27b -n 50"
exit 1
