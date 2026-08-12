#!/usr/bin/env bash
# qwen38-release-detector v2 — fires ONLY when a Qwen3.8-27B repo actually has
# downloadable WEIGHTS (real safetensors/gguf, not an empty placeholder).
# Turing (sm_75): usable = int4(awq/gptq/w4a16) or gguf-q4 or base(bf16->self-quant).
# FP8/NVFP4 are reported but do NOT trigger (won't run on 2080 Ti).
set -uo pipefail
UA='User-Agent: Mozilla/5.0'
OUT=/home/kevin/Desktop/qwen38-detected.json
LOG=/home/kevin/Desktop/qwen38-detector.log
INTERVAL=15

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
log "detector v2 started; polling every ${INTERVAL}s (needs real weights)"

# echo the repo id if it 200s AND has real weight files (>1GB safetensors/gguf)
has_weights(){ # $1 = repo id
  curl -s -H "$UA" "https://huggingface.co/api/models/$1?blobs=true" 2>/dev/null | python3 -c "
import sys,json
try: d=json.load(sys.stdin)
except: sys.exit(1)
if d.get('error'): sys.exit(1)
big=0
for s in d.get('siblings',[]):
    fn=s.get('rfilename','').lower(); sz=s.get('size') or 0
    if (fn.endswith('.safetensors') or fn.endswith('.gguf')) and sz>1_000_000_000:
        big+=1
sys.exit(0 if big>0 else 1)
"
}

# official repos, in Turing-preference order
OFFICIAL=(Qwen/Qwen3.8-27B-AWQ Qwen/Qwen3.8-27B-GPTQ-Int4 Qwen/Qwen3.8-27B-GGUF \
          Qwen/Qwen3.8-27B Qwen/Qwen3.8-27B-Instruct Qwen/Qwen3.8-27B-Instruct-AWQ)

while true; do
  found=""
  # 1) official first
  for r in "${OFFICIAL[@]}"; do
    if has_weights "$r"; then found="OFFICIAL $r"; break; fi
  done
  # 2) community with real weights + turing-usable format
  if [ -z "$found" ]; then
    for r in $(curl -s -H "$UA" 'https://huggingface.co/api/models?search=Qwen3.8-27B&limit=100' \
      | python3 -c "
import sys,json
try: d=json.load(sys.stdin)
except: sys.exit()
for m in d:
    i=m['id']; low=(i+' '+' '.join(m.get('tags',[]))).lower()
    if any(q in low for q in ['awq','gptq','int4','w4a16','gguf']):
        print(i)
" 2>/dev/null); do
      if has_weights "$r"; then found="${found:+$found, }community $r"; fi
    done
  fi

  if [ -n "$found" ]; then
    log "REAL WEIGHTS DETECTED: $found"
    printf '{"detected_utc":"%s","what":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$found" > "$OUT"
    exit 0
  fi
  sleep "$INTERVAL"
done
