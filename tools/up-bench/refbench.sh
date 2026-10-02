#!/usr/bin/env bash
# refbench.sh MODEL_DIR SERVED_NAME PORT LABEL -- upstream "reference lane": completions endpoint, pure-filler " the" prompt, output restricted to " the",
# warm-up 4K/128 excluded, 4K/128 median of 3, then 32K/512. Uses launcher's tools/profile_request.py. Unique prompt variants (no prefix reuse).
set -u
MD=$1; SN=$2; PORT=$3; LB=$4
PY=/home/kevin/Desktop/wt-integrate/.venv/bin/python; H=/home/kevin/Desktop/wt-integrate/tools/profile_request.py; S=$(date +%s)
run() { $PY $H --model-dir "$MD" --served-name "$SN" --base-url "http://127.0.0.1:$PORT/v1" --endpoint completions --prompt-tokens "$1" --gen-tokens "$2" --label "$LB-$3" --prompt-variant "up-$LB-$S-$3" --out /dev/null --ignore-eos --pure-filler --allowed-token-text " the" 2>&1 | $PY -c "
import sys,json
raw=sys.stdin.read(); dec=json.JSONDecoder(); rec=None
for i,c in enumerate(raw):
    if c!='{': continue
    try: v,_=dec.raw_decode(raw,i)
    except Exception: continue
    if isinstance(v,dict) and 'prefill_tok_s' in v: rec=v
print(json.dumps({k:rec.get(k) for k in ('prefill_tok_s','decode_tok_s','ttft_s','http_status','stream_done','completion_tokens','prompt_tokens','error')}) if rec else 'NO-RECORD '+raw[-300:])
"; }
echo "[warmup]"; run 4096 128 warm
for i in 1 2 3; do echo "4K/128 #$i: $(run 4096 128 s$i)"; done
echo "32K/512: $(run 32768 512 long)"
