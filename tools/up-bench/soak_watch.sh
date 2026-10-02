#!/usr/bin/env bash
# soak_watch.sh MINUTES -- every 30s: health, unit state, fault-ledger count, journal ERROR count, tiny-prompt TTFT
M=${1:-30}; B=/home/kevin/.local/share/vllm-qwen27b; END=$(( $(date +%s) + M*60 )); L0=$(wc -l < $B/incidents/ledger.jsonl)
echo "start $(date +%T) ledger_lines=$L0"
while [ $(date +%s) -lt $END ]; do
  h=$(curl -s -m 5 -o /dev/null -w "%{http_code}" localhost:8001/health)
  t=$(python3 - <<'PY'
import time,json,http.client
t0=time.time()
try:
    c=http.client.HTTPConnection("127.0.0.1",8001,timeout=60)
    c.request("POST","/v1/chat/completions",json.dumps(dict(model="qwen-local",messages=[{"role":"user","content":"Reply with the single word: ok"}],max_tokens=4,temperature=0,stream=True,chat_template_kwargs={"enable_thinking":False})),{"Content-Type":"application/json","X-Client":"soak-up"})
    r=c.getresponse(); r.read1(65536); print(round(time.time()-t0,2))
except Exception as e: print("ERR",type(e).__name__)
PY
)
  e=$(journalctl -u vllm-qwen27b --since "-1 min" --no-pager 2>/dev/null | grep -c -E " ERROR |Traceback")
  echo "$(date +%T) health=$h unit=$(systemctl is-active vllm-qwen27b) nrestarts=$(systemctl show vllm-qwen27b -p NRestarts --value) ledger+=$(( $(wc -l < $B/incidents/ledger.jsonl) - L0 )) ttft_tiny=$t journal_err_1m=$e"
  sleep 30
done
