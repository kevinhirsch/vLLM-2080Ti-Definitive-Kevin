#!/usr/bin/env bash
# K6 window (~25 min): (1) TP2 reference probes on the live config, (2) stream-priority mux microbench on GPU0,
# (3) single-card TP1 engine on GPU1 with the full int4 stack -> weights / KV GiB / pool tokens / prefill rate /
# decode rates / ITL-under-prefill, (4) restore EXACTLY the override env present at window start.
# Run under:  cd /home/kevin/Desktop/vLLM-2080Ti-Definitive && .venv/bin/python deploy/bin/gateway-offline.py run \
#   --reason "K6 P/D disaggregation rates + mux microbench" --by K6 --ttl 2400 -- bash /home/kevin/Desktop/wt-k6/tools/k6/k6_win.sh
set -u
K=/home/kevin/Desktop/wt-k6/tools/k6; B=/home/kevin/Desktop/wt-integrate/tools/s2-bench; O=/home/kevin/.local/share/vllm-qwen27b/v02.override.env
OUT=/home/kevin/projects/lanes/k6/win; mkdir -p $OUT; cd $K
PY=/home/kevin/Desktop/wt-integrate/.venv/bin/python
log(){ echo "$(date '+%F %T') $*" | tee -a $OUT/win.log; }
XID0=$(journalctl -k --no-pager | grep -c "NVRM: Xid")
cp -a $O $OUT/override.at_start.env; log "window start; xid=$XID0; override saved ($(wc -l < $O) lines)"
health(){ curl -s -m 3 -o /dev/null -w "%{http_code}" localhost:8001/health; }
boot(){ # boot LABEL : actuator restart with current $O, wait for health
  for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
  python3 /home/kevin/.local/share/vllm-qwen27b/engine-actuator.py restart --by K6 --reason "K6 $1" --no-drain --foreground > $OUT/boot_$1.log 2>&1 &
  sleep 20; local T0=$(date +%s)
  until [ "$(health)" = 200 ]; do
    for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
    [ $(( $(date +%s) - T0 )) -ge ${BOOT_TIMEOUT:-900} ] && { log "BOOT FAILED $1: $(journalctl -u vllm-qwen27b --since '-15 min' --no-pager | grep -iE 'error|Traceback|memory' | tail -4 | cut -c1-260 | tr '\n' '|')"; return 1; }
    sleep 4
  done
  for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
  log "booted $1: $(journalctl -u vllm-qwen27b --since '-15 min' --no-pager | grep -E 'Model loading took|Available KV cache memory|GPU KV cache size|Actual usage is|attention block size' | sed 's/.*INFO//;s/\[[a-z_.0-9:]*\]//' | cut -c1-260 | sort -u | tr '\n' '|')"
}
restore(){
  cp -a $OUT/override.at_start.env $O; boot restore || { sleep 30; boot restore2; }
  log "restored: health $(health); xid now $(journalctl -k --no-pager | grep -c 'NVRM: Xid') (start $XID0)"
}
trap 'log "abort trap -> restore"; restore; exit 1' INT TERM
# ---- (1) TP2 reference on the live config (gateway is offline => engine idle) ----
[ "$(health)" = 200 ] || boot tp2ref
log "TP2 quick: $($PY $B/quick.py 2>&1 | tail -1)"
log "TP2 prefill: $(python3 prefill_probe.py --sizes 8000 16000 30000 --reps 2 2>&1 | tail -1)"
for n in 4 8 12; do log "TP2 conc $n: $(python3 $B/conc_short.py $n 256 2>&1 | tail -1)"; done
log "TP2 itl_under_prefill: $(python3 itl_under_prefill.py --prefill 16000 2>&1 | tail -1)"
# ---- (2)+(3) TP1 engine on GPU1; mux microbench on GPU0 meanwhile ----
{ cat $OUT/override.at_start.env | grep -v -E 'CUDA_VISIBLE_DEVICES|V02_MAXLEN|VLLM_GPU_UTIL'
  echo 'export CUDA_VISIBLE_DEVICES=1'; echo 'export V02_MAXLEN=65536'; echo 'export VLLM_GPU_UTIL=0.94'
  grep -q VLLM_SERVE_EXTRA_ARGS $OUT/override.at_start.env || echo 'export VLLM_SERVE_EXTRA_ARGS=""'
  echo 'export VLLM_SERVE_EXTRA_ARGS="${VLLM_SERVE_EXTRA_ARGS} --tensor-parallel-size 1"'; } > $O
log "TP1 override: $(tr '\n' ';' < $O)"
( for i in $(seq 1 60); do u=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 1500 ] && break; sleep 3; done
  CUDA_DEVICE_ORDER=PCI_BUS_ID timeout 600 $PY $K/mux_bench.py --gpu 0 --layers 64 --decode-m 4 20 48 --out $OUT/mux_gpu0.json > $OUT/mux_gpu0.out 2>&1
  echo "$(date '+%F %T') mux bench rc=$? $(tail -c 300 $OUT/mux_gpu0.out | tr '\n' ' ')" >> $OUT/win.log ) &
MUX=$!
if boot tp1; then
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> $OUT/win.log
  log "TP1 quick: $($PY $B/quick.py 2>&1 | tail -1)"
  log "TP1 prefill: $(python3 prefill_probe.py --sizes 8000 16000 30000 --reps 2 2>&1 | tail -1)"
  for n in 4 8 12; do log "TP1 conc $n: $(python3 $B/conc_short.py $n 256 2>&1 | tail -1)"; done
  log "TP1 itl_under_prefill: $(python3 itl_under_prefill.py --prefill 16000 2>&1 | tail -1)"
fi
wait $MUX
# ---- (4) restore ----
trap - INT TERM; restore
log "window end"
