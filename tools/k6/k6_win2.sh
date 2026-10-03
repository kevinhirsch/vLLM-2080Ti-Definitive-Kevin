#!/usr/bin/env bash
# K6 window 2 (~11 min engine-stopped; +9 min with TP1=1): the parts window 1 (10-03 09:26) did not measure.
#   (a) mux_bench on GPU0 alone  (b) mux_ar_bench on both GPUs  [(c) TP1=1: single-card engine on GPU1 + probes]
#   then restore EXACTLY the override env and watchdog.timer state present at start.
# Run:  cd /home/kevin/Desktop/vLLM-2080Ti-Definitive && .venv/bin/python deploy/bin/gateway-offline.py run \
#   --reason "K6 mux + second-communicator bench" --by K6 --ttl 1500 -- bash /home/kevin/Desktop/wt-k6/tools/k6/k6_win2.sh
set -u
K=/home/kevin/Desktop/wt-k6/tools/k6; B=/home/kevin/Desktop/wt-integrate/tools/s2-bench; O=/home/kevin/.local/share/vllm-qwen27b/v02.override.env
OUT=/home/kevin/projects/lanes/k6/win2; mkdir -p $OUT; cd $K; PY=/home/kevin/Desktop/wt-integrate/.venv/bin/python
log(){ echo "$(date '+%F %T') $*" | tee -a $OUT/win.log; }
health(){ curl -s -m 3 -o /dev/null -w "%{http_code}" localhost:8001/health; }
XID0=$(journalctl -k --no-pager | grep -c "NVRM: Xid"); WD0=$(systemctl is-active vllm-qwen27b-watchdog.timer 2>/dev/null || true)
cp -a $O $OUT/override.at_start.env; log "window2 start; xid=$XID0; override $(wc -l < $O) lines; watchdog.timer=$WD0; TP1=${TP1:-0}"
boot(){ # boot LABEL: actuator restart with current $O; success = NEW MainPID + health 200
  local PID0=$(systemctl show vllm-qwen27b -p MainPID --value)
  python3 /home/kevin/.local/share/vllm-qwen27b/engine-actuator.py restart --by K6 --reason "K6 window2 arm $1 (planned bench restart)" --no-drain --foreground > $OUT/boot_$1.log 2>&1 &
  local APID=$! T0=$(date +%s); sleep 15
  until [ "$(systemctl show vllm-qwen27b -p MainPID --value)" != "$PID0" ] && [ "$(systemctl show vllm-qwen27b -p MainPID --value)" != 0 ] && [ "$(health)" = 200 ]; do
    grep -q '"refused"' $OUT/boot_$1.log 2>/dev/null && { log "BOOT REFUSED $1: $(cat $OUT/boot_$1.log)"; return 1; }
    [ $(( $(date +%s) - T0 )) -ge 900 ] && { log "BOOT FAILED $1: $(journalctl -u vllm-qwen27b --since "@$T0" --no-pager | grep -iE 'error|Traceback|memory' | tail -4 | cut -c1-260 | tr '\n' '|')"; return 1; }
    sleep 4
  done
  log "booted $1 (MainPID $PID0 -> $(systemctl show vllm-qwen27b -p MainPID --value)): $(journalctl -u vllm-qwen27b --since "@$T0" --no-pager | grep -E 'Model loading took|Available KV cache memory|GPU KV cache size|Actual usage is|attention block size' | sed 's/.*INFO//;s/\[[a-z_.0-9:]*\]//' | cut -c1-300 | sort -u | tr '\n' '|')"
}
restore(){
  cp -a $OUT/override.at_start.env $O
  if [ "$(systemctl is-active vllm-qwen27b)" = active ]; then boot restore || { sleep 20; boot restore2; }
  else sudo -n systemctl start vllm-qwen27b; T0=$(date +%s); until [ "$(health)" = 200 ] || [ $(( $(date +%s)-T0 )) -ge 900 ]; do sleep 4; done; fi
  [ "$WD0" = active ] && sudo -n systemctl start vllm-qwen27b-watchdog.timer
  log "restored: health $(health); KV $(journalctl -u vllm-qwen27b --since '-6 min' --no-pager | grep -o 'GPU KV cache size: [0-9,]* tokens' | tail -1); watchdog.timer $(systemctl is-active vllm-qwen27b-watchdog.timer) (start $WD0); xid $(journalctl -k --no-pager | grep -c 'NVRM: Xid') (start $XID0)"
}
trap 'log "abort trap -> restore"; restore; exit 1' INT TERM
gpu_free(){ local u=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 1500 ]; }
# ---- engine stopped, both GPUs free ----
[ "$WD0" = active ] && sudo -n systemctl stop vllm-qwen27b-watchdog.timer
sudo -n systemctl stop vllm-qwen27b; sleep 8
for i in 0 1; do gpu_free $i || { log "GPU$i not free after stop; aborting"; restore; exit 1; }; done
log "engine stopped; GPUs free"
CUDA_DEVICE_ORDER=PCI_BUS_ID timeout 420 $PY $K/mux_bench.py --gpu 0 --layers 64 --decode-m 4 20 48 --out $OUT/mux_gpu0.json > $OUT/mux_gpu0.out 2>&1
log "mux_bench rc=$? $(python3 -c "import json;d=json.load(open('$OUT/mux_gpu0.json'));print(json.dumps({k:{'alone':v['decode_alone_ms_p50'],'hi':v['hi_prio']['decode_slowdown'],'pf_rate':v['hi_prio']['prefill_rate_during_overlap'],'eq':v['equal_prio']['decode_slowdown']} for k,v in d['rows'].items()}))" 2>&1 | tail -1)"
CUDA_DEVICE_ORDER=PCI_BUS_ID timeout 420 $PY -m torch.distributed.run --nproc-per-node 2 --master-port 29561 $K/mux_ar_bench.py --out $OUT/mux_ar.json > $OUT/mux_ar.out 2>&1
log "mux_ar rc=$? $(grep -h -E '^\{' $OUT/mux_ar.out | head -2 | cut -c1-900 | tr '\n' ' ')"
if [ "${TP1:-0}" = 1 ]; then
  { grep -v -E 'CUDA_VISIBLE_DEVICES|V02_MAXLEN|VLLM_GPU_UTIL' $OUT/override.at_start.env
    echo 'export CUDA_VISIBLE_DEVICES=1'; echo 'export V02_MAXLEN=65536'; echo 'export VLLM_GPU_UTIL=0.94'
    grep -q VLLM_SERVE_EXTRA_ARGS $OUT/override.at_start.env || echo 'export VLLM_SERVE_EXTRA_ARGS=""'
    echo 'export VLLM_SERVE_EXTRA_ARGS="${VLLM_SERVE_EXTRA_ARGS} --tensor-parallel-size 1"'; } > $O
  sudo -n systemctl start vllm-qwen27b; T0=$(date +%s)
  until [ "$(health)" = 200 ] || [ $(( $(date +%s)-T0 )) -ge 900 ] || [ "$(systemctl is-active vllm-qwen27b)" = failed ]; do sleep 4; done
  if [ "$(health)" = 200 ] && gpu_free 0; then
    log "TP1 booted: $(journalctl -u vllm-qwen27b --since "@$T0" --no-pager | grep -E 'Model loading took|Available KV cache memory|GPU KV cache size|Actual usage is' | sed 's/.*INFO//' | cut -c1-300 | sort -u | tr '\n' '|')"
    log "TP1 quick: $($PY $B/quick.py 2>&1 | tail -1)"
    log "TP1 prefill: $(python3 prefill_probe.py --sizes 8000 16000 30000 --reps 2 2>&1 | tail -1)"
    for n in 4 8 12; do log "TP1 conc $n: $(python3 $B/conc_short.py $n 256 2>&1 | tail -1)"; done
  else log "TP1 boot failed: $(journalctl -u vllm-qwen27b --since "@$T0" --no-pager | grep -iE 'error|memory|Traceback' | tail -3 | cut -c1-260 | tr '\n' '|')"; fi
fi
trap - INT TERM; restore; log "window2 end"
