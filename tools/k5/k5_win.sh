#!/usr/bin/env bash
# K5 window (L80, decode-megakernel step 1): sm_75 fused GDN post-conv MTP decode, A/B on the live trial config.
#   0. save the current override env (restored verbatim at the end, whatever happens)
#   1. kernel correctness + microbench on GPU1 with the engine stopped (gate for step 3)
#   2. BASE arm: current override + torch profiler config -> quick.py decode, profiles at 1/4/12 streams
#   3. K5 arm: same + V02_ROOT=wt-k5 VLLM_K5_GDN_FUSED=1 -> quick.py, profiles 1/4/12, evalkit tool_call+code_exec+long_ctx, Xid/OOM counts
#   4. restore the saved override + engine-actuator restart, health check
# Run under: python3 /home/kevin/Desktop/wt-integrate/deploy/bin/gateway-offline.py run --reason "K5 GDN fused decode A/B" --by K5 --ttl 3000 --wait-s 90 -- bash /home/kevin/Desktop/wt-k5/tools/k5/k5_win.sh
set -u
L=/home/kevin/projects/lanes/k5; K5=/home/kevin/Desktop/wt-k5; I=/home/kevin/Desktop/wt-integrate
SD=/home/kevin/.local/share/vllm-qwen27b; O=$SD/v02.override.env
mkdir -p $L/win $L/win/prof_base $L/win/prof_k5; rm -f $L/test_gdn_mtp.json; cp $O $L/win/override.saved; echo "k5 window start $(date)"; cat $L/win/override.saved
XID0=$(sudo -n dmesg 2>/dev/null | grep -c -E 'Xid' || echo na)
boot() {  # boot LABEL EXTRA_LINES...   (override = saved trial override + extra export lines)
  local lab=$1; shift; cp $L/win/override.saved $O; for kv in "$@"; do echo "export $kv" >> $O; done
  for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
  python3 $SD/engine-actuator.py restart --by K5 --reason "K5 arm $lab" --no-drain --foreground > $L/win/boot_$lab.log 2>&1 &
  sleep 20; local t0=$(date +%s)
  until curl -s -m 2 -o /dev/null -w "%{http_code}" localhost:8001/health | grep -q 200; do
    for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
    [ $(( $(date +%s) - t0 )) -ge 900 ] && { echo "BOOT FAILED $lab"; journalctl -u vllm-qwen27b --since "-10 min" --no-pager | grep -iE "error|Traceback|assert" | tail -8 | cut -c1-250; return 1; }
    sleep 4
  done
  for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
  echo "booted $lab $(date +%T): $(journalctl -u vllm-qwen27b --since '-15 min' --no-pager | grep -E 'GPU KV cache size|Model loading took' | sed 's/.*INFO//' | cut -c1-90 | tr '\n' '|')"
}
prof_extra() {  # current VLLM_SERVE_EXTRA_ARGS from the saved override + profiler config (dir $1)
  local cur; cur=$(bash -c ". $L/win/override.saved; echo \"\${VLLM_SERVE_EXTRA_ARGS:-}\"")
  echo "VLLM_SERVE_EXTRA_ARGS='$cur --profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"$1\",\"torch_profiler_with_stack\":false}'"
}
measure() {  # measure LABEL
  local lab=$1; local pd=$L/win/prof_$lab
  echo "== $lab quick $(date +%T)"; python3 $I/tools/s2-bench/quick.py 2>&1 | tail -1 | tee $L/win/quick_$lab.json
  for s in 1 4 12; do
    mkdir -p $pd/s$s; echo "== $lab profile $s streams $(date +%T)"
    # point the profiler at a per-arm dir via symlink swap is not possible at runtime: traces land in $pd; move them per stream count
    python3 $K5/tools/k5/k5_prof.py $s 96 2>&1 | tail -3; sleep 10
    mv $pd/*.pt.trace.json.gz $pd/profiler_out_* $pd/s$s/ 2>/dev/null
    for f in $pd/s$s/rank0*.pt.trace.json.gz; do python3 $K5/tools/k5/k5_steps.py $f | tee -a $L/win/steps_$lab.jsonl; done
  done
}
# 1. kernel test with the engine stopped
echo "== stop engine $(date +%T)"; sudo -n systemctl stop vllm-qwen27b; sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
( . $L/envbuild.sh; cd $K5 && CUDA_VISIBLE_DEVICES=1 VLLM_K5_GDN_BUILD_DIR=$L/ext PYTHONPATH=$K5 timeout 600 python tools/k5/test_gdn_mtp.py ) 2>&1 | grep -vE "^W1003|warn" | tail -20
TEST_RC=${PIPESTATUS[0]}; KPASS=$(python3 -c "import json;print(json.load(open('$L/test_gdn_mtp.json'))['pass'])" 2>/dev/null || echo False)
echo "kernel test pass=$KPASS"
# 2. BASE arm
boot base "$(prof_extra $L/win/prof_base)" && measure base
# 3. K5 arm (only if the kernel test passed)
if [ "$KPASS" = "True" ]; then
  boot k5 "$(prof_extra $L/win/prof_k5)" "V02_ROOT=$K5" "VLLM_K5_GDN_FUSED=1" "VLLM_K5_GDN_BUILD_DIR=$L/ext" && {
    journalctl -u vllm-qwen27b --since '-15 min' --no-pager | grep -iE "k5|gdn_mtp" | tail -3
    measure k5
    echo "== k5 evalkit $(date +%T)"; (cd /home/kevin/Desktop/qwen38-evalkit && timeout 1500 python3 run_eval.py --tag k5-gdnfused --categories tool_call,code_exec,long_ctx 2>&1 | grep -E "passed=False|/60")
    echo "k5 errors in journal: $(journalctl -u vllm-qwen27b --since '-40 min' --no-pager | grep -cE 'Traceback|CUDA error|illegal memory')"
  }
else echo "SKIP K5 arm: kernel test failed"; fi
# 4. restore
echo "== restore $(date +%T)"; cp $L/win/override.saved $O
python3 $SD/engine-actuator.py restart --by K5 --reason "K5 window restore" --no-drain --foreground > $L/win/boot_restore.log 2>&1 &
sleep 20; t0=$(date +%s); until curl -s -m 2 -o /dev/null -w "%{http_code}" localhost:8001/health | grep -q 200; do [ $(( $(date +%s) - t0 )) -ge 900 ] && break; sleep 4; done
cmp -s $O $L/win/override.saved && echo "override restored verbatim"
echo "restored health $(curl -s -m3 -o /dev/null -w '%{http_code}' localhost:8001/health) Xid before=$XID0 after=$(sudo -n dmesg 2>/dev/null | grep -c -E 'Xid' || echo na) $(date)"
