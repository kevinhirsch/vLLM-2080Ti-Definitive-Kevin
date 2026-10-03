#!/usr/bin/env bash
# K9 window (L102): chunked tensor-core GDN prefill (VLLM_K9_GDN_CHUNK=1) vs FlashQLA legacy, full-engine A/B on the live config.
#   0. save the current override env (restored verbatim at the end, whatever happens); pause the watchdog timer
#   1. BASE arm: saved override            -> cold prefill x3, quick decode, det, estate pass (12 bodies: prime + 3 warm)
#   2. K9 arm:   saved override + V02_ROOT=wt-k9g (prod 7004df6ae8 + 2 K9 commits) + VLLM_K9_GDN_CHUNK=1 (prebuilt ext)
#                -> same measurements + det --cmp vs stack ref + evalkit tool_call,code_exec,long_ctx (+ needle 262K if time)
#   3. restore the saved override + engine-actuator restart, health check, watchdog timer back
# Run under: python3 /home/kevin/Desktop/wt-integrate/deploy/bin/gateway-offline.py run --reason "K9 GDN chunk prefill A/B" --by K9 --ttl 3600 --wait-s 90 -- bash /home/kevin/Desktop/wt-k9g/tools/k9/k9_win.sh > /home/kevin/projects/lanes/k9/win/win.out 2>&1
set -u
L=/home/kevin/projects/lanes/k9/win; K9=/home/kevin/Desktop/wt-k9g; I=/home/kevin/Desktop/wt-integrate; S2=/home/kevin/projects/lanes/s2-speed
SD=/home/kevin/.local/share/vllm-qwen27b; O=$SD/v02.override.env
mkdir -p $L; cp $O $L/override.saved; echo "k9 window start $(date)"; cat $L/override.saved
T_START=$(date +%s)
WD0=$(systemctl is-active vllm-qwen27b-watchdog.timer); echo "watchdog timer at start: $WD0"
[ "$WD0" = active ] && sudo -n systemctl stop vllm-qwen27b-watchdog.timer  # its Wants= would restart a stopped engine
X0=$(journalctl -k --no-pager | grep -c 'NVRM: Xid')
gpu_clean_wait() {  # wait up to 600 s for no non-engine GPU compute processes before a boot (the KV pool is sized at boot)
  local t0=$(date +%s) f
  while f=$(/home/kevin/projects/lanes/windows/gpu_foreign.sh); [ -n "$f" ]; do
    [ $(( $(date +%s) - t0 )) -ge 600 ] && { echo "WARN foreign GPU procs at boot $1 (KV pool may shrink): $(echo "$f" | tr '\n' '|')"; return 1; }
    sleep 10
  done; return 0
}
ABORT=0
boot() {  # boot LABEL EXTRA_LINES...   (override = saved override + extra export lines)
  local lab=$1; shift; cp $L/override.saved $O; for kv in "$@"; do echo "export $kv" >> $O; done
  for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
  gpu_clean_wait $lab
  python3 $SD/engine-actuator.py restart --by K9 --reason "K9 arm $lab" --no-drain --foreground > $L/boot_$lab.log 2>&1 &
  sleep 20; local t0=$(date +%s)
  until curl -s -m 2 -o /dev/null -w "%{http_code}" localhost:8001/health | grep -q 200; do
    grep -qiE '"refused"|refus' $L/boot_$lab.log 2>/dev/null && { echo "BOOT REFUSED $lab: $(head -c 300 $L/boot_$lab.log)"; ABORT=1; return 1; }
    for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
    [ $(( $(date +%s) - t0 )) -ge 900 ] && { ABORT=1; echo "BOOT FAILED $lab"; journalctl -u vllm-qwen27b --since "-10 min" --no-pager | grep -iE "error|Traceback|assert" | tail -8 | cut -c1-250; return 1; }
    sleep 4
  done
  for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
  echo "booted $lab $(date +%T): $(journalctl -u vllm-qwen27b --since '-15 min' --no-pager | grep -E 'GPU KV cache size|Model loading took|GDN prefill|k9' | sed 's/.*INFO//' | cut -c1-110 | sort -u | tr '\n' '|')"
}
measure() {  # measure LABEL
  local lab=$1
  echo "== $lab cold prefill x3 $(date +%T)"; (cd $S2 && ESTATE_FR=$S2/fr python3 /home/kevin/Desktop/wt-lp/tools/lp/cold_prefill.py 3 $L/cold_$lab.json 2>&1 | tail -4)
  echo "== $lab quick $(date +%T)"; python3 $I/tools/s2-bench/quick.py 2>&1 | tail -1 | tee $L/quick_$lab.json
  echo "== $lab det $(date +%T)"; (cd $S2 && python3 det.py $L/det_$lab.json >/dev/null 2>&1; python3 det.py --cmp det_s4_stack_ref.json $L/det_$lab.json 2>&1 | tail -7)
  echo "== $lab estate 12 bodies prime+3 $(date +%T)"; (cd $S2 && ./w11.sh k9$lab 3 "12:0" >/dev/null 2>&1; python3 analyze_cliff.py k9$lab | cut -c1-160)
}
boot base && measure base
if [ "$ABORT" = 1 ]; then echo "SKIP K9 arm: a boot failed or was refused -> straight to restore"
else
  boot k9 "V02_ROOT=$K9" "VLLM_K9_GDN_CHUNK=1" "VLLM_K9_GDN_BUILD_DIR=/home/kevin/projects/lanes/k9/gdn_build_arm" && {
    measure k9
    echo "== k9 evalkit $(date +%T)"; (cd /home/kevin/Desktop/qwen38-evalkit && timeout 1500 python3 run_eval.py --tag k9-gdnchunk --categories tool_call,code_exec,long_ctx 2>&1 | grep -E "passed=False|/60|passed,")
    if [ $(( $(date +%s) - T_START )) -lt 2700 ]; then echo "== k9 needle 262K $(date +%T)"; python3 $I/tools/up-bench/needle_long.py --tokens 262000 --depth 0.5 2>&1 | tail -1 | cut -c1-200; else echo "needle skipped: window time budget"; fi
    echo "k9 errors in journal: $(journalctl -u vllm-qwen27b --since '-45 min' --no-pager | grep -cE 'Traceback|CUDA error|illegal memory') OOMwarn=$(journalctl -u vllm-qwen27b --since '-45 min' --no-pager | grep -c 'allocation failed with OOM')"
  }
fi
echo "== restore $(date +%T)"; cp $L/override.saved $O; gpu_clean_wait restore
python3 $SD/engine-actuator.py restart --by K9 --reason "K9 window restore" --no-drain --foreground > $L/boot_restore.log 2>&1 &
sleep 20; t0=$(date +%s); until curl -s -m 2 -o /dev/null -w "%{http_code}" localhost:8001/health | grep -q 200; do [ $(( $(date +%s) - t0 )) -ge 900 ] && break; sleep 4; done
cmp -s $O $L/override.saved && echo "override restored verbatim"
[ "$WD0" = active ] && sudo -n systemctl start vllm-qwen27b-watchdog.timer; echo "watchdog timer now: $(systemctl is-active vllm-qwen27b-watchdog.timer)"
echo "restored health $(curl -s -m3 -o /dev/null -w '%{http_code}' localhost:8001/health) KV: $(journalctl -u vllm-qwen27b --since '-10 min' --no-pager | grep -E 'GPU KV cache size' | tail -1 | sed 's/.*INFO//' | cut -c1-80) Xid delta=$(( $(journalctl -k --no-pager | grep -c 'NVRM: Xid') - X0 )) $(date)"
