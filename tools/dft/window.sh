#!/usr/bin/env bash
# window.sh -- Lane DFT overnight window body. Run ONLY under:
#   cd /home/kevin/Desktop/vLLM-2080Ti-Definitive && .venv/bin/python deploy/bin/gateway-offline.py run --reason "DFT MTP drafter fine-tune (Kevin-approved 02:00)" --by DFT --ttl 16000 -- bash /home/kevin/projects/lanes/dft/window.sh
# Baseline = the production STACK (text-only + int4 lm_head + int4 MTP; V02_STACK=1 default in serve-hauhaucs-v02.sh). Tuned bf16 weights go through the same load-time int4 quantiser.
# Phases: base live A/B (engine up) -> on-policy generation (engine up) -> stop engine -> hidden-state extraction -> offline base eval (sanity vs live)
#         -> train -> offline eval base vs tuned -> boot tuned variant -> live A/B + gates -> KEEP or restore -> ALWAYS a healthy engine at exit.
cd /home/kevin/projects/lanes/dft; L=$PWD; R=$L/results
PY=/home/kevin/Desktop/wt-integrate/.venv/bin/python
SV=/home/kevin/.local/share/vllm-qwen27b
S2=/home/kevin/projects/lanes/s2-speed; export ESTATE_FR=$S2/fr
VAR=/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven-dft1
START=$(date +%s); HARD_END=$(( START + ${DFT_MAX_S:-14400} ))
ts(){ date '+%F %T'; }
say(){ echo "[$(ts)] $*"; }
spend(){ python3 - <<'P'
import json;d=json.load(open('/home/kevin/.local/share/vllm-qwen27b/gateway-spend.json'));print(d['day'],round(d['spent'],3))
P
}
say "DFT window start; spend: $(spend)"
cp -p $SV/v02.override.env $L/override.baseline.$START; cp -p $SV/active-serve $L/active-serve.baseline.$START
say "baseline override: $(tr '\n' ';' < $SV/v02.override.env)"; say "baseline active-serve: $(cat $SV/active-serve)"
rm -f $L/override.saved

restore_base(){   # restore the exact baseline override + active-serve, restart, verify healthy
  say "RESTORE baseline engine"
  cp -p $L/override.baseline.$START $SV/v02.override.env; cp -p $L/active-serve.baseline.$START $SV/active-serve; rm -f $L/override.saved
  sudo -n systemctl stop vllm-qwen27b >/dev/null 2>&1; sleep 6
  for p in $(pgrep -f "[w]armup-after-start.sh"); do kill $p; done
  BOOT_TIMEOUT=900 $L/boot_cfg.sh restore || BOOT_TIMEOUT=900 $L/boot_cfg.sh restore2
  sudo -n systemctl start vllm-qwen27b-watchdog.timer >/dev/null 2>&1
  say "restored health: $(curl -s -m3 -o /dev/null -w '%{http_code}' localhost:8001/health) $(journalctl -u vllm-qwen27b --since '-10 min' --no-pager | grep 'GPU KV cache size' | tail -1 | sed 's/.*size: //' | cut -c1-24)"
}
finish_trap(){ if ! curl -s -m3 -o /dev/null -w '%{http_code}' localhost:8001/health | grep -q 200; then restore_base; fi; say "DFT window end; spend: $(spend)"; }
trap finish_trap EXIT

if ! curl -s -m3 -o /dev/null -w '%{http_code}' localhost:8001/health | grep -q 200; then say "engine NOT healthy at start; booting baseline"; BOOT_TIMEOUT=900 $L/boot_cfg.sh start0 || { say "cannot boot baseline"; exit 1; }; fi

# ---------- P1: base live A/B + greedy reference
say "P1 base live A/B"; $PY $L/live_ab.py base $R/live_base.json 2>&1 | tail -2
say "P1 det base cold/warm"; python3 $S2/det.py $R/det_base_cold.json | tail -1; python3 $S2/det.py $R/det_base_warm.json | tail -1; python3 $S2/det.py --cmp $R/det_base_cold.json $R/det_base_warm.json | tail -1
say "spend: $(spend)"

# ---------- P2: on-policy generation on the live engine
say "P2 on-policy gen"; $PY $L/gen_onpolicy.py --n 100 --budget-s 840 2>&1 | tail -3
say "spend: $(spend)"

# ---------- P3: stop engine, free GPUs
say "P3 stop engine"
sudo -n systemctl stop vllm-qwen27b-watchdog.timer; sudo -n systemctl stop vllm-qwen27b; sleep 10
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '
# ---------- P4: hidden-state extraction (deadline 40 min)
say "P4 extract"
DL=$(( $(date +%s) + ${DFT_EXTRACT_S:-2400} ))
$L/run_extract.sh $L/data/manifest.json $L/data/hid --deadline $DL > $R/extract.log 2>&1; tail -3 $R/extract.log
NH=$(ls $L/data/hid/*.npy 2>/dev/null | wc -l); say "extracted $NH sequences"
if [ "$NH" -lt 40 ]; then say "EXTRACTION FAILED/too small ($NH)"; restore_base; exit 2; fi

# ---------- P5: offline base eval (sanity against live)
say "P5 offline base eval (sanity)"
CUDA_VISIBLE_DEVICES=0 $PY $L/eval_mtp.py --hid-dir $L/data/hid --mtp base=/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven/model-mtp.safetensors --out $R/offline_base_only.json 2>&1 | grep -v Warn | tail -4

# ---------- P6: train (2 GPUs)
say "P6 train"
rm -rf $L/out; mkdir -p $L/out
cd $L; CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port 29517 $L/train_mtp.py --hid-dir $L/data/hid --out-dir $L/out --budget-min ${DFT_TRAIN_MIN:-45} > $R/train.log 2>&1
grep -E "VAL|FINAL|WROTE|Error|error" $R/train.log | tail -15
[ -f $L/out/model-mtp.safetensors ] || { say "TRAIN FAILED"; tail -20 $R/train.log; restore_base; exit 3; }

# ---------- P7: offline eval base vs tuned (+ int4-sim)
say "P7 offline eval"
BASEF=/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven/model-mtp.safetensors
CUDA_VISIBLE_DEVICES=0 $PY $L/eval_mtp.py --hid-dir $L/data/hid --mtp base=$BASEF tuned=$L/out/model-mtp.safetensors --int4-sim --out $R/offline.json 2>&1 | grep -v Warn | tail -12
GATE=$($PY - <<'P'
import json;r=json.load(open('/home/kevin/projects/lanes/dft/results/offline.json'))
b=r['base:int4']['assistant']['accept_len_sampled'];t=r['tuned:int4']['assistant']['accept_len_sampled'];bb=r['base:bf16']['assistant']['accept_len_sampled'];tb=r['tuned:bf16']['assistant']['accept_len_sampled'];print('PASS' if t>b+0.02 else 'FAIL', 'int4 base/tuned', round(b,4), round(t,4), 'bf16 base/tuned', round(bb,4), round(tb,4))
P
)
say "offline gate: $GATE"
case "$GATE" in PASS*) ;; *) say "tuned not better offline -> not booting it"; restore_base; exit 0;; esac

# ---------- P8: boot tuned variant (same override baseline + --model variant)
say "P8 boot tuned"
$L/make_variant.sh $L/out/model-mtp.safetensors $VAR
VLLM_U2_QUANT_DEVICE=cuda:0 CUDA_VISIBLE_DEVICES=0 $PY /home/kevin/Desktop/wt-integrate/tools/u2/prebuild_headquant.py --model $VAR 2>&1 | grep -c cached
$L/dft_cfg.sh on $VAR | tail -2
if ! BOOT_TIMEOUT=900 $L/boot_cfg.sh tuned; then say "tuned boot failed"; restore_base; exit 4; fi
say "tuned booted: $(journalctl -u vllm-qwen27b --since '-12 min' --no-pager | grep -E 'GPU KV cache size|Loading weights took' | tail -2 | sed 's/.*INFO//' | cut -c1-110 | tr '\n' '|')"

# ---------- P9: live A/B + gates on tuned
say "P9 tuned live A/B"; $PY $L/live_ab.py tuned $R/live_tuned.json 2>&1 | tail -2
say "P9 det tuned"; python3 $S2/det.py $R/det_tuned.json | tail -1; python3 $S2/det.py --cmp $R/det_base_cold.json $R/det_tuned.json | tee $R/det_cmp.txt | tail -8
say "P9 evalkit"; (cd /home/kevin/Desktop/qwen38-evalkit && python3 run_eval.py --tag dft-tuned --categories tool_call,code_exec,long_ctx 2>&1 | grep -E "passed=False|passed,|/60" | tail -6) | tee $R/evalkit.txt
say "spend: $(spend)"

# ---------- P10: decide
python3 $L/decide.py $R/live_base.json $R/live_tuned.json | tee $R/verdict.json; V=$?
EK=$(grep -oE '[0-9]+/60' $R/evalkit.txt | tail -1); EKN=${EK%%/*}
say "verdict exit=$V evalkit=${EK:-none}"
if [ "$V" -eq 0 ] && [ "${EKN:-0}" -ge 59 ]; then say "KEEP tuned drafter (engine stays on the tuned variant; rollback = dft_cfg.sh off + restart)"; echo KEEP > $R/final.txt
else say "NOT KEEPING -> restore base"; echo RESTORED > $R/final.txt; restore_base; fi
