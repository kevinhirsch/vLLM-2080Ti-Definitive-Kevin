#!/usr/bin/env bash
# Lane LP window W1 (~60 min; ~42 min with LP_SECOND=0): W4A8-INT8 Marlin on the SHIPPED asymmetric W4 weights (no requant).
#   Phase A (small-footprint IDLE engine, ~10 min incl. boot): kernel microbench both GPUs (per-rank shapes, M=16..3632, correctness vs CPU emulation)
#            + 25 s sustained gate_up per format per GPU (SM clock / power / temp under int8 vs fp16).
#   Phase B (engine UP with int8 activations, ~25 min; W4A8G = per-(row,128) act scales if its kernel gate passes): B1 cold prefill x3, quick decode, 12-body estate pass x3,
#            det cold vs stack ref, evalkit tool_call+code_exec+long_ctx, needle 71K; then restore whatever override was live.
# Run under the gateway offline lease, exactly like S4 windows:
#   python3 /home/kevin/Desktop/wt-integrate/deploy/bin/gateway-offline.py run --reason "LP W1 W4A8" --by LP --ttl 3000 --wait-s 90 -- bash /home/kevin/Desktop/wt-lp/tools/lp/lp_win1.sh
# Env: LP_ONLY='<regex>' to restrict int8 to matching layers (e.g. 'gate_up_proj|linear_attn|self_attn' keeps down_proj W4A16).
set -u
OUT=/home/kevin/projects/lanes/lp; mkdir -p $OUT; cd /home/kevin/projects/lanes/s2-speed; export ESTATE_FR=$PWD/fr
OV=/home/kevin/.local/share/vllm-qwen27b/v02.override.env; cp $OV $OUT/override.before
LABEL=${LABEL:-lpw4a8}
PY=/home/kevin/Desktop/wt-integrate/.venv/bin/python
export PATH=/home/kevin/.local/share/shim-gcc15:/home/kevin/Desktop/wt-integrate/.venv/bin:/usr/local/cuda-13/bin:$PATH CUDA_HOME=/usr/local/cuda-13 CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15
X0=$(journalctl -k --no-pager | grep -c "NVRM: Xid")
echo "LP W1 start $(date)  label=$LABEL only='${LP_ONLY:-}'"
# ---- Phase A: small-footprint idle engine (S4 window-B2 pattern: util 0.45, maxlen 64K -> ~10 GiB/GPU free; gateway is offline) ----
BOOT_TIMEOUT=900 ./boot2.sh ${LABEL}-small "VLLM_GPU_UTIL=0.45" "V02_MAXLEN=65536" || echo "small boot failed; benching anyway if VRAM allows"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
for g in 0 1; do
  LP_IN_WINDOW=1 CUDA_VISIBLE_DEVICES=$g PYTHONPATH=/home/kevin/Desktop/wt-lp:/home/kevin/Desktop/wt-integrate/tools/u2 timeout 900 $PY /home/kevin/Desktop/wt-lp/tools/lp/w4a8_bench.py \
     --min-free-mib 3000 --cap-mib 2500 --iters 30 --sustain 25 --json $OUT/w4a8_bench_gpu$g.json > $OUT/w4a8_bench_gpu$g.log 2>&1
  echo "bench gpu$g rc=$? :"; grep -E "PER-CHUNK|SUSTAIN|correctness" $OUT/w4a8_bench_gpu$g.log | cut -c1-200
done
# ---- Phase A2 (~5-10 min, small idle engine still up): full-model kernel-exact fidelity gate on GPU0 (K7's --device cuda mode)
GPU_IN_WINDOW=1 CUDA_VISIBLE_DEVICES=0 timeout 900 $PY /home/kevin/Desktop/wt-lp/tools/lp/variant_gate.py --device cuda --gpu-need-mib 1500 \
   --variants w4a8e3,w4a8g,w4a8 --out $OUT/gate_gpu_win.json > $OUT/gate_gpu_win.log 2>&1; echo "gpu fidelity gate rc=$?"
grep -E "^\[.*\] (w4a8e3|w4a8g|w4a8) \{" $OUT/gate_gpu_win.log | cut -c1-260
# ---- Phase B: engine with int8 activations (from the wt-lp tree; everything else = the override that was live) ----
# Gate: the LP_A8G kernel (per-(row,128) act scales) must match its CPU emulation on every real shape and odd M (rel <= 5e-3);
# then Phase B runs W4A8G on all linears, else stock per-token W4A8 on all linears.
G8OK=$(python3 - <<PY
import json,glob
# pick the LP mode for Phase B: e (int-exponent, EMAX ${LP_EMAX:-3}) if its kernel matches emulation and is faster than stock-W4A16 on the chunk;
# else g (float scales) if correct; else 0 (stock per-token W4A8)
okg=oke=True; n=0; tot={}
for f in glob.glob("$OUT/w4a8_bench_gpu*.json"):
    d=json.load(open(f))
    for r in d["rows"]:
        for k,v in r.items():
            if k.startswith("corr_rel_w4a8g_vs_emul"): n+=1; okg = okg and v <= 5e-3
            if k.startswith("corr_rel_w4a8e2_vs_emul"): oke = oke and v <= 5e-3
    for k,v in d["chunk_ms"].items(): tot[k]=tot.get(k,0)+v
fe = tot.get("w4a8e2", 9e9) < tot.get("w4a16", 0); fg = tot.get("w4a8g", 9e9) < tot.get("w4a16", 0)
print("e" if n and oke and fe else ("g" if n and okg and fg else "0"))
PY
)
echo "LP mode chosen for Phase B (e/g/0=stock): $G8OK"
mapfile -t LIVE < <(grep -E '^export ' $OUT/override.before | sed 's/^export //')
EXTRA=("V02_ROOT=/home/kevin/Desktop/wt-lp" "VLLM_MARLIN_INPUT_DTYPE=int8")
if [ "$G8OK" != "0" ]; then EXTRA+=("VLLM_LP_W4A8G=1" "VLLM_LP_W4A8G_MODE=$G8OK" "VLLM_LP_A8E_EMAX=${LP_EMAX:-3}"); LABEL=${LABEL}$G8OK
else EXTRA+=("VLLM_LP_INT8_ONLY='gate_up_proj|linear_attn|self_attn'"); LABEL=${LABEL}nd; fi   # per-token int8 is unfit for down_proj
[ -n "${LP_ONLY:-}" ] && EXTRA+=("VLLM_LP_INT8_ONLY='${LP_ONLY}'")
BOOT_TIMEOUT=900 ./boot2.sh $LABEL "${LIVE[@]}" "${EXTRA[@]}" || { echo "BOOT FAILED"; ./boot2.sh restore-$LABEL "${LIVE[@]}" >/dev/null 2>&1; cp $OUT/override.before $OV; exit 1; }
echo "booted $(date): $(journalctl -u vllm-qwen27b --since '-15 min' --no-pager | grep -E 'GPU KV cache size|Model loading took|MarlinLinearKernel|int8' | sed 's/.*INFO//' | cut -c1-110 | sort -u | tr '\n' '|')"
python3 /home/kevin/Desktop/wt-lp/tools/lp/cold_prefill.py 3 $OUT/b1_${LABEL}.json
python3 /home/kevin/Desktop/wt-integrate/tools/s2-bench/quick.py 2>&1 | tail -1
python3 det.py det_${LABEL}_cold.json; echo "vs stack ref (cold):"; python3 det.py --cmp det_s4_stack_ref.json det_${LABEL}_cold.json | tail -7
./w11.sh $LABEL 3 "12:0" >/dev/null 2>&1; python3 analyze_cliff.py $LABEL | cut -c1-150
echo "== evalkit $(date)"; (cd /home/kevin/Desktop/qwen38-evalkit && python3 run_eval.py --tag lp-$LABEL --categories tool_call,code_exec,long_ctx 2>&1 | grep -E "passed=False|/60|passed,")
echo "== needle 71K $(date)"; python3 /home/kevin/Desktop/wt-integrate/tools/up-bench/needle_long.py --tokens 71000 --depth 0.5 2>&1 | tail -1 | cut -c1-200
echo "xid: $(( $(journalctl -k --no-pager | grep -c 'NVRM: Xid') - X0 )) OOMwarn=$(journalctl -u vllm-qwen27b --since '-45 min' --no-pager | grep -c 'allocation failed with OOM')"
# ---- Phase C (optional, LP_SECOND=1, ~18 min): STOCK per-token int8 everywhere EXCEPT mlp.down_proj (worst activation SQNR, 18 dB) ----
if [ "${LP_SECOND:-1}" = "1" ] && [ -z "${LP_ONLY:-}" ]; then
  L2=lpw4a8nd
  BOOT_TIMEOUT=900 ./boot2.sh $L2 "${LIVE[@]}" "V02_ROOT=/home/kevin/Desktop/wt-lp" "VLLM_MARLIN_INPUT_DTYPE=int8" "VLLM_LP_INT8_ONLY='gate_up_proj|linear_attn|self_attn'" \
    && { python3 /home/kevin/Desktop/wt-lp/tools/lp/cold_prefill.py 3 $OUT/b1_${L2}.json
         python3 det.py det_${L2}_cold.json; python3 det.py --cmp det_s4_stack_ref.json det_${L2}_cold.json | tail -1
         (cd /home/kevin/Desktop/qwen38-evalkit && python3 run_eval.py --tag lp-$L2 --categories tool_call,code_exec,long_ctx 2>&1 | grep -E "passed=False|/60|passed,"); } \
    || echo "phase C boot failed"
fi
# ---- restore exactly the override that was live before the window ----
cp $OUT/override.before $OV
mapfile -t LIVE < <(grep -E '^export ' $OUT/override.before | sed 's/^export //')
./boot2.sh restore-$LABEL "${LIVE[@]}" >/dev/null 2>&1; cp $OUT/override.before $OV
echo "restored health $(curl -s -m3 -o /dev/null -w '%{http_code}' localhost:8001/health) $(date)"
echo "LP W1 end $(date)"
