#!/usr/bin/env bash
# suite.sh LABEL [model_dir served_name]  -- speed suite on the engine at :8001 (call when it is quiet)
L=$1; D=/home/kevin/projects/lanes/up-integrate; MD=${2:-/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven}; SN=${3:-qwen-local}
cd $D
echo "== bench_basic $L"; python3 bench_basic.py --reps 3 --long 5600 --out ${L}_basic.json > ${L}_basic.log 2>&1; tail -12 ${L}_basic.log
echo "== refbench $L"; ./refbench.sh $MD $SN 8001 $L 2>&1 | tee ${L}_ref.log
