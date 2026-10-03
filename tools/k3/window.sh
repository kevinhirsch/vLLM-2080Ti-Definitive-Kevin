#!/usr/bin/env bash
# K3 (L78) window script: engine must be STOPPED (needs ~1 GiB free per GPU). ~10 min.
set -u
cd /home/kevin/Desktop/wt-k3
export PYTHONPATH=$PWD
PY=/home/kevin/Desktop/wt-integrate/.venv/bin/python
OUT=${1:-/home/kevin/projects/lanes/k3}; mkdir -p "$OUT"
$PY tools/k3/bench_allreduce.py --mode check   --tokens 1,8,64,256            --out $OUT/chk | tee $OUT/chk.txt
$PY tools/k3/bench_allreduce.py --mode ar      --tokens 64,256,819,1024,1816,2048,3632 --out $OUT/ar  | tee $OUT/ar.txt
$PY tools/k3/bench_allreduce.py --mode overlap --tokens 1816,3632             --out $OUT/ov  | tee $OUT/ov.txt
