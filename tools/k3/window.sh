#!/usr/bin/env bash
# K3 (L78) in-window microbench runner. Use as a windowctl step AFTER an `engine: stop` step (needs ~1 GiB free per GPU):
#   {"name":"k3-micro","run":"bash {{REL}}/tools/k3/window.sh {{REL}} {{RESULTS}}/k3","timeout_s":900,"on_fail":"continue"}
# $1 = release root (default: the checkout this script lives in), $2 = output prefix. Never touches the engine.
set -u
REL=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT=${2:-/tmp/k3_out}; mkdir -p "$(dirname "$OUT")"
PY=${PY:-$REL/.venv/bin/python}; [ -x "$PY" ] || PY=python3
export PYTHONPATH=$REL PYTHONUNBUFFERED=1
rc=0
# each mode spawns its own 2 workers with an explicit shared --rdv file and a whole-run kill timeout (see bench_allreduce.py)
$PY "$REL/tools/k3/bench_allreduce.py" --mode check   --tokens 1,8,64,256 --timeout 240 --out "$OUT.chk" | tee "$OUT.chk.txt" || rc=$?
$PY "$REL/tools/k3/bench_allreduce.py" --mode ar      --tokens 64,256,819,1024,1816,2048,3584 --timeout 420 --out "$OUT.ar" | tee "$OUT.ar.txt" || rc=$?
$PY "$REL/tools/k3/bench_allreduce.py" --mode overlap --tokens 1816,3584 --timeout 420 --out "$OUT.ov" | tee "$OUT.ov.txt" || rc=$?
exit $rc
