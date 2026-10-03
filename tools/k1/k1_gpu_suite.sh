#!/usr/bin/env bash
# K1 GPU suite: correctness gates + quiet microbenches of the K1 kernel. Touches no engine state; every python
# process re-checks ~/projects/lanes/windows/gpuok.sh (run inside a window with WINDOW_ID set, or beside a quiet
# engine). GPU = $K1_GPU (default 1). ~12 min. Output: ~/projects/lanes/k1/suite_<time>/.
set -u
G=${K1_GPU:-1}
cd "$(dirname "$0")"
O=/home/kevin/projects/lanes/k1/suite_$(date +%H%M); mkdir -p "$O"
export CUDA_VISIBLE_DEVICES=$G K1_CAP_MB=${K1_CAP_MB:-420} K1_FI_WS_MB=64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_HOME=/usr/local/cuda-13 PATH=/home/kevin/Desktop/wt-integrate/.venv/bin:/usr/local/cuda-13/bin:$PATH \
  PYTHONPATH=/home/kevin/Desktop/wt-k1:/home/kevin/projects/lanes/windows
# every GPU process runs under the enforced cap runner (own systemd unit, allocator cap, watcher kill at cap+margin)
CAP=${K1_CAP_MB:-420}
run() { local n=$1; shift; echo "== $n $(date +%T)"
  ~/projects/lanes/windows/gpuok.sh --run --lane k1 --job "$n" --gpu "$G" --cap $((CAP + 250)) --margin 150 --timeout 1500 \
    --out "$O/$n.log" -- env CUDA_VISIBLE_DEVICES=$G K1_CAP_MB=$CAP K1_FI_WS_MB=64 PYTHONPATH=$PYTHONPATH PATH=$PATH CUDA_HOME=$CUDA_HOME "$@"
  echo "rc=$? $(grep -E 'GATE|passed|failed|Hk=|ms ' "$O/$n.log" | grep -v Warn | tail -14)"; }
run gate python test_fa75_prefill.py
run pytest_cont python -m pytest -q -p no:cacheprovider --confcutdir=/home/kevin/Desktop/wt-k1/tools/k1 -c /dev/null --rootdir=/home/kevin/Desktop/wt-k1/tools/k1 -p live_guard /home/kevin/Desktop/wt-k1/tests/v1/attention/test_tq_k1_continuation.py /home/kevin/Desktop/wt-k1/tests/v1/attention/test_tq_prefix_combine_lse.py
run gqa_proxy python ablate.py --gqa-proxy
run ablate_3632x32k python ablate.py 3632 32768
run ablate_512x64k python ablate.py 512 65536
run grid python bench_fa75_prefill.py --variants 0,3,7,8 --budget-mb 300 --rounds 5 --out "$O/grid.json"
python3 fmt.py < "$O/grid.log" > "$O/grid.txt" 2>/dev/null; cat "$O/grid.txt"
echo "suite done $(date +%T) -> $O"
