#!/usr/bin/env bash
# ncu_run.sh TAG [cu_micro args]  -> profiles tq_gqa_stage1 from a fresh -lineinfo build dir build_li_TAG
TAG=$1; shift; A=${@:-3568 28000 1 4 128}
P=/home/kevin/.local/share/shim-gcc15:/home/kevin/Desktop/wt-integrate/.venv/bin:/usr/local/cuda-13/bin:/usr/bin:/bin
cd /home/kevin/projects/lanes/s2-speed
sudo -n env PATH="$P" CUDA_HOME=/usr/local/cuda-13 CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15 NVCC_CCBIN=/usr/bin/g++-15 VLLM_TQ_GQA_LINEINFO=1 VLLM_TQ_GQA_BUILD_DIR=/home/kevin/projects/lanes/s2-speed/build_li_$TAG CUDA_VISIBLE_DEVICES=0 HOME=/home/kevin PYTHONPATH=/home/kevin/Desktop/wt-integrate timeout 900 /usr/local/cuda-13/bin/ncu --kernel-name regex:tq_gqa_stage1 --launch-skip 1 --launch-count 1 --section SourceCounters --section SpeedOfLight --section LaunchStats --section Occupancy --section WarpStateStats --section SchedulerStats -f -o ncu_$TAG /home/kevin/Desktop/wt-integrate/.venv/bin/python cu_micro.py $A 2>&1 | grep -E "cuda NS|rror"
sudo -n chown -R kevin:kevin ncu_$TAG.ncu-rep build_li_$TAG
N=/usr/local/cuda-13/bin/ncu
$N --import ncu_$TAG.ncu-rep --page details 2>&1 | grep -E "Duration|Executed Ipc Active|Registers Per|Achieved Occupancy|Warp Cycles Per Issued"
$N --import ncu_$TAG.ncu-rep --page raw 2>&1 | grep -E "smsp__inst_executed.sum " | head -1
$N --import ncu_$TAG.ncu-rep --page source --csv --print-source sass 2>/dev/null > sass_$TAG.csv
( cd cub && rm -f *.cubin && /usr/local/cuda-13/bin/cuobjdump -xelf all ../build_li_$TAG/tq_gqa_sm75.so >/dev/null 2>&1 && /usr/local/cuda-13/bin/nvdisasm -g -c tq_gqa.sm_75.cubin > dis.txt 2>/dev/null )
python3 byline.py sass_$TAG.csv | head -${NLINES:-16}
