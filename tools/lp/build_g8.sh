#!/usr/bin/env bash
# Build (or verify cached) the LP W4A8 Marlin JIT extensions (g mode + e mode EMAX 1,2,3) into wt-lp/.deps. CPU only.
cd /home/kevin/Desktop/wt-lp
export PATH=/home/kevin/.local/share/shim-gcc15:/home/kevin/Desktop/wt-integrate/.venv/bin:/usr/local/cuda-13/bin:$PATH CUDA_HOME=/usr/local/cuda-13
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 NVCC_CCBIN=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=7.5
CUDA_VISIBLE_DEVICES= /home/kevin/Desktop/wt-integrate/.venv/bin/python -c "
import sys; sys.path.insert(0,'/home/kevin/Desktop/wt-lp')
from vllm.model_executor.layers.quantization.utils import lp_w4a8g as L
for m, e in [('g', 2), ('p16', 0), ('r16', 0), ('pe', 3), ('pe', 0)] + [('e', x) for x in ${LP_EMAXES:-(1, 2, 3)}]:
    print('BUILT', m, e, L.load_ext(m, e).__file__, flush=True)"
