#!/usr/bin/env bash
# Build (or verify cached) the LP_A8G Marlin JIT extension into wt-lp/.deps/lp_marlin_g8_build. CPU only.
cd /home/kevin/Desktop/wt-lp
export PATH=/home/kevin/.local/share/shim-gcc15:/home/kevin/Desktop/wt-integrate/.venv/bin:/usr/local/cuda-13/bin:$PATH CUDA_HOME=/usr/local/cuda-13
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 NVCC_CCBIN=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=7.5
CUDA_VISIBLE_DEVICES= /home/kevin/Desktop/wt-integrate/.venv/bin/python -c "
import sys; sys.path.insert(0,'tools/lp'); import lp_g8; m=lp_g8.build(verbose=True); print('BUILT', m.__file__)"
