"""JIT build of the K8 CUDA kernels (sm_75) using the same toolchain env as the serve script."""
import os
from torch.utils.cpp_extension import load
ROOT = "/home/kevin/Desktop/wt-k8"
def build(name="k8_gdn", src="gdn_spec_decode.cu", verbose=False):
    env = dict(CUDA_HOME="/usr/local/cuda-13", CC="/usr/bin/gcc-15", CXX="/usr/bin/g++-15", CUDAHOSTCXX="/usr/bin/g++-15", NVCC_CCBIN="/usr/bin/g++-15")
    os.environ.update(env)
    os.environ["PATH"] = "/home/kevin/Desktop/wt-integrate/.venv/bin:/home/kevin/.local/share/shim-gcc15:/usr/local/cuda-13/bin:" + os.environ["PATH"]
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
    d = os.environ.get("K8_EXT_DIR", "/home/kevin/projects/lanes/k8/ext"); os.makedirs(d, exist_ok=True)
    return load(name=name, sources=[f"{ROOT}/csrc/k8/{src}"], build_directory=d, extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"], verbose=verbose)
