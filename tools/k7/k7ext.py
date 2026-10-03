"""Lane K7: JIT-build + load the W4A4 sm_75 extension (tools/k7/csrc/k7_w4a4.cu).  Build cache: ~/projects/lanes/k7/build."""
import os
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
os.environ.setdefault("CC", "gcc-14"); os.environ.setdefault("CXX", "g++-14")
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = os.environ.get("K7_CUTLASS", "/home/kevin/Desktop/vLLM-2080Ti-Definitive/.deps/cutlass-src")
BUILD = os.environ.get("K7_BUILD", "/home/kevin/projects/lanes/k7/build")
os.makedirs(BUILD, exist_ok=True)
_mod = None


def ext(verbose=False):
    global _mod
    if _mod is None:
        _mod = load(name="k7_w4a4", sources=[os.path.join(HERE, "csrc/k7_w4a4.cu")], build_directory=BUILD,
                    extra_include_paths=[f"{CUTLASS}/include", f"{CUTLASS}/tools/util/include"],
                    extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-gencode=arch=compute_75,code=sm_75",
                                       "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1", "-ccbin=g++-14", "-Xptxas=-v" if verbose else "-lineinfo"],
                    verbose=verbose)
    return _mod


def pack_s4(q: torch.Tensor) -> torch.Tensor:
    """q int tensor [R, K] in [-8, 7] -> int8 [R, K/2], low nibble = even column (cutlass int4b_t order)."""
    q = q.to(torch.int16)
    lo, hi = q[:, 0::2] & 15, q[:, 1::2] & 15
    return (lo | (hi << 4)).to(torch.uint8).view(torch.int8)


def unpack_s4(p: torch.Tensor) -> torch.Tensor:
    u = p.view(torch.uint8).to(torch.int16)
    lo, hi = u & 15, (u >> 4) & 15
    lo = torch.where(lo > 7, lo - 16, lo); hi = torch.where(hi > 7, hi - 16, hi)
    return torch.stack([lo, hi], -1).reshape(p.shape[0], -1)


def gpu_gate(need_mib=800, exit_on_fail=True):
    """Shared estate GPU gate (lane RL): refuses during windows/boots/planned-offline/unhealthy engine/low free VRAM."""
    import subprocess, sys
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    r = subprocess.run([os.path.expanduser("~/projects/lanes/windows/gpuok.sh"), gpu, str(need_mib)], capture_output=True, text=True)
    if r.returncode != 0:
        msg = f"gpuok refused GPU{gpu}: {r.stdout.strip()} {r.stderr.strip()}"
        if exit_on_fail:
            sys.exit(msg)
        print(msg, flush=True)
        return False
    return True
