"""Lane LP: Python side of the MX-style W4A8 Marlin (per-(row,128) int8 activation scales), JIT-built from csrc_lp/marlin_g8.
build()                -> extension module (cached in .deps/lp_marlin_g8_build; needs nvcc, no GPU)
quant_act_g128(x)      -> (int8 [M,K], fp32 [M,K/128])      per-(row,128-group) symmetric int8 (torch reference; Triton below)
G8Linear(t, N, K)      -> layer from compressed-tensors pack-quantized asym g128 tensors; .forward(x fp16) -> fp16
"""
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.realpath(os.path.join(HERE, "../../csrc_lp/marlin_g8"))
BUILD = os.path.realpath(os.path.join(HERE, "../../.deps/lp_marlin_g8_build"))
_ext = None


def build(verbose=False):
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
        os.makedirs(BUILD, exist_ok=True)
        _ext = load(name="lp_marlin_g8", sources=[os.path.join(SRC, "lp_marlin_g8.cu")], build_directory=BUILD,
                    extra_include_paths=[SRC], verbose=verbose,
                    extra_cuda_cflags=["-O3", "-DLP_A8G", "--expt-relaxed-constexpr", "-std=c++17", "-lineinfo",
                                       "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                                       "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                                       "-gencode=arch=compute_75,code=sm_75"],
                    extra_cflags=["-O3", "-std=c++17"])
    return _ext


def quant_act_g128(x, g=128):
    M, K = x.shape
    xg = x.float().view(M, K // g, g)
    s = xg.abs().amax(-1).clamp(min=1e-8) / 127.0
    q = torch.clamp(torch.round(xg / s.unsqueeze(-1)), -127, 127).to(torch.int8).view(M, K)
    return q, s.contiguous()


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _q_g128(x_ptr, q_ptr, s_ptr, K, G: tl.constexpr):
        row = tl.program_id(0); grp = tl.program_id(1)
        offs = grp * G + tl.arange(0, G)
        x = tl.load(x_ptr + row * K + offs).to(tl.float32)
        s = tl.maximum(tl.max(tl.abs(x), 0), 1e-8) / 127.0
        q = tl.extra.cuda.libdevice.rint(x / s)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(q_ptr + row * K + offs, q.to(tl.int8))
        tl.store(s_ptr + row * (K // G) + grp, s)

    def quant_act_g128_triton(x, g=128):
        M, K = x.shape
        q = torch.empty((M, K), dtype=torch.int8, device=x.device); s = torch.empty((M, K // g), dtype=torch.float32, device=x.device)
        _q_g128[(M, K // g)](x, q, s, K, G=g)
        return q, s
except Exception:  # pragma: no cover
    quant_act_g128_triton = None


class G8Linear:
    """Holds the Marlin int8-layout tensors of one asym-W4 g128 linear and runs the LP_A8G kernel."""

    def __init__(self, t, N, K, device="cuda"):
        import sys
        sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
        from _ctlayer import make_marlin_layer
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
        os.environ["VLLM_MARLIN_INPUT_DTYPE"] = "int8"
        try:
            layer, scheme = make_marlin_layer({k: v for k, v in t.items()}, N, K, device=device)
        finally:
            os.environ.pop("VLLM_MARLIN_INPUT_DTYPE", None)
        self.N, self.K = N, K
        self.q = layer.weight_packed.data
        self.zp = layer.weight_zero_point.data
        # real fp16 scales, same permutation as the int8 path (stock W4A8 would have turned them into int16)
        s = t["weight_scale"].to(device=device, dtype=torch.float16).t().contiguous()  # [K/128, N]
        self.s = marlin_permute_scales(s, size_k=K, size_n=N, group_size=128, is_a_8bit=True).contiguous()
        self.ws = torch.zeros(1024, dtype=torch.int32, device=device)
        self.stock = (layer, scheme)

    def forward(self, x, qfn=None):
        q, s = (qfn or quant_act_g128)(x)
        return build().w4a8g_gemm(q, s, self.q, self.s, self.zp, self.ws, self.N, True)
