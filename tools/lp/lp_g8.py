"""Lane LP tool-side wrapper around vllm/model_executor/layers/quantization/utils/lp_w4a8g.py (single source of truth).
build(mode, emax)        -> JIT extension (mode 'g' float act scales, 'e' int-exponent); cached in .deps/lp_marlin_*_build
quant_act_g128(x)        -> torch reference of the g-mode quantizer; quant_act_g128_triton(x) -> Triton (engine) version
G8Linear(t, N, K, mode)  -> layer from CT pack-quantized asym g128 tensors; .forward(x fp16) -> fp16
"""
import os
import sys

import torch

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")))
from vllm.model_executor.layers.quantization.utils import lp_w4a8g as L  # noqa: E402


def build(mode="g", emax=2, verbose=False):
    return L.load_ext(mode, emax)


def quant_act_g128(x, g=128):
    M, K = x.shape
    xg = x.float().view(M, K // g, g)
    s = xg.abs().amax(-1).clamp(min=1e-8) / 127.0
    q = torch.clamp(torch.round(xg / s.unsqueeze(-1)), -127, 127).to(torch.int8).view(M, K)
    return q, s.contiguous()


quant_act_g128_triton = L.quant_g128


class G8Linear:
    def __init__(self, t, N, K, device="cuda", mode="g", emax=2, level=None):
        sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
        from _ctlayer import make_marlin_layer
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
        os.environ["VLLM_MARLIN_INPUT_DTYPE"] = "int8"
        try:
            layer, scheme = make_marlin_layer({k: v for k, v in t.items()}, N, K, device=device)
        finally:
            os.environ.pop("VLLM_MARLIN_INPUT_DTYPE", None)
        self.N, self.K, self.mode, self.emax = N, K, mode, emax
        self.q = layer.weight_packed.data
        self.zp = layer.weight_zero_point.data
        s = t["weight_scale"].to(device=device, dtype=torch.float16).t().contiguous()
        sp = marlin_permute_scales(s, size_k=K, size_n=N, group_size=128, is_a_8bit=True).contiguous()
        if mode == "g":
            self.s, self.wglob = sp, 1.0
        else:
            self.level = level or (4096 >> emax)
            self.s, self.wglob = L.process_scales_e(sp, self.level)
        self.ws = torch.zeros(1024, dtype=torch.int32, device=device)
        self.stock = (layer, scheme)
        build(mode, emax)

    def forward(self, x, qfn=None):
        if self.mode == "g":
            q, s = (qfn or quant_act_g128)(x)
            ones = torch.ones((x.shape[0],), dtype=torch.float32, device=x.device)
            return build("g").gemm(q, ones, s, self.q, self.s, self.zp, self.ws, self.N, True)
        q, e, r = L.quant_e(x, self.wglob, self.emax)
        return build("e", self.emax).gemm(q, r, e, self.q, self.s, self.zp, self.ws, self.N, True)
