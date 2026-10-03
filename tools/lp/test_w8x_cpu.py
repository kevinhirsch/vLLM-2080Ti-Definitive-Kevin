"""CPU check (TRITON_INTERPRET=1) of the W8X expand kernel on python-Marlin-packed asym g128 weights."""
import os, sys
os.environ["TRITON_INTERPRET"] = "1"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/kevin/Desktop/wt-lp"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import awq_marlin_quantize
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils import lp_w8x as W
import marlin_dequant_r2 as md
torch.manual_seed(0)
for k, n in ((256, 128), (384, 192), (512, 256)):
    w = torch.randn(k, n, dtype=torch.float16)
    w_ref, q, s, zp = awq_marlin_quantize(w, scalar_types.uint4, 128)
    st = W.W8XState(q, s, zp, k, n)
    w8 = W.expand(st)                                        # [N, K] int8
    deq = md.marlin_dequant_torch(q, s, zp, k, n, 128).float()   # [K, N]
    codes = md.unpack_marlin_codes(q, k, n).float()                       # [K, N]
    zfull = md.unpermute_zero_points(zp, k // 128, n).float().repeat_interleave(128, 0)
    sd = md.unpermute_scales(s, k, n, 128).float()
    ratio = sd / st.s_ch.reshape(1, -1)
    v = ((codes - zfull) * ratio.repeat_interleave(128, 0)).T
    v = torch.where(v >= 0, torch.floor(v + 0.5), torch.ceil(v - 0.5))
    exp = torch.clamp(v, -127, 127)
    assert torch.equal(w8.float(), exp), (k, n, (w8.float() - exp).abs().max())
    rel = ((w8.float() * st.s_ch.reshape(-1, 1) - deq.T).norm() / deq.norm()).item()
    print(k, n, "expand == reference; w8*s_ch vs W4 dequant rel err %.4f" % rel)
print("W8X_CPU_OK")
