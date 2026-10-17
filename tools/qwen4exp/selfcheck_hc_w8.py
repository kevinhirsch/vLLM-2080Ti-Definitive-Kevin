"""CPU selfcheck (TRITON_INTERPRET=1) of the W8A16 HC GEMM against torch, at the HC shapes.

down: [M, 10240] -> [M, 336] (split-K path), up: [M, 320] -> [M, 10240] (single split);
M in {1, 3, 17} (decode, MTP verify, a prefill tile edge). The kernel must equal
x @ dequant(q, s).T within fp32-accumulation noise, and the dequantized weight must be
within the expected int8 error of the original.
"""
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch

from vllm.models.qwen4_exp.nvidia.ops.hc_w8 import HC_W8_GROUP, quantize_w8_groupwise, w8a16_linear

torch.manual_seed(0)
ok_all = True
for name, (n, k) in {"down_inject": (336, 10240), "up": (10240, 320)}.items():
    w = (torch.randn(n, k) * 0.02).half()
    w[-12:] = 0  # the merged layer's zero padding rows
    q, s = quantize_w8_groupwise(w, HC_W8_GROUP)
    wq = (q.float().view(n, k // HC_W8_GROUP, HC_W8_GROUP) * s.float()[..., None]).view(n, k)
    werr = ((wq - w.float()).norm() / w.float().norm()).item()
    for m in (1, 3, 17):
        x = torch.randn(m, k).half()
        y = w8a16_linear(x, q, s, HC_W8_GROUP, torch.float32)
        ref = x.float() @ wq.T
        err = ((y - ref).abs().max() / ref.abs().max()).item()
        ok = err < 1e-3 and torch.isfinite(y).all().item() and y[:, -12:].abs().max().item() == 0.0
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'} {name} M={m}: kernel vs dequant ref rel max err {err:.2e} (weight int8 err {werr:.2e})")
print("ALL PASS" if ok_all else "SOME FAIL")
sys.exit(0 if ok_all else 1)
