#!/usr/bin/env python
"""Lane K9 (L102): the engine-level wrappers (qwen_gdn_linear_attn.flashqla_legacy_[varlen_]chunk_gated_delta_rule:
l2norm + casts) with VLLM_K9_GDN_CHUNK=1 vs the FlashQLA kernel, single and varlen, unnormalised q/k like the layer."""
import os, sys, json
os.environ["VLLM_K9_GDN_CHUNK"] = "1"
sys.path.insert(0, "/home/kevin/Desktop/wt-k9")
import torch
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as M
import vllm.model_executor.layers.mamba.gdn.k9_gdn_chunk as K9
assert K9.ENABLED
dev = torch.device("cuda"); torch.manual_seed(0)
T, Hk, Hv, D = 2000, 8, 24, 128
q = torch.randn(1, T, Hk, D, device=dev).half() * 3; k = torch.randn(1, T, Hk, D, device=dev).half() * 3
v = torch.randn(1, T, Hv, D, device=dev).half(); g = -torch.rand(1, T, Hv, device=dev) * 0.3; beta = torch.rand(1, T, Hv, device=dev).half()
def run(enabled, varlen):
    K9.ENABLED = enabled
    if varlen:
        cu = torch.tensor([0, 700, 1500, T], dtype=torch.int32, device=dev); st = torch.randn(3, Hv, D, D, device=dev) * 0.05
        return M.flashqla_legacy_varlen_chunk_gated_delta_rule(q, k, v, g, beta, st, True, cu)
    st = torch.randn(1, Hv, D, D, device=dev, generator=torch.Generator(device=dev).manual_seed(1)) * 0.05
    return M.flashqla_legacy_chunk_gated_delta_rule(q, k, v, g, beta, st, True)
for varlen in (False, True):
    torch.manual_seed(5); oA, sA = run(False, varlen); torch.manual_seed(5); oB, sB = run(True, varlen)
    print(json.dumps({"varlen": varlen, "out_shape": list(oB.shape), "dtype": str(oB.dtype), "state_dtype": str(sB.dtype),
                      "out_rel": ((oB.float() - oA.float()).norm() / oA.float().norm()).item(),
                      "state_rel": ((sB.float() - sA.float()).norm() / sA.float().norm()).item()}))
