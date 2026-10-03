"""The production merge (merge_attn_states CUDA op if importable, else its torch rule) for K1 benches."""
import math

import torch

try:
    from vllm.v1.attention.ops.merge_attn_states import merge_attn_states as _m

    def merge(out, po, pl, so, sl):  # pl/sl [T, H] base-2 FlashInfer LSE -> ln, [H, T]
        _m(out, po, (pl * math.log(2)).T.contiguous(), so, (sl * math.log(2)).T.contiguous())
except Exception:  # pragma: no cover
    def merge(out, po, pl, so, sl):
        pl, sl = pl * math.log(2), sl * math.log(2)
        m = torch.maximum(pl, sl)
        a, b = torch.exp(pl - m), torch.exp(sl - m)
        out.copy_((po.float() * (a / (a + b))[..., None] + so.float() * (b / (a + b))[..., None]).half())
