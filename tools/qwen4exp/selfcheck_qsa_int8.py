"""CPU selfcheck (TRITON_INTERPRET=1) for the ported INT8 QSA main cache.

Same sparse problem through three paths:
  ref   : torch fp32 softmax(q k^T / sqrt(d)) v over the selected tokens
  fp16  : qsa_sparse_paged_attention on an fp16 paged cache (the shipped path)
  int8  : qsa_quantize_store_kv into an int8 paged cache, then the int8 kernel path
Gate: fp16 vs ref within fp16 noise, int8 vs ref within the int8 quantization bound,
and the int8 store matches common.qsa_int8.quantize_kv_int8 bit for bit.
"""
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch

from vllm.models.qwen4_exp.common.qsa_int8 import quantize_kv_int8
import vllm.models.qwen4_exp.nvidia.ops.qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    qsa_quantize_store_kv,
    qsa_sparse_paged_attention,
)


class _TorchMerge:
    """Test-only stand-in for _qsa_merge_splitk_kernel, whose `split_mask & has_values`
    (bool tensor & scalar) the Triton CPU interpreter rejects. Same math:
    out = sum_s exp2(lse_s - max) * partial_s / sum_s exp2(lse_s - max)."""

    def __getitem__(self, grid):
        def run(partial_output, partial_lse, out, stride_row, stride_head, num_rows, **kw):
            splits = kw["NUM_SPLITS"]
            po = partial_output[:splits]  # [S, rows, heads, dim]
            lse = partial_lse[:splits]  # [S, rows, heads]
            m = lse.max(dim=0).values
            w = torch.where(torch.isfinite(lse), torch.exp2(lse - m), torch.zeros_like(lse))
            den = w.sum(dim=0).clamp_min(1e-20)
            out.copy_(((w[..., None] * po).sum(dim=0) / den[..., None]).to(out.dtype))
        return run


if os.environ.get("TRITON_INTERPRET") == "1":
    qsa_ops._qsa_merge_splitk_kernel = _TorchMerge()

torch.manual_seed(0)
dt = torch.float16
num_blocks, page, kv_heads, head_dim = 8, 16, 1, 256
q_heads = 12  # 24 query heads / TP2
context = 96  # tokens 0..95 live in the cache
rows = 3  # three decode rows of one request
width = 32  # selected columns per row (+1 trailing count column)

# Random physical layout: logical page p -> physical block perm[p]
perm = torch.randperm(num_blocks)
block_table = perm[: (context + page - 1) // page].to(torch.int32).unsqueeze(0).contiguous()
token_to_req = torch.zeros(rows, dtype=torch.int32)

k_rows = (torch.randn(context, kv_heads, head_dim) * 3.0).to(dt)
v_rows = (torch.randn(context, kv_heads, head_dim) * 2.0).to(dt)
q = (torch.randn(rows, q_heads, head_dim)).to(dt)

slots = torch.tensor(
    [int(block_table[0, t // page]) * page + t % page for t in range(context)], dtype=torch.int64
)

# fp16 cache in the owner's layout: kv_cache [blocks, page, heads, 2*dim] split along the last dim
kv16 = torch.zeros(num_blocks, page, kv_heads, 2 * head_dim, dtype=dt)
kc16, vc16 = kv16.split(head_dim, dim=-1)
for t in range(context):
    b, o = divmod(int(slots[t]), page)
    kc16[b, o] = k_rows[t]
    vc16[b, o] = v_rows[t]

# Selections: each row picks `n_valid` distinct tokens; trailing column = count
logical = torch.full((rows, width + 1), -1, dtype=torch.int32)
picks = []
for r, n_valid in enumerate((32, 20, 7)):
    sel = torch.randperm(context)[:n_valid].sort().values
    picks.append(sel)
    logical[r, :n_valid] = sel.to(torch.int32)
    logical[r, width] = n_valid


def reference():
    out = torch.zeros(rows, q_heads, head_dim, dtype=torch.float32)
    for r in range(rows):
        sel = picks[r]
        k = k_rows[sel, 0].float()  # [n, d]
        v = v_rows[sel, 0].float()
        s = q[r].float() @ k.T / head_dim**0.5  # [heads, n]
        out[r] = torch.softmax(s, dim=-1) @ v
    return out


ref = reference()

out16 = qsa_sparse_paged_attention(
    q, kc16, vc16, logical, block_table, token_to_req, use_prefill_config=False
)
err16 = (out16.float() - ref).abs().max().item()

# int8: per-layer scales from max |x| / 127 (what a calibrated checkpoint carries)
k_scale = k_rows.float().abs().max().item() / 127.0
v_scale = v_rows.float().abs().max().item() / 127.0
kv8 = torch.zeros(num_blocks, page, kv_heads, 2 * head_dim, dtype=torch.int8)
kc8, vc8 = kv8.split(head_dim, dim=-1)
qsa_quantize_store_kv(k_rows, v_rows, kc8, vc8, slots, k_scale, v_scale)

# Store must equal the torch reference quantizer bit for bit
store_ok = True
for t in range(context):
    b, o = divmod(int(slots[t]), page)
    if not torch.equal(kc8[b, o], quantize_kv_int8(k_rows[t], k_scale)) or not torch.equal(
        vc8[b, o], quantize_kv_int8(v_rows[t], v_scale)
    ):
        store_ok = False
        break

out8 = qsa_sparse_paged_attention(
    q, kc8, vc8, logical, block_table, token_to_req, use_prefill_config=False,
    k_scale=k_scale, v_scale=v_scale,
)
err8 = (out8.float() - ref).abs().max().item()
ref_mag = ref.abs().max().item()
# Bound: per-element rounding error is <= scale/2 on K and V; the output is a convex
# combination of V rows, so |dV| <= v_scale/2, plus the K error's effect on the softmax.
bound8 = v_scale * 0.5 + 0.05 * ref_mag

print(f"fp16 vs ref: max abs err {err16:.4e} (ref max {ref_mag:.3f})")
print(f"int8 vs ref: max abs err {err8:.4e} (bound {bound8:.4e}; k_scale {k_scale:.4f} v_scale {v_scale:.4f})")
print(f"int8 store == quantize_kv_int8: {store_ok}")
ok = err16 < 2e-2 * max(ref_mag, 1.0) and err8 < bound8 and store_ok
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
