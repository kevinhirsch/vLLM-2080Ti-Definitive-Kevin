"""CPU selfcheck (TRITON_INTERPRET=1): the Turing decode profile of the QSA indexer
scoring kernel (thecyberdevl 0029: 32 columns, 4 tiles per program, 1 warp) writes
the same logits as the GB300 default (64 columns, 1 tile, 2 warps)."""
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import _qsa_mqa_paged_uniform_kernel
from vllm.triton_utils import triton

torch.manual_seed(0)
reqs, dql, heads, dim, page, width = 2, 3, 4, 128, 16, 8
columns = width * page
q = torch.randn(reqs * dql, heads, dim).half()
k_cache = torch.randn(16, page, dim).half()
page_table = torch.stack([torch.randperm(16)[:width] for _ in range(reqs)]).to(torch.int32)
visible = torch.tensor([columns, columns - 5, columns - 9, 70, 71, 72], dtype=torch.int32)


def run(block_n, tiles, warps):
    logits = torch.full((reqs * dql, columns), float("nan"), dtype=torch.float32)
    grid = (reqs, triton.cdiv(columns, block_n * tiles))
    _qsa_mqa_paged_uniform_kernel[grid](
        q, k_cache, page_table, visible, logits,
        *q.stride()[:-1], *k_cache.stride()[:2], *page_table.stride()[:-1], *logits.stride()[:-1],
        PAGE_SIZE=page, PAGE_TABLE_WIDTH=width, NUM_HEADS=heads, HEAD_DIM=dim,
        DECODE_QUERY_LEN=dql, BLOCK_N=block_n, TILES_PER_PROG=tiles, STAGES=2, num_warps=warps,
    )
    return logits


a = run(64, 1, 2)
b = run(32, 4, 1)
mask = torch.arange(columns)[None, :] < visible[:, None]
nan_a, nan_b = int(torch.isnan(a[mask]).sum()), int(torch.isnan(b[mask]).sum())
diff = (a[mask] - b[mask]).abs()
scale = a[mask].abs().max().item()
print(f"nan a={nan_a} b={nan_b}  max abs diff {diff.max().item():.3e}  max logit {scale:.2f}")
# Both profiles compute each column's full dot; only the MMA tile shape differs, so
# the results may differ by fp32 summation order, never by more.
same = nan_a == 0 and nan_b == 0 and diff.max().item() <= 1e-5 * max(scale, 1.0)
print(f"visible-column logits equal within fp32 rounding: {same} ({int(mask.sum())} values)")
print("PASS" if same else "FAIL")
sys.exit(0 if same else 1)
