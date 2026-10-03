# SPDX-License-Identifier: Apache-2.0
"""[FORK][LANE EF2] Bounds + value check of the S3 continuation dequant layout (L134).

The TurboQuant continuation prefill (``TurboQuantAttentionImpl._continuation_prefill``,
lane S3 d797625753) dequantizes the cached prefix straight into ``(rows, Hk, D)``
workspace buffers through a permuted ``(1, Hk, rows, D)`` view, with
``rows = max(ceil(cached/bs), ceil(seq/bs)) * bs``. The legacy 10-01/10-02 Xid 31 faults
sat in this prefill portion, so this test pins the index math of that kernel launch:

* the block table ends on the LAST block of an exactly sized pool (a pool-end overrun
  would read the canary region placed right after the pool);
* the output buffers are views into larger buffers whose tail is a canary (an overrun
  write past ``rows`` changes the canary);
* every dequantized value equals a pure-torch reference built from the same slot bytes
  (reads come from exactly the slots the block table names);
* rows ``[cached, alloc_len)`` of the last cached page are written but are inside
  ``rows`` and are overwritten by the raw current chunk afterwards.

CPU (no GPU):  TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= pytest <this file>
GPU:           pytest <this file>   (not next to the live engine)
"""

import math
import os

import pytest
import torch

from vllm.v1.attention.ops.triton_turboquant_decode import _tq_full_dequant_kv

DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
if DEVICE == "cuda" and not torch.cuda.is_available():
    pytest.skip("needs CUDA or TRITON_INTERPRET=1", allow_module_level=True)

D, HK = 256, 2
MSE_BITS, VQB = 3, 4  # turboquant_k3v4_nc
MSE_BYTES = math.ceil(D * MSE_BITS / 8)
KPS = MSE_BYTES + 2
VAL_DATA = math.ceil(D * VQB / 8)
SLOT = KPS + VAL_DATA + 4
CANARY = 7



def _make_pool(num_blocks: int, bs: int, g: torch.Generator):
    total = num_blocks * bs * HK * SLOT
    raw = torch.full((total + 4096,), CANARY, dtype=torch.uint8)
    pool = raw[:total].view(num_blocks, bs, HK, SLOT)
    pool.copy_(torch.randint(0, 256, pool.shape, generator=g, dtype=torch.uint8))
    norms = torch.rand(num_blocks, bs, HK, generator=g) + 0.5
    scales = torch.rand(num_blocks, bs, HK, generator=g) * 0.1 + 0.01
    zeros = torch.rand(num_blocks, bs, HK, generator=g) - 0.5
    def as_bytes(x):
        return x.half().unsqueeze(-1).view(torch.uint8)  # (..., 2)

    pool[..., MSE_BYTES:MSE_BYTES + 2] = as_bytes(norms)
    pool[..., KPS + VAL_DATA:KPS + VAL_DATA + 2] = as_bytes(scales)
    pool[..., KPS + VAL_DATA + 2:KPS + VAL_DATA + 4] = as_bytes(zeros)
    return raw, pool


def _reference(pool, bt, cached, bs, cent):
    k = torch.empty(cached, HK, D)
    v = torch.empty(cached, HK, D)
    d = torch.arange(D)
    bit = d * MSE_BITS
    byte, shift = bit // 8, bit % 8
    for pos in range(cached):
        blk = int(bt[0, pos // bs])
        for h in range(HK):
            s = pool[blk, pos % bs, h].long()
            raw16 = s[byte] | (s[byte + 1] << 8)
            idx = (raw16 >> shift) & ((1 << MSE_BITS) - 1)
            c = cent[idx]
            c = c / torch.sqrt((c * c).sum() + 1e-16)
            norm = pool[blk, pos % bs, h, MSE_BYTES:MSE_BYTES + 2].view(torch.float16).float()
            k[pos, h] = norm * c
            vb = s[KPS + d // 2]
            vi = ((vb >> ((d % 2) * 4)) & 0xF).float()
            sc = pool[blk, pos % bs, h, KPS + VAL_DATA:KPS + VAL_DATA + 2].view(torch.float16).float()
            zr = pool[blk, pos % bs, h, KPS + VAL_DATA + 2:KPS + VAL_DATA + 4].view(torch.float16).float()
            v[pos, h] = vi * sc + zr
    return k, v


@pytest.mark.parametrize(
    "bs,cached,q_len",
    [
        (16, 37, 5),    # partial last cached page, small chunk
        (16, 48, 16),   # page-aligned prefix, chunk = one block (align-mode shape)
        (16, 48, 3),    # rows from ceil(seq/bs) > alloc_len
        (8, 61, 40),    # chunk spans several pages
    ],
)
def test_s3_continuation_dequant_in_bounds(bs, cached, q_len):
    import triton

    g = torch.Generator().manual_seed(bs * 1000 + cached)
    pages = math.ceil(cached / bs)
    num_blocks = pages + 3
    raw, pool = _make_pool(num_blocks, bs, g)
    # Block table: ends on the LAST block of the pool, extra width holds garbage ids
    # (the kernel must never read past `pages` columns).
    bt = torch.full((1, pages + 5), 10**6, dtype=torch.int32)
    bt[0, :pages] = torch.arange(num_blocks - pages, num_blocks, dtype=torch.int32).flip(0)
    cent = torch.linspace(-1.0, 1.0, 1 << MSE_BITS)

    seq_len = cached + q_len
    alloc_len = pages * bs
    rows = max(alloc_len, math.ceil(seq_len / bs) * bs)
    kbig = torch.full((rows + 64, HK, D), float(CANARY), dtype=torch.float16)
    vbig = torch.full((rows + 64, HK, D), float(CANARY), dtype=torch.float16)
    k_rows, v_rows = kbig[:rows], vbig[:rows]
    kc = k_rows.permute(1, 0, 2).unsqueeze(0)
    vc = v_rows.permute(1, 0, 2).unsqueeze(0)

    dev = DEVICE
    pool_d, bt_d, cent_d = pool.to(dev), bt.to(dev), cent.to(dev)
    kbig_d, vbig_d = kbig.to(dev), vbig.to(dev)
    kc_d = kbig_d[:rows].permute(1, 0, 2).unsqueeze(0)
    vc_d = vbig_d[:rows].permute(1, 0, 2).unsqueeze(0)
    _tq_full_dequant_kv[(alloc_len, HK)](
        pool_d, bt_d, cent_d, kc_d, vc_d,
        kc_d.stride(0), kc_d.stride(1), kc_d.stride(2),
        vc_d.stride(0), vc_d.stride(1), vc_d.stride(2),
        pool_d.stride(0), pool_d.stride(1), pool_d.stride(2), bt_d.stride(0),
        HEAD_DIM=D, BLOCK_SIZE=bs, NUM_KV_HEADS=HK, MSE_BYTES=MSE_BYTES, KPS=KPS,
        VQB=VQB, VAL_DATA_BYTES=VAL_DATA, MSE_BITS=MSE_BITS, KEY_FP8=0,
        BLOCK_D=triton.next_power_of_2(D), NORM_CORRECTION=1, FP8_FORMAT=0, num_warps=4,
    )
    kbig_h, vbig_h = kbig_d.cpu(), vbig_d.cpu()
    # 1. no write past `rows`
    assert torch.all(kbig_h[rows:] == CANARY) and torch.all(vbig_h[rows:] == CANARY)
    # 2. exactly [0, alloc_len) written (padding rows of the last cached page included)
    assert not torch.any(kbig_h[:alloc_len] == CANARY)
    if rows > alloc_len:
        assert torch.all(kbig_h[alloc_len:rows] == CANARY)
    # 3. values match the reference over the cached prefix
    k_ref, v_ref = _reference(pool, bt, cached, bs, cent)
    torch.testing.assert_close(kbig_h[:cached].float(), k_ref, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(vbig_h[:cached].float(), v_ref, atol=2e-3, rtol=2e-3)
    # 4. pool canary untouched (kernel never writes the cache)
    assert torch.all(raw[pool.numel():] == CANARY)
