# SPDX-License-Identifier: Apache-2.0
"""Lane S4 (2026-10-03): software stochastic rounding fp32 -> fp16 for the GDN recurrent-state stores (sm_75 has no cvt.rs).

Why: the GDN state cache is written once per verify step per accepted position with ``b_h.to(fp16)`` = round-to-nearest-even.  K8 measured that a
fp16 RNE state DRIFTS with decode length (KL vs fp32 state 0.0005 over the first 64 decode tokens -> 0.0038 over the last 64) and EF2's earlier fp16
run stalled the same way; ``--enable-mamba-cache-stochastic-rounding`` is wired for Mamba2 only (PTX cvt.rs.f16x2.f32, sm_100+).

Method (exactly unbiased):
  * normal fp16 range (|x| >= 2^-14): add 13 uniformly random low bits to the fp32 bit pattern of |x| and clear the low 13 bits (the fp32 mantissa has 23
    bits, fp16 has 10).  The result is a fp32 value that is exactly representable in fp16 and equals the lower neighbour with probability
    1 - frac and the upper neighbour with probability frac, so E[y] = x.
  * fp16 sub-normal range (|x| < 2^-14, fixed grid 2^-24): y = floor(|x| * 2^24 + u) * 2^-24 with u uniform in [0, 1).
  * inf / nan pass through; values that would round above the fp16 maximum are clamped to 65504.
``rand`` is a uint32/int32 tensor of random bits (tl.randint); the caller owns seeding.
"""
import os

import torch

from vllm.triton_utils import tl, triton

_SR_SEEDS: dict = {}


def sr_enabled() -> bool:
    """VLLM_GDN_SR=1 turns on stochastic rounding of fp16 GDN state stores (off by default)."""
    return os.environ.get("VLLM_GDN_SR", "0") == "1"


def get_sr_seed(device) -> torch.Tensor:
    """Persistent int32[1] device buffer the SR kernels read their seed from.  Kernels only READ it (graph-safe: the captured graph holds the
    pointer, the value changes between replays); bump_sr_seed() advances it once per scheduler step, outside any graph."""
    key = str(device)
    buf = _SR_SEEDS.get(key)
    if buf is None:
        buf = torch.randint(1, 2**30, (1,), dtype=torch.int32, device=device)
        _SR_SEEDS[key] = buf
    return buf


def bump_sr_seed(device) -> None:
    get_sr_seed(device).add_(1)  # int32 wraps; pure device op, no host sync


@triton.jit
def sr_fp32_to_fp16(x, rand):
    bits = x.to(tl.int32, bitcast=True)
    mag = bits & 0x7FFFFFFF
    neg = bits < 0
    noise = rand.to(tl.int32) & 0x1FFF
    mag_n = (mag + noise) & 0x7FFFE000
    y_norm = mag_n.to(tl.float32, bitcast=True)
    ax = tl.abs(x)
    u = (rand.to(tl.int32) & 0xFFFFFF).to(tl.float32) * (1.0 / 16777216.0)
    y_sub = tl.floor(ax * 16777216.0 + u) * (1.0 / 16777216.0)
    y = tl.where(mag >= 0x38800000, y_norm, y_sub)  # 0x38800000 = 2^-14
    y = tl.minimum(y, 65504.0)
    y = tl.where(neg, -y, y)
    y = tl.where(mag >= 0x7F800000, x, y)  # inf / nan unchanged
    return y.to(tl.float16)


@triton.jit
def sr_hash(seed, offset):
    """Cheap 32-bit mixing hash (lowbias32, Chris Wellons) of (seed, element offset) -> uint32 random bits.  ~8 integer ops per element versus ~50
    for the 10-round Philox in tl.randint (which also wastes 3 of its 4 outputs); plenty for SR dithering (needs uniform low 13/24 bits)."""
    x = (offset.to(tl.uint32) + seed.to(tl.uint32) * tl.cast(2654435761, tl.uint32))
    x = x ^ (x >> 16)
    x = x * tl.cast(2146121005, tl.uint32)
    x = x ^ (x >> 15)
    x = x * tl.cast(2221713035, tl.uint32)
    x = x ^ (x >> 16)
    return x
