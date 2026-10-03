# SPDX-License-Identifier: Apache-2.0
"""Lane S4: software stochastic rounding fp32->fp16 (GDN state stores).  Runs on CPU with TRITON_INTERPRET=1 (set below) or on a CUDA device."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")
import numpy as np
import torch

from vllm.third_party.flash_linear_attention.ops.sr_convert import sr_fp32_to_fp16, sr_hash
from vllm.triton_utils import tl, triton


@triton.jit
def _k(x_ptr, out_ptr, seed, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    r = sr_hash(seed, offs)
    tl.store(out_ptr + offs, sr_fp32_to_fp16(x, r), mask=m)


def sr(x: torch.Tensor, seed: int) -> torch.Tensor:
    out = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    n = x.numel()
    _k[(triton.cdiv(n, 256),)](x, out, seed, n, BLOCK=256)
    return out


def test_representable_values_are_exact():
    x = torch.tensor([0.0, 1.0, -1.0, 0.5, 2.0 ** -14, -(2.0 ** -24), 3.0 * 2.0 ** -24, 65504.0, -65504.0, 0.099975586], dtype=torch.float32)
    x = x.half().float()  # exactly representable in fp16
    for seed in range(5):
        assert torch.equal(sr(x, seed).float(), x)


def test_only_neighbours_and_unbiased():
    g = torch.Generator().manual_seed(0)
    mags = torch.tensor([3.1e-3, 0.0713, 0.9, 1.3, 17.77, 123.456, 4096.7])
    x = (mags * torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])).float()
    reps = 4000
    xs = x.repeat(reps)
    y = sr(xs, 1234).float().view(reps, -1)
    h = x.half().float()
    ulp = torch.tensor([torch.finfo(torch.float16).eps * 2.0 ** int(np.floor(np.log2(abs(v)))) for v in x.tolist()])
    assert (y - x).abs().max() <= ulp.max() * 1.0001
    # every output is one of the two fp16 neighbours of x
    for j in range(x.numel()):
        vals = torch.unique(y[:, j])
        assert len(vals) <= 2 and all(abs(float(v) - float(x[j])) < float(ulp[j]) * 1.0001 for v in vals)
    # unbiased: mean error within 5 sigma (sigma of one draw <= ulp/2)
    err = (y.mean(0) - x).abs()
    assert bool((err < 5 * ulp / 2 / np.sqrt(reps)).all()), (err, ulp)
    # plain RNE is NOT unbiased for these points (sanity that the test can tell the difference)
    assert (h - x).abs().max() > 0


def test_subnormal_grid_and_sign():
    x = torch.tensor([1.234e-6, -3.3e-7, 5.0e-5, -5.0e-5], dtype=torch.float32)
    reps = 8000
    y = sr(x.repeat(reps), 7).float().view(reps, -1)
    step = 2.0 ** -24
    assert bool(((y / step).round() * step - y).abs().max() < 1e-12)  # outputs live on the 2^-24 grid
    assert bool(((y.mean(0) - x).abs() < 5 * step / 2 / np.sqrt(reps)).all())
    assert bool((torch.sign(y) * torch.sign(x) >= 0).all())


def test_inf_nan_and_clamp():
    x = torch.tensor([float("inf"), float("-inf"), float("nan"), 65519.0, -70000.0], dtype=torch.float32)
    y = sr(x, 3).float()
    assert y[0] == float("inf") and y[1] == float("-inf") and torch.isnan(y[2])
    # |x| above the fp16 maximum is clamped to +-65504 (never wraps to a small value or inf)
    assert float(y[3]) == 65504.0 and float(y[4]) == -65504.0
