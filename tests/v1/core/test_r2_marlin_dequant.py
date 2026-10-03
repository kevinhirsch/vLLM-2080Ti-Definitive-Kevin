# SPDX-License-Identifier: Apache-2.0
"""Lane R2 / L55: Marlin-layout -> dense fp16 dequant (reference + Triton under the CPU interpreter)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools", "r2"))
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (  # noqa: E402
    awq_marlin_quantize,
    marlin_quantize,
)
from vllm.scalar_type import scalar_types  # noqa: E402

import marlin_dequant as md  # noqa: E402


@pytest.mark.parametrize("k,n,gs", [(256, 128, 128), (512, 256, 128), (256, 192, 64)])
def test_asymmetric_zp_inverse_matches_quantizer_reference(k, n, gs):
    torch.manual_seed(0)
    w = torch.randn(k, n, dtype=torch.float16)
    w_ref, q, s, zp = awq_marlin_quantize(w, scalar_types.uint4, gs)
    out = md.marlin_dequant_torch(q, s, zp, k, n, gs)
    assert torch.allclose(out.float(), w_ref.float(), atol=2e-3, rtol=0)


@pytest.mark.parametrize("k,n,gs", [(256, 128, 128), (512, 256, 128)])
def test_symmetric_inverse_matches_quantizer_reference(k, n, gs):
    torch.manual_seed(1)
    w = torch.randn(k, n, dtype=torch.float16)
    w_ref, q, s = marlin_quantize(w, scalar_types.uint4b8, gs)
    out = md.marlin_dequant_torch(q, s, None, k, n, gs)
    assert torch.allclose(out.float(), w_ref.float(), atol=2e-3, rtol=0)


_CHILD = r"""
import sys, torch
sys.path.insert(0, sys.argv[1])
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import awq_marlin_quantize
from vllm.scalar_type import scalar_types
import marlin_dequant as md
torch.manual_seed(2)
for k, n, gs in ((256, 128, 128), (256, 192, 64)):
    w = torch.randn(k, n, dtype=torch.float16)
    w_ref, q, s, zp = awq_marlin_quantize(w, scalar_types.uint4, gs)
    ref = md.marlin_dequant_torch(q, s, zp, k, n, gs)
    got = md.marlin_dequant_triton(q, s, zp, k, n, gs)
    assert torch.equal(got, ref), (k, n, gs)
    assert torch.equal(md.marlin_dequant_triton(q, s, zp, k, n, gs, out_nk=True), ref.T), (k, n, gs)
    assert torch.allclose(got.float(), w_ref.float(), atol=2e-3, rtol=0)
print("TRITON_OK")
"""


@pytest.mark.skipif(not md.HAVE_TRITON, reason="triton missing")
def test_triton_kernel_equals_reference_in_interpreter():
    """TRITON_INTERPRET must be set before triton is imported, hence a child process."""
    import subprocess

    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    tools = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools", "r2")
    r = subprocess.run([sys.executable, "-c", _CHILD, tools], env=env, capture_output=True, text=True, timeout=600)
    assert "TRITON_OK" in r.stdout, r.stdout[-500:] + r.stderr[-1500:]
