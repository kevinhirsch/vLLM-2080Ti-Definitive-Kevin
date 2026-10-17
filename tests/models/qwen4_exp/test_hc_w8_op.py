"""CPU tests (TRITON_INTERPRET): the HC W8A16 GEMM is a registered custom op, compiles once, and never JIT-compiles
mid-serving.

NM-9 W2j measured HC W8 at C1 23.4 tok/s against 44.5 for the fp16 projections, with 4 runtime compiles of
_w8a16_gemm_kernel. Every other HC op here is a registered custom op (ops/hc.py, low_latency_gemm.py); the W8 path
launched the Triton kernel from Python with a split-K choice computed from the token count, which torch.compile
cannot keep dynamic. Now: one opaque op (fake impl for tracing), no specialization on runtime ints, and every
split-K variant compiled at the first eager call.
"""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm.models.qwen4_exp.nvidia import model as _model  # noqa: E402,F401  (import order: avoids the qsa cycle)
from vllm.models.qwen4_exp.nvidia.ops import hc_w8 as W  # noqa: E402
from vllm.utils.torch_utils import vllm_lib  # noqa: E402

# The engine registers the op for the platform's key (CUDA) plus Meta; give it the same implementation on CPU so the
# op dispatch itself is what these tests exercise.
try:
    vllm_lib.impl("qwen4_exp_hc_w8a16_linear", W._w8a16_linear_impl, dispatch_key="CPU")
except RuntimeError:  # already registered in this process
    pass

# Kernel launches need the interpreter enabled before Triton is first imported: tl.zeros and the other stdlib
# helpers are @jit functions decorated at that import, and the root conftest imports vLLM (and Triton) before this
# module runs. Run with TRITON_INTERPRET=1 in the environment; the other tests run either way.
import triton.language as _tl  # noqa: E402

_KERNEL = W._w8a16_gemm_kernel
requires_interpreter = pytest.mark.skipif(
    type(_tl.zeros).__name__ != "InterpretedFunction",
    reason="Triton was imported before TRITON_INTERPRET=1: set it in the environment to run the kernel on CPU",
)


def _weights(n=48, k=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k, generator=g) * 0.05
    return W.quantize_w8_groupwise(w, W.HC_W8_GROUP)


def _reference(x, q, s):
    deq = q.float() * s.float().repeat_interleave(W.HC_W8_GROUP, dim=1)
    return (x.float() @ deq.T).to(x.dtype)


@requires_interpreter
def test_op_matches_dequantized_reference():
    q, s = _weights()
    for m in (1, 3, 17):
        x = torch.randn(m, q.shape[1], generator=torch.Generator().manual_seed(m)).half()
        y = W.w8a16_linear(x, q, s)
        assert y.shape == (m, q.shape[0]) and y.dtype == x.dtype
        assert torch.allclose(y.float(), _reference(x, q, s).float(), rtol=2e-2, atol=2e-3), m
        assert torch.equal(y, W._w8a16_linear_impl(x, q, s, W.HC_W8_GROUP))  # the op is the implementation


def test_fake_impl_shape():
    q, _ = _weights()
    x = torch.empty(5, 2, q.shape[1], dtype=torch.float16)
    assert W._w8a16_linear_fake(x, q, None, W.HC_W8_GROUP).shape == (5, 2, q.shape[0])


@requires_interpreter
def test_torch_compile_fullgraph_dynamic_compiles_once():
    q, s = _weights()
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    def f(x):
        return W.w8a16_linear(x, q, s) * 2

    torch._dynamo.reset()
    cf = torch.compile(f, backend=backend, fullgraph=True, dynamic=True)
    # M >= 2: Dynamo specializes size-1 dims on its own (M = 1 gets its own graph whatever the op does)
    for m in (2, 3, 5, 17, 33, 64):  # the decode and prefill batch sizes the engine sees
        x = torch.randn(m, q.shape[1], generator=torch.Generator().manual_seed(m)).half()
        assert torch.equal(cf(x), f(x)), m
    assert len(graphs) == 1  # one graph, M symbolic: no graph break, no recompile per batch size
    assert any("qwen4_exp_hc_w8a16_linear" in str(n.target) for n in graphs[0].graph.nodes)


def test_kernel_unspecializes_only_what_varies_at_runtime():
    """M and x's alignment vary per call; N, K and the strides are per-projection constants whose divisibility
    Triton needs to vectorize the int8 weight loads (W2l: unspecializing them cut C1 23.4 -> 15.2 tok/s)."""
    import inspect

    fn = getattr(_KERNEL, "fn", _KERNEL)
    assert set(W._W8_NO_SPECIALIZE) <= set(inspect.signature(fn).parameters)
    assert set(W._W8_NO_SPECIALIZE) == {"x_ptr", "M"}
    for kp in getattr(_KERNEL, "params", ()):  # the real JITFunction when the root conftest imported it first
        if kp.name in ("N", "K", "stride_qn", "q_ptr"):
            assert not kp.do_not_specialize, kp.name
        if kp.name in W._W8_NO_SPECIALIZE:
            assert kp.do_not_specialize, kp.name


def test_warm_up_covers_every_split_the_dispatch_can_pick():
    launched, gemv = [], []
    real, real_gemv = W._launch, W._launch_gemv
    W._launch = lambda x2, q, s, group, split: launched.append(triton_cdiv(q.shape[1] // group, split) * group)
    W._launch_gemv = lambda x2, q, s, group, split, rows: gemv.append((split, rows))
    try:
        for n, k in ((336, 10240), (10240, 320)):  # Flash-Next's HC down (block-inject merged) and up projections
            q = torch.zeros(n, k, dtype=torch.int8)
            s = torch.zeros(n, k // W.HC_W8_GROUP, dtype=torch.float16)
            x = torch.zeros(1, k, dtype=torch.float16)
            W._W8_WARMED.clear()
            launched.clear()
            gemv.clear()
            assert W._warm_up(x, q, s, W.HC_W8_GROUP) == len(launched) + len(gemv) > 0
            if W._GEMV:  # the one-token path: its single split for this shape, every row bucket
                assert sorted(r for _, r in gemv) == [1, 2, 4]
                assert {sp for sp, _ in gemv} == {W._gemv_split(n, k, W.HC_W8_GROUP)}
            assert len(set(launched)) == len(launched)  # each variant once
            assert W._warm_up(x, q, s, W.HC_W8_GROUP) == 0  # memoized per shape
            warmed = set(launched)
            for m in range(1, 4097):  # every batch size: its K_PER_SPLIT was compiled at load
                split = W._split_k(m, n, k, W._BLOCK_M, W._BLOCK_N, W.HC_W8_GROUP)
                assert triton_cdiv(k // W.HC_W8_GROUP, split) * W.HC_W8_GROUP in warmed, (n, k, m, split)
    finally:
        W._launch, W._launch_gemv = real, real_gemv
        W._W8_WARMED.clear()


def triton_cdiv(a, b):
    return -(-a // b)


@requires_interpreter
def test_gemv_path_matches_reference_for_every_row_bucket():
    """M <= 4 takes the one-token GEMV (K-contiguous tiles, tl.sum, split-K by group divisors)."""
    for n, k in ((48, 256), (70, 640)):  # the second needs split-K and has a partial last N block
        q, s = _weights(n, k, seed=n)
        split = W._gemv_split(n, k, W.HC_W8_GROUP)
        assert (k // W.HC_W8_GROUP) % split == 0
        for m in (1, 2, 3, 4):
            x = torch.randn(m, k, generator=torch.Generator().manual_seed(10 * n + m)).half()
            partial = W._launch_gemv(x, q, s, W.HC_W8_GROUP, split, W._gemv_rows(m))
            y = partial.sum(dim=0).to(x.dtype)
            assert torch.allclose(y.float(), _reference(x, q, s).float(), rtol=2e-2, atol=2e-3), (n, k, m)


def test_gemv_split_divides_the_groups_and_fills_the_sms():
    for n, k in ((336, 10240), (10240, 320), (48, 256)):
        groups = k // W.HC_W8_GROUP
        split = W._gemv_split(n, k, W.HC_W8_GROUP)
        assert groups % split == 0
        programs = triton_cdiv(n, W._GEMV_BLOCK_N) * split
        assert programs >= 136 or split == groups, (n, k, split)
    assert W._gemv_split(336, 10240, W.HC_W8_GROUP) == 16  # the HC down projection: 11 x 16 = 176 programs
    assert W._gemv_split(10240, 320, W.HC_W8_GROUP) == 1  # the up projection: 320 programs already


def test_small_m_dispatches_to_gemv_and_large_m_to_the_gemm():
    calls = []
    real, real_gemv, real_warm = W._launch, W._launch_gemv, W._warm_up
    W._warm_up = lambda *a: 0

    def gemm(x2, q, s, group, split):
        calls.append(("gemm", x2.shape[0]))
        return torch.zeros(split, x2.shape[0], q.shape[0])

    def gemv(x2, q, s, group, split, rows):
        calls.append(("gemv", x2.shape[0], rows))
        return torch.zeros(split, x2.shape[0], q.shape[0])

    W._launch, W._launch_gemv = gemm, gemv
    try:
        q, s = _weights()
        for m in (1, 2, 3, 4, 5, 16):
            W._w8a16_linear_impl(torch.zeros(m, q.shape[1], dtype=torch.float16), q, s, W.HC_W8_GROUP)
    finally:
        W._launch, W._launch_gemv, W._warm_up = real, real_gemv, real_warm
    if W._GEMV:
        assert calls == [("gemv", 1, 1), ("gemv", 2, 2), ("gemv", 3, 4), ("gemv", 4, 4), ("gemm", 5), ("gemm", 16)]
    else:
        assert all(c[0] == "gemm" for c in calls)
