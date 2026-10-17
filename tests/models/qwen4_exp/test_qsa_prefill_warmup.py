"""CPU tests: the QSA indexer's prefill logits kernel compiles once per cache layout, at boot.

NM-9 W2p traced eight variants of _qsa_mqa_paged_prefill_kernel in one night, compiled mid-serving. Their TTIR differed
only in the 16-divisibility of per-batch arguments (metadata pointers at batch offsets, the page-table stride and
width, the logits row stride) and in PAGE_SIZE (392 or 400, one per layout). Those arguments are now unspecialized,
and a warm-up compiles the layout's variant with the launch's own profile.
"""
import inspect

import torch

from vllm.models.qwen4_exp.nvidia import model as _model  # noqa: F401  (import order: avoids the qsa cycle)
from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as Q

_KERNEL = Q._qsa_mqa_paged_prefill_kernel


def test_every_per_batch_argument_is_unspecialized():
    fn = getattr(_KERNEL, "fn", _KERNEL)
    params = set(inspect.signature(fn).parameters)
    assert set(Q._PREFILL_NO_SPECIALIZE) <= params
    assert {"query_start_loc_ptr", "visible_blocks_ptr", "stride_table_req", "page_table_width",
            "stride_logits_row"} <= set(Q._PREFILL_NO_SPECIALIZE)
    for kp in getattr(_KERNEL, "params", ()):  # the real JITFunction
        if kp.name in Q._PREFILL_NO_SPECIALIZE:
            assert kp.do_not_specialize, kp.name
        if kp.name in ("q_ptr", "k_cache_ptr", "page_table_ptr", "logits_ptr", "stride_q_row", "stride_q_head",
                       "stride_cache_block", "stride_cache_token"):
            assert not kp.do_not_specialize, kp.name  # constant per layout: keep the vectorization they allow


class _Capture:
    """Stands in for the JITFunction: records what the launch and the warm-up pass."""

    def __init__(self):
        self.calls = []

    def warmup(self, *args, **kwargs):
        self.calls.append(("warmup", args, kwargs))

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append(("launch", args, kwargs))
        return launch


def _constexprs(kwargs):
    return {k: v for k, v in kwargs.items() if k.isupper() or k == "num_warps"}


def test_warm_up_compiles_the_variant_the_launch_uses(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(Q, "_qsa_mqa_paged_prefill_kernel", cap)
    page_size, heads, head_dim = 392, 4, 128
    k_cache = torch.zeros(9, page_size, head_dim, dtype=torch.float16)
    page_table = torch.zeros(3, 5, dtype=torch.int32)
    Q.warmup_qsa_mqa_paged_prefill(k_cache, page_table, num_heads=heads, head_dim=head_dim,
                                   q_dtype=torch.float16)
    num_queries = 37
    q = torch.zeros(num_queries, heads, head_dim, dtype=torch.float16)
    Q._prefill_logits(q, k_cache, page_table, torch.tensor([0, 10, 20, 37], dtype=torch.int32),
                      torch.ones(num_queries, dtype=torch.int32), max_query_len=17, logits_width=128,
                      query_offset=0, num_queries=num_queries)
    (_, wargs, wkw), (_, largs, lkw) = cap.calls
    assert _constexprs(wkw) == _constexprs(lkw)  # PAGE_SIZE, heads, dim, TILE_R, BLOCK_N, K_TILES, STAGES, warps
    assert wkw["PAGE_SIZE"] == page_size
    # the specialized integer arguments (q strides, cache strides) carry the same values
    names = list(inspect.signature(getattr(Q.__dict__.get("_qsa_mqa_paged_prefill_kernel"), "fn", None)
                                   or _KERNEL.fn).parameters)
    for i, name in enumerate(names[:len(largs)]):
        if name.startswith("stride_") and name not in Q._PREFILL_NO_SPECIALIZE:
            assert wargs[i] == largs[i], name


def test_warm_up_returns_the_shared_profile(monkeypatch):
    monkeypatch.setattr(Q, "_qsa_mqa_paged_prefill_kernel", _Capture())
    k_cache = torch.zeros(4, 400, 128, dtype=torch.float16)  # int8-KV layout: 1600-token blocks / 4
    got = Q.warmup_qsa_mqa_paged_prefill(k_cache, torch.zeros(2, 3, dtype=torch.int32), num_heads=4, head_dim=128)
    assert got == Q._prefill_profile(torch.float16)
