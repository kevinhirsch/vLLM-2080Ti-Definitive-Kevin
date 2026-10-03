# SPDX-License-Identifier: Apache-2.0
"""[FORK][LANE EF2] plan() indptr sources are private copies (L134 pinned-buffer race).

Legacy-tree regression: the per-impl pinned scratch indptr buffers are rewritten by the
next request before FlashInfer's non_blocking H2D copy runs (2026-10-01 14:40 Xid 31).
CPU-only; the GPU proof is tools/ef2/l134_probe.py --part race on ef2/fault-roots.
"""

import torch

from vllm.v1.attention.backends import turboquant_attn as tq


def test_private_copies_are_isolated_from_caller_mutation():
    qo = torch.tensor([0, 843], dtype=torch.int32)
    kv = torch.tensor([0, 40091], dtype=torch.int32)
    out = tq._tq_private_plan_indptrs({"qo_indptr": qo, "kv_indptr": kv, "causal": True})
    qo[1] = 1087
    kv[1] = 1087
    assert out["qo_indptr"].tolist() == [0, 843]
    assert out["kv_indptr"].tolist() == [0, 40091]
    assert out["causal"] is True


def test_shared_identity_preserved():
    qsl = torch.tensor([0, 9], dtype=torch.int32)
    out = tq._tq_private_plan_indptrs({"qo_indptr": qsl, "kv_indptr": qsl})
    assert out["qo_indptr"] is out["kv_indptr"] and out["qo_indptr"] is not qsl


def test_legacy_plan_cache_miss_uses_private_copy(monkeypatch):
    seen = {}

    class FakeWrapper:
        def __init__(self, *a, **k):
            pass

        def plan(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(tq, "BatchPrefillWithRaggedKVCacheWrapper", FakeWrapper)
    monkeypatch.setattr(tq, "_get_shared_flashinfer_prefill_workspace",
                        lambda d, b: torch.empty(1, dtype=torch.uint8))
    monkeypatch.setattr(tq, "_TQ_FI_PREFILL_CUDAGRAPH_SAFE", False)
    monkeypatch.setattr(tq, "_DEFAULT_TQ_FI_PLAN_CACHE", True)
    tq._TQ_FI_PREFILL_WRAPPERS.clear()
    impl = tq.TurboQuantAttentionImpl.__new__(tq.TurboQuantAttentionImpl)
    impl._use_flashinfer_prefill = True
    impl._fi_prefill_backend = "fa2"
    scratch_qo = torch.tensor([0, 843], dtype=torch.int32)
    scratch_kv = torch.tensor([0, 40091], dtype=torch.int32)
    impl._get_or_plan_flashinfer_prefill_wrapper(
        torch.device("cpu"), ("continuation", 843, 40091),
        {"qo_indptr": scratch_qo, "kv_indptr": scratch_kv},
    )
    scratch_qo[1] = 1087  # next request in the same step
    scratch_kv[1] = 1087
    assert seen["qo_indptr"].tolist() == [0, 843]
    assert seen["kv_indptr"].tolist() == [0, 40091]
    tq._TQ_FI_PREFILL_WRAPPERS.clear()
