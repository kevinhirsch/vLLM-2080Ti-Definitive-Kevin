# SPDX-License-Identifier: Apache-2.0
"""[FORK][LANE EF2] plan() indptr sources are private copies (L134 pinned-buffer race).

FlashInfer plan() copies qo/kv indptr host->device with non_blocking=True. If the
caller's buffer is rewritten before the queued DMA executes, the cached wrapper keeps
another request's lengths on the device (legacy 2026-10-01 14:40 Xid 31). The TQ plan
helper must therefore hand plan() tensors that no caller can mutate afterwards.
CPU-only test; the GPU proof is tools/ef2/l134_probe.py --part race.
"""

import torch

from vllm.v1.attention.backends import turboquant_attn as tq


def test_private_copies_are_isolated_from_caller_mutation():
    qo = torch.tensor([0, 843], dtype=torch.int32)
    kv = torch.tensor([0, 40091], dtype=torch.int32)
    kwargs = {"qo_indptr": qo, "kv_indptr": kv, "num_qo_heads": 12}
    out = tq._tq_private_plan_indptrs(kwargs)
    assert out["qo_indptr"] is not qo and out["kv_indptr"] is not kv
    # the next request rewrites the caller's scratch buffer in place
    qo[1] = 1087
    kv[1] = 1087
    assert out["qo_indptr"].tolist() == [0, 843]
    assert out["kv_indptr"].tolist() == [0, 40091]
    assert out["num_qo_heads"] == 12
    assert kwargs["qo_indptr"] is qo  # caller's dict untouched


def test_shared_qo_kv_identity_preserved():
    qsl = torch.tensor([0, 7, 19], dtype=torch.int32)
    out = tq._tq_private_plan_indptrs({"qo_indptr": qsl, "kv_indptr": qsl})
    assert out["qo_indptr"] is out["kv_indptr"]
    assert out["qo_indptr"] is not qsl
    qsl[2] = 99
    assert out["kv_indptr"].tolist() == [0, 7, 19]


def test_pinned_flag_preserved_when_available():
    if not torch.cuda.is_available():
        return
    qo = torch.tensor([0, 5], dtype=torch.int32, pin_memory=True)
    out = tq._tq_private_plan_indptrs({"qo_indptr": qo, "kv_indptr": qo.clone()})
    assert out["qo_indptr"].is_pinned()


def test_plan_helper_routes_through_private_copy(monkeypatch):
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
    tq._TQ_FI_PREFILL_WRAPPERS.clear()
    qo = torch.tensor([0, 843], dtype=torch.int32)
    kv = torch.tensor([0, 40091], dtype=torch.int32)
    tq._get_or_plan_tq_flashinfer_prefill_wrapper(
        torch.device("cpu"), ("ef2-test", 843, 40091), {"qo_indptr": qo, "kv_indptr": kv}
    )
    assert seen["qo_indptr"] is not qo and seen["kv_indptr"] is not kv
    qo[1] = 1
    assert seen["qo_indptr"].tolist() == [0, 843]
    tq._TQ_FI_PREFILL_WRAPPERS.clear()
