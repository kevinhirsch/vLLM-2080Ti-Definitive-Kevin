# SPDX-License-Identifier: Apache-2.0
"""[FORK][LANE EF2] FlashInfer sampling-failure sentinel guard (L135).

FlashInfer returns the vocab size for a row it cannot sample (NaN / fully masked probs).
The guard must replace exactly those ids with a valid NaN-safe argmax, leave valid ids
untouched, never sync per call, and report the count. CPU-only.
"""

import logging

import torch

from vllm.v1.sample.ops import topk_topp_sampler as ts


def _reset():
    ts._sentinel_state.update(calls=0, count=None, logged=0)


def test_valid_ids_untouched():
    _reset()
    logits = torch.randn(3, 11)
    ids = torch.tensor([0, 5, 10])
    assert torch.equal(ts.guard_sampling_sentinel(ids, logits), ids)


def test_sentinel_and_negative_replaced_by_nan_safe_argmax():
    _reset()
    V = 8
    logits = torch.full((3, V), float("-inf"))
    logits[0, 3] = 2.0
    logits[1] = float("nan")
    logits[1, 6] = 1.0           # NaN row with one finite logit
    logits[2, 1] = 0.5
    ids = torch.tensor([V, V, -1])  # FlashInfer sentinel x2, garbage x1
    out = ts.guard_sampling_sentinel(ids, logits)
    assert out.tolist() == [3, 6, 1]
    assert int(ts._sentinel_state["count"]) == 3


def test_all_masked_row_still_valid_id():
    _reset()
    V = 5
    logits = torch.full((1, V), float("nan"))
    out = ts.guard_sampling_sentinel(torch.tensor([V]), logits)
    assert 0 <= int(out[0]) < V


def test_rate_limited_log(caplog, monkeypatch):
    _reset()
    monkeypatch.setattr(ts, "_SENTINEL_LOG_EVERY", 2)
    logits = torch.zeros(1, 4)
    with caplog.at_level(logging.ERROR):
        ts.guard_sampling_sentinel(torch.tensor([4]), logits)
        ts.guard_sampling_sentinel(torch.tensor([1]), logits)
    assert ts._sentinel_state["logged"] == 1


def test_flashinfer_sample_applies_guard_when_enabled(monkeypatch):
    _reset()
    import sys
    import types

    V = 6
    fake = types.SimpleNamespace(sampling=types.SimpleNamespace(
        top_k_top_p_sampling_from_logits=lambda logits, k, p, deterministic: torch.tensor([V, 2])))
    monkeypatch.setitem(sys.modules, "flashinfer", fake)
    logits = torch.zeros(2, V)
    logits[0, 4] = 1.0
    monkeypatch.setattr(ts, "_SENTINEL_GUARD", True)
    out = ts.flashinfer_sample(logits, torch.tensor([3, 3]), torch.tensor([0.9, 0.9]))
    assert out.tolist() == [4, 2]
    monkeypatch.setattr(ts, "_SENTINEL_GUARD", False)
    out = ts.flashinfer_sample(logits, torch.tensor([3, 3]), torch.tensor([0.9, 0.9]))
    assert out.tolist() == [V, 2]  # default (off) keeps upstream behaviour
