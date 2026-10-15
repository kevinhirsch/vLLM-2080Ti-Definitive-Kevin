# SPDX-License-Identifier: Apache-2.0
"""FR1: CPU tests for the reduced-vocab MTP draft head mapping logic."""
import json

import pytest
import torch

from vllm.model_executor.layers.quantization import u2_headquant as u2
from vllm.v1.spec_decode import draft_vocab as dv


def test_normalize_sorts_dedupes_pads(tmp_path):
    ids = dv.normalize_ids([5, 3, 3, 9, 1000], vocab_size=4096)
    assert ids.numel() % dv.PAD_MULTIPLE == 0
    assert ids.is_contiguous() and ids.dtype == torch.int64
    assert torch.equal(ids, torch.sort(ids).values)
    assert len(set(ids.tolist())) == ids.numel()
    assert {3, 5, 9, 1000} <= set(ids.tolist())
    # filler = smallest unused ids
    assert set(ids.tolist()) - {3, 5, 9, 1000} == set(
        [i for i in range(4096) if i not in (3, 5, 9, 1000)][: ids.numel() - 4]
    )


def test_normalize_rejects_out_of_range():
    with pytest.raises(ValueError):
        dv.normalize_ids([1, 5000], vocab_size=4096)
    with pytest.raises(ValueError):
        dv.normalize_ids([], vocab_size=4096)


def test_load_json_and_text(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(list(range(0, 600, 2))))
    t = tmp_path / "b.txt"
    t.write_text("\n".join(str(i) for i in range(0, 600, 2)))
    a = dv.load_draft_vocab_ids(str(p), 1024)
    b = dv.load_draft_vocab_ids(str(t), 1024)
    assert torch.equal(a, b)


def test_map_subset_argmax_matches_masked_full_argmax():
    torch.manual_seed(0)
    V, rows = 2048, 32
    ids = dv.normalize_ids(torch.randperm(V)[:700].tolist(), V)
    full = torch.randn(rows, V)
    sub = full[:, ids]
    out = dv.map_subset_argmax(sub, ids)
    masked = torch.full_like(full, float("-inf"))
    masked[:, ids] = full[:, ids]
    assert torch.equal(out, masked.argmax(-1))
    assert set(out.tolist()) <= set(ids.tolist())


def test_subset_quantized_is_bit_identical_to_row_subset():
    torch.manual_seed(1)
    V, D = 512, 256
    w = torch.randn(V, D)
    full = u2.quantize_linear(w)
    ids = dv.normalize_ids(torch.randperm(V)[:200].tolist(), V)
    sub = dv.subset_quantized(full, ids)
    assert sub["weight_packed"].shape == (ids.numel(), D // 8)
    assert sub["weight_zero_point"].shape == (ids.numel() // 8, D // 128)
    assert tuple(sub["weight_shape"].tolist()) == (ids.numel(), D)
    ref = u2.dequantize_linear(full)[ids]
    got = u2.dequantize_linear(sub)
    assert torch.equal(ref, got)
    # and equals quantizing the selected rows directly
    direct = u2.dequantize_linear(u2.quantize_linear(w[ids]))
    assert torch.equal(direct, got)


def test_rejection_is_lossless_for_any_subset_proposal():
    """One-hot draft q (draft_probs None): accept w.p. p(d); else resample from p without d.
    Exact output law == p for every proposal d, so a restricted proposal set cannot bias output."""
    torch.manual_seed(2)
    p = torch.softmax(torch.randn(50), 0)
    for d in (0, 7, 49):
        law = torch.zeros(50)
        law[d] += p[d]  # accepted
        rest = p.clone()
        rest[d] = 0
        law += (1 - p[d]) * rest / rest.sum()  # recovered, excludes d
        assert torch.allclose(law, p, atol=1e-6)


def test_no_lazy_state_after_init():
    class FakeLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.randn(256, 16)
            self.quant_method = self

        def apply(self, layer, x, bias):
            return x @ layer.w.t()

    ids = dv.normalize_ids(list(range(256)), 1024)
    head = dv.DraftVocabHead(FakeLinear(), ids)
    assert head.ids.dtype == torch.int64
    x = torch.randn(4, 16)
    n_bufs = len(list(head.buffers()))
    out = head.greedy(x)
    assert out.shape == (4,)
    assert len(list(head.buffers())) == n_bufs
