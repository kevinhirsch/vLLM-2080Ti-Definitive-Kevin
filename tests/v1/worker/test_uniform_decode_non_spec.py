# SPDX-License-Identifier: Apache-2.0
"""Lane S4 (2026-10-03): a prefill / continuation chunk whose length equals 1 + num_speculative_tokens must not be
classified as a uniform spec-decode batch (vllm-project/vllm#53051).  Production incident: engine death at 06:27:29 with
`CUDAGRAPH-REFRESH mismatch dst shape=(0,) src shape=(4,)` (captured FULL graph = all spec rows, live step = a 4-token
non-spec prefill row => GDN non_spec_token_indx 0 vs 4 elements)."""
import numpy as np

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

K = 3  # num_speculative_tokens -> uniform_decode_query_len 4
Q = K + 1


class _Meta:  # stands in for SpecDecodeMetadata (only truthiness matters)
    pass


def uniform(max_q, ntok, nreqs, non_spec):
    return GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=max_q,
        uniform_decode_query_len=Q,
        num_tokens=ntok,
        num_reqs=nreqs,
        has_non_spec_rows=non_spec,
    )


def has_non_spec(meta, mask, k=K):
    arr = np.array(mask, dtype=np.int32)
    return GPUModelRunner._batch_has_non_spec_rows(meta, arr, len(arr), k)


def test_pure_spec_decode_batch_is_uniform():
    assert uniform(Q, Q * 5, 5, has_non_spec(_Meta(), [K] * 5)) is True


def test_four_token_prefill_alone_is_not_uniform_decode():
    # one virgin/continuation request with exactly 4 scheduled tokens, no scheduled drafts
    assert has_non_spec(None, [-1]) is True
    assert uniform(Q, Q, 1, has_non_spec(None, [-1])) is False


def test_four_token_prefill_mixed_with_spec_rows_is_not_uniform_decode():
    mask = [K, K, -1, K]
    assert has_non_spec(_Meta(), mask) is True
    assert uniform(Q, Q * 4, 4, True) is False


def test_without_speculation_nothing_changes():
    assert has_non_spec(None, [-1, -1], k=0) is False
    assert GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=1, uniform_decode_query_len=1, num_tokens=2, num_reqs=2,
        has_non_spec_rows=has_non_spec(None, [-1, -1], k=0),
    ) is True


def test_capture_force_flag_still_wins():
    assert GPUModelRunner._is_uniform_decode(
        max_num_scheduled_tokens=7, uniform_decode_query_len=Q, num_tokens=7, num_reqs=1,
        force_uniform_decode=True, has_non_spec_rows=True,
    ) is True


def test_shape_mismatch_is_never_uniform():
    assert uniform(Q, Q * 3 + 1, 3, False) is False
    assert uniform(Q - 1, (Q - 1) * 3, 3, False) is False
