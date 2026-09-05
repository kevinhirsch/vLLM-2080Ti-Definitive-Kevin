# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RANK-4 backlog ("chunked continuation-dequant, +93K pool tokens"):
CPU-only tests for the workspace-reservation shrink in
``TurboQuantAttentionImpl._reserve_continuation_workspace``
(vllm/v1/attention/backends/turboquant_attn.py).

This exercises the REAL reserve function (not a re-implementation) against
a real, CPU-device WorkspaceManager (torch.empty on a CPU tensor needs no
CUDA) and a minimal fake VllmConfig, to pin down the actual claim behind
this backlog item: with
``VLLM_TURBOQUANT_CONTINUATION_CHUNK_TOKENS`` unset (default), the reserve
formula is untouched (byte-identical); once it is set, the reserve
collapses from max_model_len-class token counts down to the chunk size,
unless an operator has explicitly set
``VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS``, which still
wins (same precedence the unchunked formula already gave that override).

Module-level env-derived constants (``_TQ_CONTINUATION_CHUNK_TOKENS``,
``_TQ_CONTINUATION_WORKSPACE_RESERVE_TOKENS``) are monkeypatched directly
on the imported module rather than via environment variables + reimport,
since they are read once at import time.
"""

import types

import pytest
import torch

import vllm.v1.attention.backends.turboquant_attn as tqa
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    reset_workspace_manager,
    workspace_manager_total_bytes,
)

pytestmark = pytest.mark.cpu_test


class _FakeAttnImpl:
    """Stand-in for TurboQuantAttentionImpl exposing only the attributes
    _reserve_continuation_workspace reads/writes."""

    def __init__(self, num_kv_heads: int, head_size: int):
        self._continuation_workspace_reserved = False
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size


def _fake_vllm_config(max_batched_tokens: int, block_size: int, max_model_len: int):
    return types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(
            max_num_batched_tokens=max_batched_tokens
        ),
        cache_config=types.SimpleNamespace(block_size=block_size),
        model_config=types.SimpleNamespace(max_model_len=max_model_len),
    )


@pytest.fixture(autouse=True)
def _clean_workspace_manager():
    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    yield
    reset_workspace_manager()


def _run_reserve(
    monkeypatch,
    *,
    chunk_tokens: int,
    reserve_override: int,
    max_batched_tokens: int,
    block_size: int,
    max_model_len: int,
    num_kv_heads: int = 2,
    head_size: int = 256,
) -> tuple[int, _FakeAttnImpl]:
    monkeypatch.setattr(tqa, "_TQ_CONTINUATION_CHUNK_TOKENS", chunk_tokens)
    monkeypatch.setattr(
        tqa, "_TQ_CONTINUATION_WORKSPACE_RESERVE_TOKENS", reserve_override
    )
    monkeypatch.setattr(
        tqa,
        "get_current_vllm_config",
        lambda: _fake_vllm_config(max_batched_tokens, block_size, max_model_len),
    )
    fake = _FakeAttnImpl(num_kv_heads=num_kv_heads, head_size=head_size)
    tqa.TurboQuantAttentionImpl._reserve_continuation_workspace(fake)
    return workspace_manager_total_bytes(), fake


def _expected_bytes(reserve_tokens, block_size, num_kv_heads, head_size):
    """K+V, fp16, (1, Hk, reserve_cached_len, D): with num_kv_heads=2,
    head_size=256 (this file's default), 1*Hk*D*2 bytes/token = 1024, an
    exact multiple of get_simultaneous's 256-byte alignment, so no padding
    slack -- the reservation's byte count is then an exact multiple of the
    per-token cost with no fudge factor needed in these assertions."""
    import math

    reserve_cached_len = math.ceil(reserve_tokens / block_size) * block_size
    per_token_bytes = 2 * num_kv_heads * head_size * 2  # 2 buffers, fp16
    return per_token_bytes * reserve_cached_len


def test_default_unset_reserve_is_untouched_max_model_len_formula(monkeypatch):
    """Chunking off (default 0), no explicit reserve override: byte-for-byte
    the pre-existing formula, reserve_tokens = max(max_batched, max_model_len)."""
    total_bytes, fake = _run_reserve(
        monkeypatch,
        chunk_tokens=0,
        reserve_override=0,
        max_batched_tokens=8192,
        block_size=16,
        max_model_len=524288,
    )
    assert fake._continuation_workspace_reserved is True
    assert total_bytes == _expected_bytes(524288, 16, 2, 256)
    assert total_bytes == 1024 * 1024 * 1024  # 1 GiB, matches the in-code comment


def test_chunking_on_shrinks_reserve_to_chunk_size_with_no_override(monkeypatch):
    """The RANK-4 claim: turning chunking on (no explicit reserve override)
    collapses the reserve from max_model_len-class down to the chunk size."""
    total_bytes, _ = _run_reserve(
        monkeypatch,
        chunk_tokens=16384,
        reserve_override=0,
        max_batched_tokens=8192,
        block_size=16,
        max_model_len=524288,
    )
    assert total_bytes == _expected_bytes(16384, 16, 2, 256)
    baseline_bytes = _expected_bytes(524288, 16, 2, 256)
    assert total_bytes == baseline_bytes // 32  # 524288 / 16384 == 32


def test_explicit_override_still_wins_over_chunk_default(monkeypatch):
    """An operator-set VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS
    takes precedence over the chunk-derived default, exactly like it already
    took precedence over max_model_len before this change."""
    total_bytes, _ = _run_reserve(
        monkeypatch,
        chunk_tokens=16384,
        reserve_override=32768,
        max_batched_tokens=8192,
        block_size=16,
        max_model_len=524288,
    )
    assert total_bytes == _expected_bytes(32768, 16, 2, 256)


def test_override_alone_unaffected_by_chunking_flag(monkeypatch):
    """Sanity: the override branch's own formula doesn't change whether
    chunking is on or off -- only which branch is reached when the override
    is absent changes."""
    with_chunk_off, _ = _run_reserve(
        monkeypatch,
        chunk_tokens=0,
        reserve_override=32768,
        max_batched_tokens=8192,
        block_size=16,
        max_model_len=524288,
    )
    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    with_chunk_on, _ = _run_reserve(
        monkeypatch,
        chunk_tokens=16384,
        reserve_override=32768,
        max_batched_tokens=8192,
        block_size=16,
        max_model_len=524288,
    )
    assert with_chunk_off == with_chunk_on


def test_max_batched_tokens_floor_still_applies_under_chunking(monkeypatch):
    """reserve_tokens = max(max_batched_tokens, chunk_tokens): a batch-size
    floor larger than the configured chunk still wins, same as it already
    does against max_model_len in the unchunged formula."""
    total_bytes, _ = _run_reserve(
        monkeypatch,
        chunk_tokens=4096,
        reserve_override=0,
        max_batched_tokens=8192,
        block_size=16,
        max_model_len=524288,
    )
    assert total_bytes == _expected_bytes(8192, 16, 2, 256)


def test_below_decode_threshold_skips_reservation_entirely(monkeypatch):
    """max_num_batched_tokens <= _CONTINUATION_DECODE_THRESHOLD (128): the
    continuation dequant path is never entered at all, chunked or not, so
    no workspace should be reserved -- unrelated to this change, pinned
    here as a regression guard since this function is now touched."""
    total_bytes, fake = _run_reserve(
        monkeypatch,
        chunk_tokens=16384,
        reserve_override=0,
        max_batched_tokens=64,
        block_size=16,
        max_model_len=524288,
    )
    assert total_bytes == 0
    assert fake._continuation_workspace_reserved is True


def test_reserved_flag_prevents_double_reservation(monkeypatch):
    """Calling twice with the flag already set must be a no-op (idempotent
    process_weights_after_loading re-entry guard)."""
    monkeypatch.setattr(tqa, "_TQ_CONTINUATION_CHUNK_TOKENS", 16384)
    monkeypatch.setattr(tqa, "_TQ_CONTINUATION_WORKSPACE_RESERVE_TOKENS", 0)
    monkeypatch.setattr(
        tqa,
        "get_current_vllm_config",
        lambda: _fake_vllm_config(8192, 16, 524288),
    )
    fake = _FakeAttnImpl(num_kv_heads=2, head_size=256)
    fake._continuation_workspace_reserved = True
    tqa.TurboQuantAttentionImpl._reserve_continuation_workspace(fake)
    assert workspace_manager_total_bytes() == 0
