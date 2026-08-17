# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K-aware config-validation regression tests (fork-local, 2080Ti).

Covers the three boot-assert lessons measured on hardware:

  1. The Mamba align-mode ``block_size <= max_num_batched_tokens`` check now
     raises a ValueError that states the K-derived block_size, the governing
     formula, and the minimal valid max_num_batched_tokens (was a bare assert
     that only printed the two numbers).
  2. Under speculative decoding, the default cudagraph capture size is
     auto-capped to a rounded ``max_num_seqs * (1 + K)`` instead of the flat
     512 default that needlessly inflates the profiling minimal-KV alloc.
  3. The env escape hatch ``VLLM_KEEP_DEFAULT_CAPTURE_SIZE`` restores legacy.

Pure Python: no engine, no CUDA. ``validate_block_size`` is exercised against a
SimpleNamespace stub used as ``self`` (same pattern as
test_mamba_align_mtp_prefix_cache.py).
"""
import importlib
import os
from types import SimpleNamespace

import vllm.envs as envs
from vllm.config.vllm import (
    VllmConfig,
    default_spec_cudagraph_capture_size,
    mamba_align_block_size_error,
)


def _legacy_default(max_num_seqs: int, k: int) -> int:
    return min(max_num_seqs * (1 + k) * 2, 512)


# ---------------------------------------------------------------------------
# Lesson 2: auto-capped default cudagraph capture size
# ---------------------------------------------------------------------------
def test_capture_size_matches_measured_boot():
    # K=16, max_num_seqs=16: base = 16 * 17 = 272 -> round_up(.,32) = 288.
    # This is exactly the value that let K=16/seqs=16 boot on the 2080Ti.
    assert default_spec_cudagraph_capture_size(16, 16) == 288


def test_capture_size_rounds_up_to_align():
    # base = 16 * 3 = 48 -> next multiple of 32 = 64.
    assert default_spec_cudagraph_capture_size(16, 2) == 64
    # base = 8 * 2 = 16 -> next multiple of 32 = 32.
    assert default_spec_cudagraph_capture_size(8, 1) == 32
    # base already a multiple of 32 stays put: 32 * 3 = 96.
    assert default_spec_cudagraph_capture_size(32, 2) == 96


def test_capture_size_clamped_to_ceiling():
    # base = 64 * 17 = 1088 -> aligned 1088 -> clamped to legacy ceiling 512.
    assert default_spec_cudagraph_capture_size(64, 16) == 512


def test_capture_size_never_exceeds_legacy_default():
    # The auto-cap must only ever *lower* the capture size vs legacy behavior.
    for seqs in (1, 4, 8, 16, 32, 64, 128, 256):
        for k in (1, 2, 4, 8, 16, 32):
            got = default_spec_cudagraph_capture_size(seqs, k)
            assert got <= _legacy_default(seqs, k), (seqs, k, got)
            assert got % 32 == 0 or got == _legacy_default(seqs, k)


def test_capture_size_is_capturable_multiple_of_align():
    # Below the ceiling the result is always a multiple of the align stride.
    for seqs, k in [(16, 16), (16, 2), (8, 1), (4, 4)]:
        got = default_spec_cudagraph_capture_size(seqs, k)
        if got < 512:
            assert got % 32 == 0


# ---------------------------------------------------------------------------
# Lesson 3: env escape hatch is wired
# ---------------------------------------------------------------------------
def test_keep_default_capture_size_env_flag():
    importlib.reload(envs)
    assert envs.VLLM_KEEP_DEFAULT_CAPTURE_SIZE is False
    prev = os.environ.get("VLLM_KEEP_DEFAULT_CAPTURE_SIZE")
    try:
        os.environ["VLLM_KEEP_DEFAULT_CAPTURE_SIZE"] = "1"
        importlib.reload(envs)
        assert envs.VLLM_KEEP_DEFAULT_CAPTURE_SIZE is True
    finally:
        if prev is None:
            os.environ.pop("VLLM_KEEP_DEFAULT_CAPTURE_SIZE", None)
        else:
            os.environ["VLLM_KEEP_DEFAULT_CAPTURE_SIZE"] = prev
        importlib.reload(envs)


# ---------------------------------------------------------------------------
# Lesson 1: the align-mode error message and the ValueError it raises
# ---------------------------------------------------------------------------
def test_align_error_message_states_k_formula_and_minimum():
    msg = mamba_align_block_size_error(
        block_size=3856, max_num_batched_tokens=2048, num_speculative_tokens=16
    )
    # states the K-derived block size ...
    assert "3856" in msg
    # ... the K it was derived with and the decode-width formula ...
    assert "num_speculative_tokens" in msg
    assert "(1 + 16)" in msg and "= 17" in msg
    # ... the governing constraint ...
    assert "max_num_batched_tokens >= block_size" in msg
    # ... and the minimal valid max_num_batched_tokens (== block_size).
    assert "Minimal valid max_num_batched_tokens: 3856" in msg
    assert "at least 3856" in msg


def _align_stub(block_size, mnbt, k, *, long_prefill=0, disable_mm=False):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size, mamba_cache_mode="align"
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=mnbt,
            long_prefill_token_threshold=long_prefill,
            disable_chunked_mm_input=disable_mm,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        num_speculative_tokens=k,
    )


def test_validate_block_size_raises_valueerror_with_remedy():
    stub = _align_stub(block_size=3856, mnbt=2048, k=16)
    try:
        VllmConfig.validate_block_size(stub)
    except ValueError as e:
        text = str(e)
        assert "3856" in text
        assert "2048" in text
        assert "(1 + 16) = 17" in text
        assert "Minimal valid max_num_batched_tokens: 3856" in text
    else:
        raise AssertionError("expected ValueError for block_size > mnbt")


def test_validate_block_size_passes_when_within_budget():
    # block_size <= mnbt in align mode: must not raise.
    stub = _align_stub(block_size=2048, mnbt=8192, k=16)
    assert VllmConfig.validate_block_size(stub) is None


def test_validate_block_size_noop_when_not_align():
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=99999, mamba_cache_mode="none"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        num_speculative_tokens=0,
    )
    # block_size (99999) >> mnbt (8) but mode is not align -> no constraint.
    assert VllmConfig.validate_block_size(stub) is None


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
