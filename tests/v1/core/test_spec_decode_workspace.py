# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-python unit tests for the speculative-decode verify-workspace reserve.

Validates the sizing formula (shapes -> bytes) in
``vllm/v1/core/spec_decode_workspace.py`` against the measured OOM facts, without
importing torch / CUDA. The module is stdlib-only, so it is loaded directly by
path.

Measured facts being pinned (Qwen3.8-27B GPTQ-Int4 TP=2, vocab=248320,
mtp num_speculative_tokens config):
  * K=16, max_num_seqs=16: boot OOM'd post-profiling; actual allocations blew
    ~5 GiB past what profiling budgeted.
  * K=16 booted only after dropping max_num_seqs 16 -> 4 (~4x less overshoot).
  * K=2 at max_num_seqs=16 profiles fine (no meaningful overshoot).

Run: ``python3 tests/v1/core/test_spec_decode_workspace.py`` (stdlib only).
"""

import importlib.util
import os
import sys

GIB = 1 << 30
VOCAB = 248320  # Qwen3.8-27B


def _load_module():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    path = os.path.join(repo, "vllm", "v1", "core", "spec_decode_workspace.py")
    name = "_spec_decode_workspace"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the frozen dataclass's string annotations
    # (`from __future__ import annotations`) resolve their module.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_module()
reserve_bytes = M.spec_verify_reserve_bytes
reserve = M.spec_verify_reserve


def _gib(b):
    return b / GIB


# ---------------------------------------------------------------------------

def test_predicts_observed_5gib_gap_at_k16_seqs16():
    """The default reserve must cover the observed ~5 GiB post-profiling gap."""
    r = reserve_bytes(16, 16, VOCAB)  # default overshoot_mult=24
    assert r >= 5 * GIB, f"expected >=5 GiB, got {_gib(r):.3f} GiB"
    # sanity upper bound so a typo can't silently reserve the whole card
    assert r <= 8 * GIB, f"unexpectedly large reserve {_gib(r):.3f} GiB"


def test_small_at_k2_seqs16():
    """K=2 profiles fine on hardware -> reserve must be small and << K=16."""
    r2 = reserve_bytes(2, 16, VOCAB)
    r16 = reserve_bytes(16, 16, VOCAB)
    assert r2 < 0.5 * GIB, f"K=2 reserve not small: {_gib(r2):.3f} GiB"
    assert r2 < r16 / 10, f"K=2 ({_gib(r2):.3f}) not << K=16 ({_gib(r16):.3f})"


def test_seqs_scaling_matches_boot_evidence():
    """K=16 booted after seqs 16->4; overshoot must scale ~linearly with seqs."""
    r_s16 = reserve_bytes(16, 16, VOCAB)
    r_s4 = reserve_bytes(16, 4, VOCAB)
    # (K-1)*seqs scaling => exactly 4x for seqs 16 vs 4.
    assert abs(r_s16 - 4 * r_s4) <= 1, (
        f"seqs scaling off: s16={_gib(r_s16):.3f} s4={_gib(r_s4):.3f}"
    )
    # and s4 must be small enough that it booted (well under the s16 gap).
    assert r_s4 < 2 * GIB, f"K16/s4 reserve too large to have booted: {_gib(r_s4):.3f}"


def test_noop_below_or_at_k1_and_no_spec():
    """Nothing beyond the profiled K=1 baseline -> zero reserve."""
    assert reserve_bytes(1, 16, VOCAB) == 0
    assert reserve_bytes(0, 16, VOCAB) == 0
    assert reserve_bytes(2, 0, VOCAB) == 0
    assert reserve_bytes(2, 16, 0) == 0
    assert reserve_bytes(16, 16, VOCAB, overshoot_mult=0) == 0


def test_per_request_delta_width_is_k_minus_one():
    """Profiler covers 2 positions/req; real verify width is K+1 -> delta K-1."""
    for k in (2, 4, 8, 16):
        b = reserve(k, 16, VOCAB)
        assert b.per_req_delta_width == k - 1, (k, b.per_req_delta_width)
        assert b.delta_positions == (k - 1) * 16


def test_monotonic_in_k_and_seqs():
    prev = -1
    for k in (2, 3, 4, 8, 16, 32):
        r = reserve_bytes(k, 16, VOCAB)
        assert r > prev, f"not increasing in K at K={k}"
        prev = r
    prev = -1
    for s in (1, 2, 4, 8, 16, 32):
        r = reserve_bytes(16, s, VOCAB)
        assert r >= prev, f"not increasing in seqs at seqs={s}"
        prev = r


def test_mechanistic_floor_is_honest_1_to_2_gib():
    """The physics-only reserve (concurrent verify buffers, mult=6) is ~1.3 GiB.

    This documents that the raw verify buffers CANNOT be ~5 GiB (the arithmetic
    ceiling is ~2 GiB at this vocab/batch); the default's larger envelope covers
    the un-instrumented residual (GDN align-decode workspace + allocator
    fragmentation), not more logits.
    """
    r = reserve_bytes(16, 16, VOCAB, overshoot_mult=M.MECHANISTIC_OVERSHOOT_MULT)
    assert 1.0 * GIB <= r <= 2.0 * GIB, f"mechanistic floor {_gib(r):.3f} GiB"


def test_exact_bytes_k16_seqs16():
    """Pin the exact arithmetic: 24 * (16-1)*16 * 248320 * 4 bytes."""
    expected = 24 * (16 - 1) * 16 * VOCAB * 4
    assert reserve_bytes(16, 16, VOCAB) == expected
    assert expected == 5_721_292_800  # 5.328 GiB


# ---------------------------------------------------------------------------

def _print_table():
    print(f"{'K':>4} {'seqs':>5} {'mult':>5} {'delta_pos':>10} {'reserve_GiB':>12}")
    for (k, s, mult) in [
        (2, 16, 24),
        (16, 4, 24),
        (16, 16, 24),
        (16, 16, M.MECHANISTIC_OVERSHOOT_MULT),
        (16, 32, 24),
        (32, 16, 24),
    ]:
        b = reserve(k, s, VOCAB, overshoot_mult=mult)
        print(
            f"{k:>4} {s:>5} {mult:>5} {b.delta_positions:>10} "
            f"{_gib(b.reserve_bytes):>12.3f}"
        )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print()
    _print_table()
    print()
    if failed:
        raise SystemExit(f"{failed}/{len(tests)} tests failed")
    print(f"All {len(tests)} tests passed.")
