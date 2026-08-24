# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-045b: unit tests for the Xid31 instrumentation's pure sizing arithmetic.

These pin the int32 / 2 GiB boundary math used to rank candidates and to project
runtime high-water offsets. No torch/CUDA required -- the arithmetic helpers in
``vllm.v1.worker.xid31_trace`` are pure Python.
"""

from vllm.v1.worker.xid31_trace import (
    INT32_BYTE_LIMIT,
    crosses_in_interval,
    int32_overflow_maxlen,
    projected_offset_bytes,
    worst_case_byte_offset,
)


def test_int32_byte_limit_is_2gib() -> None:
    assert INT32_BYTE_LIMIT == 2**31 == 2_147_483_648


def test_worst_case_offset_contiguous() -> None:
    # Contiguous [rows, cols] int32 => worst offset == (numel-1)*itemsize.
    shape = (8, 31744)  # max_num_reqs x cdiv(507904, 16)
    strides = (31744, 1)
    itemsize = 4
    got = worst_case_byte_offset(shape, strides, itemsize)
    assert got == (8 * 31744 - 1) * 4
    # A block-table this size is ~0.94 MiB worst offset -- nowhere near 2 GiB,
    # which exonerates the block table itself as an int32-overflow site.
    assert got < INT32_BYTE_LIMIT


def test_worst_case_offset_strided_row() -> None:
    # A [rows, big_row] float16 view: outer stride dominates the worst offset.
    # rows crossing 2 GiB requires row_stride_bytes * (rows-1) >= 2**31.
    shape = (2, 524288)
    strides = (524288, 1)
    itemsize = 2
    got = worst_case_byte_offset(shape, strides, itemsize)
    assert got == (524288 + 524288 - 1) * 2


def test_worst_case_offset_empty_and_zero_dims() -> None:
    assert worst_case_byte_offset((), (), 4) == 0
    assert worst_case_byte_offset((0, 16), (16, 1), 4) == 0


def test_projected_offset_scales_with_index_and_stride() -> None:
    # offset = high_water_block * row_stride_elems * itemsize
    assert projected_offset_bytes(0, 4096, 2) == 0
    assert projected_offset_bytes(10, 4096, 2) == 10 * 4096 * 2
    # Doubling either factor doubles the projected offset (linear).
    base = projected_offset_bytes(1000, 4096, 2)
    assert projected_offset_bytes(2000, 4096, 2) == 2 * base
    assert projected_offset_bytes(1000, 8192, 2) == 2 * base


def test_int32_overflow_maxlen_matches_2gib() -> None:
    # A 4096-byte per-token stride crosses exactly 2**31 at max_model_len 2**19.
    assert int32_overflow_maxlen(4096) == 524288 == 2**19
    # Guard against a zero/negative stride.
    assert int32_overflow_maxlen(0) == float("inf")


def test_fingerprint_interval_stride_band() -> None:
    """The 2 GiB crossing lands inside the (460800, 507904] fingerprint interval
    iff the per-token byte stride is in ~[4228.13, 4660.34].

    (2**31/507904 == 2097152/496 == 4228.129...;
     2**31/460800 == 2097152/450 == 4660.337...)"""
    lo_stride = INT32_BYTE_LIMIT / 507904  # ~4228.13 -> crossing at 507904
    hi_stride = INT32_BYTE_LIMIT / 460800  # ~4660.34 -> crossing at 460800
    assert 4228.0 < lo_stride < 4229.0
    assert 4660.0 < hi_stride < 4661.0

    # A stride strictly inside the band crosses inside the interval.
    mid = (lo_stride + hi_stride) / 2
    assert crosses_in_interval(mid)

    # 4096 bytes/token crosses at 524288 -- ABOVE the interval (exonerated for
    # the interval, but note it is exactly where MTP-OFF@524288 sits at 2**19).
    assert not crosses_in_interval(4096)
    assert int32_overflow_maxlen(4096) == 524288

    # A much larger per-token stride (e.g. 8192 B) crosses far below 460800,
    # so it would already fault at max_model_len=460800 (which is CLEAN) ->
    # exonerated by the interval fingerprint.
    assert int32_overflow_maxlen(8192) == 262144
    assert not crosses_in_interval(8192)


def test_crosses_in_interval_interior_and_exterior() -> None:
    # Pick max_model_len targets away from the exact endpoints so float
    # round-trip error cannot flip the boolean, then derive the stride that
    # crosses 2**31 at that target.
    def stride_crossing_at(mml: int) -> float:
        return INT32_BYTE_LIMIT / mml

    # Interior of (460800, 507904] -> crosses.
    assert crosses_in_interval(stride_crossing_at(480000))
    assert crosses_in_interval(stride_crossing_at(500000))
    # Below 460800 (would already fault at the CLEAN 460800 config) -> exonerated.
    assert not crosses_in_interval(stride_crossing_at(450000))
    assert not crosses_in_interval(stride_crossing_at(262144))
    # Above 507904 (e.g. the 2**19=524288 config) -> outside the interval.
    assert not crosses_in_interval(stride_crossing_at(520000))
    assert not crosses_in_interval(stride_crossing_at(524288))
