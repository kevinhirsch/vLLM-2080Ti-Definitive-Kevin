# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tq-continuation-crash lane: CPU-only tests for the decode-kernel-family
diagnostic added to triton_turboquant_decode.py.

`_tq_decode_stage1` / `_tq_decode_stage1_qtile` (the Triton kernels
themselves) have no equivalent bounds check at all today -- see
/home/kevin/projects/lanes/tq-crash/ANALYSIS.md Section 3 and
MECHANISMS.md Rank 1. This only tests the pure Python/CPU-tensor
diagnostic wrapper (`_tq_decode_trace_pre_launch`) and the env-flag
plumbing; it cannot exercise the Triton kernels, which need a GPU.
"""

import logging

import torch

from vllm.v1.attention.ops.triton_turboquant_decode import (
    _tq_decode_trace_pre_launch,
    _TQ_DECODE_BLOCK_TABLE_CLAMP,
    _TQ_DECODE_TRACE,
)


def test_trace_flags_default_off():
    # Both new env-gated behaviors must default to off (zero behavior
    # change for anyone who doesn't opt in) per the task's requirement.
    assert _TQ_DECODE_BLOCK_TABLE_CLAMP is False
    assert _TQ_DECODE_TRACE is False


def test_trace_pre_launch_empty_block_table_is_a_no_op(caplog):
    bt = torch.empty((0, 0), dtype=torch.int32)
    seq_lens = torch.empty((0,), dtype=torch.int32)
    with caplog.at_level(logging.INFO):
        _tq_decode_trace_pre_launch("stage1", bt, seq_lens, num_blocks=50)
    # Nothing to report for an empty table; must not raise or fabricate a
    # bogus bt_min/bt_max from an empty reduction.
    assert not any("decode-launch" in r.message for r in caplog.records)


def test_trace_pre_launch_flags_out_of_range_ids(caplog):
    bt = torch.tensor([[0, 1, 2, 999]])
    seq_lens = torch.tensor([50])
    with caplog.at_level(logging.INFO):
        _tq_decode_trace_pre_launch("stage1", bt, seq_lens, num_blocks=50)
    records = [r.message for r in caplog.records if "decode-launch" in r.message]
    assert len(records) == 1
    assert "bt_max=999" in records[0]
    assert "bt_in_range=False" in records[0]


def test_trace_pre_launch_reports_in_range_when_valid(caplog):
    bt = torch.tensor([[0, 1, 2, 49]])
    seq_lens = torch.tensor([50])
    with caplog.at_level(logging.INFO):
        _tq_decode_trace_pre_launch("qtile", bt, seq_lens, num_blocks=50)
    records = [r.message for r in caplog.records if "decode-launch" in r.message]
    assert len(records) == 1
    assert "site=qtile" in records[0]
    assert "bt_in_range=True" in records[0]


def test_decode_block_table_clamp_helper_semantics():
    # The fix itself is a single `.clamp()` call at the launcher call site
    # (triton_turboquant_decode_attention); pin the exact semantics it
    # relies on directly against torch, since that's the only "logic" in
    # the fix that isn't already covered by
    # test_tq_continuation_bounds.py's _tq_clamp_block_table_pages tests.
    bt = torch.tensor([[0, 1, -1, 999]])
    num_blocks = 50
    clamped = bt.clamp(0, max(num_blocks - 1, 0))
    assert clamped.tolist() == [[0, 1, 0, 49]]
    # Values already in range are untouched (no-op in the common case).
    bt_valid = torch.tensor([[0, 1, 2, 49]])
    assert bt_valid.clamp(0, num_blocks - 1).tolist() == bt_valid.tolist()


def test_decode_block_table_clamp_handles_zero_num_blocks_without_crashing():
    # max(num_blocks - 1, 0) guards the num_blocks==0 edge case (e.g. a
    # unit test or a config-less path) from becoming clamp(0, -1), which
    # would silently produce an all-zero-or-invalid range; confirm the
    # guarded form used at the call site is well-defined instead.
    bt = torch.tensor([[0, 1, 2]])
    num_blocks = 0
    clamped = bt.clamp(0, max(num_blocks - 1, 0))
    assert clamped.tolist() == [[0, 0, 0]]
