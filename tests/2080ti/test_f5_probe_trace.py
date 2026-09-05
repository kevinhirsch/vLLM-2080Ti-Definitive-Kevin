# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""F-5 (backlog RANK 2, f5-forced-k lane): unit tests for the forced-K
probe trace in ``vllm.v1.worker.f5_probe_trace``.

No torch/CUDA required -- the module itself has no torch dependency (by
design: see its module docstring), and every test here exercises it with
plain Python ints/lists, exactly as its real call sites do after their own
``.tolist()``/``.item()`` conversions.

These tests pin two properties:

  1. **True no-op when disabled** (the default): no file is created, no
     exception escapes, regardless of what garbage is passed in.
  2. **Correct, valid-JSONL output when enabled**: each hook writes the
     documented event shape (PROBE.md "Instrumentation: env-gated trace").

They do NOT and cannot validate that the trace correctly diagnoses the
real doubled-token defect on GPU hardware -- only that its own logging
logic is faithful to what its call sites pass it. See LANE/REPORT.md for
what remains unvalidated without a GPU.
"""

import json

import pytest

from vllm.v1.worker import f5_probe_trace


@pytest.fixture(autouse=True)
def _reset_trace_state(monkeypatch, tmp_path):
    """Isolate every test: force a fresh trace path per test and reset the
    module's mutable state (step counter, install-once flag) so tests don't
    leak into each other regardless of execution order."""
    trace_path = tmp_path / "f5_probe_trace.jsonl"
    monkeypatch.setattr(f5_probe_trace, "_PATH", str(trace_path))
    monkeypatch.setattr(f5_probe_trace, "_step", 0)
    monkeypatch.setattr(f5_probe_trace, "_installed", False)
    return trace_path


def _read_events(path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestDisabledIsANoOp:
    """ENABLED defaults False (VLLM_F5_PROBE_TRACE unset); every public
    entry point must early-return without writing anything or raising."""

    def test_default_enabled_is_false(self) -> None:
        # Guards against an accidental flip of the default in envs parsing.
        assert f5_probe_trace.ENABLED is False

    def test_install_is_noop_when_disabled(self, _reset_trace_state) -> None:
        f5_probe_trace.install(mamba_block_size=16, num_spec=3)
        assert not _reset_trace_state.exists()

    def test_log_dispatch_is_noop_when_disabled(self, _reset_trace_state) -> None:
        f5_probe_trace.log_dispatch(
            cudagraph_mode="FULL", num_tokens=4, num_reqs=8, uniform=True
        )
        assert not _reset_trace_state.exists()
        # Must not even advance the step counter when disabled.
        assert f5_probe_trace.current_step() == 0

    def test_log_preprocess_is_noop_when_disabled(self, _reset_trace_state) -> None:
        f5_probe_trace.log_preprocess(
            req_id="r1",
            prev_state_idx=0,
            curr_state_idx=1,
            boundary_crossed=True,
            accepted_pre=3,
            accepted_post=1,
            block_size=16,
            num_speculative_blocks=3,
        )
        assert not _reset_trace_state.exists()

    def test_log_fanout_is_noop_when_disabled(self, _reset_trace_state) -> None:
        f5_probe_trace.log_fanout(
            num_accepted_tokens=[1, 4],
            state_indices=[[1, 2, 3, 4], [5, 6, 7, 8]],
        )
        assert not _reset_trace_state.exists()

    def test_disabled_never_raises_on_bad_input(self, _reset_trace_state) -> None:
        # A diagnostic must not be able to take the engine down. Even with
        # disabled=True this documents the never-raise contract; the
        # mismatched-length lists below would be a caller bug.
        f5_probe_trace.log_fanout(num_accepted_tokens=[1, 2, 3], state_indices=[[1]])


class TestEnabledLogging:
    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch):
        monkeypatch.setattr(f5_probe_trace, "ENABLED", True)

    def test_install_writes_header_once(self, _reset_trace_state) -> None:
        f5_probe_trace.install(mamba_block_size=16, num_spec=3)
        f5_probe_trace.install(mamba_block_size=16, num_spec=3)  # idempotent
        events = _read_events(_reset_trace_state)
        assert len(events) == 1
        assert events[0]["event"] == "init"
        assert events[0]["mamba_block_size"] == 16
        assert events[0]["num_speculative_tokens"] == 3

    def test_dispatch_advances_step_and_records_shape(self, _reset_trace_state) -> None:
        f5_probe_trace.log_dispatch(
            cudagraph_mode="FULL", num_tokens=4, num_reqs=8, uniform=True
        )
        assert f5_probe_trace.current_step() == 1
        f5_probe_trace.log_dispatch(
            cudagraph_mode="PIECEWISE", num_tokens=1, num_reqs=1, uniform=False
        )
        assert f5_probe_trace.current_step() == 2
        events = _read_events(_reset_trace_state)
        assert [e["step"] for e in events] == [1, 2]
        assert events[0]["cudagraph_mode"] == "FULL"
        assert events[0]["num_reqs"] == 8
        assert events[1]["cudagraph_mode"] == "PIECEWISE"

    def test_preprocess_and_fanout_share_dispatchs_step(self, _reset_trace_state) -> None:
        f5_probe_trace.log_dispatch(
            cudagraph_mode="FULL", num_tokens=4, num_reqs=1, uniform=True
        )
        f5_probe_trace.log_preprocess(
            req_id="cmpl-abc-0",
            prev_state_idx=41,
            curr_state_idx=42,
            boundary_crossed=True,
            accepted_pre=3,
            accepted_post=1,
            block_size=16,
            num_speculative_blocks=3,
        )
        f5_probe_trace.log_fanout(
            num_accepted_tokens=[1],
            state_indices=[[903, 904, 905, 906]],
        )
        events = _read_events(_reset_trace_state)
        assert len(events) == 3
        dispatch, preprocess, fanout = events
        assert dispatch["step"] == preprocess["step"] == fanout["step"] == 1

        assert preprocess["req_id"] == "cmpl-abc-0"
        assert preprocess["boundary_crossed"] is True
        assert preprocess["accepted_pre"] == 3
        assert preprocess["accepted_post"] == 1

        assert fanout["rows"] == [
            {
                "row_index": 0,
                "num_accepted_tokens": 1,
                "state_indices_row": [903, 904, 905, 906],
                "fanout_select_index": 0,
                "fanout_select_block_id": 903,
            }
        ]

    def test_fanout_select_index_matches_accepted_minus_one(
        self, _reset_trace_state
    ) -> None:
        # This is the exact arithmetic the GDN kernels perform on-device
        # (fused_sigmoid_gating.py: i_t = num_accepted_tokens[i_n] - 1); the
        # trace must mirror it precisely or its "which slot was read"
        # conclusion would be off by construction.
        f5_probe_trace.log_fanout(
            num_accepted_tokens=[1, 2, 4],
            state_indices=[[10, 11], [20, 21, 22], [30, 31, 32, 33]],
        )
        [event] = _read_events(_reset_trace_state)
        rows = event["rows"]
        assert rows[0]["fanout_select_index"] == 0
        assert rows[0]["fanout_select_block_id"] == 10
        assert rows[1]["fanout_select_index"] == 1
        assert rows[1]["fanout_select_block_id"] == 21
        assert rows[2]["fanout_select_index"] == 3
        assert rows[2]["fanout_select_block_id"] == 33

    def test_fanout_out_of_range_select_index_is_null_not_a_crash(
        self, _reset_trace_state
    ) -> None:
        # num_accepted_tokens == 0 (a discarded/stale async row, per the
        # vllm-project/vllm#51508 guard already in gdn_attn.py) would select
        # index -1; a caller upstream nulls these rows before this point in
        # practice, but the trace itself must degrade to `None`, never raise.
        f5_probe_trace.log_fanout(num_accepted_tokens=[0], state_indices=[[10, 11]])
        [event] = _read_events(_reset_trace_state)
        assert event["rows"][0]["fanout_select_index"] == -1
        assert event["rows"][0]["fanout_select_block_id"] is None

    def test_every_line_is_valid_json_with_a_timestamp(self, _reset_trace_state) -> None:
        f5_probe_trace.install(mamba_block_size=16, num_spec=3)
        f5_probe_trace.log_dispatch(
            cudagraph_mode="FULL", num_tokens=4, num_reqs=1, uniform=True
        )
        with open(_reset_trace_state) as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)  # raises if malformed
            assert "t" in record  # wall-clock timestamp on every record
