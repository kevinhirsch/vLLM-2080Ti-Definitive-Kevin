# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the [FORK] port of vllm-project/vllm#50021 ("Bound
accepted-token state lookups in GDN/KDA spec decode"), which completes the
already-landed vllm-project/vllm#51508 port in
``vllm.v1.attention.backends.gdn_attn``.

Background (see LANE/ARTIFACTS.md and LANE/PORT.md in
projects/lanes/upstream-fixes for the full research trail): the #51508 port
nulls a stale (``num_accepted_tokens <= 0``) spec-decode row's own
``spec_state_indices_tensor`` entries, but leaves
``num_accepted_tokens[row] == 0`` in place. Every kernel consumer of that
tensor (``fused_recurrent_gated_delta_rule_fwd_kernel``,
``fused_sigmoid_gating_delta_rule_update_kernel``, ``causal_conv1d_update``'s
spec-decode path) computes ``i_t = num_accepted_tokens[row] - 1`` *before*
indexing ``ssm_state_indices`` / ``conv_state_indices`` with it, so a stale
row's ``i_t`` becomes -1 -- which, for a plain row-major
``[batch, num_spec + 1]`` buffer, is column ``num_spec`` of row
``row - 1`` (or truly out of bounds for row 0), not this row's own
(already-nulled) column 0.

None of this touches a `@triton.jit` kernel -- everything under test is a
plain torch tensor op, runnable and meaningful on CPU tensors
(``CUDA_VISIBLE_DEVICES=""``); the only thing a GPU window can add is
confirming the *kernel* behaves as this file assumes, which
LANE/PORT.md tracks separately.
"""

import os

import torch

from vllm.v1.attention.backends.gdn_attn import neutralize_stale_accept_rows

NUM_SPEC = 3  # matches MTP K=3 in prod (deploy/bin/serve-tqk8v4-fg.sh)


class TestNeutralizeStaleAcceptRows:
    """Direct tests of the extracted, pure fix function."""

    def test_no_stale_rows_is_a_pure_passthrough(self):
        counts = torch.tensor([1, 2, 4, 1], dtype=torch.int32)
        stale = counts <= 0
        assert not bool(stale.any())
        out = neutralize_stale_accept_rows(counts, stale)
        assert out is counts, "no-stale-rows path must not allocate/copy"
        assert torch.equal(out, counts)

    def test_single_stale_row_forced_to_one(self):
        counts = torch.tensor([2, 0, 4, 1], dtype=torch.int32)
        stale = counts <= 0
        out = neutralize_stale_accept_rows(counts, stale)
        assert out.tolist() == [2, 1, 4, 1]

    def test_multiple_stale_rows_all_forced_to_one(self):
        counts = torch.tensor([0, 0, 3, 0], dtype=torch.int32)
        stale = counts <= 0
        out = neutralize_stale_accept_rows(counts, stale)
        assert out.tolist() == [1, 1, 3, 1]

    def test_negative_counts_are_also_stale(self):
        # The call site's mask is `num_accepted_tokens <= 0`, not `== 0`;
        # confirm negative values (defensive -- not known to occur today,
        # but the mask covers them) are neutralized identically to zero.
        counts = torch.tensor([-1, 5, -3], dtype=torch.int32)
        stale = counts <= 0
        out = neutralize_stale_accept_rows(counts, stale)
        assert out.tolist() == [1, 5, 1]

    def test_all_rows_stale(self):
        counts = torch.zeros(5, dtype=torch.int32)
        stale = counts <= 0
        out = neutralize_stale_accept_rows(counts, stale)
        assert out.tolist() == [1, 1, 1, 1, 1]

    def test_does_not_mutate_input(self):
        counts = torch.tensor([0, 2], dtype=torch.int32)
        original = counts.clone()
        stale = counts <= 0
        neutralize_stale_accept_rows(counts, stale)
        assert torch.equal(counts, original), "must not mutate its input"

    def test_idempotent(self):
        counts = torch.tensor([0, 2, 0, 5], dtype=torch.int32)
        stale = counts <= 0
        once = neutralize_stale_accept_rows(counts, stale)
        twice = neutralize_stale_accept_rows(once, once <= 0)
        assert torch.equal(once, twice)

    def test_matches_the_codebases_own_neutral_convention(self):
        # gdn_attn.py fills cudagraph padding rows with `.fill_(1)`
        # (decode_cudagraph_max_bs padding) and mamba_utils.py resets
        # `num_accepted_tokens_cpu[i] = 1` on a block-boundary crossing --
        # this fix must produce that exact same sentinel, not e.g. 0 or -1.
        counts = torch.tensor([0], dtype=torch.int32)
        out = neutralize_stale_accept_rows(counts, counts <= 0)
        assert out.item() == 1


class TestOffByOneRowSemantics:
    """Mechanically replicates the kernels' row-major indexing arithmetic
    (``ssm_state_indices[row, num_accepted_tokens[row] - 1]``) on a plain
    CPU/numpy-style buffer to make the off-by-one-row bug -- and the fix --
    concrete and falsifiable without a Triton kernel or a GPU.
    """

    def _make_table(self, num_rows: int) -> torch.Tensor:
        # Row-major [batch, num_spec+1] int32 buffer, exactly as allocated
        # in GDNAttentionMetadataBuilder.__init__:
        #   torch.empty((decode_cudagraph_max_bs, self.num_spec + 1), ...)
        # Row i's block ids are distinguishable: 100*i + column.
        table = torch.empty((num_rows, NUM_SPEC + 1), dtype=torch.int32)
        for r in range(num_rows):
            for c in range(NUM_SPEC + 1):
                table[r, c] = 100 * r + c + 1  # +1 so every entry is > 0
        return table

    def _kernel_read_addr_flat_index(
        self, table: torch.Tensor, row: int, i_t: int
    ) -> int:
        """Same address arithmetic as fused_recurrent.py:117
        (``ssm_state_indices + i_n * stride_indices_seq + i_t``), returned
        as a flat offset into the buffer's underlying storage so we can
        assert exactly which row it lands in.
        """
        stride_indices_seq = table.stride(0)
        return row * stride_indices_seq + i_t

    def test_stale_row_i_t_minus_one_reads_previous_rows_last_column(self):
        """Reproduces the bug: for row > 0, a stale row's unclamped i_t=-1
        resolves to the *previous* row's last column, not this row's own
        (nulled) column 0 -- so nulling only this row's entries (the
        existing #51508 port) cannot make this read safe.
        """
        table = self._make_table(num_rows=4)
        flat = table.reshape(-1)

        row = 2
        i_t = -1  # num_accepted_tokens[row] == 0, unclamped
        addr = self._kernel_read_addr_flat_index(table, row, i_t)
        read_value = flat[addr].item()

        prev_row_last_col_addr = self._kernel_read_addr_flat_index(
            table, row - 1, NUM_SPEC
        )
        this_row_col0_addr = self._kernel_read_addr_flat_index(table, row, 0)

        assert addr == prev_row_last_col_addr
        assert addr != this_row_col0_addr
        # And that previous row's data is a normal-looking, real-looking
        # positive block id (from an unrelated, presumably-live request) --
        # not NULL_BLOCK_ID -- so the kernel's `state_idx <= 0: return`
        # guard does NOT catch it. This is the silent cross-request read.
        assert read_value > 0
        assert read_value == table[row - 1, NUM_SPEC].item()

    def test_row_zero_stale_i_t_minus_one_is_out_of_bounds(self):
        """For row 0 there is no previous row at all: the same address
        arithmetic resolves to a negative flat offset, i.e. genuinely
        before the tensor's own storage -- consistent with a wild-address
        illegal-memory-access fault on a real CUDA buffer, rather than a
        readable (if wrong) neighbour row.
        """
        table = self._make_table(num_rows=4)
        row = 0
        i_t = -1
        addr = self._kernel_read_addr_flat_index(table, row, i_t)
        assert addr < 0

    def test_clamped_row_i_t_zero_reads_its_own_nulled_column(self):
        """After the fix: num_accepted_tokens[row] is forced to 1, so
        i_t == 0, which reads row's *own* column 0 -- exactly the column
        the existing #51508 port already set to NULL_BLOCK_ID for this
        row -- so the kernel's existing guard now correctly fires in place.
        """
        NULL_BLOCK_ID = 0
        table = self._make_table(num_rows=4)
        row = 2
        table[row, :] = NULL_BLOCK_ID  # the existing #51508 null-write

        counts = torch.tensor([1, 1, 0, 1], dtype=torch.int32)
        stale = counts <= 0
        clamped = neutralize_stale_accept_rows(counts, stale)
        i_t = int(clamped[row].item()) - 1
        assert i_t == 0

        addr = self._kernel_read_addr_flat_index(table, row, i_t)
        this_row_col0_addr = self._kernel_read_addr_flat_index(table, row, 0)
        assert addr == this_row_col0_addr
        read_value = table.reshape(-1)[addr].item()
        assert read_value == NULL_BLOCK_ID, (
            "post-fix, the stale row's own (nulled) data is what gets "
            "read -- the kernel's `state_idx <= 0: return` guard now "
            "fires correctly, touching no other request's state"
        )


class TestEnvVarDefaultAndRoundTrip:
    def test_default_is_off(self, monkeypatch):
        monkeypatch.delenv("VLLM_GDN_SPEC_ACCEPT_INDEX_CLAMP", raising=False)
        import importlib

        import vllm.envs as envs

        importlib.reload(envs)
        assert envs.VLLM_GDN_SPEC_ACCEPT_INDEX_CLAMP is False

    def test_enables_on_truthy_values(self, monkeypatch):
        import importlib

        import vllm.envs as envs

        for value in ("1", "true", "TRUE", "yes", "on"):
            monkeypatch.setenv("VLLM_GDN_SPEC_ACCEPT_INDEX_CLAMP", value)
            importlib.reload(envs)
            assert envs.VLLM_GDN_SPEC_ACCEPT_INDEX_CLAMP is True, value

        monkeypatch.delenv("VLLM_GDN_SPEC_ACCEPT_INDEX_CLAMP", raising=False)
        importlib.reload(envs)


def test_cpu_only_no_gpu_allocation_guard():
    """Sanity guard for this file itself: everything above must run with no
    CUDA device visible, matching the lane's CPU-only constraint.
    """
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" or not torch.cuda.is_available()
