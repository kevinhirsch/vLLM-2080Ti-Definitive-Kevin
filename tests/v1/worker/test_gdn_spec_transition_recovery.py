# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contract tests for the vllm-project/vllm#55504 fork port
("[Bugfix] Fix V1 GDN state recovery across mixed speculative batches"),
in ``vllm/v1/attention/backends/gdn_attn.py``
(``GDNAttentionMetadataBuilder.build``) and
``vllm/model_executor/layers/mamba/gdn_linear_attn.py`` (``_forward_core``,
``_forward_core_decode_non_spec``).

Following the precedent set by the sibling #52078/#53077 port test
(``test_gdn_metadata_builder_spec_mask.py``): ``GDNAttentionMetadataBuilder
.build`` as a whole is CUDA-only (device tensors, Triton FLA chunk ops
downstream in the same method) and not meaningfully unit-testable on CPU.
This instead replicates the two new index-arithmetic branches verbatim on
plain CPU tensors -- keep these in sync with ``build()`` if it changes.

[FORK DEVIATION] Upstream scopes #55504 to ``mamba_cache_mode == "none"``.
This fork's align mode performs the equivalent conv/ssm state recovery
earlier and generically, in ``mamba_utils.preprocess_mamba``'s
host-orchestrated block relocation (``accept_token_bias``), and always
resets ``num_accepted_tokens`` to 1 before this builder runs; align mode's
block-table columns are also separate per-checkpoint physical blocks, not
the fixed draft-position slots this recovery assumes. The port therefore
adds a `self._recover_spec_transition_state = (mamba_cache_mode != "align")`
gate in ``GDNAttentionMetadataBuilder.__init__``, verified directly below
alongside the arithmetic.
"""

import torch


def _recover_spec_transition_state(mamba_cache_mode: str) -> bool:
    """Verbatim replica of the one-line gate set in
    ``GDNAttentionMetadataBuilder.__init__``. Keep in sync with it."""
    return mamba_cache_mode != "align"


def _pure_decode_recovery(
    num_accepted_tokens: torch.Tensor,
    block_table_tensor: torch.Tensor,
    num_decodes: int,
    num_prefills: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verbatim replica of the ``spec_sequence_masks is None`` branch's
    recovery block in ``GDNAttentionMetadataBuilder.build`` -- callers here
    are assumed to already be past the
    ``self._recover_spec_transition_state and self.use_spec_decode and
    num_accepted_tokens is not None and num_decodes > 0`` gate. Keep in
    sync with ``build()``."""
    col_indices = (num_accepted_tokens[:num_decodes] - 1).clamp(min=0)
    spec_decode_src_indices = block_table_tensor[
        torch.arange(num_decodes, device=block_table_tensor.device),
        col_indices,
    ]
    recovered_num_accepted = num_accepted_tokens[:num_decodes]
    if num_prefills > 0:
        recovered_num_accepted = torch.cat(
            [
                recovered_num_accepted,
                torch.ones(
                    num_prefills,
                    dtype=recovered_num_accepted.dtype,
                    device=recovered_num_accepted.device,
                ),
            ]
        )
    return spec_decode_src_indices, recovered_num_accepted


def _mixed_batch_recovery(
    num_accepted_tokens: torch.Tensor,
    block_table_tensor: torch.Tensor,
    non_spec_sequence_masks_cpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verbatim replica of the mixed-batch (``else:``) branch's recovery
    block, gated the same way as ``_pure_decode_recovery`` above. Keep in
    sync with ``build()``."""
    non_spec_num_accepted = num_accepted_tokens[non_spec_sequence_masks_cpu].clamp(
        min=1
    )
    non_spec_block_rows = block_table_tensor[non_spec_sequence_masks_cpu]
    source_columns = non_spec_num_accepted - 1
    spec_decode_src_indices = non_spec_block_rows[
        torch.arange(non_spec_block_rows.size(0), device=block_table_tensor.device),
        source_columns,
    ]
    return spec_decode_src_indices, non_spec_num_accepted


class TestRecoverSpecTransitionStateGate:
    def test_align_mode_disables_recovery(self):
        assert _recover_spec_transition_state("align") is False

    def test_other_modes_enable_recovery(self):
        assert _recover_spec_transition_state("none") is True
        assert _recover_spec_transition_state("all") is True


class TestPureDecodeRecovery:
    def test_finds_true_source_column_per_row(self):
        # 3 decode rows: accepted counts 1, 3, 2 -> source columns 0, 2, 1.
        num_accepted_tokens = torch.tensor([1, 3, 2], dtype=torch.int32)
        block_table_tensor = torch.tensor(
            [
                [10, 11, 12, 13],
                [20, 21, 22, 23],
                [30, 31, 32, 33],
            ],
            dtype=torch.int32,
        )
        src, recovered = _pure_decode_recovery(
            num_accepted_tokens, block_table_tensor, num_decodes=3, num_prefills=0
        )
        assert src.tolist() == [10, 22, 31]
        assert recovered.tolist() == [1, 3, 2]

    def test_clamps_a_stale_zero_count_to_column_zero(self):
        # A stale (num_accepted_tokens <= 0) row must not produce a
        # negative index -- clamp(min=0) keeps it at column 0, matching the
        # neutral convention the rest of the codebase uses for stale rows
        # (see the #51508/#50021 ports in the same file).
        num_accepted_tokens = torch.tensor([0, 2], dtype=torch.int32)
        block_table_tensor = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
        src, _ = _pure_decode_recovery(
            num_accepted_tokens, block_table_tensor, num_decodes=2, num_prefills=0
        )
        assert src.tolist() == [10, 21]

    def test_pads_reclassified_prefill_rows_with_neutral_one(self):
        # Rows beyond num_decodes were reclassified to prefills (see
        # gdn_attn.py's `num_decodes > 0 and num_spec_decodes > 0` merge);
        # they get the neutral accepted-count of 1, not a real column.
        num_accepted_tokens = torch.tensor([2], dtype=torch.int32)
        block_table_tensor = torch.tensor([[10, 11, 12]], dtype=torch.int32)
        _, recovered = _pure_decode_recovery(
            num_accepted_tokens, block_table_tensor, num_decodes=1, num_prefills=2
        )
        assert recovered.tolist() == [2, 1, 1]


class TestMixedBatchRecovery:
    def test_preserves_non_spec_counts_and_their_source_columns(self):
        # 3 rows: spec, non-spec(accepted=2), non-spec(accepted=1).
        num_accepted_tokens = torch.tensor([3, 2, 1], dtype=torch.int32)
        block_table_tensor = torch.tensor(
            [
                [10, 11, 12],
                [20, 21, 22],
                [30, 31, 32],
            ],
            dtype=torch.int32,
        )
        non_spec_mask = torch.tensor([False, True, True])
        src, non_spec_num_accepted = _mixed_batch_recovery(
            num_accepted_tokens, block_table_tensor, non_spec_mask
        )
        assert non_spec_num_accepted.tolist() == [2, 1]
        assert src.tolist() == [21, 30]

    def test_clamps_stale_non_spec_rows_to_neutral_one(self):
        num_accepted_tokens = torch.tensor([3, 0], dtype=torch.int32)
        block_table_tensor = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
        non_spec_mask = torch.tensor([False, True])
        src, non_spec_num_accepted = _mixed_batch_recovery(
            num_accepted_tokens, block_table_tensor, non_spec_mask
        )
        assert non_spec_num_accepted.tolist() == [1]
        assert src.tolist() == [20]


class TestForwardCoreSsmStateCorrection:
    """Verbatim replica of the ssm_state copy added to `_forward_core` and
    `_forward_core_decode_non_spec` in gdn_linear_attn.py -- keep in sync."""

    @staticmethod
    def _apply(ssm_state, non_spec_state_indices_tensor, spec_decode_src_indices):
        if spec_decode_src_indices is not None:
            num_corrected_states = spec_decode_src_indices.shape[0]
            destination_indices = non_spec_state_indices_tensor[:num_corrected_states]
            ssm_state[destination_indices] = ssm_state[spec_decode_src_indices]
        return ssm_state

    def test_none_is_a_pure_no_op(self):
        ssm_state = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        before = ssm_state.clone()
        out = self._apply(ssm_state, torch.tensor([0, 1]), None)
        assert torch.equal(out, before)

    def test_copies_true_source_block_into_canonical_column(self):
        ssm_state = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        original = ssm_state.clone()
        # Row 2 is the canonical ("column 0") slot for decode row 0; row 3
        # holds the true accepted state from its previous spec step.
        non_spec_state_indices_tensor = torch.tensor([2])
        spec_decode_src_indices = torch.tensor([3])
        out = self._apply(
            ssm_state, non_spec_state_indices_tensor, spec_decode_src_indices
        )
        assert torch.equal(out[2], original[3])
        # Every other row (including the untouched source) is unaffected.
        assert torch.equal(out[0], original[0])
        assert torch.equal(out[1], original[1])
        assert torch.equal(out[3], original[3])
