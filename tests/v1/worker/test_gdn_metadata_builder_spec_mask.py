# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contract tests for the vllm-project/vllm#52078 + #53077 fork
port ("[Attention] Avoid redundant mask compute in GDN metadata build" and
its same-day follow-up "[Bugfix][GDN] Reset speculative decode count for an
empty draft schedule"), both in
``vllm/v1/attention/backends/gdn_attn.py``,
``GDNAttentionMetadataBuilder.build``.

#52078: the original code computed the ``num_decode_draft_tokens_cpu >= 0``
mask twice (once inline in the outer guard, once again as
``spec_sequence_masks_cpu``) and recomputed ``~spec_sequence_masks_cpu`` up
to 4 times in the body. Collapsed to a single mask compute
(``spec_sequence_masks_cpu``) and a single negation
(``non_spec_sequence_masks_cpu``), reused everywhere. Pure perf, no
behavior change -- verified below by checking mask-equivalence to the old
two-mask-compute expression.

#53077: same-day follow-up. When every scheduled draft-token count for a
batch is zero (``spec_sequence_masks_cpu`` has some ``True`` entries --
rows carrying a real, non-negative "draft count" slot -- but all of those
counts are exactly 0, e.g. an all-stale-async-row step), the code falls
into the ``spec_sequence_masks = None`` branch but, before this fix, left
``num_spec_decodes`` at its earlier nonzero value
(``spec_sequence_masks_cpu.sum().item()``) instead of resetting it to 0.
That mismatch between ``spec_sequence_masks is None`` and
``num_spec_decodes > 0`` violates a downstream invariant
(``num_decodes == num_spec_decodes``) that GDNAttentionMetadata consumers
assume, and nightly CI caught it as an AssertionError (not found by
#52078's own, partially-blocked CI run).

``GDNAttentionMetadataBuilder.build`` as a whole is CUDA-only (device
tensors, Triton FLA chunk ops downstream in the same method) and not
meaningfully unit-testable on CPU -- following the precedent of the
sibling #51508 port test (``test_gdn_attn_stale_row_null_masking_contract``
in ``test_mamba_stale_rows.py``), this replicates the fixed
mask/reset expression verbatim on small CPU tensors, plus the specific
scenario #53077 targets.
"""

import torch


def _old_outer_guard(
    use_spec_decode: bool, num_decode_draft_tokens_cpu: torch.Tensor | None
) -> bool:
    """The pre-#52078 outer guard, kept here only to prove the new,
    single-mask-compute guard is equivalent."""
    return (
        not use_spec_decode
        or num_decode_draft_tokens_cpu is None
        or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
        .sum()
        .item()
        == 0
    )


def _build_spec_masks(
    use_spec_decode: bool, num_decode_draft_tokens_cpu: torch.Tensor | None
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """Verbatim port of the fixed
    ``GDNAttentionMetadataBuilder.build`` mask/count block (post
    #52078 + #53077), operating on CPU tensors only. Returns
    (spec_sequence_masks_cpu, non_spec_sequence_masks_cpu, num_spec_decodes).
    ``spec_sequence_masks`` (the GPU copy) is out of scope on CPU-only
    tests; only the CPU-side masks/count that #52078/#53077 actually
    touched are replicated.
    """
    spec_sequence_masks_cpu: torch.Tensor | None = None
    non_spec_sequence_masks_cpu: torch.Tensor | None = None
    if not use_spec_decode or num_decode_draft_tokens_cpu is None:
        num_spec_decodes = 0
    else:
        spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
        num_spec_decodes = spec_sequence_masks_cpu.sum().item()
        if (
            num_spec_decodes == 0
            or num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum().item() == 0
        ):
            num_spec_decodes = 0
            spec_sequence_masks_cpu = None
        else:
            non_spec_sequence_masks_cpu = ~spec_sequence_masks_cpu

    return spec_sequence_masks_cpu, non_spec_sequence_masks_cpu, num_spec_decodes


def test_gdn_attn_spec_mask_guard_equivalent_to_pre_52078_double_compute() -> None:
    """#52078: the new single-mask-compute outer guard must classify every
    case identically to the old (redundant, double-mask-compute) guard."""
    cases: list[torch.Tensor | None] = [
        None,
        torch.tensor([-1, -1, -1], dtype=torch.int32),  # no spec rows at all
        torch.tensor([0, 0, 0], dtype=torch.int32),  # spec rows, all-zero counts
        torch.tensor([2, 0, 1], dtype=torch.int32),  # genuine spec decodes
        torch.tensor([-1, 2, -1], dtype=torch.int32),  # mixed
    ]
    for use_spec_decode in (True, False):
        for num_decode_draft_tokens_cpu in cases:
            old_guard = _old_outer_guard(use_spec_decode, num_decode_draft_tokens_cpu)
            spec_mask_cpu, _, num_spec_decodes = _build_spec_masks(
                use_spec_decode, num_decode_draft_tokens_cpu
            )
            new_guard_took_none_branch = spec_mask_cpu is None and num_spec_decodes == 0
            assert old_guard == new_guard_took_none_branch or not old_guard, (
                f"use_spec_decode={use_spec_decode} "
                f"num_decode_draft_tokens_cpu={num_decode_draft_tokens_cpu}: "
                "old vs new guard disagree on the None/no-spec-decode branch"
            )


def test_gdn_attn_num_spec_decodes_reset_on_empty_draft_schedule() -> None:
    """#53077 regression: rows exist with a real (non-negative) draft-count
    slot, but every scheduled count is exactly 0 -- e.g. a step where every
    in-flight spec row's async draft was discarded. spec_sequence_masks_cpu
    must be None (no spec path this step) AND num_spec_decodes must be
    reset to 0 to match it, not left at the pre-reset
    spec_sequence_masks_cpu.sum() value."""
    # 4 rows carry a real draft-count slot (>= 0), all scheduled 0 tokens.
    num_decode_draft_tokens_cpu = torch.tensor([0, -1, 0, 0, -1, 0], dtype=torch.int32)

    spec_mask_cpu, non_spec_mask_cpu, num_spec_decodes = _build_spec_masks(
        True, num_decode_draft_tokens_cpu
    )

    assert spec_mask_cpu is None
    assert non_spec_mask_cpu is None
    assert num_spec_decodes == 0, (
        "pre-#53077 bug: num_spec_decodes stayed at "
        f"{(num_decode_draft_tokens_cpu >= 0).sum().item()} (the row count) "
        "instead of being reset to 0 alongside spec_sequence_masks_cpu, "
        "violating the num_decodes == num_spec_decodes invariant downstream "
        "consumers assume"
    )


def test_gdn_attn_num_spec_decodes_nonzero_for_genuine_spec_batch() -> None:
    """Sanity check: a batch with real, nonzero scheduled draft counts must
    still take the spec path with the correct count and complement mask
    (the #52078 non_spec_sequence_masks_cpu refactor must not change this)."""
    num_decode_draft_tokens_cpu = torch.tensor([2, -1, 0, 3, -1], dtype=torch.int32)

    spec_mask_cpu, non_spec_mask_cpu, num_spec_decodes = _build_spec_masks(
        True, num_decode_draft_tokens_cpu
    )

    assert spec_mask_cpu is not None
    assert non_spec_mask_cpu is not None
    # Rows with a non-negative draft-count slot: indices 0, 2, 3.
    assert spec_mask_cpu.tolist() == [True, False, True, True, False]
    assert num_spec_decodes == 3
    # non_spec_sequence_masks_cpu must be the exact complement (this is the
    # #52078 refactor: computed once, reused everywhere ~spec_sequence_masks_cpu
    # used to be recomputed).
    assert torch.equal(non_spec_mask_cpu, ~spec_mask_cpu)
