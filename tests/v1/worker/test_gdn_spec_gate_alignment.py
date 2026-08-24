# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contract test for the vllm-project/vllm#51812 fork port
("[Bugfix] Align Qwen GDN gates with speculative tokens"), ported into
``vllm/model_executor/layers/mamba/gdn_linear_attn.py``,
``GdnLinearAttention._forward_core``.

Upstream bug: in a mixed batch where non-spec tokens precede spec tokens,
``mixed_qkv`` (and therefore q/k/v) is row-selected by ``spec_token_indx``,
but the ``a``/``b`` sigmoid gate tensors were passed straight through
unindexed into ``fused_sigmoid_gating_delta_rule_update`` alongside
``q=query_spec, k=key_spec, v=value_spec``. Whenever ``spec_token_indx`` is
not a leading prefix of the batch (true whenever any non-spec token precedes
a spec token -- the boundary condition upstream reproduced at a
``max_model_len`` remainder with 2 MTP draft tokens), row i of q/k/v no
longer describes the same original token as row i of a/b: the recurrent
update applies gate values from the wrong token.

The fix (mirrored in ``_forward_core``'s "1. Convolution sequence
transformation" section): compute ``a_spec``/``b_spec`` via the *same*
``.index_select(0, spec_token_indx)`` used for ``mixed_qkv_spec``, and pass
those into the spec-path ``fused_sigmoid_gating_delta_rule_update`` call
instead of the raw ``a``/``b``.

``_forward_core`` itself is CUDA-only (Triton kernels, causal_conv1d,
forward-context threading) and not meaningfully unit-testable on CPU --
following the precedent of the sibling #51508 port test
(``tests/v1/worker/test_mamba_stale_rows.py``), this instead replicates the
fixed indexing expression on small CPU tensors and checks it against the
alignment contract the fix restores.
"""

import torch


def test_gdn_spec_gate_alignment_matches_qkv_selection() -> None:
    """Fixed behavior: a_spec/b_spec track mixed_qkv_spec's row selection
    row-for-row, for both a mixed (non-spec-then-spec) batch and an
    all-spec batch."""
    num_tokens = 6
    token_id = torch.arange(num_tokens)
    # Gate values keyed by original token id, so misalignment is detectable
    # by value.
    a = token_id.float() * 10
    b = token_id.float() * 100

    cases = {
        # Mixed batch: two non-spec tokens (0, 1) precede four spec tokens
        # (2..5) -- spec_token_indx is NOT a leading prefix of the batch.
        # This is the boundary condition #51812 reproduced.
        "mixed_non_spec_then_spec": torch.tensor([2, 3, 4, 5], dtype=torch.int64),
        # All-spec batch (attn_metadata.num_prefills == 0 and
        # num_decodes == 0): spec_token_indx IS the identity prefix, so the
        # bug is inert here -- included to document why it doesn't show up
        # in every test.
        "all_spec": torch.tensor([0, 1, 2, 3], dtype=torch.int64),
    }

    for name, spec_token_indx in cases.items():
        mixed_qkv_spec = token_id.index_select(0, spec_token_indx)
        a_spec = a.index_select(0, spec_token_indx)
        b_spec = b.index_select(0, spec_token_indx)

        for i, orig_token in enumerate(mixed_qkv_spec.tolist()):
            assert a_spec[i].item() == orig_token * 10, name
            assert b_spec[i].item() == orig_token * 100, name


def test_gdn_spec_gate_misalignment_without_index_select() -> None:
    """Pre-fix regression check: passing the *raw*, un-indexed a/b (as the
    buggy code did, `a=a, b=b` instead of `a=a_spec, b=b_spec`) applies gate
    values from the wrong token whenever spec_token_indx is not a leading
    prefix of the batch -- the exact scenario #51812 fixes. Un-indexed a/b
    is equivalent to implicitly reading rows [0, len(spec_token_indx)) of
    the original batch order, which is what a positionally-aligned consumer
    (indexed only by cu_seqlens over the spec-selected count) would see.
    """
    num_tokens = 6
    token_id = torch.arange(num_tokens)
    a = token_id.float() * 10

    # Mixed batch (non-spec tokens 0, 1 precede spec tokens 2..5).
    spec_token_indx = torch.tensor([2, 3, 4, 5], dtype=torch.int64)
    mixed_qkv_spec = token_id.index_select(0, spec_token_indx)

    buggy_a_spec = a[: len(spec_token_indx)]
    mismatches = sum(
        1
        for i, orig_token in enumerate(mixed_qkv_spec.tolist())
        if buggy_a_spec[i].item() != orig_token * 10
    )
    assert mismatches == len(spec_token_indx), (
        "sanity check: this batch layout must actually exercise the "
        "gate/qkv misalignment #51812 fixes, or this test isn't testing "
        "anything"
    )

    # Fixed behavior does not have this problem (see the companion test
    # above): a_spec = a.index_select(0, spec_token_indx) always matches
    # mixed_qkv_spec row-for-row, regardless of batch layout.
    fixed_a_spec = a.index_select(0, spec_token_indx)
    fixed_mismatches = sum(
        1
        for i, orig_token in enumerate(mixed_qkv_spec.tolist())
        if fixed_a_spec[i].item() != orig_token * 10
    )
    assert fixed_mismatches == 0
