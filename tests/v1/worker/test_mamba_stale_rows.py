# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for the vllm-project/vllm#51508 fork port (commit
b2e84e2, "GDN: port vllm-project#51508 -- skip stale zero-accept async
rows").

A stale async spec-decode row reports ``num_accepted_tokens <= 0``. Two sites
were patched to skip such rows instead of silently corrupting state (search
for ``[FORK] vllm-project/vllm#51508`` / ``[FORK] Port of vllm-project/vllm
#51508``):

* ``vllm/v1/worker/mamba_utils.py`` -- ``preprocess_mamba`` and
  ``postprocess_mamba`` each ``continue`` past a stale row before it can be
  folded into the Mamba running-state copy bookkeeping.
* ``vllm/v1/attention/backends/gdn_attn.py`` (~lines 317-337, inside
  ``GdnAttentionMetadataBuilder.build``) -- nulls the stale row's
  ``spec_state_indices_tensor`` entry instead of indexing with ``count - 1 ==
  -1`` (a *valid* Python index that would silently read the wrong/neighbour
  block).

The ``mamba_utils`` tests below call the real functions with SimpleNamespace
stubs (following the style in
``tests/v1/kv_connector/unit/test_mamba_cpu_offload_compat.py``) and a spy in
place of ``collect_mamba_copy_meta`` / ``do_mamba_copy_block`` (both of which
otherwise fan out into GPU-only Triton memcpy kernels). The ``gdn_attn``
test does not import ``GdnAttentionMetadataBuilder.build`` -- it is heavy and
CUDA-only -- and instead replicates the masking expression verbatim as a
contract test on small CPU tensors.
"""

from types import SimpleNamespace

import torch

from vllm.v1.worker import mamba_utils


def _make_copy_bufs(block_size: int, num_speculative_blocks: int = 0):
    """Minimal stand-in for MambaCopyBuffers: only the fields preprocess_mamba
    / postprocess_mamba read directly (the rest is forwarded, opaque, into
    the mocked collect_mamba_copy_meta)."""
    return SimpleNamespace(
        offset=0,
        mamba_group_ids=[0],
        mamba_spec=SimpleNamespace(
            block_size=block_size,
            num_speculative_blocks=num_speculative_blocks,
        ),
    )


def _install_collect_spy(monkeypatch):
    """Patch the module-global `collect_mamba_copy_meta` / `do_mamba_copy_block`
    names that preprocess_mamba/postprocess_mamba call unqualified, so the
    real (GPU-only) memcpy machinery never runs. Returns the list the spy
    appends call records to."""
    calls: list[dict] = []

    def _spy_collect(
        copy_bufs,
        kv_cache_config,
        mamba_state_copy_funcs,
        mamba_group_ids,
        src_block_idx,
        dest_block_idx,
        accept_token_bias,
        req_state,
        forward_context,
    ):
        calls.append(
            dict(src=src_block_idx, dest=dest_block_idx, bias=accept_token_bias)
        )

    monkeypatch.setattr(mamba_utils, "collect_mamba_copy_meta", _spy_collect)
    monkeypatch.setattr(mamba_utils, "do_mamba_copy_block", lambda copy_bufs: None)
    return calls


def test_preprocess_mamba_skips_stale_row_but_processes_normal_row(monkeypatch):
    """Batch of two requests, one stale (num_accepted_tokens == 0) and one
    normal (num_accepted_tokens == 3). Both have a previous running-state
    block (index 0) distinct from this step's block (index 1), so *if*
    processed, both would call collect_mamba_copy_meta.

    Expected: only the normal row is collected; the stale row's
    num_accepted_tokens_cpu entry is left untouched (not reset to 1); the
    normal row's entry is reset to 1 after being folded into the block copy.
    """
    calls = _install_collect_spy(monkeypatch)

    block_size = 4
    copy_bufs = _make_copy_bufs(block_size=block_size)

    scheduler_output = SimpleNamespace(
        finished_req_ids=(),
        preempted_req_ids=(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=()),
        num_scheduled_tokens={"req_normal": 4, "req_stale": 4},
    )
    cache_config = SimpleNamespace(enable_prefix_caching=True)
    mamba_state_idx = {"req_normal": 0, "req_stale": 0}
    input_batch = SimpleNamespace(
        req_ids=["req_normal", "req_stale"],
        num_accepted_tokens_cpu=[3, 0],  # req_stale is the stale (<=0) row
    )
    requests = {
        "req_normal": SimpleNamespace(num_computed_tokens=4),
        "req_stale": SimpleNamespace(num_computed_tokens=4),
    }

    mamba_utils.preprocess_mamba(
        scheduler_output=scheduler_output,
        kv_cache_config=None,
        cache_config=cache_config,
        mamba_state_idx=mamba_state_idx,
        input_batch=input_batch,
        requests=requests,
        forward_context={},
        mamba_state_copy_funcs=(),
        copy_bufs=copy_bufs,
    )

    # Only the normal row triggered a state-copy collection.
    assert len(calls) == 1, f"expected exactly 1 collect call, got {calls}"
    assert calls[0] == dict(src=0, dest=1, bias=2)  # bias = accepted(3) - 1

    # Stale row: num_accepted_tokens_cpu left untouched (no copy collected /
    # bookkeeping not mutated for a discarded step).
    assert input_batch.num_accepted_tokens_cpu[1] == 0

    # Normal row: reset to 1 after being folded into the running state.
    assert input_batch.num_accepted_tokens_cpu[0] == 1

    # The *position* bookkeeping (mamba_state_idx) is updated unconditionally
    # for both rows -- only the state COPY is skipped for the stale row.
    assert mamba_state_idx == {"req_normal": 1, "req_stale": 1}


def test_postprocess_mamba_skips_stale_row_but_processes_normal_row(monkeypatch):
    """Same shape for postprocess_mamba. The stale row deliberately has no
    `mamba_state_idx` entry: the guard must `continue` before
    `mamba_state_idx[req_id]` is ever read, so a missing entry must not raise
    KeyError.
    """
    calls = _install_collect_spy(monkeypatch)

    block_size = 4
    copy_bufs = _make_copy_bufs(block_size=block_size)

    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req_normal": 4, "req_stale": 4},
        scheduled_spec_decode_tokens={},
    )
    mamba_state_idx = {"req_normal": 0}  # no entry for req_stale on purpose
    input_batch = SimpleNamespace(
        req_ids=["req_normal", "req_stale"],
        num_accepted_tokens_cpu=[3, 0],
    )
    requests = {
        "req_normal": SimpleNamespace(num_computed_tokens=2),
        "req_stale": SimpleNamespace(num_computed_tokens=2),
    }

    mamba_utils.postprocess_mamba(
        scheduler_output=scheduler_output,
        kv_cache_config=None,
        input_batch=input_batch,
        requests=requests,
        mamba_state_idx=mamba_state_idx,
        forward_context={},
        mamba_state_copy_funcs=(),
        copy_bufs=copy_bufs,
    )

    # num_tokens_running_state = 2 + 4 - 0 = 6
    # new_num_computed_tokens  = 6 + 3 - 1 = 8
    # aligned_new_computed_tokens = 8 // 4 * 4 = 8  (>= 6 -> collected)
    # accept_token_bias = 8 - 6 = 2; dest_block_idx = 8 // 4 - 1 = 1
    assert len(calls) == 1, f"expected exactly 1 collect call, got {calls}"
    assert calls[0] == dict(src=0, dest=1, bias=2)

    # Stale row untouched, and no KeyError despite the missing
    # mamba_state_idx entry -- proof the skip happens before that lookup.
    assert input_batch.num_accepted_tokens_cpu[1] == 0
    assert "req_stale" not in mamba_state_idx


def test_gdn_attn_stale_row_null_masking_contract():
    """Contract test for the masking expression in
    ``vllm/v1/attention/backends/gdn_attn.py`` (~lines 317-337, the
    ``[FORK] Port of vllm-project/vllm#51508`` block)::

        stale_rows = num_accepted_tokens <= 0
        if bool(stale_rows.any()):
            spec_state_indices_tensor[
                stale_rows.to(spec_state_indices_tensor.device)
            ] = NULL_BLOCK_ID

    This intentionally does NOT import/exercise
    ``GdnAttentionMetadataBuilder.build`` (heavy, CUDA-only). It replicates
    the expression verbatim on small CPU tensors so the masking semantics
    stay covered without a GPU. If the real expression in gdn_attn.py
    changes, this test must be updated to match it.
    """
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    num_accepted_tokens = torch.tensor([2, 0, 1], dtype=torch.int32)
    spec_state_indices_tensor = torch.arange(1, 13, dtype=torch.int64).reshape(3, 4)
    original = spec_state_indices_tensor.clone()

    # --- verbatim contract expression from gdn_attn.py ---
    stale_rows = num_accepted_tokens <= 0
    if bool(stale_rows.any()):
        spec_state_indices_tensor[
            stale_rows.to(spec_state_indices_tensor.device)
        ] = NULL_BLOCK_ID
    # --- end contract expression ---

    assert bool(stale_rows.tolist() == [False, True, False])

    # Row 1 (num_accepted_tokens == 0) is fully nulled.
    assert torch.all(spec_state_indices_tensor[1] == NULL_BLOCK_ID)
    # Rows 0 and 2 (counts 2 and 1, both > 0) are untouched.
    assert torch.equal(spec_state_indices_tensor[0], original[0])
    assert torch.equal(spec_state_indices_tensor[2], original[2])


def test_gdn_attn_stale_row_null_masking_contract_no_stale_rows_is_noop():
    """Same contract expression, no stale rows: the tensor must be left
    completely untouched."""
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    num_accepted_tokens = torch.tensor([2, 1, 3], dtype=torch.int32)
    spec_state_indices_tensor = torch.arange(1, 13, dtype=torch.int64).reshape(3, 4)
    original = spec_state_indices_tensor.clone()

    stale_rows = num_accepted_tokens <= 0
    if bool(stale_rows.any()):
        spec_state_indices_tensor[
            stale_rows.to(spec_state_indices_tensor.device)
        ] = NULL_BLOCK_ID

    assert torch.equal(spec_state_indices_tensor, original)
