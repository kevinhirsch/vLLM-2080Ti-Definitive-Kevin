# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""F-5 (backlog RANK 2): pure-Python mamba align-mode block-boundary
detection, shared by:

  1. ``vllm.v1.worker.mamba_utils.preprocess_mamba`` (the real per-step
     running-state block bookkeeping this module's formulas are extracted
     from -- see ``compute_curr_state_idx``/``crosses_boundary`` docstrings
     for the exact line-for-line correspondence), and
  2. the opt-in ``VLLM_F5_REFUSE_CAPTURE_ON_ACCEPTANCE_BOUNDARY`` mitigation
     wired into ``gpu_model_runner.py``'s decode cudagraph dispatch.

Why this exists (see LANE/MAP.md and LANE/PROBE.md in the f5-forced-k lane
for the full investigation): under ``VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1``
(FULL decode cudagraphs + MTP spec-decode + Mamba/GDN hybrid models,
``--mamba-cache-mode align``), a decode step whose running-state block is
promoted this step (``preprocess_mamba`` resets that request's
``num_accepted_tokens`` to the align-mode "already copied" neutral value of
1) shares its GPU-resident ``num_accepted_tokens`` buffer with the GDN/Mamba
recurrent kernels' own K+1-way "fan-out then select" state indexing
(``fused_sigmoid_gating.py`` / ``fused_recurrent.py`` /
``mamba_ssm.py`` / ``causal_conv1d.py``: ``i_t = num_accepted_tokens[i_n] -
1`` selects which of K+1 candidate state slots to read as this step's
initial state). Both mechanisms independently derive "which mamba cache
block is the current one" from the same underlying quantities
(``num_computed_tokens``, ``block_size``) but via two separately-maintained
formulas, and nothing at runtime cross-checks that they agree. A step where
a request's running-state block boundary is crossed is exactly the step
where that agreement matters; this module lets both the diagnostic trace
and the opt-in mitigation identify such steps *before* the forward pass
runs, using only data already available at cudagraph-dispatch time (no GPU
access, no forward pass, no new synchronization).

Every function here is pure Python on plain ints/dicts/sequences -- no
torch import, no CUDA context required -- so it is fully unit-testable on
CPU-only hardware (see ``tests/2080ti/test_f5_accept_boundary.py``). This
is deliberate: the actual runtime hazard this module reasons about can only
be confirmed or refuted by the forced-K probe (PROBE.md) on real GPU
hardware; what is proven here is only that the *shape/keying logic* matches
the production formula it is extracted from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from vllm.utils.math_utils import cdiv


def compute_curr_state_idx(
    num_computed_tokens: int,
    num_scheduled_tokens: int,
    block_size: int,
    num_speculative_blocks: int,
) -> int:
    """Which mamba-cache block index holds "the" running state for a
    request after this step's tokens are computed, under
    ``mamba_cache_mode="align"``.

    Extracted verbatim (arithmetically) from
    ``vllm.v1.worker.mamba_utils.preprocess_mamba``::

        num_blocks = (
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size)
            + num_speculative_blocks
        )
        curr_state_idx = num_blocks - 1 - num_speculative_blocks

    which simplifies to the one-liner below (the ``+ num_speculative_blocks``
    /``- num_speculative_blocks`` cancel; kept as a separate parameter here
    anyway so a future change to either side of that cancellation in the
    original function is easy to notice and re-derive against this copy).
    """
    num_blocks = (
        cdiv(num_computed_tokens + num_scheduled_tokens, block_size)
        + num_speculative_blocks
    )
    return num_blocks - 1 - num_speculative_blocks


def crosses_boundary(prev_state_idx: int | None, curr_state_idx: int) -> bool:
    """Whether this step promotes the request's running-state block.

    Mirrors ``mamba_utils.py``'s
    ``if prev_state_idx != -1 and prev_state_idx != curr_state_idx:`` guard
    exactly, with ``None`` standing in for that function's "new/resumed
    request" sentinel (``mamba_state_idx.get(req_id)`` returning ``None``,
    which ``preprocess_mamba`` then replaces with a synthetic
    ``(num_computed_tokens - 1) // block_size`` -- callers of this function
    that have that synthetic value on hand should pass it as
    ``prev_state_idx`` directly; ``None`` here only covers the case where no
    prior value is available to the caller at all).
    """
    if prev_state_idx is None or prev_state_idx == -1:
        return False
    return prev_state_idx != curr_state_idx


def request_crosses_boundary_this_step(
    prev_state_idx: int | None,
    num_computed_tokens: int,
    num_scheduled_tokens: int,
    block_size: int,
    num_speculative_blocks: int,
) -> tuple[bool, int]:
    """Convenience wrapper: returns ``(crosses, curr_state_idx)`` for one
    request. ``curr_state_idx`` is returned alongside the boolean so callers
    (the trace, in particular) can log it without recomputing.
    """
    curr_state_idx = compute_curr_state_idx(
        num_computed_tokens, num_scheduled_tokens, block_size, num_speculative_blocks
    )
    return crosses_boundary(prev_state_idx, curr_state_idx), curr_state_idx


def batch_has_pending_boundary_crossing(
    req_ids: Sequence[str],
    num_computed_tokens_by_req: Mapping[str, int],
    num_scheduled_tokens_by_req: Mapping[str, int],
    mamba_state_idx: Mapping[str, int],
    block_size: int,
    num_speculative_blocks: int,
) -> bool:
    """Batch-level OR: would *any* scheduled request cross its mamba
    running-state block boundary if the step ran right now?

    This is the direct input to the ``VLLM_F5_REFUSE_CAPTURE_ON_ACCEPTANCE_BOUNDARY``
    mitigation: when true, the caller downgrades this step's cudagraph
    dispatch away from FULL (see ``gpu_model_runner.py``'s
    ``dispatch_cudagraph`` call site) rather than replay a captured graph on
    a step this lane's investigation flags as the hazard window for H1 in
    PROBE.md. It intentionally errs conservative (batch-wide, not
    per-request) because cudagraph dispatch is a whole-batch decision in
    this codebase -- there is no per-request graph selection to fall back
    to.

    A request missing from ``mamba_state_idx`` (new/resumed, no prior
    recorded block) never counts as crossing here, matching
    ``preprocess_mamba``'s own treatment: a request's first mamba step has
    nothing to have been "captured from" yet, so there is no stale-slot
    hazard for it specifically (mirrors ``prev_state_idx == -1`` in
    ``crosses_boundary`` above; ``preprocess_mamba`` synthesizes
    ``(num_computed_tokens - 1) // block_size`` for brand-new requests
    instead of leaving it absent, but that synthetic value by construction
    can never disagree with the same step's freshly-derived
    ``curr_state_idx`` when ``num_computed_tokens`` reflects zero prior
    decode steps, so treating "absent" as "does not cross" is equivalent
    here, not just a simplification).
    """
    for req_id in req_ids:
        if req_id not in num_computed_tokens_by_req:
            continue  # not a scheduled/known request; nothing to check
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            continue
        crosses, _ = request_crosses_boundary_this_step(
            prev_state_idx,
            num_computed_tokens_by_req[req_id],
            num_scheduled_tokens_by_req.get(req_id, 0),
            block_size,
            num_speculative_blocks,
        )
        if crosses:
            return True
    return False
