# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[FORK] Boot-time guard for ``VLLM_MTP_DRAFT_CAP`` (M-7).

Kept torch-free on purpose, same rationale as ``spec_decode_workspace.py``: the
pure check (:func:`validate_mtp_draft_cap`) is unit-tested directly (loaded by
path) without importing torch/CUDA. The config-facing call site lives in
``vllm/config/speculative.py`` (``SpeculativeConfig._verify_args``) and only
passes scalars in.

Root cause (M-7, empirically proven on prod hardware 2026-08-23, 7-experiment
bisect)
-----------------------------------------------------------------------------
``VLLM_MTP_DRAFT_CAP`` (read in ``llm_base_proposer.SpecDecodeBaseProposer.
__init__``) lets the learned drafter (MTP/EAGLE) chain a *shorter* prefix than
``num_speculative_tokens`` (K) so a CPU suffix-tree overlay can still propose
full-K-wide drafts on top of it. In practice only ``cap == K`` has ever worked;
every other relationship between the two is fatal, but the two failure modes
look nothing alike and neither points at the env var:

* ``cap < K`` (measured: cap=2, K=3): a per-step ``RuntimeError`` -- "The size
  of tensor a (3) must match the size of tensor b (2)" -- inside
  ``GPUModelRunner._copy_draft_token_ids_to_cpu``
  (``vllm/v1/worker/gpu_model_runner.py``). The pinned CPU draft-token buffer
  and the GPU draft-token tensor end up sized from different widths (one from
  K, one from the capped proposer width) and the async ``.copy_()`` between
  them fails immediately -- loud, but hours into a serving session, not at
  boot.
* ``cap == K`` (measured: cap=3, K=3): no cap in effect (min(K, cap) == K), so
  this is silently equivalent to leaving ``VLLM_MTP_DRAFT_CAP`` unset. Fine,
  but pointless to set.
* ``cap > K``: never validated against K anywhere, so a stale/typo'd cap wider
  than K silently proposes/copies extra width -- observed as a decode
  deadlock (requests accepted, 0 tok/s forever, no errors) rather than a crash.

Because both broken shapes surface late (mid-run RuntimeError, or a silent
hang with zero diagnostic signal) and far from the env var that caused them,
the fix is to make the *only* supported relationship (cap unset, or
cap == K) the only one that can boot at all. This module does not attempt to
make cap < K or cap > K actually work -- that is explicitly out of scope; see
``docs/`` for the wider MTP-cap design notes.
"""

from __future__ import annotations


def validate_mtp_draft_cap(raw_cap: str | None, num_speculative_tokens: int) -> None:
    """Raise ``ValueError`` if ``VLLM_MTP_DRAFT_CAP`` is set to anything other
    than ``num_speculative_tokens`` (K).

    ``raw_cap`` is the raw string value as read from the environment (or
    ``None``/``""`` if unset) -- pass ``os.environ.get("VLLM_MTP_DRAFT_CAP")``
    directly. Parsing mirrors ``SpecDecodeBaseProposer.__init__``
    (``int(os.environ.get("VLLM_MTP_DRAFT_CAP", "0"))``, where ``0`` means "no
    cap") so this guard and the runtime cap agree on what "set" means:
    unset, ``""``, and ``"0"`` are all no-ops here.

    Args:
        raw_cap: the raw ``VLLM_MTP_DRAFT_CAP`` env var value, or ``None``.
        num_speculative_tokens: the resolved speculative-config K.

    Raises:
        ValueError: if the cap is set (> 0) and does not equal K.
    """
    if raw_cap is None or raw_cap == "":
        return
    cap = int(raw_cap)
    if cap <= 0:
        return
    if cap != num_speculative_tokens:
        raise ValueError(
            f"VLLM_MTP_DRAFT_CAP={cap} is set but does not equal "
            f"num_speculative_tokens (K)={num_speculative_tokens}. [M-7] "
            "This combination is fatal, not merely suboptimal: cap < K raises "
            "a per-step RuntimeError in _copy_draft_token_ids_to_cpu "
            "(draft-token tensor width mismatch), and cap > K silently "
            "deadlocks decode (requests accepted, 0 tok/s forever, no "
            "errors) -- both empirically reproduced on prod hardware. Only "
            "cap == K has ever worked, which makes the cap a no-op anyway. "
            f"Unset VLLM_MTP_DRAFT_CAP, or set it to exactly "
            f"{num_speculative_tokens}."
        )
