# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""F-5 (backlog RANK 2): forced-K probe instrumentation for the FULL
cudagraph + MTP spec-decode + Mamba/GDN "doubled token" defect.

See LANE/MAP.md and LANE/PROBE.md (f5-forced-k lane,
``/home/kevin/projects/lanes/f5-forced-k/``) for the full investigation
this supports. Short version: under
``VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1`` the captured recurrent-state
update for hybrid Mamba/GDN models may go stale under varying speculative-
acceptance patterns; this module logs, per decode step, exactly which
mamba running-state block boundary events and which K+1-way fan-out
selections occurred and whether the step was cudagraph-replayed or eager,
so a forced-K probe run can attribute an observed doubled/repeated output
token to a specific tensor/slot rather than re-deriving the mechanism from
source alone.

Style follows ``vllm/v1/worker/xid31_trace.py``: a pure diagnostic, NO-OP
unless ``VLLM_F5_PROBE_TRACE`` is truthy (checked once at import, so the
hot path pays nothing beyond one boolean test per call site when disabled),
log-only, never raises out of its own code, and safe to leave staged
permanently. Nothing here changes engine numerics; every public entry
point early-returns when disabled.

Output is newline-delimited JSON (one event object per line) appended to
``VLLM_F5_PROBE_TRACE_PATH``, so a probe run can be parsed with a one-line
``jsonlines``/``json.loads`` loop -- no bespoke log-scraping. Event
records are intentionally NOT merged into one big per-step object inside
the engine process (that would require threading extra state through
several unrelated call sites); a `step` counter present on every event
lets offline analysis (PROBE.md "Analysis procedure") join them back up
per decode step.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    try:
        return bool(int(os.getenv(name, default)))
    except (TypeError, ValueError):
        return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# Read once at import; cheap and constant for the process lifetime, matching
# xid31_trace.py's convention.
ENABLED: bool = _env_flag("VLLM_F5_PROBE_TRACE", "0")
_PATH: str = os.getenv("VLLM_F5_PROBE_TRACE_PATH", "/tmp/f5_probe_trace.jsonl")

_lock = threading.Lock()
_step: int = 0
_installed: bool = False


def _write(record: dict[str, Any]) -> None:
    """Best-effort append of one JSON line. Never raises: a diagnostic must
    not be able to take the engine down, and a torn/missing trace line is
    vastly preferable to a crashed decode step."""
    record.setdefault("t", time.time())
    try:
        line = json.dumps(record, default=str)
    except (TypeError, ValueError) as e:
        logger.debug("f5_probe_trace: failed to serialize %r: %s", record, e)
        return
    try:
        with _lock, open(_PATH, "a") as f:
            f.write(line)
            f.write("\n")
    except OSError as e:
        logger.debug("f5_probe_trace: failed to write %s: %s", _PATH, e)


def install(mamba_block_size: int | None, num_spec: int) -> None:
    """Log a header event once (idempotent). Cheap enough to call
    unconditionally from engine init; real cost is one `if ENABLED` test
    when disabled.
    """
    global _installed
    if not ENABLED or _installed:
        return
    _installed = True
    _write(
        {
            "event": "init",
            "mamba_block_size": mamba_block_size,
            "num_speculative_tokens": num_spec,
            "pid": os.getpid(),
        }
    )
    logger.info(
        "F-5 probe trace ARMED (VLLM_F5_PROBE_TRACE=1): writing to %s. "
        "This is a diagnostic for the FULL-cudagraph+MTP doubled-token "
        "investigation; do not leave enabled on a production route "
        "(unbounded file growth, per-decode-step file I/O).",
        _PATH,
    )


def next_step() -> int:
    """Advance and return the monotonic decode-step counter. Called once
    per forward step (Hook A, at cudagraph dispatch time); later hooks in
    the same step read `current_step()` instead of advancing again, so all
    events for one step share a `step` value.
    """
    global _step
    _step += 1
    return _step


def current_step() -> int:
    return _step


def log_dispatch(
    cudagraph_mode: str,
    num_tokens: int,
    num_reqs: int,
    uniform: bool,
) -> None:
    """Hook A: right after ``dispatch_cudagraph()`` decides this step's
    replay mode (``gpu_model_runner.py``, ``_determine_batch_execution_and_padding``).
    Advances the step counter -- call exactly once per forward step.
    """
    if not ENABLED:
        return
    step = next_step()
    _write(
        {
            "event": "dispatch",
            "step": step,
            "cudagraph_mode": cudagraph_mode,
            "num_tokens": num_tokens,
            "num_reqs": num_reqs,
            "uniform": uniform,
        }
    )


def log_preprocess(
    req_id: str,
    prev_state_idx: int,
    curr_state_idx: int,
    boundary_crossed: bool,
    accepted_pre: int,
    accepted_post: int,
    block_size: int,
    num_speculative_blocks: int,
) -> None:
    """Hook B: inside ``mamba_utils.preprocess_mamba``'s per-request loop,
    after computing this step's running-state block and applying the
    align-mode neutral-value reset (if any). ``accepted_pre`` /
    ``accepted_post`` differ iff the reset fired this step for this
    request -- see PROBE.md H1.
    """
    if not ENABLED:
        return
    _write(
        {
            "event": "preprocess",
            "step": current_step(),
            "req_id": req_id,
            "prev_state_idx": prev_state_idx,
            "curr_state_idx": curr_state_idx,
            "boundary_crossed": boundary_crossed,
            "accepted_pre": accepted_pre,
            "accepted_post": accepted_post,
            "block_size": block_size,
            "num_speculative_blocks": num_speculative_blocks,
        }
    )


def log_fanout(
    num_accepted_tokens: list[int],
    state_indices: list[list[int]],
) -> None:
    """Hook C: right before the GDN/Mamba recurrent kernel call
    (``gdn_linear_attn.py``, immediately before
    ``fused_sigmoid_gating_delta_rule_update``), logging exactly what that
    kernel launch is about to read: every spec-decode row's K+1-wide
    fan-out row and the accepted-count that selects which column is read
    as the initial state (``i_t = num_accepted_tokens[row] - 1``).

    Logged as one record per step covering all rows (rather than one call
    per request) because at this call site rows are identified only by
    position within the spec-decode sub-batch -- ``GDNAttentionMetadata``
    carries no request-id mapping this deep into the mixer forward. Row
    index is therefore only unambiguous against Hook B's (req_id-keyed)
    events when exactly one spec-decode request is active this step, e.g.
    PROBE.md's single-stream probe arms. Cross-batch (concurrent multi-
    request) attribution would need a request-id to be threaded down
    through ``attn_metadata`` -- not done here to keep this a read-only,
    additive probe rather than a metadata-shape change.

    Callers should pass already-materialized Python values (``.tolist()``),
    not GPU tensors -- this module intentionally has no torch dependency,
    both to stay import-light and to force the (cheap, small: at most
    ``max_num_seqs x (K+1)`` ints) device-to-host copy to happen at the
    call site where the caller already knows whether it's on a hot path.
    """
    if not ENABLED:
        return
    rows = []
    for row_index, (accepted, row) in enumerate(
        zip(num_accepted_tokens, state_indices)
    ):
        select_index = accepted - 1
        select_block_id = row[select_index] if 0 <= select_index < len(row) else None
        rows.append(
            {
                "row_index": row_index,
                "num_accepted_tokens": accepted,
                "state_indices_row": row,
                "fanout_select_index": select_index,
                "fanout_select_block_id": select_block_id,
            }
        )
    _write(
        {
            "event": "fanout",
            "step": current_step(),
            "rows": rows,
        }
    )
