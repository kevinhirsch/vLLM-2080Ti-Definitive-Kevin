# SPDX-License-Identifier: Apache-2.0
"""[FORK][LANE EF] Cheap attribution fences for mixed prefill+spec-decode steps.

Asynchronous CUDA faults (Xid 31 / cudaErrorIllegalAddress) are reported at the
next host sync, which is usually many kernels (and layers) after the faulting
one: the 2026-10-01 20:45:58 fault surfaced at a ``seq_lens.tolist()`` in the
TurboQuant mixed-batch branch. ``fence()`` drains the stream at chosen points
of steps that contain prefill tokens (long steps, ~seconds, so a few hundred
microseconds per fence is noise) and, if the context is already poisoned,
logs WHICH stage first observed it plus the batch shape before re-raising.

Pure-decode steps (the latency-critical hot path) are never fenced.
Disable with VLLM_EF_FENCE=0.
"""

import os
from collections.abc import Callable
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_EF_FENCE", "1") != "0"
_count = 0


def fence(tag: str, ctx: Callable[[], Any] | None = None) -> None:
    global _count
    if not ENABLED:
        return
    try:
        torch.cuda.synchronize()
        _count += 1
    except Exception as exc:  # noqa: BLE001 - re-raised below
        try:
            detail = ctx() if ctx is not None else None
        except Exception:  # pragma: no cover
            detail = "<ctx failed>"
        logger.error("EF-FENCE first observed CUDA fault at %s: %s ctx=%s", tag, exc, detail)
        raise
