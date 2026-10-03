# SPDX-License-Identifier: Apache-2.0
"""Lane K6, lead M (L91): intra-GPU prefill/decode multiplexing. Plumbing only. Env-gated, default OFF.

Idea (design: tools/k6/DESIGN-M.md): long prefill chunks run in a background "prefill lane" (its own
thread, its own low-priority CUDA stream, its own TP communicator) while the engine's main loop keeps
replaying decode-only CUDA graphs on a high-priority stream. Both lanes share the weights, the KV
pool and the prefix cache; they touch disjoint KV blocks / GDN state slots.

This module holds the per-thread lane state that the hooks consult:
  * forward context   (vllm/forward_context.py)      -> each lane has its own ForwardContext
  * TP group          (vllm/distributed/parallel_state.py: get_tp_group + the compiled all_reduce /
                       all_gather / reduce_scatter custom ops resolve the lane's own GroupCoordinator,
                       so at most ONE collective is in flight per communicator, K3's constraint)
  * Marlin workspace  (mixed_precision/marlin.py)    -> the prefill lane uses its own lock workspace;
                       two lanes running the same layer would otherwise share the split-K locks
  * stream priorities -> decode lane at the device's greatest priority, never below K3's all-reduce
                       side stream (VLLM_K3_AR_OVERLAP_PRIO, default -1); prefill lane at 0

  VLLM_K6_MUX=1                     enable the plumbing (hooks become lane-aware)
  VLLM_K6_MUX_DECODE_PRIO=<int>     decode-lane stream priority (default: greatest the device offers)
  VLLM_K6_MUX_PREFILL_PRIO=<int>    prefill-lane stream priority (default 0 = lowest)

With VLLM_K6_MUX unset every hook is a single module-level bool test and behaviour is unchanged.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Any

ENABLED: bool = os.environ.get("VLLM_K6_MUX", "0") == "1"

DECODE = "decode"
PREFILL = "prefill"

_TLS = threading.local()
_UNSET = object()


def current_lane() -> str:
    return getattr(_TLS, "lane", DECODE)


def in_prefill_lane() -> bool:
    return getattr(_TLS, "lane", DECODE) == PREFILL


@contextmanager
def lane(name: str, *, groups: dict[str, Any] | None = None, stream=None):
    """Run the body as lane ``name`` on this thread.

    groups: {original GroupCoordinator.unique_name: lane GroupCoordinator}, e.g. {"tp:0": tp_b}.
    stream: optional torch.cuda.Stream made current for the body.
    """
    prev = (getattr(_TLS, "lane", DECODE), getattr(_TLS, "groups", None), getattr(_TLS, "fc", _UNSET))
    _TLS.lane = name
    _TLS.groups = dict(groups) if groups else None
    _TLS.fc = None
    try:
        if stream is not None:
            import torch

            with torch.cuda.stream(stream):
                yield
        else:
            yield
    finally:
        _TLS.lane, _TLS.groups = prev[0], prev[1]
        if prev[2] is _UNSET:
            if hasattr(_TLS, "fc"):
                del _TLS.fc
        else:
            _TLS.fc = prev[2]


# ---- forward context (per lane) -------------------------------------------------------------------
def lane_forward_context():
    return getattr(_TLS, "fc", None)


def set_lane_forward_context(ctx) -> Any:
    prev = getattr(_TLS, "fc", None)
    _TLS.fc = ctx
    return prev


# ---- process groups (per lane) --------------------------------------------------------------------
def group_override(unique_name: str):
    g = getattr(_TLS, "groups", None)
    return g.get(unique_name) if g else None


def resolve_group(unique_name: str, default):
    """Lane-aware lookup used by parallel_state's compiled collective ops."""
    g = getattr(_TLS, "groups", None)
    if g:
        o = g.get(unique_name)
        if o is not None:
            return o
    return default


# ---- Marlin lock workspace (per lane) -------------------------------------------------------------
_LANE_WS: dict[tuple[str, int], Any] = {}
_LANE_WS_LOCK = threading.Lock()


def marlin_workspace(default):
    """Decode lane (main thread) keeps the layer's own workspace (its CUDA graphs captured that
    address). Any other lane gets one private workspace per (lane, original workspace): the
    workspace holds split-K reduction locks, so concurrent GEMMs must not share it."""
    name = getattr(_TLS, "lane", DECODE)
    if name == DECODE:
        return default
    key = (name, id(default))
    ws = _LANE_WS.get(key)
    if ws is None:
        with _LANE_WS_LOCK:
            ws = _LANE_WS.get(key)
            if ws is None:
                ws = default.new_zeros(default.shape)
                _LANE_WS[key] = ws
    return ws


# ---- stream priorities -----------------------------------------------------------------------------
def k3_priority() -> int:
    return int(os.environ.get("VLLM_K3_AR_OVERLAP_PRIO", "-1"))


def choose_priorities(priority_range: tuple[int, int], decode: int | None = None,
                      prefill: int | None = None, k3: int | None = None) -> dict[str, int]:
    """Pure helper. CUDA: lower number = higher priority; torch reports (least, greatest), e.g. (0, -5).
    Order enforced: decode <= k3 (decode at least as urgent as K3's prefill all-reduce side stream)
    and prefill >= k3. Values are clamped into the device range."""
    least, greatest = priority_range
    k3 = k3_priority() if k3 is None else k3

    def clamp(p: int) -> int:
        return max(greatest, min(least, p))

    if decode is None:
        env = os.environ.get("VLLM_K6_MUX_DECODE_PRIO")
        decode = int(env) if env is not None else greatest
    if prefill is None:
        prefill = int(os.environ.get("VLLM_K6_MUX_PREFILL_PRIO", str(least)))
    decode, prefill, k3c = clamp(decode), clamp(prefill), clamp(k3)
    if decode > k3c:  # decode would sit below K3's comm stream: raise it to K3's level
        decode = k3c
    if prefill < k3c:  # prefill compute must not outrank its own all-reduce side stream
        prefill = k3c
    return {"decode": decode, "k3_allreduce": k3c, "prefill": prefill}
