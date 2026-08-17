# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-045b: instrumentation to name the max-model-len-gated structure behind
the residual Xid31 (MMU FAULT_PDE) crash class on the 2080Ti turboquant+MTP fork.

The crash fingerprint (from the EXP-045 constraint matrix):

  * Xid31 FAULT_PDE on BOTH GPUs, async (cudagraph-replay OR eager), NOT VRAM
    exhaustion (~3.2 GB free at death), fault DEPTH varies at fixed request size
    (allocator-layout dependent) -> an out-of-bounds device access, not an OOM.
  * The crash tracks ``max_model_len``, NOT the KV pool size: a 449K request is
    CLEAN at ``max-model-len=460800`` (615-741K pool) but CRASHES at
    ``max-model-len=507904`` (smaller 514K pool). So the faulting structure is
    sized/strided by ``max_model_len`` (or ``cdiv(max_model_len, block_size)``),
    NOT by the physical block count.
  * Spec-config dependent: MTP-OFF at 524288 ran 465K/491K clean; MTP-ON crashes.
  * Threshold in ``(460800, 507904]`` -- the fingerprint interval. A candidate
    whose byte-offset / extent does NOT change behaviour across that interval is
    exonerated.

This module is a pure diagnostic. It is a NO-OP unless ``VLLM_TQ_XID31_TRACE`` is
truthy, so it can stay staged in the hot path with negligible overhead. When on it:

  1. At engine init, logs every registered tensor/buffer > ``THRESHOLD_MB`` with
     name / shape / dtype / device / data-ptr range, and computes each buffer's
     worst-case linear BYTE offset -- the quantity a 32-bit kernel index would
     overflow at 2**31 (2 GiB). Buffers whose worst-case byte offset lands within
     ``NEAR_FRAC`` of 2**31 are flagged as live overflow candidates.
  3. During execution, cheaply (every ``EVERY_N`` steps) re-projects the current
     high-water sequence position onto each ``max_model_len``-scaled suspect and
     asserts (log-only, never raises) that the projected offset stays under both
     the buffer's allocated extent and 2**31.
  4. Enables ``torch.cuda.memory._record_memory_history`` and installs an
     excepthook + faulthandler that dump a memory snapshot on any exception
     (including the CUDA error raised after an async Xid31), so the allocation
     that owned the faulting address can be recovered post-mortem.

Nothing here changes engine numerics; every public entry point early-returns when
disabled.
"""

from __future__ import annotations

import os
import sys
import weakref
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# 2 GiB -- the int32 byte-offset boundary. A CUDA/Triton kernel that computes a
# linear address as ``int32 index * stride`` wraps negative here, producing the
# unmapped-page access that surfaces as Xid31 FAULT_PDE.
INT32_BYTE_LIMIT = 2**31  # 2_147_483_648
# Report buffers whose worst-case byte offset is within this fraction of the
# limit -- these are the live overflow candidates for the current max_model_len.
NEAR_FRAC = 0.75


def _env_flag(name: str, default: str = "0") -> bool:
    try:
        return bool(int(os.getenv(name, default)))
    except (TypeError, ValueError):
        return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Read once at import; cheap and constant for the process lifetime.
_ENABLED: bool = _env_flag("VLLM_TQ_XID31_TRACE", "0")
# Log any registered buffer at least this large (MiB).
_THRESHOLD_MB: int = _env_int("VLLM_TQ_XID31_TRACE_MIN_MB", 64)
# Periodic projection cadence, in execute_model steps.
_EVERY_N: int = max(1, _env_int("VLLM_TQ_XID31_TRACE_EVERY_N", 64))
# Where to write the CUDA allocator snapshot on fault.
_SNAPSHOT_PATH: str = os.getenv(
    "VLLM_TQ_XID31_TRACE_SNAPSHOT", "/tmp/xid31_mem_snapshot.pickle"
)

_installed: bool = False
_scanned: bool = False
_step_counter: int = 0
# name -> (weakref(tensor), static_meta_dict)
_registry: "dict[str, tuple[Any, dict[str, Any]]]" = {}


def enabled() -> bool:
    """True iff instrumentation is armed via VLLM_TQ_XID31_TRACE."""
    return _ENABLED


# --------------------------------------------------------------------------- #
# Pure sizing arithmetic (unit-tested; no torch/CUDA needed).
# --------------------------------------------------------------------------- #
def worst_case_byte_offset(
    shape: "tuple[int, ...]",
    strides_elems: "tuple[int, ...]",
    itemsize: int,
) -> int:
    """Largest linear BYTE offset addressable in a strided tensor.

    This is the value a kernel computes for the last element:
    ``sum_i (shape_i - 1) * stride_i * itemsize``. When a kernel accumulates this
    in int32 and it reaches/exceeds 2**31, the address wraps negative -> unmapped
    page -> Xid31 FAULT_PDE. For a contiguous tensor this equals
    ``(numel - 1) * itemsize``.

    Args:
        shape: tensor shape.
        strides_elems: per-dim strides in ELEMENTS (torch ``tensor.stride()``).
        itemsize: bytes per element.
    """
    if not shape:
        return 0
    off_elems = 0
    for dim, stride in zip(shape, strides_elems):
        if dim <= 0:
            return 0
        off_elems += (dim - 1) * stride
    return off_elems * itemsize


def projected_offset_bytes(
    high_water_index: int,
    stride_elems: int,
    itemsize: int,
) -> int:
    """Byte offset a kernel reaches when indexing a ``max_model_len``-scaled row
    stride at a given high-water index (e.g. the deepest per-request block row).

    ``high_water_index * stride_elems * itemsize``. This is the runtime
    projection used to test the fingerprint: it grows with BOTH the request depth
    (``high_water_index``) and ``max_model_len`` (which sets ``stride_elems`` for
    ``[*, cdiv(max_model_len, block_size)]`` structures).
    """
    return int(high_water_index) * int(stride_elems) * int(itemsize)


def int32_overflow_maxlen(stride_bytes_per_token: float) -> float:
    """The ``max_model_len`` at which a per-token byte stride crosses 2**31.

    ``2**31 / stride_bytes_per_token``. Used to check whether a candidate's 2 GiB
    crossing lands inside the fingerprint interval (460800, 507904]. A per-token
    stride in ``[2**31/507904, 2**31/460800] == [4228.13, 4660.34]`` bytes crosses
    exactly inside the interval and is the arithmetic signature to look for.
    """
    if stride_bytes_per_token <= 0:
        return float("inf")
    return INT32_BYTE_LIMIT / float(stride_bytes_per_token)


def crosses_in_interval(
    stride_bytes_per_token: float,
    lo: int = 460800,
    hi: int = 507904,
) -> bool:
    """True iff the 2**31 crossing for this per-token stride is in (lo, hi]."""
    mml = int32_overflow_maxlen(stride_bytes_per_token)
    return lo < mml <= hi


# --------------------------------------------------------------------------- #
# Runtime plumbing (all guarded on _ENABLED / best-effort; never raises).
# --------------------------------------------------------------------------- #
def install() -> None:
    """Arm memory-history recording + fault dumping. Idempotent; no-op if off."""
    global _installed
    if not _ENABLED or _installed:
        return
    _installed = True
    try:
        import faulthandler

        faulthandler.enable()
    except Exception:  # pragma: no cover - defensive
        logger.exception("[XID31] faulthandler.enable() failed")

    try:
        import torch

        # Record the allocation call-stack for every block so the snapshot dumped
        # on fault names which tensor owned the faulting address.
        torch.cuda.memory._record_memory_history(
            enabled="all", stacks="python", max_entries=100_000
        )
        logger.warning(
            "[XID31] memory-history recording ARMED "
            "(min_mb=%d every_n=%d snapshot=%s int32_limit=%d)",
            _THRESHOLD_MB,
            _EVERY_N,
            _SNAPSHOT_PATH,
            INT32_BYTE_LIMIT,
        )
    except Exception:  # pragma: no cover - defensive
        logger.exception("[XID31] _record_memory_history() failed")

    # Chain an excepthook so a snapshot lands even on an uncaught CUDA error.
    prev_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        dump_snapshot(tag="excepthook")
        prev_hook(exc_type, exc, tb)

    sys.excepthook = _hook


def dump_snapshot(tag: str = "manual") -> None:
    """Dump the CUDA allocator snapshot to ``_SNAPSHOT_PATH``. No-op if off."""
    if not _ENABLED:
        return
    try:
        import torch

        path = f"{_SNAPSHOT_PATH}.{tag}"
        torch.cuda.memory._dump_snapshot(path)
        logger.warning("[XID31] memory snapshot dumped -> %s", path)
    except Exception:  # pragma: no cover - defensive
        logger.exception("[XID31] snapshot dump failed (tag=%s)", tag)


def _tensor_report(name: str, t: Any) -> "dict[str, Any] | None":
    """Build a static report dict for one tensor; None if too small / invalid."""
    try:
        numel = t.numel()
        itemsize = t.element_size()
        nbytes = numel * itemsize
        if nbytes < _THRESHOLD_MB * 1024 * 1024:
            return None
        shape = tuple(t.shape)
        strides = tuple(t.stride())
        try:
            ptr = t.data_ptr()
        except Exception:
            ptr = 0
        wbytes = worst_case_byte_offset(shape, strides, itemsize)
        return {
            "shape": shape,
            "strides": strides,
            "dtype": str(t.dtype),
            "device": str(t.device),
            "itemsize": itemsize,
            "nbytes": nbytes,
            "ptr_lo": ptr,
            "ptr_hi": ptr + nbytes,
            "worst_byte_off": wbytes,
        }
    except Exception:  # pragma: no cover - defensive
        return None


def register_buffer(name: str, tensor: Any) -> None:
    """Register one tensor for tracking + log it if it clears the threshold."""
    if not _ENABLED or tensor is None:
        return
    try:
        rep = _tensor_report(name, tensor)
        if rep is None:
            return
        _registry[name] = (weakref.ref(tensor), rep)
        near = rep["worst_byte_off"] >= INT32_BYTE_LIMIT * NEAR_FRAC
        logger.warning(
            "[XID31] buf %-40s shape=%s dtype=%s dev=%s bytes=%.1fMiB "
            "ptr=[%#x,%#x) worst_off=%.3fGiB%s",
            name,
            rep["shape"],
            rep["dtype"],
            rep["device"],
            rep["nbytes"] / (1024 * 1024),
            rep["ptr_lo"],
            rep["ptr_hi"],
            rep["worst_byte_off"] / (1024**3),
            "  <== NEAR-2GiB-INT32-LIMIT" if near else "",
        )
    except Exception:  # pragma: no cover - defensive
        logger.exception("[XID31] register_buffer failed for %s", name)


def _iter_named_tensors(root: Any, prefix: str, depth: int = 0):
    """Yield (name, tensor) for tensors reachable from a small set of roots.

    Deliberately shallow: known containers on the model runner and its drafter,
    plus attention-group metadata builders. Avoids a full object-graph walk so it
    stays cheap and predictable.
    """
    import torch

    if root is None or depth > 3:
        return
    if isinstance(root, torch.Tensor):
        yield prefix, root
        return
    if isinstance(root, (list, tuple)):
        for i, v in enumerate(root):
            yield from _iter_named_tensors(v, f"{prefix}[{i}]", depth + 1)
        return
    if isinstance(root, dict):
        for k, v in root.items():
            yield from _iter_named_tensors(v, f"{prefix}.{k}", depth + 1)
        return
    d = getattr(root, "__dict__", None)
    if not d:
        return
    for k, v in list(d.items()):
        if isinstance(v, torch.Tensor):
            yield f"{prefix}.{k}", v
        elif isinstance(v, (list, tuple, dict)):
            yield from _iter_named_tensors(v, f"{prefix}.{k}", depth + 1)


def scan_runner(runner: Any) -> None:
    """Register the large persistent buffers of a GPUModelRunner. No-op if off.

    Covers: KV caches, attention-group metadata builders (block-table /
    state-index tensors sized by ``cdiv(max_model_len, block_size)``), the spec
    drafter's persistent buffers, and the input batch's block table.
    """
    global _scanned
    if not _ENABLED:
        return
    _scanned = True
    try:
        import torch  # noqa: F401

        roots = {
            "runner.kv_caches": getattr(runner, "kv_caches", None),
            "runner.attn_groups": getattr(runner, "attn_groups", None),
            "runner.drafter": getattr(runner, "drafter", None),
        }
        ib = getattr(runner, "input_batch", None)
        if ib is not None:
            roots["runner.input_batch.block_table"] = getattr(
                ib, "block_table", None
            )
        seen: set[int] = set()
        for root_name, root in roots.items():
            for name, t in _iter_named_tensors(root, root_name):
                key = id(t)
                if key in seen:
                    continue
                seen.add(key)
                register_buffer(name, t)
        _log_config(runner)
    except Exception:  # pragma: no cover - defensive
        logger.exception("[XID31] scan_runner failed")


def _log_config(runner: Any) -> None:
    """Log the sizing inputs and the exact 2 GiB / interval arithmetic."""
    try:
        mml = int(getattr(runner, "max_model_len", 0))
        cache_cfg = getattr(getattr(runner, "vllm_config", None), "cache_config", None)
        block_size = int(getattr(cache_cfg, "block_size", 0) or 0)
        spec = getattr(runner, "speculative_config", None)
        num_spec = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec else 0
        wbc = None
        if block_size > 0:
            wbc = -(-mml // block_size)  # cdiv
        logger.warning(
            "[XID31] config: max_model_len=%d block_size=%s "
            "cdiv(mml,bs)=%s num_speculative_tokens=%d | "
            "int32 2GiB crosses at per-token-stride in [%.1f,%.1f] bytes "
            "for interval (460800,507904]",
            mml,
            block_size,
            wbc,
            num_spec,
            2**31 / 507904,
            2**31 / 460800,
        )
    except Exception:  # pragma: no cover - defensive
        logger.exception("[XID31] _log_config failed")


def periodic(runner: Any) -> None:
    """Cheap per-step check: project the high-water position onto each suspect
    and log-assert it stays under the buffer extent and 2**31. No-op if off."""
    if not _ENABLED:
        return
    install()
    if not _scanned:
        scan_runner(runner)
    global _step_counter
    _step_counter += 1
    if _step_counter % _EVERY_N != 0:
        return
    try:
        high_water = _current_high_water(runner)
        cache_cfg = getattr(getattr(runner, "vllm_config", None), "cache_config", None)
        block_size = int(getattr(cache_cfg, "block_size", 0) or 0) or 1
        # Deepest per-request block row a kernel could index this step.
        hw_block = high_water // block_size
        flagged = 0
        for name, (ref, rep) in list(_registry.items()):
            t = ref()
            if t is None:
                continue
            # Row stride in elements (dim-1 stride for a [rows, cols] table, else
            # the outer stride). Projected offset = hw_block * row_stride * items.
            strides = rep["strides"]
            row_stride = strides[0] if strides else 0
            proj = projected_offset_bytes(hw_block, row_stride, rep["itemsize"])
            if proj >= INT32_BYTE_LIMIT * NEAR_FRAC or proj >= rep["worst_byte_off"]:
                flagged += 1
                logger.warning(
                    "[XID31] step=%d hw_pos=%d hw_block=%d buf=%s "
                    "proj_off=%.3fGiB extent=%.3fGiB %s",
                    _step_counter,
                    high_water,
                    hw_block,
                    name,
                    proj / (1024**3),
                    rep["worst_byte_off"] / (1024**3),
                    "OVER-2GiB" if proj >= INT32_BYTE_LIMIT else "near-limit",
                )
        if flagged == 0:
            logger.info(
                "[XID31] step=%d hw_pos=%d hw_block=%d suspects_ok=%d",
                _step_counter,
                high_water,
                hw_block,
                len(_registry),
            )
    except Exception:  # pragma: no cover - defensive
        logger.exception("[XID31] periodic failed")


def _current_high_water(runner: Any) -> int:
    """Best-effort deepest sequence position among running requests."""
    try:
        ib = getattr(runner, "input_batch", None)
        if ib is None:
            return 0
        # InputBatch tracks per-request depth as ``num_tokens_no_spec`` (a numpy
        # array indexed by request slot); there is no plain ``num_tokens``.
        nt = getattr(ib, "num_tokens_no_spec", None)
        if nt is not None:
            try:
                return int(max(nt[: ib.num_reqs])) if ib.num_reqs else 0
            except Exception:
                pass
        # Fallback to the runner's ceiling.
        return int(getattr(runner, "max_model_len", 0))
    except Exception:  # pragma: no cover - defensive
        return 0
