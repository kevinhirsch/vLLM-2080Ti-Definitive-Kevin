#!/usr/bin/env python3
"""Lane IR: hardened --kv-transfer-config builder for the opt-in host-RAM / SSD prefix-KV tier
(OffloadingConnector + TieringOffloadingSpec) used by serve-hauhaucs-v02.sh when V02_SSD_KV_DIR is set.

Why this exists (all from reading code + public reports, nothing measured on the live engine):
  * lane UP measured the tier at a 4 GiB host staging size: cold +25 %, warm pass WORSE (72 s vs 36.6 s).  The estate
    working set is ~12 bodies x ~15 chunk-keys x ~52 MB = ~9.4 GB, so 4 GiB is a cyclic-LRU thrash (the same cliff the
    GPU pool shows), i.e. net I/O loss.  A public same-hardware report (Aiakos1818/qwen3-8-27b-dual-2080ti-vllm) measured
    exactly this: staging smaller than the working chain = restore voided and I/O wasted; big enough = 98.7 % hit, 5 s.
  * vLLM's default kv_load_failure_policy is "fail": one failed block load ABORTS the request.  "recompute" degrades to a
    normal prefill instead (same report; field exists in vllm/config/kv_transfer.py).
  * the shared staging region is /dev/shm/vllm_offload_<engine_id>.mmap and is only unlinked on a clean exit; a random
    engine_id per start leaks one cpu_bytes-sized file per crash and eventually /dev/shm fills (cuMemHostRegister then
    fails and poisons the CUDA context).  We pin engine_id and delete orphaned files that no live process holds open.

Pure stdlib.  CLI:
  offload_kv_config.py config --root DIR --cpu-bytes N --fingerprint FP [--store-threshold K] [--allow-small]
  offload_kv_config.py clean-stale [--shm-dir /dev/shm] [--dry-run]
  offload_kv_config.py recommend [--bodies 12] [--chunks-per-body 15]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

GIB = 1 << 30
# One CPU chunk slot = worker_kv_bytes_per_block x ranks.  For Qwen3.8-27B TQ k3v4_nc TP2: one KV block id spans 16 pool
# tensors x one 1.64 MB page (RF note: "page ~26 MB per GPU") = ~26.2 MB per GPU, x2 ranks = ~52.4 MB.  Override with
# --chunk-bytes (the engine's startup log prints the real aligned number on the OffloadingSpec line when it is known).
DEFAULT_CHUNK_BYTES = int(2 * 26.2e6)
# estate pass: 12 recorded bodies, each ~9 attention chunks + 1-3 recurrent snapshots x 3 recurrent groups (~12-18 keys).
DEFAULT_BODIES = 12
DEFAULT_CHUNKS_PER_BODY = 15
HEADROOM = 1.25  # LRU needs slack: working set at 100 % of capacity is the textbook zero-hit cyclic cliff
ENGINE_ID = "hauhaucs-v02-kv"


def recommended_cpu_bytes(
    bodies: int = DEFAULT_BODIES,
    chunks_per_body: int = DEFAULT_CHUNKS_PER_BODY,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    headroom: float = HEADROOM,
) -> int:
    return int(bodies * chunks_per_body * chunk_bytes * headroom)


def check_cpu_bytes(cpu_bytes: int, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> tuple[bool, str]:
    """(ok, message).  Not ok when the tier cannot hold the estate working set (cyclic-LRU thrash = net loss)."""
    need = recommended_cpu_bytes(chunk_bytes=chunk_bytes)
    chunks = cpu_bytes // chunk_bytes
    if cpu_bytes >= need:
        return True, f"cpu_bytes {cpu_bytes / GIB:.1f} GiB = {chunks} chunks >= recommended {need / GIB:.1f} GiB"
    return False, (
        f"cpu_bytes {cpu_bytes / GIB:.1f} GiB = {chunks} chunks is below the estate working set "
        f"({need / GIB:.1f} GiB recommended: {DEFAULT_BODIES} bodies x {DEFAULT_CHUNKS_PER_BODY} keys x "
        f"{chunk_bytes / 1e6:.1f} MB x {HEADROOM}). A tier smaller than the cyclic working set thrashes (lane UP: 4 GiB was a net loss)."
    )


def build_config(
    root_dir: str,
    cpu_bytes: int,
    fingerprint: str,
    *,
    store_threshold: int = 0,
    failure_policy: str = "recompute",
    engine_id: str = ENGINE_ID,
) -> dict:
    if failure_policy not in ("recompute", "fail"):
        raise ValueError("failure_policy must be 'recompute' or 'fail'")
    extra: dict = {
        "cpu_bytes_to_use": int(cpu_bytes),
        "spec_name": "TieringOffloadingSpec",
        "secondary_tiers": [{"type": "fs", "root_dir": os.path.join(root_dir, "checkpoint-" + fingerprint)}],
    }
    if store_threshold >= 2:  # only store chunks offered >= K times: replayed estate prefixes yes, one-off tool output no
        extra["store_threshold"] = int(store_threshold)
    return {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "engine_id": engine_id,
        "kv_load_failure_policy": failure_policy,
        "kv_connector_extra_config": extra,
    }


def _open_paths_by_any_process() -> set[str] | None:
    """Targets of every readable /proc/*/fd symlink.  None if /proc is unusable (then callers must not delete)."""
    if not os.path.isdir("/proc/self/fd"):
        return None
    held: set[str] = set()
    for fd_dir in glob.glob("/proc/[0-9]*/fd"):
        try:
            for name in os.listdir(fd_dir):
                try:
                    held.add(os.readlink(os.path.join(fd_dir, name)))
                except OSError:
                    pass
        except OSError:
            continue  # other user's process / vanished: its files are simply not proven free
    return held


def stale_offload_files(shm_dir: str = "/dev/shm", held: set[str] | None = None) -> list[str]:
    """vllm_offload_*.mmap files in shm_dir that no readable process has open.  Files held by an unreadable
    (other-user) process cannot be proven free, so a conservative caller passes held=None only when /proc works."""
    files = sorted(glob.glob(os.path.join(shm_dir, "vllm_offload_*.mmap")))
    if held is None:
        held = _open_paths_by_any_process()
        if held is None:
            return []
    return [f for f in files if os.path.realpath(f) not in held and f not in held]


def clean_stale(shm_dir: str = "/dev/shm", dry_run: bool = False, held: set[str] | None = None) -> list[str]:
    removed = []
    for f in stale_offload_files(shm_dir, held):
        if not dry_run:
            try:
                os.unlink(f)
            except OSError:
                continue
        removed.append(f)
    return removed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("config")
    c.add_argument("--root", required=True)
    c.add_argument("--cpu-bytes", type=int, required=True)
    c.add_argument("--fingerprint", required=True)
    c.add_argument("--store-threshold", type=int, default=0)
    c.add_argument("--failure-policy", default="recompute")
    c.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    c.add_argument("--allow-small", action="store_true", help="do not refuse a tier below the working-set estimate")
    s = sub.add_parser("clean-stale")
    s.add_argument("--shm-dir", default="/dev/shm")
    s.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("recommend")
    r.add_argument("--bodies", type=int, default=DEFAULT_BODIES)
    r.add_argument("--chunks-per-body", type=int, default=DEFAULT_CHUNKS_PER_BODY)
    r.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    a = ap.parse_args(argv)
    if a.cmd == "config":
        ok, msg = check_cpu_bytes(a.cpu_bytes, a.chunk_bytes)
        print(("OK: " if ok else "WARN: ") + msg, file=sys.stderr)
        if not ok and not a.allow_small:
            print("refusing: pass --allow-small (V02_SSD_KV_ALLOW_SMALL=1) to override", file=sys.stderr)
            return 2
        print(json.dumps(build_config(a.root, a.cpu_bytes, a.fingerprint, store_threshold=a.store_threshold,
                                      failure_policy=a.failure_policy), separators=(",", ":")))
        return 0
    if a.cmd == "clean-stale":
        removed = clean_stale(a.shm_dir, a.dry_run)
        for f in removed:
            print(("would remove " if a.dry_run else "removed ") + f, file=sys.stderr)
        return 0
    print(recommended_cpu_bytes(a.bodies, a.chunks_per_body, a.chunk_bytes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
