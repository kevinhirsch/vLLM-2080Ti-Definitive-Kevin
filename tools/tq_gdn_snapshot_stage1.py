# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-038 Stage-1 driver — attn KV snapshot is just pinning.

Standalone, MANUAL script. It is NOT a pytest test (no ``test_`` prefix, not
under ``tests/``) and does nothing on import — everything is under
``if __name__ == "__main__"``. Run it ONLY in a scheduled maintenance window on
a THROWAWAY 2-GPU engine. NEVER point it at the production :8001 serve, and
NEVER run it while :8001 is serving on the same GPUs.

What it proves (Stage-1): the attn side of SNAPSHOT is pure refcount pinning.
``block_pool.touch`` a finished prefix's KV blocks and they survive the
producing request's ``free()`` — ``ref_cnt >= 1`` and resident (not back in the
free queue) AFTER the source request finishes. This is the block-pinning layer
that lets a RESTORE reattach a foreign-but-valid block table (Stage 2). It
exercises the EngineCore-side utility pair added on ``feat-gdn-snapshot``:
``pin_request_kv_blocks`` / ``verify_pinned_blocks`` / ``unpin_kv_blocks``,
reachable from the ``LLM`` driver via the same utility-RPC path as
``reset_prefix_cache`` (the block pool lives in the scheduler process, not the
worker — so this is NOT a ``collective_rpc``).

Example (throwaway engine, same flags as the deploy):

    VLLM_TQ_GDN_SNAPSHOT=1 \
    VLLM_TQ_GDN_SNAPSHOT_CONFIRM_THROWAWAY=1 \
    python tools/tq_gdn_snapshot_stage1.py \
        --model /path/to/Qwen3.8-27B-hybrid \
        --tensor-parallel-size 2 \
        --kv-cache-dtype turboquant_k8v4 \
        --mamba-cache-mode align \
        --dtype half \
        --gpu-memory-utilization 0.82 \
        --block-aligned-tokens 2112
"""

from __future__ import annotations

import argparse
import sys


def _build_block_aligned_prompt(n_tokens: int) -> list[int]:
    """A deterministic, block-aligned dummy prompt (token ids). Block-aligned
    (N x block_size) prompts avoid the partial-block / accept_token_bias corner
    cases, isolating the pin primitive (see EXP-038 Risk #2)."""
    return [13] * n_tokens


def main() -> int:
    import os
    import threading
    import time

    import vllm.envs as envs
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--kv-cache-dtype", default="turboquant_k8v4")
    parser.add_argument("--mamba-cache-mode", default="align")
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument(
        "--block-aligned-tokens",
        type=int,
        default=2112,
        help="prompt length; keep a multiple of the mamba block_size",
    )
    args = parser.parse_args()

    if not envs.VLLM_TQ_GDN_SNAPSHOT:
        print(
            "REFUSING TO RUN: set VLLM_TQ_GDN_SNAPSHOT=1 to enable the "
            "snapshot feature (it is inert by default).",
            file=sys.stderr,
        )
        return 2

    # Hard guard: this must never share GPUs with the production serve.
    if os.getenv("VLLM_TQ_GDN_SNAPSHOT_CONFIRM_THROWAWAY") != "1":
        print(
            "REFUSING TO RUN: set VLLM_TQ_GDN_SNAPSHOT_CONFIRM_THROWAWAY=1 to "
            "confirm this is a THROWAWAY engine, not the :8001 serve.",
            file=sys.stderr,
        )
        return 2

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        kv_cache_dtype=args.kv_cache_dtype,
        mamba_cache_mode=args.mamba_cache_mode,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enforce_eager=True,
        max_model_len=8192,
        additional_config={"gdn_prefill_backend": "flashqla_legacy"},
    )

    # Drive a block-aligned prefill, then PIN mid-generation: the scheduler
    # drops a request's block table the moment it finishes and frees, so the
    # pin must be taken while the request is genuinely in flight. Decode enough
    # tokens to give the polling loop a comfortable window under enforce_eager.
    prompt = _build_block_aligned_prompt(args.block_aligned_tokens)
    gen_done = threading.Event()
    gen_error: list[BaseException] = []

    def _generate() -> None:
        try:
            llm.generate(
                {"prompt_token_ids": prompt},
                SamplingParams(max_tokens=256, temperature=0.0),
            )
        except BaseException as exc:  # noqa: BLE001 - surface to the main thread
            gen_error.append(exc)
        finally:
            gen_done.set()

    gen_thread = threading.Thread(target=_generate, daemon=True)
    gen_thread.start()

    # Poll until we catch the sequence in flight and pin its blocks. The pin is
    # a scheduler-side (EngineCore) utility, driven straight off the LLM driver
    # (NOT collective_rpc). A pin taken atomically inside one utility call has
    # no intra-step race: once ref_cnt is bumped the blocks survive regardless.
    handle: dict | None = None
    while not gen_done.is_set() and handle is None:
        time.sleep(0.5)
        try:
            attempt = llm.pin_request_kv_blocks()
        except Exception as exc:  # noqa: BLE001 - transient mid-gen RPC issues
            print(f"pin attempt failed transiently: {exc!r}")
            continue
        if attempt.get("num_pinned_blocks", 0) > 0:
            handle = attempt

    if handle is None:
        print("never caught the sequence in flight to pin", file=sys.stderr)
        return 1

    handle_id = handle["handle_id"]
    attn_groups = {
        gid: g for gid, g in handle["groups"].items() if g["spec"] == "attn"
    }
    mamba_groups = {
        gid: g for gid, g in handle["groups"].items() if g["spec"] == "mamba"
    }
    n_attn = sum(len(g["block_ids"]) for g in attn_groups.values())
    n_mamba = sum(len(g["block_ids"]) for g in mamba_groups.values())
    print(
        f"PINNED handle={handle_id} req={handle['req_id']} "
        f"num_computed_tokens={handle['num_computed_tokens']} "
        f"attn_blocks={n_attn} mamba_blocks={n_mamba} "
        f"total={handle['num_pinned_blocks']}"
    )

    # Let the source request run to completion; on finish the scheduler frees
    # its block table (decrementing ref_cnt once). The pin's extra ref must
    # keep the blocks resident.
    gen_thread.join(timeout=180)
    if not gen_done.is_set():
        print("source generation did not finish in time", file=sys.stderr)
        llm.unpin_kv_blocks(handle_id)
        return 1
    if gen_error:
        print(
            f"source generation raised ({gen_error[0]!r}); the pin's "
            "survive-free claim was never exercised — failing rather than "
            "reporting a false pass",
            file=sys.stderr,
        )
        llm.unpin_kv_blocks(handle_id)
        return 1
    # Give the engine a beat to run the finish/free step for the request.
    time.sleep(1.0)

    # Stage-1 proof: verify residency AFTER the source request has finished and
    # freed. This is the whole claim — the pin (not luck) kept the blocks alive.
    verdict = llm.verify_pinned_blocks(handle_id)
    ok = bool(verdict.get("ok"))
    min_ref = verdict.get("min_ref_cnt")
    free_blocks = verdict.get("num_free_blocks")

    # The full-attn groups are the Stage-1 target of the design; assert every
    # attn-group block is individually resident with ref_cnt >= 1.
    attn_all_resident = True
    for gid, g in verdict.get("groups", {}).items():
        if g.get("spec") != "attn":
            continue
        for row in g.get("blocks", []):
            if not row.get("resident") or row.get("ref_cnt", 0) < 1:
                attn_all_resident = False
                print(
                    f"  NOT RESIDENT: group={gid} block={row.get('block_id')} "
                    f"ref_cnt={row.get('ref_cnt')}"
                )

    print(
        f"POST-FREE verify: ok={ok} min_ref_cnt={min_ref} "
        f"num_free_blocks={free_blocks} attn_all_resident={attn_all_resident}"
    )

    # Demonstrative teardown: release the pin and confirm the accounting
    # (exactly one free per pin, per EXP-038 Risk #1).
    released = llm.unpin_kv_blocks(handle_id)
    print(
        f"UNPIN: ok={released.get('ok')} "
        f"num_freed_blocks={released.get('num_freed_blocks')}"
    )

    passed = (
        ok
        and attn_all_resident
        and (min_ref or 0) >= 1
        and released.get("ok") is True
        and released.get("num_freed_blocks") == handle["num_pinned_blocks"]
    )
    print(f"\nSTAGE-1 {'PASS' if passed else 'FAIL'} "
          f"(attn KV blocks survive source free, pinned resident: {passed})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
