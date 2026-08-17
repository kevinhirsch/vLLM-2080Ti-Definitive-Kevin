# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-038 Stage-2 driver — restore + byte-exact continuation (the proof).

Standalone, MANUAL script. It is NOT a pytest test (no ``test_`` prefix, not
under ``tests/``) and does nothing on import — everything is under
``if __name__ == "__main__"``. Run it ONLY in a scheduled maintenance window on
a THROWAWAY 2-GPU engine. NEVER point it at the production :8001 serve, and
NEVER run it while :8001 is serving on the same GPUs.

Variant implemented: the ZERO-KERNEL restore (EXP-038 design lines 26-30, 103).
Rationale for choosing it over the manual-seed variant is in the RISKS/GAP
banner printed at the end and in the branch commit message: the manual variant
(hand-seed a fresh req's block table from pinned block_ids + worker-side
``restore_mamba_state`` batch_memcpy + a single controlled decode step with
``torch.equal`` on raw logits) requires bridging the scheduler-process block
pool and the worker-process GDN state for a request built OUTSIDE the normal
add_request lifecycle — process-boundary plumbing that is too invasive for
staged-only work. The zero-kernel variant proves SNAPSHOT+RESTORE end-to-end
through the DEPLOYED align-mode path with a strictly public LLM API.

What it proves:
  (a) BYTE-EXACT CONTINUATION. A block-aligned prefix is pinned mid-generation
      (attn + mamba groups), the source request finishes and frees, then the
      identical token_ids are resubmitted. ``find_longest_cache_hit`` returns
      the pinned attn blocks AND the pinned by-hash-cached GDN state block; the
      align-mode preprocess copies that GDN state into the new running slot for
      free. The resubmitted continuation must equal the reference continuation
      token-for-token (and logprob-for-logprob) — the exact
      "restored == prefix-cache-hit" byte-exactness argument (design line 84).
  (b) THE PIN IS LOAD-BEARING. The restore request's prefix KV blocks are the
      SAME physical block_ids that were pinned (set-intersection), proving the
      continuation reused the pinned state rather than coincidentally
      re-caching. With ``--churn-fillers N`` the cache is pressured between the
      two runs so an UNPINNED prefix would have been evicted — making the pin
      the reason the blocks survived.

Example (throwaway engine, same flags as the deploy):

    VLLM_TQ_GDN_SNAPSHOT=1 \
    VLLM_TQ_GDN_SNAPSHOT_CONFIRM_THROWAWAY=1 \
    python tools/tq_gdn_snapshot_stage2.py \
        --model /path/to/Qwen3.8-27B-hybrid \
        --tensor-parallel-size 2 \
        --kv-cache-dtype turboquant_k8v4 \
        --mamba-cache-mode align \
        --dtype half \
        --gpu-memory-utilization 0.82 \
        --block-aligned-tokens 2112 \
        --continuation-tokens 32
"""

from __future__ import annotations

import argparse
import sys


def _build_block_aligned_prompt(n_tokens: int, seed: int = 13) -> list[int]:
    """A deterministic, block-aligned dummy prompt (token ids). Block-aligned
    (N x block_size) prompts are fully cacheable as FULL blocks, so a resubmit
    hits the whole prefix and no partial-block copy is needed (EXP-038 Risk #2).
    ``seed`` lets filler prompts differ from the payload prompt."""
    return [seed] * n_tokens


def _extract_continuation(request_output) -> tuple[list[int], list[float]]:
    """Pull (token_ids, chosen-token logprobs) out of a RequestOutput.

    logprobs are the fine-grained byte-exactness signal on top of the token-id
    match; if logprobs weren't requested/populated the list is empty and the
    caller falls back to token-id equality alone."""
    comp = request_output.outputs[0]
    token_ids = list(comp.token_ids)
    logprobs: list[float] = []
    if getattr(comp, "logprobs", None):
        for step_idx, tid in enumerate(token_ids):
            try:
                step = comp.logprobs[step_idx]
                lp = step.get(tid)
                logprobs.append(float(lp.logprob) if lp is not None else float("nan"))
            except Exception:  # noqa: BLE001 - best-effort fine-grained signal
                logprobs.append(float("nan"))
    return token_ids, logprobs


def _run_and_pin(llm, prompt, sampling_params, do_pin, stop_on_first):
    """Run ``generate(prompt)`` in a background thread; while it is in flight,
    poll ``do_pin()`` (a scheduler-side utility) and return its result. Returns
    (request_output, pin_result).

    ``stop_on_first=True`` for a MUTATING op (pin_request_kv_blocks) — fire it
    exactly once, or repeated polls would create multiple pins and leak
    refcounts. ``stop_on_first=False`` for a READ-ONLY probe
    (get_request_kv_block_ids) — poll aggressively and keep the last catch,
    because the RESTORE run's prefill is skipped by the cache hit and can
    finish in only ``continuation-tokens`` decode steps."""
    import threading
    import time

    result_box: dict = {}
    gen_done = threading.Event()

    def _generate() -> None:
        try:
            outs = llm.generate({"prompt_token_ids": prompt}, sampling_params)
            result_box["output"] = outs[0]
        finally:
            gen_done.set()

    gen_thread = threading.Thread(target=_generate, daemon=True)
    gen_thread.start()

    pin_result = None
    while not gen_done.is_set() and do_pin is not None:
        try:
            attempt = do_pin()
        except Exception:  # noqa: BLE001 - transient mid-gen RPC issues
            time.sleep(0.15)
            continue
        if attempt is not None:
            pin_result = attempt  # keep the last successful catch
            if stop_on_first:
                break
        time.sleep(0.15)

    gen_thread.join(timeout=240)
    if not gen_done.is_set():
        raise RuntimeError("generation did not finish in time")
    return result_box.get("output"), pin_result


def main() -> int:
    import os
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
        help="prefix length; keep a multiple of BOTH the attn and mamba "
        "block_size so the whole prefix caches as full blocks",
    )
    parser.add_argument(
        "--continuation-tokens",
        type=int,
        default=64,
        help="greedy decode steps to compare byte-for-byte (also the restore "
        "run's window for catching its block table mid-generation)",
    )
    parser.add_argument(
        "--churn-fillers",
        type=int,
        default=0,
        help="number of distinct filler prompts to submit between the "
        "reference and restore runs to pressure the cache (makes the pin "
        "provably load-bearing). 0 = no pressure.",
    )
    args = parser.parse_args()

    if not envs.VLLM_TQ_GDN_SNAPSHOT:
        print(
            "REFUSING TO RUN: set VLLM_TQ_GDN_SNAPSHOT=1 to enable the "
            "snapshot feature (it is inert by default).",
            file=sys.stderr,
        )
        return 2
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

    prompt = _build_block_aligned_prompt(args.block_aligned_tokens)
    greedy = SamplingParams(
        max_tokens=args.continuation_tokens, temperature=0.0, logprobs=1
    )

    # ---- Reference run + pin (attn + mamba blocks) ----------------------
    print("[1/4] reference run + pin ...")
    ref_out, handle = _run_and_pin(
        llm, prompt, greedy,
        do_pin=lambda: llm.pin_request_kv_blocks(),
        stop_on_first=True,
    )
    if ref_out is None or handle is None or handle.get("num_pinned_blocks", 0) <= 0:
        print("failed to pin the reference request in flight", file=sys.stderr)
        return 1
    handle_id = handle["handle_id"]
    ref_tokens, ref_logprobs = _extract_continuation(ref_out)
    pinned_attn = {
        int(b)
        for gid, g in handle["groups"].items()
        if g["spec"] == "attn"
        for b in g["block_ids"]
    }
    pinned_mamba = {
        int(b)
        for gid, g in handle["groups"].items()
        if g["spec"] == "mamba"
        for b in g["block_ids"]
    }
    print(
        f"  pinned handle={handle_id} attn_blocks={len(pinned_attn)} "
        f"mamba_blocks={len(pinned_mamba)} ref_continuation={ref_tokens[:8]}..."
    )

    # Let the finish/free step land, then confirm the pin held across the free.
    time.sleep(1.0)
    v = llm.verify_pinned_blocks(handle_id)
    print(
        f"[2/4] post-free verify: ok={v.get('ok')} "
        f"min_ref_cnt={v.get('min_ref_cnt')} "
        f"num_free_blocks={v.get('num_free_blocks')}"
    )
    if not v.get("ok"):
        print("pinned blocks did not survive source free", file=sys.stderr)
        llm.unpin_kv_blocks(handle_id)
        return 1

    # ---- Optional cache pressure ---------------------------------------
    if args.churn_fillers > 0:
        print(f"[..] churning cache with {args.churn_fillers} filler prompts ...")
        for i in range(args.churn_fillers):
            filler = _build_block_aligned_prompt(
                args.block_aligned_tokens, seed=100 + i
            )
            llm.generate(
                {"prompt_token_ids": filler},
                SamplingParams(max_tokens=1, temperature=0.0),
            )
        v2 = llm.verify_pinned_blocks(handle_id)
        print(
            f"  post-churn verify: ok={v2.get('ok')} "
            f"min_ref_cnt={v2.get('min_ref_cnt')}"
        )
        if not v2.get("ok"):
            print("pinned blocks evicted under churn (unexpected)", file=sys.stderr)
            llm.unpin_kv_blocks(handle_id)
            return 1

    # ---- Restore run: resubmit identical token_ids ----------------------
    print("[3/4] restore run (resubmit identical token_ids) ...")
    res_out, restore_blocks = _run_and_pin(
        llm, prompt, greedy,
        do_pin=lambda: llm.get_request_kv_block_ids(),
        stop_on_first=False,
    )
    if res_out is None:
        print("restore run produced no output", file=sys.stderr)
        llm.unpin_kv_blocks(handle_id)
        return 1
    res_tokens, res_logprobs = _extract_continuation(res_out)
    # THE restore-vs-recompute discriminator: a cache-hit restore skips prefill
    # for the cached prefix; a recompute has num_cached_tokens ~ 0.
    print(
        f"  restore num_cached_tokens={getattr(res_out, 'num_cached_tokens', None)} "
        f"(prompt len {len(prompt)}; cached>0 = rode the cache, 0 = recompute)"
    )

    restore_attn: set[int] = set()
    restore_mamba: set[int] = set()
    if restore_blocks is not None:
        for gid, g in restore_blocks.get("groups", {}).items():
            if g["spec"] == "attn":
                restore_attn.update(int(b) for b in g["block_ids"])
            elif g["spec"] == "mamba":
                restore_mamba.update(int(b) for b in g["block_ids"])

    # ---- Verdict --------------------------------------------------------
    tokens_equal = res_tokens == ref_tokens
    logprobs_equal = (
        bool(ref_logprobs)
        and bool(res_logprobs)
        and len(ref_logprobs) == len(res_logprobs)
        and all(
            (a != a and b != b)  # both NaN
            or a == b
            for a, b in zip(ref_logprobs, res_logprobs)
        )
    )
    attn_reused = len(pinned_attn & restore_attn)
    mamba_reused = len(pinned_mamba & restore_mamba)
    # reuse_checked: did we actually catch the restore request's block table
    # mid-generation? The restore run's prefill is skipped by the cache hit, so
    # a fast decode can slip past the poll. A MISSED catch is inconclusive
    # corroboration (not a failure); a CAUGHT-but-empty intersection is a real
    # NEGATIVE (restore recomputed instead of reusing the pinned state).
    reuse_checked = restore_blocks is not None and (
        len(restore_attn) > 0 or len(restore_mamba) > 0
    )
    # Fable re-adjudication 2026-08-16: block-table intersection is the WRONG
    # observable for load-bearing. (a) attn: only FULL blocks can cache-hit, so
    # the partial decode block never intersects — attn_reused > 0 is already the
    # maximum signal. (b) mamba: align-mode restore COPIES state out of the
    # cached block into a FRESH running slot — the pinned source never appears
    # in the new block table, by design. The decisive observable is
    # num_cached_tokens: a cache-hit restore skips prefill for the cached
    # prefix; a recompute has ~0.
    cached_tokens = int(getattr(res_out, "num_cached_tokens", 0) or 0)
    cached_frac = cached_tokens / max(1, len(prompt))
    pin_load_bearing = cached_frac > 0.5 and attn_reused > 0

    llm.unpin_kv_blocks(handle_id)

    print("[4/4] verdict:")
    print(f"  byte-exact continuation (token ids): {tokens_equal}")
    print(
        f"  logprobs match: {logprobs_equal} "
        f"(ref n={len(ref_logprobs)}, res n={len(res_logprobs)})"
    )
    print(
        f"  reuse checked (caught restore block table mid-gen): {reuse_checked}"
    )
    print(
        f"  pinned blocks in restore table: attn {attn_reused}/{len(pinned_attn)} "
        f"(partial block can never hit), mamba {mamba_reused}/{len(pinned_mamba)} "
        f"(copy-out by design, absence expected)"
    )
    print(
        f"  cache-hit discriminator: num_cached_tokens={cached_tokens}/{len(prompt)} "
        f"({cached_frac:.0%}) -> load_bearing={pin_load_bearing}"
    )
    if not tokens_equal:
        print(f"  ref[:16]={ref_tokens[:16]}")
        print(f"  res[:16]={res_tokens[:16]}")

    # PASS = byte-exact continuation (the flagship claim). logprobs corroborate.
    passed = tokens_equal and (not ref_logprobs or logprobs_equal)

    # A genuine NEGATIVE: we DID observe the restore's block table and it did
    # NOT reuse the pinned blocks despite a byte-exact continuation -> the
    # continuation came from recompute, not restore. Flag for re-adjudication
    # rather than rubber-stamping (frontier rule).
    if passed and not pin_load_bearing:
        print(
            "\nSTAGE-2 NEGATIVE — needs Fable re-adjudication: continuation is "
            "byte-exact but num_cached_tokens shows the restore RECOMPUTED the "
            "prefix instead of riding the cache (cached_frac <= 0.5 or no full "
            "attn block reused). Inspect find_longest_cache_hit (hash/eviction)."
        )
        return 1

    if passed and not reuse_checked:
        print(
            "\nNOTE: pinned-block reuse was INCONCLUSIVE (missed the restore "
            "request's block table mid-generation — the cache-hit decode "
            "outran the poll). The byte-exact continuation still holds. To "
            "make the pin provably load-bearing, re-run with --churn-fillers "
            ">0 and/or a larger --continuation-tokens window."
        )

    print(
        f"\nSTAGE-2 {'PASS' if passed else 'FAIL'} "
        f"(zero-kernel restore reproduces the continuation byte-exactly via "
        f"the pinned attn+GDN state: {passed}; pin_load_bearing="
        f"{pin_load_bearing}, reuse_checked={reuse_checked})"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
