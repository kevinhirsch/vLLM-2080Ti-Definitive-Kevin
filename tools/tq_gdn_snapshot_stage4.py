# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-038 Stage-4 driver — scheduler-level FORK API (fork WITHOUT resubmit).

Standalone, MANUAL script. It is NOT a pytest test (no ``test_`` prefix, not
under ``tests/``) and does nothing on import — everything is under
``if __name__ == "__main__"``. Run it ONLY in a scheduled maintenance window on
a THROWAWAY 2-GPU engine. NEVER point it at the production :8001 serve, and
NEVER run it while :8001 is serving on the same GPUs.

What this proves vs Stage-3: Stage-3 forked via RESUBMIT (identical token_ids
through ``llm.generate``). Stage-4 forks via the new ``llm.fork_from_handle``
utility — children are built INSIDE EngineCore from the pinned handle's cached
prefix and adopt the pinned donor blocks through the local prefix cache (the
proven Stage-2/3 adoption path). No resubmit, no re-tokenize, no per-child API
round-trip. See docs/exp038-stage4-fork-api.md for the insertion-point analysis
and the honest gap list.

The claim (PASS = A and B and C and D):
  (A) BYTE-EXACT greedy child. fork(handle, n=3) with THREE DIFFERENT sampling
      params (greedy + two temperatures); the GREEDY child's continuation is
      byte-exact (token ids) vs the reference continuation. Token-id argmax is
      batch-shape-invariant (Stage-3 Fable adjudication), so co-scheduling with
      two temperature siblings does not perturb the greedy argmax.
  (B) ZERO PREFILL COMPUTE. Every child's prompt phase shows
      ``num_cached_tokens`` ~= the full prefix (a recompute would be ~0). This is
      the Stage-2 restore-vs-recompute discriminator, applied per child. NB it is
      the ~zero-*compute* proxy, not literally zero scheduled tokens — see the
      residual-gap note in the verdict.
  (C) DISTINCT MAMBA SLOTS + SHARED ATTN. The fork's mid-gen block-table snapshot
      (captured engine-side while all children co-run) shows a SHARED full-attn
      cache-hit block across children and a DISTINCT mamba running slot per child.
      A shared mamba running slot = two sequences on one recurrent state = a hard
      NEGATIVE (Stage-3 rule).
  (D) NO LEAKED BLOCKS. After all children finish + free, the pool's free-block
      count returns to the pre-fork baseline (the pinned handle stays pinned;
      only unpin releases it).

Per-child sampling/max_tokens divergence is also demonstrated (child 2 is given a
distinct, smaller max_tokens). Honest note (see the doc): divergence is NOT unique
to fork — resubmit supports it too because block hashes exclude sampling params.

Example (throwaway engine, same flags as the deploy):

    VLLM_TQ_GDN_SNAPSHOT=1 \
    VLLM_TQ_GDN_SNAPSHOT_CONFIRM_THROWAWAY=1 \
    python tools/tq_gdn_snapshot_stage4.py \
        --model /path/to/Qwen3.8-27B-hybrid \
        --tensor-parallel-size 2 \
        --kv-cache-dtype turboquant_k8v4 \
        --mamba-cache-mode align \
        --dtype half \
        --gpu-memory-utilization 0.82 \
        --block-aligned-tokens 2112 \
        --continuation-tokens 64 \
        --churn-fillers 4
"""

from __future__ import annotations

import argparse
import sys


def _split_groups(groups: dict) -> tuple[set[int], set[int]]:
    """(attn_block_ids, mamba_block_ids) from a ``{gid: {spec, block_ids}}``."""
    attn: set[int] = set()
    mamba: set[int] = set()
    for g in groups.values():
        target = attn if g["spec"] == "attn" else mamba
        target.update(int(b) for b in g["block_ids"])
    return attn, mamba


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
        help="greedy decode steps to compare byte-for-byte across the reference "
        "and the greedy fork child",
    )
    parser.add_argument(
        "--churn-fillers",
        type=int,
        default=4,
        help="distinct filler prompts submitted between the reference and the "
        "fork to pressure the cache (makes the pin provably load-bearing). "
        "0 = no pressure.",
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

    # Reuse the Stage-2 helpers verbatim (block-aligned prompt builder, the
    # continuation extractor, and the single-request run+pin harness). Stage-2's
    # module scope is import-side-effect-free (all runtime work is under its own
    # ``__main__`` guard). sys.path[0] is this script's dir (tools/) when run as
    # ``python tools/...stage4.py``.
    from tq_gdn_snapshot_stage2 import (
        _build_block_aligned_prompt,
        _extract_continuation,
        _run_and_pin,
    )

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

    # ---- [1/6] Reference run + pin (attn + mamba blocks) ----------------
    print("[1/6] reference run + pin ...")
    ref_out, handle = _run_and_pin(
        llm, prompt, greedy,
        do_pin=lambda: llm.pin_request_kv_blocks(),
        stop_on_first=True,
    )
    if ref_out is None or handle is None or handle.get("num_pinned_blocks", 0) <= 0:
        print("failed to pin the reference request in flight", file=sys.stderr)
        return 1
    handle_id = handle["handle_id"]
    ref_tokens, _ref_logprobs = _extract_continuation(ref_out)
    pinned_attn, pinned_mamba = _split_groups(handle["groups"])
    n_prefix = len(prompt)
    print(
        f"  pinned handle={handle_id} attn_blocks={len(pinned_attn)} "
        f"mamba_blocks={len(pinned_mamba)} ref_continuation={ref_tokens[:8]}..."
    )

    # Let the finish/free step land, then confirm the pin held across the free.
    time.sleep(1.0)
    v = llm.verify_pinned_blocks(handle_id)
    print(
        f"[2/6] post-free verify: ok={v.get('ok')} "
        f"min_ref_cnt={v.get('min_ref_cnt')} "
        f"num_free_blocks={v.get('num_free_blocks')}"
    )
    if not v.get("ok"):
        print("pinned blocks did not survive source free", file=sys.stderr)
        llm.unpin_kv_blocks(handle_id)
        return 1

    # ---- [3/6] Optional cache pressure ----------------------------------
    if args.churn_fillers > 0:
        print(f"[3/6] churning cache with {args.churn_fillers} filler prompts ...")
        for i in range(args.churn_fillers):
            filler = _build_block_aligned_prompt(
                args.block_aligned_tokens, seed=100 + i
            )
            llm.generate(
                {"prompt_token_ids": filler},
                SamplingParams(max_tokens=1, temperature=0.0),
            )
        vv = llm.verify_pinned_blocks(handle_id)
        if not vv.get("ok"):
            print("pinned blocks evicted under churn (unexpected)", file=sys.stderr)
            llm.unpin_kv_blocks(handle_id)
            return 1
    else:
        print("[3/6] no cache churn (--churn-fillers 0)")

    # ---- [4/6] FORK via fork_from_handle: 3 DIFFERENT sampling params ----
    # child 0: greedy  (must reproduce the reference continuation byte-exactly)
    # child 1: temperature 0.7, fixed seed (divergent sampling)
    # child 2: temperature 1.0, fixed seed, DISTINCT (smaller) max_tokens
    child2_max = max(8, args.continuation_tokens // 2)
    child_specs = [
        {"temperature": 0.0, "max_tokens": args.continuation_tokens, "logprobs": 1},
        {"temperature": 0.7, "top_p": 0.95, "seed": 1234,
         "max_tokens": args.continuation_tokens},
        {"temperature": 1.0, "top_p": 0.95, "seed": 5678,
         "max_tokens": child2_max},
    ]
    print(
        f"[4/6] fork_from_handle(n=3): greedy(max={args.continuation_tokens}), "
        f"temp0.7(max={args.continuation_tokens}), temp1.0(max={child2_max}) ..."
    )
    fork = llm.fork_from_handle(handle_id, child_specs)
    children = fork.get("children", [])
    if len(children) != 3:
        print(
            f"fork returned {len(children)} children (expected 3)", file=sys.stderr
        )
        llm.unpin_kv_blocks(handle_id)
        return 1
    c0, c1, c2 = children
    for idx, c in enumerate(children):
        print(
            f"  child{idx}: num_cached_tokens={c['num_cached_tokens']}/{n_prefix} "
            f"({c['num_cached_tokens'] / max(1, n_prefix):.0%})  "
            f"out_tokens={c['num_output_tokens']}  finish={c['finish_reason']}  "
            f"temp={c['spec'].get('temperature')}"
        )
    mid = fork.get("midgen_block_tables", {})
    n_running_caught = int(mid.get("n_running", 0))
    pre_free = int(fork.get("pre_free_blocks", -1))
    post_free = int(fork.get("post_free_blocks", -2))
    print(
        f"  widest mid-gen catch: {n_running_caught} children co-running; "
        f"pre_free_blocks={pre_free} post_free_blocks={post_free} "
        f"steps={fork.get('steps')}"
    )

    # ---- [5/6] Assertions -----------------------------------------------
    # (A) greedy child byte-exact vs reference (token ids — the state-correctness
    # signal per Stage-3 Fable adjudication).
    greedy_tokens = [int(t) for t in c0["token_ids"]]
    cond_a = greedy_tokens == ref_tokens

    # (B) every child rode the cache (~zero prefill compute).
    per_child_cached = [c["num_cached_tokens"] > 0.5 * n_prefix for c in children]
    cond_b = all(per_child_cached)

    # (C) mid-gen: shared attn full block across children + distinct mamba
    # running slot per child. Only decisive if we caught >=2 children co-running.
    reqs = mid.get("requests", {})
    caught = list(reqs.keys())
    shared_attn: set[int] = set()
    per_child_run_mamba: list[set[int]] = []
    shared_mamba_running = False
    if len(caught) >= 2:
        attn_sets = []
        for rid in caught:
            a, m = _split_groups(reqs[rid]["groups"])
            attn_sets.append(a)
            per_child_run_mamba.append(m - pinned_mamba)  # running slot(s)
        # SHARED full-attn cache-hit block = in every caught child AND pinned.
        shared_attn = set.intersection(*attn_sets) & pinned_attn if attn_sets else set()
        # Distinct running mamba slots: pairwise-disjoint across children.
        seen: set[int] = set()
        for rm in per_child_run_mamba:
            if rm & seen:
                shared_mamba_running = True
            seen |= rm
    cond_c = (
        len(caught) >= 2
        and len(shared_attn) > 0
        and not shared_mamba_running
        and all(len(rm) > 0 for rm in per_child_run_mamba)
    )

    # (D) leak invariant: free-block count returns to the pre-fork baseline.
    cond_d = pre_free >= 0 and post_free >= 0 and post_free == pre_free

    # Divergence demonstration (informational): child 2 honored a distinct
    # max_tokens, and the temperature children need not equal the greedy child.
    child2_len_ok = c2["num_output_tokens"] <= child2_max
    temp_diverged = (
        [int(t) for t in c1["token_ids"]] != greedy_tokens
        or [int(t) for t in c2["token_ids"]] != greedy_tokens
    )

    # ---- [6/6] Post-fork verify, then release the pin --------------------
    post = llm.verify_pinned_blocks(handle_id)
    released = llm.unpin_kv_blocks(handle_id)
    print(
        f"[6/6] post-fork verify: ok={post.get('ok')} "
        f"min_ref_cnt={post.get('min_ref_cnt')} | unpin ok={released.get('ok')} "
        f"num_freed_blocks={released.get('num_freed_blocks')}"
    )

    # ---- Verdict --------------------------------------------------------
    print("verdict:")
    print(f"  (A) greedy child byte-exact vs reference: {cond_a}")
    print(
        f"  (B) every child ~zero prefill compute (num_cached>50%): {cond_b} "
        f"(per_child={per_child_cached})"
    )
    print(
        f"  (C) shared attn + distinct mamba running slots: {cond_c} "
        f"(caught={len(caught)} shared_attn_blocks={len(shared_attn)} "
        f"distinct_mamba={not shared_mamba_running} "
        f"per_child_running_mamba={[sorted(rm) for rm in per_child_run_mamba]})"
    )
    print(
        f"  (D) no leaked blocks (post_free==pre_free): {cond_d} "
        f"(pre={pre_free} post={post_free})"
    )
    print(
        f"  [info] per-child divergence: child2_max_tokens_honored={child2_len_ok} "
        f"(<= {child2_max}); temperature_children_diverged_from_greedy="
        f"{temp_diverged}"
    )
    if not cond_a:
        print(f"    ref[:16]    ={ref_tokens[:16]}")
        print(f"    greedy[:16] ={greedy_tokens[:16]}")

    # Hard NEGATIVE (Stage-3 rule): caught children sharing a mamba running slot.
    if len(caught) >= 2 and shared_mamba_running:
        print(
            "\nSTAGE-4 NEGATIVE — needs Fable re-adjudication: forked children "
            "were caught SHARING a mamba running slot. align mode must copy the "
            "cached GDN state into a DISTINCT fresh running slot per child; a "
            "shared running slot means two sequences mutate one recurrent state. "
            "Inspect restore/preprocess_mamba slot allocation."
        )
        return 1

    # Byte-exact but a child did not ride the cache: FAIL-inconclusive (the pin
    # was available; likely a same-step recompute), not a correctness NEGATIVE.
    if cond_a and not cond_b:
        print(
            "\nSTAGE-4 FAIL (inconclusive, not a NEGATIVE): greedy child is "
            f"byte-exact but at least one child did NOT ride the cache "
            f"(num_cached={[c['num_cached_tokens'] for c in children]} of "
            f"{n_prefix}). Re-run with larger --churn-fillers / "
            "--continuation-tokens to force a clean cache-hit window."
        )
        return 1

    if not cond_d:
        print(
            "\nSTAGE-4 FAIL: block leak — the pool's free-block count did NOT "
            f"return to the pre-fork baseline (pre={pre_free}, post={post_free}). "
            "A forked child freed fewer/more blocks than it took, or the pin was "
            "perturbed. Inspect per-child free accounting."
        )
        return 1

    if cond_a and cond_b and cond_d and len(caught) < 2:
        print(
            "\nNOTE: corroboration (C) INCONCLUSIVE — the engine-side mid-gen "
            "catch saw <2 children co-running (a cache-hit decode is short). "
            "(A)+(B)+(D) still hold. Re-run with a larger --continuation-tokens "
            "window to witness the shared attn block / distinct mamba slots."
        )

    passed = cond_a and cond_b and cond_c and cond_d

    # Frontier rule — honest gaps flagged for re-adjudication regardless of PASS:
    print(
        "\nNEGATIVE — needs Fable re-adjudication (premise correction): the "
        "task framed per-child sampling divergence as the win over resubmit. In "
        "vLLM V1 the prefix-cache block hash EXCLUDES sampling params, so "
        "resubmit ALSO supports per-child sampling divergence. The genuine wins "
        "of fork_from_handle are: one atomic engine-side call, no ZMQ re-ship of "
        "token_ids, no client-side re-tokenize. (docs/exp038-stage4-fork-api.md "
        "§1)."
    )
    print(
        "NEGATIVE — needs Fable re-adjudication (residual gap): this fork still "
        "incurs ONE internal cache-hit prefill schedule step per child (num_new "
        "= num_tokens - num_cached ~= 1 block). num_cached_tokens ~= full prefix "
        "is the ~zero-COMPUTE proxy, not literally zero scheduled tokens. "
        "Eliminating it needs a direct scheduler.running seed (rejected as too "
        "invasive for staged work). (docs/exp038-stage4-fork-api.md §5, §7)."
    )

    print(
        f"\nSTAGE-4 {'PASS' if passed else 'FAIL'} "
        f"(fork_from_handle forked the pinned handle into 3 children with "
        f"divergent sampling; greedy byte-exact vs reference, all rode the "
        f"cache, distinct mamba slots, no leaked blocks: {passed})"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
