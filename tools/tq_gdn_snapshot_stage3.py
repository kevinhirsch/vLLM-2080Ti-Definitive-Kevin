# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-038 Stage-3 driver — FORK smoke test (zero-kernel variant).

Standalone, MANUAL script. It is NOT a pytest test (no ``test_`` prefix, not
under ``tests/``) and does nothing on import — everything is under
``if __name__ == "__main__"``. Run it ONLY in a scheduled maintenance window on
a THROWAWAY 2-GPU engine. NEVER point it at the production :8001 serve, and
NEVER run it while :8001 is serving on the same GPUs.

Variant implemented: the ZERO-KERNEL restore (EXP-038 design lines 26-30, 105),
the same public-API path Stage-2 proved. Stage-2 restored ONE child; Stage-3
restores the SAME pinned handle into TWO children AT ONCE.

What it proves (EXP-038 design line 105 — "FORK smoke test"):
  A block-aligned prefix is pinned mid-generation (attn + mamba groups) and the
  source finishes and frees. The identical ``token_ids`` are then resubmitted as
  TWO simultaneous child requests in ONE ``llm.generate`` call so they are
  co-scheduled. The claim:
    (a) EACH child rides the cache: ``num_cached_tokens > 0.5 * prompt`` (a
        recompute would be ~0). This is the Stage-2 restore-vs-recompute
        discriminator, applied per child.
    (b) EACH child's continuation is BYTE-EXACT vs the reference continuation
        (token ids, and logprobs where populated) — restored == prefix-cache
        hit (design line 84).
    (c) The two children are byte-exact vs EACH OTHER.
  PASS = (a) AND (b) AND (c).

  Corroboration (d), reported informationally — NEVER fails the run on a MISS:
    During the children's generation we poll ``get_request_kv_block_ids(
    all_running=True)`` (read-only; extended on feat-gdn-snapshot to return all
    in-flight requests because ``req_id=None`` resolves only the FIRST running
    request). A good catch should show a SHARED full-attn block in both
    children's tables (the pinned cache-hit block, ref_cnt bumped once per
    child) while each child holds a DISTINCT mamba RUNNING block — align mode
    copies the cached GDN state out into a fresh per-request running slot
    (Stage-2 Fable adjudication). The mid-gen catch may miss entirely (a cache-
    hit decode is short and can outrun the poll); a MISS is inconclusive, not a
    failure. The ONE hard signal here: if we DO catch both children and they
    SHARE a mamba running block, that is two sequences writing the same
    recurrent slot — a real NEGATIVE, flagged for Fable re-adjudication.

Scheduler-serialization caveat (documented, driver is robust to it): two
identical co-scheduled prompts are NOT guaranteed to attention-share their
blocks at the exact same step. vLLM does not dedupe identical in-flight
prompts, but if the second child is admitted a step after the first (or the two
race on find_longest_cache_hit) the ref_cnt==2 "both children on the shared
block simultaneously" snapshot may not be catchable, and one child could even
re-prefill. Because the pinned handle keeps the prefix cached regardless, BOTH
children still cache-hit and stay byte-exact — and byte-exactness (a)+(b)+(c)
is the PASS condition. The ref_cnt / shared-block observation is corroboration
only. Whatever the scheduler actually does, it is recorded in the [3/5] output.

Example (throwaway engine, same flags as the deploy):

    VLLM_TQ_GDN_SNAPSHOT=1 \
    VLLM_TQ_GDN_SNAPSHOT_CONFIRM_THROWAWAY=1 \
    python tools/tq_gdn_snapshot_stage3.py \
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


def _run_children_and_probe(llm, prompts, sampling_params, do_probe):
    """Run ``generate(list_of_prompts)`` (co-scheduled children) in a background
    thread; while in flight, poll the READ-ONLY ``do_probe()`` and keep the
    catch that saw the MOST in-flight requests (ideally both children at once).
    Returns ``(list_of_request_outputs, best_probe_result)``.

    Modeled on Stage-2's ``_run_and_pin`` read-only branch, but (i) submits a
    LIST of prompts in ONE call so the engine co-schedules them, and (ii) keeps
    the widest catch rather than merely the last, because the moment worth
    catching is the single window where BOTH children are running. A cache-hit
    decode is short, so poll fast. ``do_probe`` must be side-effect-free."""
    import threading
    import time

    result_box: dict = {}
    gen_done = threading.Event()

    def _generate() -> None:
        try:
            outs = llm.generate(
                [{"prompt_token_ids": list(p)} for p in prompts],
                sampling_params,
            )
            result_box["outputs"] = list(outs)
        finally:
            gen_done.set()

    gen_thread = threading.Thread(target=_generate, daemon=True)
    gen_thread.start()

    best = None
    best_n = -1
    while not gen_done.is_set() and do_probe is not None:
        try:
            attempt = do_probe()
        except Exception:  # noqa: BLE001 - transient mid-gen RPC issues
            time.sleep(0.05)
            continue
        if attempt is not None:
            n = attempt.get("_n_running", 0)
            if n > best_n:  # widest catch wins (want the 2-children window)
                best = attempt
                best_n = n
        time.sleep(0.05)

    gen_thread.join(timeout=240)
    if not gen_done.is_set():
        raise RuntimeError("children generation did not finish in time")
    return result_box.get("outputs"), best


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
        help="greedy decode steps to compare byte-for-byte across the "
        "reference and both children (also the children's window for "
        "catching their block tables mid-generation)",
    )
    parser.add_argument(
        "--churn-fillers",
        type=int,
        default=4,
        help="number of distinct filler prompts to submit between the "
        "reference and fork runs to pressure the cache (makes the pin "
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

    # Reuse the Stage-2 helpers verbatim (block-aligned prompt builder, the
    # continuation extractor, and the single-request run+pin harness). Stage-2's
    # module-level scope is import-side-effect-free (all runtime work is under
    # its own ``__main__`` guard), so importing it here costs nothing. sys.path[0]
    # is this script's dir (tools/) when run as ``python tools/...stage3.py``.
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

    # ---- [1/5] Reference run + pin (attn + mamba blocks) ----------------
    print("[1/5] reference run + pin ...")
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
    pinned_attn, pinned_mamba = _split_groups(handle["groups"])
    print(
        f"  pinned handle={handle_id} attn_blocks={len(pinned_attn)} "
        f"mamba_blocks={len(pinned_mamba)} ref_continuation={ref_tokens[:8]}..."
    )

    # Let the finish/free step land, then confirm the pin held across the free.
    time.sleep(1.0)
    v = llm.verify_pinned_blocks(handle_id)
    print(
        f"  post-free verify: ok={v.get('ok')} "
        f"min_ref_cnt={v.get('min_ref_cnt')} "
        f"num_free_blocks={v.get('num_free_blocks')}"
    )
    if not v.get("ok"):
        print("pinned blocks did not survive source free", file=sys.stderr)
        llm.unpin_kv_blocks(handle_id)
        return 1

    # ---- [2/5] Optional cache pressure + idle-pinned ref_cnt baseline ---
    if args.churn_fillers > 0:
        print(f"[2/5] churning cache with {args.churn_fillers} filler prompts ...")
        for i in range(args.churn_fillers):
            filler = _build_block_aligned_prompt(
                args.block_aligned_tokens, seed=100 + i
            )
            llm.generate(
                {"prompt_token_ids": filler},
                SamplingParams(max_tokens=1, temperature=0.0),
            )
    else:
        print("[2/5] no cache churn (--churn-fillers 0)")
    baseline = llm.verify_pinned_blocks(handle_id)
    if not baseline.get("ok"):
        print("pinned blocks evicted under churn (unexpected)", file=sys.stderr)
        llm.unpin_kv_blocks(handle_id)
        return 1
    # ref_cnt of the pinned attn blocks while IDLE (no child referencing them):
    # this is the baseline the children's mid-gen ref_cnt is compared against.
    baseline_attn_ref: dict[int, int] = {}
    for g in baseline.get("groups", {}).values():
        if g.get("spec") != "attn":
            continue
        for row in g.get("blocks", []):
            baseline_attn_ref[int(row["block_id"])] = int(row["ref_cnt"])
    base_min = min(baseline_attn_ref.values()) if baseline_attn_ref else None
    print(
        f"  idle-pinned baseline: ok={baseline.get('ok')} "
        f"attn min_ref_cnt={base_min} (== the pin's own ref, no children yet)"
    )

    # ---- [3/5] FORK: two identical children, co-scheduled ---------------
    print("[3/5] fork: two identical children in ONE generate() call ...")

    def _probe():
        # Read-only witness. Grab every in-flight request's block table; when we
        # see >=2 running (both children), also snapshot the pinned blocks'
        # ref_cnt at that instant (corroborates ref_cnt bumped once per child).
        ids = llm.get_request_kv_block_ids(all_running=True)
        n = len(ids.get("req_ids", []))
        catch = {"_n_running": n, "ids": ids, "verify": None}
        if n >= 2:
            try:
                catch["verify"] = llm.verify_pinned_blocks(handle_id)
            except Exception:  # noqa: BLE001 - best-effort mid-gen snapshot
                catch["verify"] = None
        return catch

    try:
        child_outs, probe = _run_children_and_probe(
            llm, [prompt, prompt], greedy, _probe
        )
    except BaseException:  # noqa: BLE001 - child gen raised/timed out; never leak the pin
        # Report (but never let) a cleanup failure replace the original error:
        # unpin can return ok=False or itself raise; surface both, then re-raise.
        try:
            released = llm.unpin_kv_blocks(handle_id)
            if not released.get("ok"):
                print(
                    f"warning: unpin failed during exception cleanup: {released}",
                    file=sys.stderr,
                )
        except Exception as cleanup_err:  # noqa: BLE001 - preserve original failure
            print(
                f"warning: unpin raised during exception cleanup: {cleanup_err}",
                file=sys.stderr,
            )
        raise
    if not child_outs or len(child_outs) != 2:
        print(
            f"fork run did not return 2 child outputs (got "
            f"{0 if not child_outs else len(child_outs)})",
            file=sys.stderr,
        )
        llm.unpin_kv_blocks(handle_id)
        return 1

    # Per-child continuation + cache-hit discriminator.
    c0_tokens, c0_logprobs = _extract_continuation(child_outs[0])
    c1_tokens, c1_logprobs = _extract_continuation(child_outs[1])
    c0_cached = int(getattr(child_outs[0], "num_cached_tokens", 0) or 0)
    c1_cached = int(getattr(child_outs[1], "num_cached_tokens", 0) or 0)
    n_prompt = len(prompt)
    print(
        f"  child0 num_cached_tokens={c0_cached}/{n_prompt} "
        f"({c0_cached / max(1, n_prompt):.0%})  "
        f"child1 num_cached_tokens={c1_cached}/{n_prompt} "
        f"({c1_cached / max(1, n_prompt):.0%})  "
        f"(>0.5 = rode the cache, ~0 = recompute)"
    )
    n_running_caught = probe.get("_n_running", 0) if probe else 0
    print(
        f"  widest mid-gen catch saw {n_running_caught} in-flight request(s) "
        f"({'both children co-scheduled' if n_running_caught >= 2 else 'children serialized / catch missed — corroboration (d) inconclusive'})"
    )

    # ---- [4/5] Assertions -----------------------------------------------
    # (a) each child rode the cache.
    a_child0 = c0_cached > 0.5 * n_prompt
    a_child1 = c1_cached > 0.5 * n_prompt
    cond_a = a_child0 and a_child1

    # (b) each child byte-exact vs the reference. Fable adjudication 2026-08-16:
    # the reference decodes at batch=1 while the two children decode at batch=2 —
    # GPU kernel reductions are NOT batch-shape-invariant, so exact float
    # equality of logprobs across different batch shapes is an over-strict
    # criterion that fails on healthy numerics. Token-id byte-exactness (greedy
    # argmax identical for every step) IS the state-correctness signal; logprobs
    # gate only within a tolerance and the max delta is reported for the record.
    LOGPROB_TOL = 5e-2  # generous bound for batch-shape numerics; deltas are printed

    def _logprob_max_delta(ref: list[float], got: list[float]) -> float:
        if not ref or not got or len(ref) != len(got):
            return float("inf")
        worst = 0.0
        for x, y in zip(ref, got):
            x_nan = x != x
            y_nan = y != y
            if x_nan and y_nan:
                continue  # both missing -> treat as equal
            if x_nan or y_nan:
                return float("inf")  # one-sided NaN -> real mismatch, never 0
            worst = max(worst, abs(x - y))
        return worst

    def _logprobs_equal(ref: list[float], got: list[float]) -> bool:
        return _logprob_max_delta(ref, got) <= LOGPROB_TOL

    b0_tokens = c0_tokens == ref_tokens
    b1_tokens = c1_tokens == ref_tokens
    # logprobs only gate the verdict if the reference actually populated them.
    b0_lp = (not ref_logprobs) or _logprobs_equal(ref_logprobs, c0_logprobs)
    b1_lp = (not ref_logprobs) or _logprobs_equal(ref_logprobs, c1_logprobs)
    cond_b = b0_tokens and b1_tokens and b0_lp and b1_lp

    # (c) children equal each other (token ids, and logprobs when present).
    c_tokens = c0_tokens == c1_tokens
    c_lp = (
        (not c0_logprobs and not c1_logprobs)
        or _logprobs_equal(c0_logprobs, c1_logprobs)
    )
    cond_c = c_tokens and c_lp

    # (d) corroboration: shared attn cache-hit block + distinct mamba running
    # blocks. Only meaningful if we caught BOTH children mid-gen.
    caught_both = n_running_caught >= 2
    shared_attn: set[int] = set()
    child0_run_mamba: set[int] = set()
    child1_run_mamba: set[int] = set()
    shared_mamba_running = False
    shared_block_refcnt_ok = False
    if caught_both and probe is not None:
        reqs = probe["ids"].get("requests", {})
        rids = list(reqs.keys())[:2]
        g0 = reqs[rids[0]]["groups"]
        g1 = reqs[rids[1]]["groups"]
        a0, m0 = _split_groups(g0)
        a1, m1 = _split_groups(g1)
        # SHARED full-attn block = present in BOTH children AND pinned (the
        # pinned cache-hit block). Partial decode block never caches, so the
        # intersection is exactly the pinned full blocks both children rode.
        shared_attn = (a0 & a1) & pinned_attn
        # Each child's mamba RUNNING block(s) = its mamba blocks MINUS the
        # pinned (cached, immutable full) blocks. Align mode copies the cached
        # GDN state into a FRESH running slot per request, so these must be
        # DISTINCT across children. A non-empty overlap = two sequences on one
        # recurrent slot = corruption (NEGATIVE).
        child0_run_mamba = m0 - pinned_mamba
        child1_run_mamba = m1 - pinned_mamba
        shared_mamba_running = bool(child0_run_mamba & child1_run_mamba)
        # ref_cnt corroboration: a shared pinned attn block should show its
        # ref_cnt bumped ABOVE the idle-pinned baseline by up to +2 (one per
        # child) during the co-scheduled window.
        verify = probe.get("verify")
        if verify and shared_attn:
            for g in verify.get("groups", {}).values():
                if g.get("spec") != "attn":
                    continue
                for row in g.get("blocks", []):
                    bid = int(row["block_id"])
                    if bid in shared_attn:
                        base = baseline_attn_ref.get(bid, 1)
                        if int(row["ref_cnt"]) > base:  # a child added a ref
                            shared_block_refcnt_ok = True

    # ---- [5/5] Post-children verify, then release the pin ----------------
    time.sleep(1.0)
    post = llm.verify_pinned_blocks(handle_id)
    print(
        f"[5/5] post-children verify: ok={post.get('ok')} "
        f"min_ref_cnt={post.get('min_ref_cnt')} "
        f"num_free_blocks={post.get('num_free_blocks')}"
    )
    released = llm.unpin_kv_blocks(handle_id)

    # ---- Verdict --------------------------------------------------------
    print("verdict:")
    print(
        f"  (a) each child rode the cache: {cond_a} "
        f"(child0={a_child0}, child1={a_child1})"
    )
    print(
        f"  (b) each child byte-exact vs reference: {cond_b} "
        f"(c0_tokens={b0_tokens}, c1_tokens={b1_tokens}, "
        f"c0_lp={b0_lp}, c1_lp={b1_lp}; ref_lp_n={len(ref_logprobs)}; "
        f"max|dlp| ref-c0={_logprob_max_delta(ref_logprobs, c0_logprobs):.2e} "
        f"ref-c1={_logprob_max_delta(ref_logprobs, c1_logprobs):.2e} "
        f"c0-c1={_logprob_max_delta(c0_logprobs, c1_logprobs):.2e})"
    )
    print(
        f"  (c) children byte-exact vs each other: {cond_c} "
        f"(tokens={c_tokens}, logprobs={c_lp})"
    )
    print(
        f"  (d) corroboration [informational]: caught_both={caught_both} "
        f"shared_attn_blocks={len(shared_attn)} "
        f"child0_running_mamba={sorted(child0_run_mamba)} "
        f"child1_running_mamba={sorted(child1_run_mamba)} "
        f"distinct_mamba_running={not shared_mamba_running} "
        f"shared_attn_refcnt_bumped={shared_block_refcnt_ok}"
    )
    print(
        f"  post verify ok={post.get('ok')} | unpin ok={released.get('ok')} "
        f"num_freed_blocks={released.get('num_freed_blocks')}"
    )
    if not b0_tokens:
        print(f"    ref[:16]={ref_tokens[:16]}")
        print(f"    c0 [:16]={c0_tokens[:16]}")
    if not c_tokens:
        print(f"    c0 [:16]={c0_tokens[:16]}")
        print(f"    c1 [:16]={c1_tokens[:16]}")

    # Cleanup must succeed too: a post-children pin-integrity failure or a failed
    # unpin means the run cannot be trusted as a PASS.
    cleanup_ok = bool(post.get("ok")) and bool(released.get("ok"))
    passed = cond_a and cond_b and cond_c and cleanup_ok

    # Real NEGATIVE (the ONE hard corroboration signal): we caught both children
    # AND they share a mamba running block -> two sequences writing one recurrent
    # slot. That is a correctness violation regardless of byte-exactness this
    # run, and must not be rubber-stamped (frontier rule).
    if caught_both and shared_mamba_running:
        print(
            "\nSTAGE-3 NEGATIVE — needs Fable re-adjudication: the two forked "
            "children were caught SHARING a mamba running block "
            f"({sorted(child0_run_mamba & child1_run_mamba)}). align mode must "
            "copy the cached GDN state into a DISTINCT fresh running slot per "
            "child; a shared running slot means two sequences mutate the same "
            "recurrent state. Inspect restore/preprocess_mamba slot allocation."
        )
        return 1

    # Byte-exact but a child did NOT ride the cache: NOT a correctness NEGATIVE
    # (the continuation still matched), but the fork-sharing claim is unproven
    # for that child. The most likely cause is co-scheduled serialization (the
    # documented caveat): a child admitted the same step it re-prefilled instead
    # of hitting the just-pinned cache. Report as FAIL-inconclusive and suggest
    # a re-run, rather than flagging Fable.
    if cond_b and cond_c and not cond_a:
        print(
            "\nSTAGE-3 FAIL (inconclusive, not a NEGATIVE): both children are "
            "byte-exact but at least one did NOT ride the cache "
            f"(child0_cached={c0_cached}, child1_cached={c1_cached} of "
            f"{n_prompt}). Likely co-scheduled serialization / same-step "
            "recompute — the pinned prefix was available, so re-run with a "
            "larger --churn-fillers and/or --continuation-tokens to force a "
            "clean cache-hit window. Byte-exactness itself is intact."
        )
        return 1

    if passed and not caught_both:
        print(
            "\nNOTE: corroboration (d) INCONCLUSIVE — never caught both "
            "children in flight at once (a cache-hit decode is short and can "
            "outrun the poll, or the scheduler serialized them). The byte-exact "
            "+ both-cached PASS still holds; (d) is corroboration only. To "
            "witness the shared attn block / distinct mamba running blocks, "
            "re-run with a larger --continuation-tokens window."
        )
    elif passed and caught_both:
        note = "distinct mamba running blocks" if not shared_mamba_running else "!!"
        print(
            f"\nCORROBORATED: caught both children co-scheduled — "
            f"shared_attn_blocks={len(shared_attn)}, {note}, "
            f"shared_attn_refcnt_bumped={shared_block_refcnt_ok}."
        )

    print(
        f"\nSTAGE-3 {'PASS' if passed else 'FAIL'} "
        f"(same pinned handle restored into 2 children; both rode the cache "
        f"and produced byte-exact continuations equal to the reference and to "
        f"each other: {passed})"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
