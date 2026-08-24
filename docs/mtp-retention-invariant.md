# MTP align-mode retention: safety invariant (PR-B evidence, part a)

Answers the maintainer question from weicj/vLLM-2080Ti-Definitive#126: *"What
invariant proves that retaining this block is safe when an uncached prompt
tail remains?"* Companion to a performance A/B (part b) to be attached to the
follow-up PR.

## The back-off being relaxed

`Scheduler._align_prefill_chunk` (vllm/v1/core/sched/scheduler.py:287-301):
under EAGLE, the last matched full-attention block is dropped
(`last_cache_position -= block_size`) so all hybrid groups describe the same
prefix. EAGLE needs this because its cache hits are shifted by one
(`shift_computed_tokens=1`): the effective resume point `hit-1` is not
block-aligned, align mode has no Mamba state materialized there, and the only
consistent boundary is one block earlier.

MTP inherits this path (`use_eagle=True` for `method="mtp"`), paying a full
block (block_size tokens, ~2-4K at our shapes) of re-prefill per resumed
request — without needing to.

## The invariant

> A retained aligned block is safe iff the scheduled chunk resumes exactly at
> a materialized Mamba boundary AND at least one prompt token remains after it
> (a non-empty uncached tail). Under that condition:
>
> 1. **Full-attention KV** for every retained block is complete and
>    position-exact: the blocks end at `last_cache_position =
>    round_down(num_tokens, block_size)`, they were filled by an earlier
>    aligned prefill, and align mode's chunking rule (every non-final chunk
>    ends on a block boundary) guarantees no partially-filled retained block.
> 2. **Mamba recurrent state** at the boundary is valid by align mode's own
>    contract: a cached block at index i represents the recurrent state after
>    exactly `(i+1) * block_size` tokens — the resume point.
> 3. **Proposer state**: the MTP head is stateless across scheduler steps; the
>    only hidden state it consumes is that of the final prompt position, which
>    lies strictly inside the uncached tail and is produced by the tail's own
>    forward pass. Unlike EAGLE's shifted-hit geometry, the proposer never
>    reads a hidden state at or before the retained boundary.

The gate in the change enforces exactly the tail-nonempty condition:
`retain_final_mtp_block = retain_flag and last_cache_position <
request.num_tokens`.

## Edge cases

| Case | What happens | Safe? |
| --- | --- | --- |
| Prompt length exactly block-aligned (`num_tokens % block_size == 0`) | `last_cache_position == num_tokens` → gate false → EAGLE back-off preserved | ✅ retention self-disables; no boundary-state ambiguity |
| Tail of exactly 1 token | Tail forward pass runs, produces the final hidden state; chunk math (`prefill_end = num_tokens - 1` on resume) unchanged | ✅ minimal covered case |
| No prefix-cache hit at all | `num_computed_tokens < last_cache_position` path unchanged; retention only widens what may be *kept*, never what is *hit* | ✅ no new hit surface |
| EAGLE proper (`method != "mtp"`) | `retain_mamba_align_mtp_cache_block` requires `method == "mtp"` at construction | ✅ EAGLE semantics untouched |
| Offload/external hits with non-divisor group sizes | Orthogonal: hit boundaries re-aligned by the `_align_hit_boundary` fix (weicj/vLLM-2080Ti-Definitive#133) | ✅ composes |
| Async scheduling + zero-accept stale rows | Pre-existing, independent defect in the align CPU path (vllm-project/vllm#51508, port planned); occurs with or without retention | ⚠️ tracked separately |

## What EAGLE needs that MTP does not

The one-block back-off protects EAGLE's *shifted* resume geometry: with
`shift_computed_tokens=1`, EAGLE's proposer requires the hidden state of the
last *cached* position, which a boundary-exact resume cannot supply — hence
dropping the block to force its recomputation. MTP's proposer requires only
the hidden state of the last *prompt* position, supplied by the tail. The
back-off is therefore load-bearing for EAGLE and pure waste for MTP with a
non-empty tail — which is precisely the case the flag gates.

## Default posture

`VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK` defaults to off; the change is
inert unless explicitly enabled. Promotion upstream waits on part (b): a
measured A/B on the dual-2080 Ti reference rig quantifying saved re-prefill
per resumed request and confirming no quality delta (evalkit gate).
