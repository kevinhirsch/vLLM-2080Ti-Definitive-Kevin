# SPDX-License-Identifier: Apache-2.0
"""Tree speculative decoding: the tree sampler and walk (E16 / P6).

Contract: ``/home/kevin/projects/lanes/coexec/results/tree-contract.md`` §5 P6.
Normative semantics: ``tree_reference.tree_walk_ref`` (walk) and
``tree_reference.tree_emit_distribution_ref`` (exact emitted distribution).

Verification is "sample-and-match": draw a target sample ``y`` at **every**
verify row, then walk the tree from the root, following the child whose draft
token equals the sample at the current node.  Every emitted token is an exact
draw from the target conditional at the node it was drawn from, so the walk
is lossless (for drafts that do not depend on ``y``).

Row roles on a tree step (rows chain-first ``[x0, c1..cD, leaves..]``):

* **bonus row** = chain row ``D``.  It is sampled exactly like today's chain
  bonus token: through the full ``Sampler`` (every logits processor, the
  min-p/argmax-invariant processors, top-k/top-p, its own randomness), with
  the chain ``c1..cD`` as the penalty context.
* **target rows** = every other row (``0..D-1`` and every leaf row), i.e.
  ``M - 1`` per request.  They get today's chain *target* treatment
  (allowed-token mask, bad words, min-tokens, penalties, thinking budget,
  temperature, top-k, top-p) but with **path** contexts: row ``r`` sees the
  committed tokens plus the draft tokens on ``path_rows(r)[1:]``.  Then one
  independent sample per row (greedy = argmax; random = exponential race on
  the processed distribution, one fresh draw per row, per-request generator
  when the request is seeded).

With a chain-only spec (``L = 0``) the target rows are exactly today's chain
target rows, the walk is today's acceptance, and greedy output is identical
to the chain ``RejectionSampler``.

Opt-out: ``opt_out_reqs`` (structured output / grammar, decided by the
caller) and requests tracked by the thinking-budget holder get every leaf
draft set to -1 for the walk, so only the chain can be accepted for them --
exactly the chain semantics, which is all their logits processors support.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import torch
import torch.nn as nn

from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words_with_drafts
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.rejection_sampler import (
    GREEDY_TEMPERATURE,
    PLACEHOLDER_TOKEN_ID,
    apply_sampling_constraints,
)
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.tree_spec import TreeSpec, TreeStepOutputs


# ------------------------------------------------------------------ tables
def tree_target_rows(spec: TreeSpec) -> tuple[int, ...]:
    """Rows sampled with the chain *target* treatment: every row but the
    bonus row D, in row order (chain rows 0..D-1, then the leaves)."""
    return tuple(r for r in range(spec.M) if r != spec.D)


def tree_path_draft_cols(spec: TreeSpec) -> tuple[tuple[int, ...], ...]:
    """Per row r: the draft columns (row - 1) of ``path_rows(r)[1:]``, i.e.
    the tokens appended to the committed context when sampling at row r."""
    return tuple(tuple(k - 1 for k in spec.path_rows(r)[1:]) for r in range(spec.M))


def tree_row_contexts(
    spec: TreeSpec,
    output_token_ids: list[list[int]],
    draft_token_ids: list[list[int]],
    rows: Iterable[int] | None = None,
) -> list[list[int]]:
    """Path-aware penalty / bad-words context for each (request, row),
    request-major over ``rows`` (default: the target rows).  Row r's context
    is the committed tokens plus the drafts on the path to r (not a linear
    prefix of the draft list).  A -1 (opted-out) draft stays -1: penalties
    map it to the pad id and bad words never match it."""
    rows = tree_target_rows(spec) if rows is None else tuple(rows)
    cols = tree_path_draft_cols(spec)
    out: list[list[int]] = []
    for committed, drafts in zip(output_token_ids, draft_token_ids):
        for r in rows:
            out.append([*committed, *(drafts[c] for c in cols[r])])
    return out


def apply_leaf_opt_out(
    spec: TreeSpec, draft_token_ids: torch.Tensor, req_indices: Iterable[int]
) -> torch.Tensor:
    """Return drafts [B, M-1] with every leaf column set to -1 for the given
    batch rows (structured output, grammar, thinking budget...).  A -1 leaf
    is never matched by the walk, so those requests can only accept the
    chain.  Returns the input unchanged when there is nothing to opt out."""
    idx = sorted({int(i) for i in req_indices if 0 <= int(i) < draft_token_ids.shape[0]})
    if not idx or spec.L == 0:
        return draft_token_ids
    out = draft_token_ids.clone()
    rows = torch.tensor(idx, dtype=torch.long, device=out.device)
    out[rows, spec.D :] = PLACEHOLDER_TOKEN_ID
    return out


# -------------------------------------------------------------- walk kernel
@triton.jit
def _tree_walk_kernel(
    y_ptr,  # [B, M] int64: target sample at every row
    drafts_ptr,  # [B, M-1] int64: row r's draft at column r-1; -1 = opt-out
    leaf_rows_ptr,  # [max(L,1)] int32
    leaf_depth_ptr,  # [max(L,1)] int32
    out_ptr,  # [B, M] int64 (output) path-ordered, -1 padded
    num_acc_ptr,  # [B] int32 (output)
    rescue_leaf_ptr,  # [B] int32 (output) -1 = none
    rescue_depth_ptr,  # [B] int32 (output) 0 = none
    cvr_ptr,  # [B, D+1] int32 (output) chain view rows
    D: tl.constexpr,
    M: tl.constexpr,
    L: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per request.  Branch-free walk over the static depth loop;
    the output row and the chain view are built in registers and stored
    once (no ordering hazards between scalar and vector stores)."""
    b = tl.program_id(0).to(tl.int64)
    y_row = y_ptr + b * M
    dr_row = drafts_ptr + b * (M - 1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    out = tl.full([BLOCK_M], -1, tl.int64)
    cvr = offs_d.to(tl.int32)
    cur = tl.full((), 0, tl.int32)
    n = tl.full((), 0, tl.int32)
    alive = tl.full((), 1, tl.int32)
    rleaf = tl.full((), -1, tl.int32)
    rdepth = tl.full((), 0, tl.int32)

    for d in tl.static_range(1, D + 1):
        t = tl.load(y_row + cur)
        c = tl.load(dr_row + (d - 1))
        is_alive = alive != 0
        match = (t == c) & is_alive
        miss = (t != c) & is_alive
        hit = tl.full((), -1, tl.int32)
        hit_row = tl.full((), 0, tl.int32)
        for l in tl.static_range(L):
            ld = tl.load(leaf_depth_ptr + l)
            lr = tl.load(leaf_rows_ptr + l)
            lt = tl.load(dr_row + (lr - 1))
            take = (hit < 0) & (ld == d) & (lt == t) & (lt >= 0)
            hit = tl.where(take, l, hit)
            hit_row = tl.where(take, lr, hit_row)
        rescue = miss & (hit >= 0)
        # Emit y[cur] (the matched chain token, the rescued leaf token, or the
        # correction) at position n.
        out = tl.where((offs_m == n) & is_alive, t, out)
        n = n + alive
        # Rescue: also emit the sample at the leaf row (the leaf's "bonus").
        y_leaf = tl.load(y_row + hit_row)
        out = tl.where((offs_m == n) & rescue, y_leaf, out)
        n = n + rescue.to(tl.int32)
        rleaf = tl.where(rescue, hit, rleaf)
        rdepth = tl.where(rescue, d, rdepth)
        cvr = tl.where((offs_d == d) & rescue, hit_row, cvr)
        cur = tl.where(match, d, cur)
        alive = match.to(tl.int32)

    # Full chain accepted: emit the bonus y[row D].
    t_bonus = tl.load(y_row + cur)
    out = tl.where((offs_m == n) & (alive != 0), t_bonus, out)
    n = n + alive

    tl.store(out_ptr + b * M + offs_m, out, mask=offs_m < M)
    tl.store(cvr_ptr + b * (D + 1) + offs_d, cvr, mask=offs_d < D + 1)
    tl.store(num_acc_ptr + b, n)
    tl.store(rescue_leaf_ptr + b, rleaf)
    tl.store(rescue_depth_ptr + b, rdepth)


def tree_walk(
    spec: TreeSpec,
    draft_token_ids: torch.Tensor,  # [B, M-1] int64
    y: torch.Tensor,  # [B, M] int64
    leaf_rows: torch.Tensor,  # int32 [max(L,1)] on y.device
    leaf_depth: torch.Tensor,  # int32 [max(L,1)] on y.device
    out: TreeStepOutputs | None = None,
) -> TreeStepOutputs:
    """Launch the walk.  ``out`` (if given) holds persistent [>=B, ..] buffers;
    the returned TreeStepOutputs are [:B] views of them."""
    B = y.shape[0]
    D, M = spec.D, spec.M
    assert y.shape == (B, M) and draft_token_ids.shape == (B, M - 1)
    y = y.to(torch.int64).contiguous()
    drafts = draft_token_ids.to(torch.int64).contiguous()
    dev = y.device
    if out is None:
        out = TreeStepOutputs(
            output_token_ids=torch.empty((B, M), dtype=torch.int64, device=dev),
            num_accepted=torch.empty(B, dtype=torch.int32, device=dev),
            rescue_leaf=torch.empty(B, dtype=torch.int32, device=dev),
            rescue_depth=torch.empty(B, dtype=torch.int32, device=dev),
            chain_view_rows=torch.empty((B, D + 1), dtype=torch.int32, device=dev),
        )
    res = TreeStepOutputs(
        output_token_ids=out.output_token_ids[:B],
        num_accepted=out.num_accepted[:B],
        rescue_leaf=out.rescue_leaf[:B],
        rescue_depth=out.rescue_depth[:B],
        chain_view_rows=out.chain_view_rows[:B],
    )
    if B == 0:
        return res
    _tree_walk_kernel[(B,)](
        y,
        drafts,
        leaf_rows,
        leaf_depth,
        res.output_token_ids,
        res.num_accepted,
        res.rescue_leaf,
        res.rescue_depth,
        res.chain_view_rows,
        D=D,
        M=M,
        L=spec.L,
        BLOCK_M=triton.next_power_of_2(M),
        BLOCK_D=triton.next_power_of_2(D + 1),
    )
    return res


# ------------------------------------------------------------------ sampler
def _min_tokens_tree(
    proc: MinTokensLogitsProcessor,
    logits: torch.Tensor,  # [B * len(row_depths), V]
    row_depths: tuple[int, ...],
    num_reqs: int,
) -> None:
    """Depth-aware min-tokens: the row at depth d predicts output position d,
    so its stop tokens stay masked iff d < min_tokens - len(output).  On a
    chain this is exactly ``apply_with_spec_decode`` (leading rows)."""
    if not proc.min_toks:
        return
    R = len(row_depths)
    rows: list[int] = []
    toks: list[int] = []
    rrows: list[int] = []
    rtoks: list[int] = []
    for req_idx, (min_tok, out_ids, stop_ids, structured) in proc.min_toks.items():
        if req_idx >= num_reqs or not stop_ids:
            continue
        remaining = min_tok - len(out_ids)
        if remaining <= 0:
            continue
        stop = list(stop_ids)
        for i, dep in enumerate(row_depths):
            if dep < remaining:
                row = req_idx * R + i
                rows += [row] * len(stop)
                toks += stop
                if structured:
                    rrows += [row] * len(stop)
                    rtoks += stop
    if not rows:
        return
    dev = logits.device
    sl = (
        torch.tensor(rows, dtype=torch.long, device=dev),
        torch.tensor(toks, dtype=torch.long, device=dev),
    )
    if rrows:
        rsl = (
            torch.tensor(rrows, dtype=torch.long, device=dev),
            torch.tensor(rtoks, dtype=torch.long, device=dev),
        )
        proc._mask_stop_token_logits(logits, sl, rsl)
    else:
        logits.index_put_(sl, torch.tensor(float("-inf"), device=dev))


class TreeSampler(nn.Module):
    """Sampler for tree steps.  Owns its persistent [B_tree, ..] output
    buffers (built once at init, never inside graph capture).

    forward(logits [B*M, V] row order, tree_draft_token_ids [B, M-1] int64,
            sampling_metadata, opt_out_reqs=()) -> (SamplerOutput, TreeStepOutputs)

    ``SamplerOutput.sampled_token_ids`` is int32 [B, M], path-ordered, -1
    padded (the same tokens as ``TreeStepOutputs.output_token_ids``, which is
    int64 per the contract).  Logprobs (if requested) are gathered along the
    emitted path: output position k comes from row ``chain_view_rows[b, k]``.
    """

    def __init__(
        self,
        sampler: Sampler,
        spec: TreeSpec,
        device: torch.device | str,
        max_batch: int | None = None,
    ):
        super().__init__()
        spec.validate()
        self.sampler = sampler
        self.spec = spec
        self.device = torch.device(device)
        self.use_fp64_gumbel = getattr(sampler, "use_fp64_gumbel", False)
        logprobs_mode = sampler.logprobs_mode
        self.is_processed_logprobs_mode = logprobs_mode in PROCESSED_LOGPROBS_MODES
        self.is_logits_logprobs_mode = logprobs_mode in (
            "raw_logits",
            "processed_logits",
        )
        D, M = spec.D, spec.M
        Bmax = int(max_batch or spec.B_tree)
        self.max_batch = Bmax
        dev = self.device
        self.target_rows = tree_target_rows(spec)
        self.T = len(self.target_rows)  # = M - 1
        self.target_depths = tuple(spec.rows_depth[r] for r in self.target_rows)
        tr = torch.tensor(self.target_rows, dtype=torch.long)
        ar = torch.arange(Bmax, dtype=torch.long)
        # Flat row index (b*M + r) of each target row / bonus row.
        self._target_idx = (ar[:, None] * M + tr[None, :]).reshape(-1).to(dev)
        self._bonus_idx = (ar * M + D).to(dev)
        self._repeat_idx = ar.repeat_interleave(self.T).to(dev)
        self._cu_targets = ((ar + 1) * self.T).to(torch.int32).to(dev)
        # chain view rows padded to M columns (for logprob gather).
        self._pad_cols = M - (D + 1)
        lr = list(spec.leaf_rows) or [0]
        ld = list(spec.leaf_depth) or [0]
        self._leaf_rows = torch.tensor(lr, dtype=torch.int32, device=dev)
        self._leaf_depth = torch.tensor(ld, dtype=torch.int32, device=dev)
        self.outputs = TreeStepOutputs(
            output_token_ids=torch.full(
                (Bmax, M), PLACEHOLDER_TOKEN_ID, dtype=torch.int64, device=dev
            ),
            num_accepted=torch.zeros(Bmax, dtype=torch.int32, device=dev),
            rescue_leaf=torch.full((Bmax,), -1, dtype=torch.int32, device=dev),
            rescue_depth=torch.zeros(Bmax, dtype=torch.int32, device=dev),
            chain_view_rows=torch.zeros((Bmax, D + 1), dtype=torch.int32, device=dev),
        )
        self._y = torch.zeros(Bmax * M, dtype=torch.int64, device=dev)

    def warmup(self) -> None:
        """Compile/launch the walk kernel once at init (never lazily on the
        first tree step).  Writes only the persistent output buffers."""
        M = self.spec.M
        y = torch.zeros((1, M), dtype=torch.int64, device=self.device)
        drafts = torch.full((1, M - 1), -1, dtype=torch.int64, device=self.device)
        tree_walk(self.spec, drafts, y, self._leaf_rows, self._leaf_depth, self.outputs)

    # ---------------------------------------------------------------- utils
    def _draft_lists(
        self, sampling_metadata: SamplingMetadata, drafts: torch.Tensor
    ) -> list[list[int]]:
        spec_ids = sampling_metadata.spec_token_ids
        B = drafts.shape[0]
        if spec_ids is not None and len(spec_ids) >= B and all(
            len(s) == self.spec.M - 1 for s in spec_ids[:B]
        ):
            return [list(s) for s in spec_ids[:B]]
        return drafts.tolist()

    def _tracked_thinking_reqs(self, sampling_metadata: SamplingMetadata) -> set[int]:
        holder = sampling_metadata.thinking_budget_state_holder
        if holder is None or not holder.has_tracked_requests():
            return set()
        return set(getattr(holder, "_state", {}).keys())

    # ------------------------------------------------------------- forward
    def apply_target_processors(
        self,
        target_logits: torch.Tensor,  # [B*T, V] fp32, request-major target rows
        sampling_metadata: SamplingMetadata,
        draft_lists: list[list[int]] | None,
        B: int,
    ) -> torch.Tensor:
        """Today's chain target-row processors with path contexts."""
        spec = self.spec
        has_penalties = not sampling_metadata.no_penalties
        bad_words = sampling_metadata.bad_words_token_ids
        contexts: list[list[int]] | None = None
        if has_penalties or bad_words:
            assert draft_lists is not None
            contexts = tree_row_contexts(
                spec,
                sampling_metadata.output_token_ids[:B],
                draft_lists,
                self.target_rows,
            )
        if has_penalties or sampling_metadata.allowed_token_ids_mask is not None:
            repeat_idx = self._repeat_idx[: B * self.T]
            if has_penalties:
                assert sampling_metadata.prompt_token_ids is not None
                assert contexts is not None
                target_logits = apply_all_penalties(
                    target_logits,
                    sampling_metadata.prompt_token_ids[repeat_idx],
                    sampling_metadata.presence_penalties[repeat_idx],
                    sampling_metadata.frequency_penalties[repeat_idx],
                    sampling_metadata.repetition_penalties[repeat_idx],
                    contexts,
                )
            if sampling_metadata.allowed_token_ids_mask is not None:
                mask = sampling_metadata.allowed_token_ids_mask[repeat_idx]
                target_logits.masked_fill_(mask, float("-inf"))
        if bad_words:
            assert contexts is not None
            apply_bad_words_with_drafts(
                target_logits, bad_words, contexts, [self.T] * B
            )
        for proc in sampling_metadata.logitsprocs.non_argmax_invariant:
            if isinstance(proc, MinTokensLogitsProcessor):
                _min_tokens_tree(proc, target_logits, self.target_depths, B)
        holder = sampling_metadata.thinking_budget_state_holder
        if holder is not None and holder.has_tracked_requests():
            # Chain semantics on the chain target rows only (their layout is
            # exactly today's: D rows per request); tracked requests are
            # opted out of the leaves by the caller.
            D, V = spec.D, target_logits.shape[-1]
            view = target_logits.view(B, self.T, V)
            chain_rows = view[:, :D].reshape(B * D, V)
            chain_spec = (
                [list(s[:D]) for s in draft_lists] if draft_lists is not None else None
            )
            chain_rows = holder.apply_to_logits(chain_rows, False, chain_spec)
            view[:, :D] = chain_rows.view(B, D, V)
        return target_logits

    def sample_target_rows(
        self,
        target_logits: torch.Tensor,  # [B*T, V] processed + constrained
        sampling_metadata: SamplingMetadata,
        B: int,
    ) -> torch.Tensor:
        """One independent target sample per row: argmax for greedy
        requests, an exponential race (fresh draw per row; seeded requests
        use their own generator) on the processed distribution otherwise."""
        greedy = target_logits.argmax(dim=-1)
        if sampling_metadata.all_greedy:
            return greedy
        probs = target_logits.softmax(dim=-1, dtype=torch.float32)
        q_dtype = torch.float64 if self.use_fp64_gumbel else torch.float32
        q = torch.empty_like(probs, dtype=q_dtype)
        q.exponential_()
        T = self.T
        for i, gen in sampling_metadata.generators.items():
            if i < B:
                q[i * T : (i + 1) * T].exponential_(generator=gen)
        rand = (probs.to(q_dtype) / q).argmax(dim=-1)
        if sampling_metadata.all_random:
            return rand
        assert sampling_metadata.temperature is not None
        is_greedy = (sampling_metadata.temperature[:B] == GREEDY_TEMPERATURE)[
            self._repeat_idx[: B * T]
        ]
        return torch.where(is_greedy, greedy, rand)

    def forward(
        self,
        logits: torch.Tensor,  # [B*M, V], row order
        tree_draft_token_ids: torch.Tensor,  # [B, M-1] int64, -1 = opt-out
        sampling_metadata: SamplingMetadata,
        opt_out_reqs: Iterable[int] = (),
    ) -> tuple[SamplerOutput, TreeStepOutputs]:
        spec = self.spec
        D, M = spec.D, spec.M
        B = tree_draft_token_ids.shape[0]
        assert 1 <= B <= self.max_batch, (B, self.max_batch)
        assert logits.shape[0] == B * M, (logits.shape, B, M)
        T = self.T
        target_idx = self._target_idx[: B * T]
        bonus_idx = self._bonus_idx[:B]

        draft_lists: list[list[int]] | None = None
        holder = sampling_metadata.thinking_budget_state_holder
        if (
            not sampling_metadata.no_penalties
            or sampling_metadata.bad_words_token_ids
            or (holder is not None and holder.has_tracked_requests())
        ):
            draft_lists = self._draft_lists(sampling_metadata, tree_draft_token_ids)

        # ---- bonus row D: today's bonus path, chain context.
        bonus_md = sampling_metadata
        if sampling_metadata.spec_token_ids is not None:
            bonus_md = replace(
                sampling_metadata,
                spec_token_ids=[list(s[:D]) for s in sampling_metadata.spec_token_ids],
            )
        bonus_out = self.sampler(
            logits=logits[bonus_idx],
            sampling_metadata=replace(bonus_md, max_num_logprobs=-1),
            predict_bonus_token=True,
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_tokens = bonus_out.sampled_token_ids.view(-1).to(torch.int64)

        # ---- target rows: chain target treatment with path contexts.
        raw_target = logits[target_idx].to(torch.float32)
        target = raw_target if self.is_processed_logprobs_mode else raw_target.clone()
        target = self.apply_target_processors(target, sampling_metadata, draft_lists, B)
        target = apply_sampling_constraints(
            target, self._cu_targets[:B], sampling_metadata
        )
        y_target = self.sample_target_rows(target, sampling_metadata, B)

        y = self._y[: B * M]
        y[target_idx] = y_target
        y[bonus_idx] = bonus_tokens
        y = y.view(B, M)

        # ---- opt-out and walk.
        opt = set(int(i) for i in opt_out_reqs)
        opt |= self._tracked_thinking_reqs(sampling_metadata)
        drafts = apply_leaf_opt_out(spec, tree_draft_token_ids, opt)
        res = tree_walk(spec, drafts, y, self._leaf_rows, self._leaf_depth, self.outputs)

        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            logprobs_tensors = self._path_logprobs(
                sampling_metadata.max_num_logprobs,
                logits,
                target if self.is_processed_logprobs_mode else raw_target,
                bonus_out.logprobs_tensors.logprobs,
                res,
                B,
            )
        return (
            SamplerOutput(
                sampled_token_ids=res.output_token_ids.to(torch.int32),
                logprobs_tensors=logprobs_tensors,
            ),
            res,
        )

    def _path_logprobs(
        self,
        max_num_logprobs: int,
        logits: torch.Tensor,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
        res: TreeStepOutputs,
        B: int,
    ) -> LogprobsTensors:
        """Logprobs gathered along the emitted path (position k <- row
        chain_view_rows[b, k]); padded positions are filtered by
        ``RejectionSampler.parse_output`` exactly as for a chain."""
        M = self.spec.M
        final = torch.zeros(
            (B * M, logits.shape[-1]), dtype=torch.float32, device=logits.device
        )
        final[self._target_idx[: B * self.T]] = target_logits.to(torch.float32)
        final[self._bonus_idx[:B]] = bonus_logits.to(torch.float32)
        rows = res.chain_view_rows.to(torch.int64)
        if self._pad_cols:
            rows = torch.cat(
                [rows, rows.new_zeros((B, self._pad_cols))], dim=1
            )
        base = torch.arange(B, device=rows.device, dtype=torch.int64)[:, None] * M
        idx = (base + rows).reshape(-1)
        tokens = res.output_token_ids.reshape(-1).clone()
        tokens[tokens == PLACEHOLDER_TOKEN_ID] = 0
        picked = final[idx]
        lp = picked if self.is_logits_logprobs_mode else self.sampler.compute_logprobs(
            picked
        )
        return self.sampler.gather_logprobs(lp, max_num_logprobs, tokens)
