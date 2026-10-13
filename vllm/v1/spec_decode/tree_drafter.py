# SPDX-License-Identifier: Apache-2.0
"""Tree speculative decoding: drafter side (E16 / P7).

Contract: ``/home/kevin/projects/lanes/coexec/results/tree-contract.md`` §5 P7.

Two pieces, both used by ``SpecDecodeBaseProposer`` only when a TreeSpec is
active (``resolve_tree_spec`` is not None):

* **Leaves.**  At each of the D existing drafter sampling sites the chain
  token ``c_d`` stays the argmax (bit-identical chain).  The leaves at depth d
  are ``topk(logits with c_d masked to -inf, max rank at d)``, picking column
  ``leaf_rank - 1``, so a leaf never equals its chain sibling.  The drafter
  returns ``cat([c_1..c_D], leaves in leaf order)`` = [B, M-1] int64.

* **Chain view.**  After a tree step the drafter's first pass must see the
  *accepted* path as a plain D+1-row chain.  ``tree_chain_view`` gathers the
  verify rows with ``TreeStepOutputs.chain_view_rows`` (tokens, positions,
  hidden states), keeps the drafter KV slots of sequence indices
  ``base + 0..D`` (row-indexed slots of chain rows 0..D: a rescued leaf at
  depth j lands at sequence index base + j, which is exactly chain row j's
  slot), and builds a D+1-rows-per-request CommonAttentionMetadata in the
  same shape ``prepare_inputs_padded`` produces for a chain step.  Everything
  after that is the unchanged padded-drafter code.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.tree_spec import TreeSpec


# ------------------------------------------------------------------ leaves
def tree_leaf_ranks_by_depth(spec: TreeSpec) -> dict[int, tuple[int, ...]]:
    """depth -> ranks (in leaf order) of the leaves at that depth."""
    out: dict[int, tuple[int, ...]] = {}
    for d in range(1, spec.D + 1):
        ls = spec.leaves_at_depth(d)
        if ls:
            out[d] = tuple(spec.leaf_rank[l] for l in ls)
    return out


def tree_leaf_tokens(
    logits: torch.Tensor,  # [B, V] drafter logits at depth d
    chain_tokens: torch.Tensor,  # [B] the chain token c_d sampled from them
    ranks: tuple[int, ...],  # leaf ranks at depth d (>= 1), leaf order
) -> torch.Tensor:
    """[B, len(ranks)] int64 leaf tokens: rank r -> the r-th best token
    other than c_d.  ``logits`` is not modified."""
    k = max(ranks)
    masked = logits.scatter(
        -1, chain_tokens.view(-1, 1).to(torch.int64), float("-inf")
    )
    top = masked.topk(k, dim=-1).indices
    cols = [r - 1 for r in ranks]
    if cols == list(range(k)):
        return top
    return top[:, cols]


def assemble_tree_drafts(
    spec: TreeSpec,
    chain: torch.Tensor,  # [B, D]
    leaves_by_depth: dict[int, torch.Tensor] | None,
) -> torch.Tensor:
    """[B, M-1] int64 = chain then leaves in leaf order (depth, rank).
    Missing depths (or ``leaves_by_depth=None``) give -1 leaves (opt-out)."""
    B = chain.shape[0]
    cols = [chain.to(torch.int64)]
    for d, ranks in tree_leaf_ranks_by_depth(spec).items():
        t = None if leaves_by_depth is None else leaves_by_depth.get(d)
        if t is None:
            t = torch.full((B, len(ranks)), -1, dtype=torch.int64, device=chain.device)
        cols.append(t.to(torch.int64))
    out = torch.cat(cols, dim=1)
    assert out.shape == (B, spec.M - 1), (out.shape, spec.M)
    return out


# -------------------------------------------------------------- chain view
class TreeChainView(NamedTuple):
    """Drop-in for what ``prepare_inputs_padded`` + the runner's target
    slicing give ``propose()`` on a chain step."""

    common_attn_metadata: CommonAttentionMetadata  # D+1 rows / request
    token_indices_to_sample: torch.Tensor  # [B] int32
    num_rejected_tokens_gpu: torch.Tensor  # [B] int32
    target_token_ids: torch.Tensor  # [B*(D+1)]
    target_positions: torch.Tensor  # [B*(D+1)] or [3, B*(D+1)]
    target_hidden_states: torch.Tensor  # [B*(D+1), H]


class TreeChainViewBuffers:
    """Static index tables for the chain view, built once (proposer init),
    sized for ``max_batch`` requests."""

    def __init__(self, spec: TreeSpec, max_batch: int, device: torch.device | str):
        D, M = spec.D, spec.M
        dev = torch.device(device)
        ar = torch.arange(max_batch + 1, dtype=torch.int32)
        self.qsl_cpu = ar * (D + 1)
        self.qsl = self.qsl_cpu.to(dev)
        b = torch.arange(max_batch, dtype=torch.int64)
        # flat verify-row base of each request and the chain-row slot index.
        self.row_base = (b * M).to(dev)  # [Bmax]
        self.chain_slot_idx = (
            (b[:, None] * M + torch.arange(D + 1)[None, :]).reshape(-1).to(dev)
        )
        self.req_last = ((b + 1) * (D + 1) - 1).to(torch.int32).to(dev)
        self.max_batch = max_batch


def tree_chain_view(
    spec: TreeSpec,
    cad: CommonAttentionMetadata,  # the tree step's CAD (M rows / request)
    chain_view_rows: torch.Tensor,  # [B, D+1] int32 (TreeStepOutputs)
    valid_sampled_tokens_count: torch.Tensor,  # [B] (prepare_next_token_ids_padded)
    target_token_ids: torch.Tensor,  # [B*M]
    target_positions: torch.Tensor,  # [B*M] or [3, B*M]
    target_hidden_states: torch.Tensor,  # [>=B*M, H]
    buffers: TreeChainViewBuffers | None = None,
) -> TreeChainView:
    """Gather the accepted path of a tree step into a D+1-row chain view.

    Row k of request b's view is verify row ``chain_view_rows[b, k]`` for the
    token / position / hidden state, and verify row k for the KV slot (the
    sequence index base + k).  ``num_rejected = D + 1 - valid_count`` and
    ``token_indices_to_sample = b*(D+1) + D - num_rejected``: identical to ``eagle_prepare_inputs_padded_kernel`` on a
    chain with D drafts per request.
    """
    D, M, L = spec.D, spec.M, spec.L
    B = cad.num_reqs
    assert cad.num_actual_tokens >= B * M, (cad.num_actual_tokens, B, M)
    assert chain_view_rows.shape == (B, D + 1)
    assert cad.dcp_local_seq_lens is None, "tree + DCP is not supported"
    dev = chain_view_rows.device
    if buffers is None:
        buffers = TreeChainViewBuffers(spec, B, dev)
    assert B <= buffers.max_batch

    gather = (buffers.row_base[:B, None] + chain_view_rows.to(torch.int64)).reshape(-1)
    slot_idx = buffers.chain_slot_idx[: B * (D + 1)]

    tok = target_token_ids[gather]
    pos = target_positions[..., gather]
    hid = target_hidden_states[gather]

    valid = valid_sampled_tokens_count.to(torch.int32)
    num_rejected = (D + 1) - valid  # D >= 1 drafts on every tree request
    token_indices_to_sample = (buffers.req_last[:B] - num_rejected).to(torch.int32)

    ub = cad.seq_lens_cpu_upper_bound
    view_cad = CommonAttentionMetadata(
        query_start_loc=buffers.qsl[: B + 1],
        query_start_loc_cpu=buffers.qsl_cpu[: B + 1],
        seq_lens=cad.seq_lens - L,
        seq_lens_cpu_upper_bound=None if ub is None else ub - L,
        num_reqs=B,
        num_actual_tokens=B * (D + 1),
        max_query_len=D + 1,
        max_seq_len=max(cad.max_seq_len - L, 1),
        block_table_tensor=cad.block_table_tensor,
        slot_mapping=cad.slot_mapping[slot_idx],
        causal=True,
        dcp_local_seq_lens=cad.dcp_local_seq_lens,
    )
    return TreeChainView(
        common_attn_metadata=view_cad,
        token_indices_to_sample=token_indices_to_sample,
        num_rejected_tokens_gpu=num_rejected.to(torch.int32),
        target_token_ids=tok,
        target_positions=pos,
        target_hidden_states=hid,
    )
