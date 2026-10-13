# SPDX-License-Identifier: Apache-2.0
"""Tree speculative decoding: the shared static tree description (E16 / P1).

Everything every tree package (runner inputs, TurboQuant mask, GDN kernels,
commit, sampler walk, drafter, dispatcher) needs about the tree's *shape* lives
here, precomputed once.  The contract document is
``/home/kevin/projects/lanes/coexec/results/tree-contract.md``.

Vocabulary (fixed for every package):

* ``D``  chain depth = ``num_speculative_tokens`` = drafter passes = GDN spec
  state slots - 1.  Unchanged by the tree.
* ``L``  number of leaves (rescue leaves in Phase 1).
* ``M``  verify rows per request on a *tree step* = ``1 + D + L``.
* Rows are ordered **chain-first**: ``[x0, c1, .., cD, leaf_0, .., leaf_{L-1}]``.
  Row 0 is the root (the bonus token sampled last step), row ``d`` (1..D) is the
  chain draft at depth ``d``; leaf rows follow, sorted by (depth, rank).
  Truncating a tree step's rows to the first ``D + 1`` therefore *is* the chain.
* ``rows_depth[r]`` is the RoPE / sequence-position offset of row ``r`` from
  the request's first scheduled position.  KV slot and token gather stay
  row-indexed (``base + r``).
* ``anc_mask[r]`` has bit ``k`` set iff row ``k`` is an ancestor of row ``r``
  or ``r`` itself.  Row ``r`` may attend (in the current chunk) to row ``k``
  iff that bit is set.

Config syntax is upstream's ``speculative_token_tree`` (rank paths): the path
``(0, 0, 1)`` is the rank-1 (second best) drafter token at depth 3 whose
parent is the depth-2 chain token.  Phase 1 accepts only "chain + leaves":
the full chain ``(0,)*d`` for d = 1..D plus any leaves ``(0,)*(d-1) + (r,)``,
r >= 1.  Internal non-chain nodes (Phase 2, branch expansion) are rejected
with a clear error so later work can lift the restriction in one place.

Gating (default OFF):

* ``SpeculativeConfig.speculative_token_tree`` (CLI: inside
  ``--speculative-config``), or env ``VLLM_SPEC_TREE``:
  unset / "" / "0" -> off (unless the config field is set);
  "1" -> chain + one rank-1 leaf at every depth (M = 2D + 1);
  otherwise a literal rank-path list, e.g. "[(0,),(1,),(0,0),(0,1)]".
  The env value wins over the config field when both are set.
* ``SpeculativeConfig.speculative_tree_max_batch`` or env
  ``VLLM_SPEC_TREE_BMAX`` (env wins): ``B_tree``, the largest number of
  requests for which a step is a tree step.  Default 4.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig

# Hard cap on rows per request.  K5 GDN kernel shared arrays are sized
# kMaxTok = 8 today; masks are int32.  Raising this needs every kernel to
# agree, so it is a contract constant.
TREE_MAX_ROWS = 16
TREE_DEFAULT_BMAX = 4

ENV_TREE = "VLLM_SPEC_TREE"
ENV_TREE_BMAX = "VLLM_SPEC_TREE_BMAX"

Path = tuple[int, ...]


def _chain_path(d: int) -> Path:
    return (0,) * d


@dataclass(frozen=True)
class TreeSpec:
    """Static description of one tree topology.  Immutable and hashable.

    All per-row tuples have length ``M`` and are in canonical row order.
    """

    D: int
    M: int
    rows_parent: tuple[int, ...]  # -1 for row 0
    rows_depth: tuple[int, ...]  # 0 for row 0, d for chain row d, depth of leaf
    rows_rank: tuple[int, ...]  # drafter rank of the row's token (0 = chain)
    anc_mask: tuple[int, ...]  # ancestor-or-self bitmask per row
    leaf_rows: tuple[int, ...]  # row index of leaf l (= D + 1 + l)
    leaf_depth: tuple[int, ...]  # depth (1..D) of leaf l
    leaf_rank: tuple[int, ...]  # drafter rank (>=1) of leaf l
    leaf_row_of_depth: tuple[int, ...]  # len D+1; first leaf row at depth d, -1
    B_tree: int = TREE_DEFAULT_BMAX
    choices: tuple[Path, ...] = field(default=(), compare=False)

    # ------------------------------------------------------------------ props
    @property
    def L(self) -> int:
        return self.M - 1 - self.D

    @property
    def num_drafts(self) -> int:
        """Draft tokens per request on a tree step (= M - 1)."""
        return self.M - 1

    @property
    def chain_rows(self) -> tuple[int, ...]:
        return tuple(range(self.D + 1))

    @property
    def max_leaves_per_depth(self) -> int:
        if not self.leaf_depth:
            return 0
        return max(self.leaf_depth.count(d) for d in set(self.leaf_depth))

    def leaves_at_depth(self, d: int) -> tuple[int, ...]:
        """Leaf indices (not rows) at depth d, in rank order."""
        return tuple(i for i, ld in enumerate(self.leaf_depth) if ld == d)

    def path_rows(self, r: int) -> tuple[int, ...]:
        """Rows from the root to row r inclusive."""
        out = []
        while r >= 0:
            out.append(r)
            r = self.rows_parent[r]
        return tuple(reversed(out))

    def to_choices(self) -> list[Path]:
        """Upstream rank-path list, BFS-sorted, for logging/config echo."""
        paths = [_chain_path(d) for d in range(1, self.D + 1)]
        for d, r in zip(self.leaf_depth, self.leaf_rank):
            paths.append(_chain_path(d - 1) + (r,))
        return sorted(paths, key=lambda t: (len(t), t))

    # ------------------------------------------------------------ builders
    @staticmethod
    def chain_plus_leaves(
        D: int,
        leaf_depths: tuple[int, ...] | None = None,
        width: int = 2,
        B_tree: int = TREE_DEFAULT_BMAX,
    ) -> TreeSpec:
        """Chain of depth D plus ranks 1..width-1 as leaves at each depth in
        ``leaf_depths`` (default: every depth 1..D)."""
        if leaf_depths is None:
            leaf_depths = tuple(range(1, D + 1))
        choices = [_chain_path(d) for d in range(1, D + 1)]
        for d in leaf_depths:
            for r in range(1, width):
                choices.append(_chain_path(d - 1) + (r,))
        return TreeSpec.from_choices(choices, B_tree=B_tree)

    @staticmethod
    def from_choices(choices: Any, B_tree: int = TREE_DEFAULT_BMAX) -> TreeSpec:
        """Build from an upstream-style rank-path list (or its string)."""
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        paths: list[Path] = []
        for c in choices:
            if isinstance(c, int):
                c = (c,)
            p = tuple(int(x) for x in c)
            if not p or any(x < 0 for x in p):
                raise ValueError(f"speculative_token_tree: bad path {c!r}")
            paths.append(p)
        if len(set(paths)) != len(paths):
            raise ValueError("speculative_token_tree: duplicate paths")
        chain = [p for p in paths if all(x == 0 for x in p)]
        D = max((len(p) for p in chain), default=0)
        if D < 1:
            raise ValueError("speculative_token_tree: needs a chain (0,) ..")
        if sorted(chain, key=len) != [_chain_path(d) for d in range(1, D + 1)]:
            raise ValueError(
                "speculative_token_tree: chain paths (0,)*d must be present "
                f"for every d = 1..{D}"
            )
        leaves: list[tuple[int, int]] = []  # (depth, rank)
        for p in paths:
            if all(x == 0 for x in p):
                continue
            d = len(p)
            if any(x != 0 for x in p[:-1]):
                raise NotImplementedError(
                    f"speculative_token_tree: path {p} has a non-chain parent; "
                    "only chain + leaves is supported (Phase 1)"
                )
            if d > D:
                raise ValueError(
                    f"speculative_token_tree: leaf {p} deeper than chain D={D}"
                )
            leaves.append((d, p[-1]))
        leaves.sort()
        ranks_by_depth: dict[int, list[int]] = {}
        for d, r in leaves:
            ranks_by_depth.setdefault(d, []).append(r)
        for d, rs in ranks_by_depth.items():
            # Ranks need not be contiguous, but each must be distinct (set
            # above).  Rank 0 at a leaf would duplicate the chain token.
            assert all(r >= 1 for r in rs)
        M = 1 + D + len(leaves)
        if M > TREE_MAX_ROWS:
            raise ValueError(f"speculative_token_tree: M={M} > {TREE_MAX_ROWS}")
        if B_tree < 1:
            raise ValueError("speculative_tree_max_batch must be >= 1")

        parent = [-1] + list(range(D))
        depth = list(range(D + 1))
        rank = [0] * (D + 1)
        leaf_rows, leaf_depth, leaf_rank = [], [], []
        lrod = [-1] * (D + 1)
        for i, (d, r) in enumerate(leaves):
            row = D + 1 + i
            parent.append(d - 1)
            depth.append(d)
            rank.append(r)
            leaf_rows.append(row)
            leaf_depth.append(d)
            leaf_rank.append(r)
            if lrod[d] < 0:
                lrod[d] = row
        anc = []
        for r in range(M):
            m, x = 0, r
            while x >= 0:
                m |= 1 << x
                x = parent[x]
            anc.append(m)
        spec = TreeSpec(
            D=D,
            M=M,
            rows_parent=tuple(parent),
            rows_depth=tuple(depth),
            rows_rank=tuple(rank),
            anc_mask=tuple(anc),
            leaf_rows=tuple(leaf_rows),
            leaf_depth=tuple(leaf_depth),
            leaf_rank=tuple(leaf_rank),
            leaf_row_of_depth=tuple(lrod),
            B_tree=int(B_tree),
            choices=tuple(paths),
        )
        spec.validate()
        return spec

    # ---------------------------------------------------------- invariants
    def validate(self) -> None:
        D, M = self.D, self.M
        assert M == 1 + D + len(self.leaf_rows) <= TREE_MAX_ROWS
        for name in ("rows_parent", "rows_depth", "rows_rank", "anc_mask"):
            assert len(getattr(self, name)) == M, name
        assert len(self.leaf_row_of_depth) == D + 1
        assert self.rows_parent[0] == -1 and self.rows_depth[0] == 0
        for r in range(1, M):
            p = self.rows_parent[r]
            assert 0 <= p < r, "parents precede children (topological rows)"
            assert self.rows_depth[r] == self.rows_depth[p] + 1
            assert self.anc_mask[r] == self.anc_mask[p] | (1 << r)
        for d in range(1, D + 1):  # chain-first
            assert self.rows_parent[d] == d - 1 and self.rows_rank[d] == 0
        for i, row in enumerate(self.leaf_rows):
            assert row == D + 1 + i
            assert self.rows_depth[row] == self.leaf_depth[i]
            assert self.rows_parent[row] == self.leaf_depth[i] - 1
            assert self.leaf_rank[i] >= 1
            assert self.anc_mask[row] == self.anc_mask[self.leaf_depth[i] - 1] | (
                1 << row
            )
        sib = {(self.rows_parent[r], self.rows_rank[r]) for r in range(1, M)}
        assert len(sib) == M - 1, "siblings have distinct ranks"

    # -------------------------------------------------------------- tensors
    def device_tensors(self, device: torch.device | str) -> TreeTensors:
        """Build the static tensors ONCE (runner init / warmup, never inside
        CUDA-graph capture).  Kernels capture these by address."""
        dev = torch.device(device)

        def t(x, dtype=torch.int32):
            return torch.tensor(list(x), dtype=dtype, device=dev)

        return TreeTensors(
            rows_parent=t(self.rows_parent),
            rows_depth=t(self.rows_depth),
            rows_depth_i64=t(self.rows_depth, torch.int64),
            anc_mask=t(self.anc_mask),
            leaf_rows=t(self.leaf_rows),
            leaf_depth=t(self.leaf_depth),
            leaf_rank=t(self.leaf_rank),
            leaf_parent_row=t(d - 1 for d in self.leaf_depth),
            leaf_row_of_depth=t(self.leaf_row_of_depth),
            chain_rows=t(range(self.D + 1)),
        )

    def describe(self) -> str:
        return (
            f"TreeSpec(D={self.D}, M={self.M}, L={self.L}, B_tree={self.B_tree}, "
            f"choices={self.to_choices()})"
        )


@dataclass(frozen=True)
class TreeTensors:
    """Device copies of the TreeSpec tables.  int32 unless noted, shape [M]
    for rows_*, [L] for leaf_*, [D+1] for leaf_row_of_depth / chain_rows."""

    rows_parent: torch.Tensor
    rows_depth: torch.Tensor
    rows_depth_i64: torch.Tensor
    anc_mask: torch.Tensor
    leaf_rows: torch.Tensor
    leaf_depth: torch.Tensor
    leaf_rank: torch.Tensor
    leaf_parent_row: torch.Tensor
    leaf_row_of_depth: torch.Tensor
    chain_rows: torch.Tensor


# ---------------------------------------------------------------- gating
def _parse_env_tree(value: str, D: int, B_tree: int) -> TreeSpec | None:
    v = value.strip()
    if v in ("", "0", "off", "false", "False"):
        return None
    if v in ("1", "on", "true", "True", "leaves"):
        return TreeSpec.chain_plus_leaves(D, B_tree=B_tree)
    spec = TreeSpec.from_choices(v, B_tree=B_tree)
    return spec


def resolve_tree_spec(
    speculative_config: SpeculativeConfig | None,
    environ: dict[str, str] | None = None,
) -> TreeSpec | None:
    """Return the TreeSpec for this engine, or None (tree off = today).

    Pure function of (config, env).  Every component (scheduler, runner,
    builders, drafter, sampler) must call this (or receive its result) so
    they cannot disagree.
    """
    env = os.environ if environ is None else environ
    if speculative_config is None:
        return None
    D = speculative_config.num_speculative_tokens
    if not D:
        return None
    cfg_tree = getattr(speculative_config, "speculative_token_tree", None)
    cfg_b = getattr(speculative_config, "speculative_tree_max_batch", None)
    b_env = env.get(ENV_TREE_BMAX, "").strip()
    B_tree = int(b_env) if b_env else int(cfg_b or TREE_DEFAULT_BMAX)

    env_tree = env.get(ENV_TREE)
    if env_tree is not None and env_tree.strip() != "":
        spec = _parse_env_tree(env_tree, D, B_tree)
    elif cfg_tree:
        spec = _parse_env_tree(str(cfg_tree), D, B_tree)
    else:
        spec = None
    if spec is None:
        return None
    if spec.D != D:
        raise ValueError(
            f"speculative_token_tree chain depth {spec.D} != "
            f"num_speculative_tokens {D}"
        )
    method = getattr(speculative_config, "method", None)
    if method != "mtp":
        raise ValueError(f"tree speculation supports method 'mtp' only, got {method}")
    if getattr(speculative_config, "parallel_drafting", False):
        raise ValueError("tree speculation is incompatible with parallel_drafting")
    if getattr(speculative_config, "num_speculative_tokens_per_batch_size", None):
        raise ValueError(
            "tree speculation is incompatible with num_speculative_tokens_per_batch_size"
        )
    if getattr(speculative_config, "use_local_argmax_reduction", False):
        raise ValueError(
            "tree speculation needs full-vocab draft logits "
            "(use_local_argmax_reduction must be off)"
        )
    return spec


# ------------------------------------------------------- step decision
def decide_tree_step(
    spec: TreeSpec | None,
    num_reqs: int,
    all_pure_decode: bool,
    has_new_or_resumed_reqs: bool = False,
    all_full_drafts: bool = True,
) -> bool:
    """THE tree-step rule, shared by scheduler, runner and builders.

    A step is a tree step iff tree is enabled, every scheduled request is a
    decode (num_computed >= num_prompt, no prefill chunk), no request is new
    or resumed this step, every request carries exactly M - 1 drafts (full
    chain; leaves may be -1), and 1 <= num_reqs <= B_tree.  Otherwise every
    request's drafts are truncated to the chain (first D) -- see
    ``truncate_drafts_to_chain``.
    """
    return (
        spec is not None
        and all_pure_decode
        and not has_new_or_resumed_reqs
        and all_full_drafts
        and 1 <= num_reqs <= spec.B_tree
    )


def truncate_drafts_to_chain(spec: TreeSpec, drafts: list[int]) -> list[int]:
    """Drafts are stored chain-first ([c1..cD, leaves..]); the chain view of
    a tree draft list is its first D entries.  Shorter lists (e.g. grammar
    truncated) are returned unchanged up to D."""
    return drafts[: spec.D]


def opt_out_leaves(spec: TreeSpec, drafts: list[int]) -> list[int]:
    """Per-request opt-out (structured output, bad words, ...): keep the chain,
    set every leaf to -1.  A -1 leaf occupies its row but is never accepted."""
    chain = drafts[: spec.D]
    if len(chain) < spec.D:
        return chain  # cannot be a tree request; scheduler must chain-step
    return chain + [-1] * spec.L


# ------------------------------------------------------------ leaf stash
@dataclass
class TreeStash:
    """Persistent leaf scratch shared by GDN verify (P4a/P4c, producers) and
    the commit kernel (P5, consumer).  Allocated ONCE by the model runner
    after model load and before CUDA-graph capture; kernels capture the
    per-layer views by address.  Indexed by the request's batch row b
    (0..B-1 in the step's request order), never by state slot.

    gdn_leaf    fp32       [n_gdn_layers, B_tree, L, HV_rank, K+V+2]
                last dim = k[0:K] | v[K:K+V] | decay=exp(g) [K+V] | beta [K+V+1]
    conv_leaf_x model dtype [n_gdn_layers, B_tree, L, conv_dim_rank]
                pre-conv input row (mixed_qkv after in_proj, before conv)
    layer_index layer prefix (e.g. "model.layers.0.linear_attn") -> layer idx
    """

    gdn_leaf: torch.Tensor
    conv_leaf_x: torch.Tensor
    layer_index: dict[str, int]

    @staticmethod
    def allocate(
        spec: TreeSpec,
        layer_names: list[str],
        num_v_heads_rank: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_dim_rank: int,
        conv_dtype: torch.dtype,
        device: torch.device | str,
    ) -> TreeStash:
        n = len(layer_names)
        L = max(spec.L, 1)
        return TreeStash(
            gdn_leaf=torch.zeros(
                n,
                spec.B_tree,
                L,
                num_v_heads_rank,
                head_k_dim + head_v_dim + 2,
                dtype=torch.float32,
                device=device,
            ),
            conv_leaf_x=torch.zeros(
                n, spec.B_tree, L, conv_dim_rank, dtype=conv_dtype, device=device
            ),
            layer_index={name: i for i, name in enumerate(layer_names)},
        )

    def gdn_view(self, layer_name: str) -> torch.Tensor:
        """[B_tree, L, HV_rank, K+V+2] view for one layer."""
        return self.gdn_leaf[self.layer_index[layer_name]]

    def conv_view(self, layer_name: str) -> torch.Tensor:
        """[B_tree, L, conv_dim_rank] view for one layer."""
        return self.conv_leaf_x[self.layer_index[layer_name]]


@dataclass
class TreeStepOutputs:
    """Produced by the tree sampler (P6) on a tree step, consumed by the
    commit (P5), runner bookkeeping and the drafter chain view (P7).
    All on device; B = num_reqs of the step (<= B_tree).  Persistent
    buffers of size B_tree are owned by the runner; these are [:B] views."""

    output_token_ids: torch.Tensor  # int64 [B, M], path + bonus, -1 padded
    num_accepted: torch.Tensor  # int32 [B], = (output_token_ids != -1).sum(1)
    rescue_leaf: torch.Tensor  # int32 [B], leaf index or -1
    rescue_depth: torch.Tensor  # int32 [B], 1..D or 0
    chain_view_rows: torch.Tensor  # int32 [B, D+1]
