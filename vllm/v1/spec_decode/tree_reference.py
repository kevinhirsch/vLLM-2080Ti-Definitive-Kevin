# SPDX-License-Identifier: Apache-2.0
"""CPU torch reference for tree verification semantics (E16 / P1).

Every tree package tests against these functions.  They are slow, simple,
fp32 and written straight from the definitions in ``tree-contract.md``.
Nothing here is imported by the engine at runtime.

Index conventions (see ``tree_spec.TreeSpec``):
  B requests, M rows per request in canonical chain-first order,
  D chain depth, L leaves.  ``draft_token_ids[b, r-1]`` is the token of row r
  (row 0 is the root x0 = bonus token from the previous step).

Contents
  positions / chain-view gather ......... tree_positions, gather_chain_view
  full attention (TurboQuant semantic) .. tree_attention_ref
  GDN delta rule (temporal) ............. gdn_sequence_ref, gdn_tree_ref,
                                          gdn_commit_ref
  GDN causal conv1d ..................... conv_tree_ref, conv_commit_ref,
                                          conv_next_hist_ref
  sampling walk ......................... tree_walk_ref
  exact output distribution (tests) ..... tree_emit_distribution_ref
"""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

from vllm.v1.spec_decode.tree_spec import TreeSpec

# Leaf GDN stash layout, last dim (fp32).  K = head_k_dim, V = head_v_dim.
#   [0:K)        k   (l2-normalised if the layer normalises q/k; NOT scaled)
#   [K:K+V)      v   (raw)
#   [K+V]        decay = exp(g)          (multiplicative, already exponentiated)
#   [K+V+1]      beta  = sigmoid(b)
def gdn_stash_width(K: int, V: int) -> int:
    return K + V + 2


# --------------------------------------------------------------- positions
def tree_positions(spec: TreeSpec, base_positions: torch.Tensor) -> torch.Tensor:
    """[B] first-row position -> [B, M] RoPE positions (depth based)."""
    depth = torch.tensor(spec.rows_depth, dtype=torch.int64)
    return base_positions.to(torch.int64)[:, None] + depth[None, :]


def tree_token_offsets(spec: TreeSpec) -> torch.Tensor:
    """Row r of request b is token index base + r for input gather and KV slot
    mapping (row indexed, NOT depth indexed)."""
    return torch.arange(spec.M, dtype=torch.int64)


def gather_chain_view(x: torch.Tensor, chain_view_rows: torch.Tensor) -> torch.Tensor:
    """x [B, M, ...] -> [B, D+1, ...]; the accepted chain view.  The same
    primitive defines the KV commit (dst slot base+j <- src slot
    base+chain_view_rows[b, j]) and the drafter's chain-view inputs."""
    idx = chain_view_rows.to(torch.int64)
    idx = idx.view(*idx.shape, *([1] * (x.dim() - 2))).expand(
        idx.shape[0], idx.shape[1], *x.shape[2:]
    )
    return torch.gather(x, 1, idx)


# ---------------------------------------------------------- full attention
def tree_attention_ref(
    spec: TreeSpec,
    q: torch.Tensor,  # [B, M, Hq, Dh]   (RoPE already applied at tree positions)
    k: torch.Tensor,  # [B, M, Hkv, Dh]  current-chunk raw K
    v: torch.Tensor,  # [B, M, Hkv, Dh]
    prefix_k: list[torch.Tensor],  # per request [P_b, Hkv, Dh] committed prefix
    prefix_v: list[torch.Tensor],
    scale: float,
) -> torch.Tensor:
    """Row r sees the whole committed prefix plus chunk rows k with
    anc_mask[r] bit k set.  Returns [B, M, Hq, Dh] fp32."""
    B, M, Hq, Dh = q.shape
    Hkv = k.shape[2]
    G = Hq // Hkv
    anc = torch.tensor(spec.anc_mask, dtype=torch.int64)
    bits = torch.arange(M, dtype=torch.int64)
    chunk_mask = ((anc[:, None] >> bits[None, :]) & 1).bool()  # [M(row), M(col)]
    out = torch.empty(B, M, Hq, Dh, dtype=torch.float32)
    for b in range(B):
        kk = torch.cat([prefix_k[b].float(), k[b].float()], 0)  # [P+M, Hkv, Dh]
        vv = torch.cat([prefix_v[b].float(), v[b].float()], 0)
        P = prefix_k[b].shape[0]
        mask = torch.cat([torch.ones(M, P, dtype=torch.bool), chunk_mask], 1)
        kk = kk.repeat_interleave(G, dim=1)  # [P+M, Hq, Dh]
        vv = vv.repeat_interleave(G, dim=1)
        s = torch.einsum("mhd,nhd->hmn", q[b].float(), kk) * scale
        s = s.masked_fill(~mask[None], float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[b] = torch.einsum("hmn,nhd->mhd", p, vv)
    return out


# ------------------------------------------------------------- GDN temporal
def _gdn_token(h, q, k, v, decay, beta):
    """One gated-delta-rule step, matching fused_sigmoid_gating /
    K5 (state h is [HV, V, K]).  q, k already normalised/scaled & GQA-expanded
    to [HV, K]; v [HV, V]; decay, beta [HV].  Returns (o [HV, V], h')."""
    h = h * decay[:, None, None]
    vn = (v - torch.einsum("hvk,hk->hv", h, k)) * beta[:, None]
    h = h + vn[:, :, None] * k[:, None, :]
    o = torch.einsum("hvk,hk->hv", h, q)
    return o, h


def _gdn_prep(q, k, g, b_raw_or_beta, scale, use_qk_l2norm, beta_is_raw, HV):
    H = q.shape[-2]
    rep = HV // H
    q = q.float()
    k = k.float()
    if use_qk_l2norm:
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q * scale
    q = q.repeat_interleave(rep, dim=-2)
    k = k.repeat_interleave(rep, dim=-2)
    decay = torch.exp(g.float())
    beta = torch.sigmoid(b_raw_or_beta.float()) if beta_is_raw else b_raw_or_beta.float()
    return q, k, decay, beta


def gdn_sequence_ref(h0, q, k, v, g, beta, scale, use_qk_l2norm=True, beta_is_raw=False):
    """Plain chain recurrence over T tokens of ONE request.
    h0 [HV,V,K]; q,k [T,H,K]; v [T,HV,V]; g,beta [T,HV].
    Returns (o [T,HV,V], states [T,HV,V,K] post-token)."""
    HV = v.shape[-2]
    q, k, decay, beta = _gdn_prep(q, k, g, beta, scale, use_qk_l2norm, beta_is_raw, HV)
    h = h0.float()
    outs, states = [], []
    for t in range(q.shape[0]):
        o, h = _gdn_token(h, q[t], k[t], v[t].float(), decay[t], beta[t])
        outs.append(o)
        states.append(h)
    return torch.stack(outs), torch.stack(states)


def gdn_tree_ref(
    spec: TreeSpec,
    h0: torch.Tensor,  # [B, HV, V, K] state before row 0 (= slot idx[b, acc_prev-1])
    q: torch.Tensor,  # [B, M, H, K] raw (pre-l2norm, pre-scale)
    k: torch.Tensor,  # [B, M, H, K]
    v: torch.Tensor,  # [B, M, HV, V]
    g: torch.Tensor,  # [B, M, HV] log decay (fp32, <= 0)
    beta: torch.Tensor,  # [B, M, HV] (post-sigmoid unless beta_is_raw)
    scale: float,
    use_qk_l2norm: bool = True,
    beta_is_raw: bool = False,
) -> dict[str, torch.Tensor]:
    """Tree verify semantics of a GDN layer.

    Row r's state is computed from its parent's post-state (row 0 from h0).
    Returns
      o            [B, M, HV, V]       output of every row
      chain_states [B, D+1, HV, V, K]  post-state of chain rows 0..D, i.e.
                                       exactly what the kernel writes to
                                       spec_state_indices[b, 0..D] (as today)
      leaf_states  [B, L, HV, V, K]    reference only; kernels do NOT store
      leaf_stash   [B, L, HV, K+V+2]   what the kernel MUST write per leaf
    """
    B, M, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    D, L = spec.D, spec.L
    qn, kn, decay, bt = _gdn_prep(q, k, g, beta, scale, use_qk_l2norm, beta_is_raw, HV)
    o = torch.empty(B, M, HV, V)
    post = torch.empty(B, M, HV, V, K)
    for b in range(B):
        for r in range(M):
            p = spec.rows_parent[r]
            h_in = h0[b].float() if p < 0 else post[b, p]
            o[b, r], post[b, r] = _gdn_token(
                h_in, qn[b, r], kn[b, r], v[b, r].float(), decay[b, r], bt[b, r]
            )
    leaf_rows = list(spec.leaf_rows)
    stash = torch.empty(B, L, HV, gdn_stash_width(K, V))
    if L:
        stash[..., :K] = kn[:, leaf_rows]
        stash[..., K : K + V] = v[:, leaf_rows].float()
        stash[..., K + V] = decay[:, leaf_rows]
        stash[..., K + V + 1] = bt[:, leaf_rows]
    return {
        "o": o,
        "chain_states": post[:, : D + 1].clone(),
        "leaf_states": post[:, leaf_rows].clone(),
        "leaf_stash": stash,
    }


def gdn_commit_ref(
    spec: TreeSpec,
    chain_states: torch.Tensor,  # [B, D+1, HV, V, K] slots after verify
    leaf_stash: torch.Tensor,  # [B, L, HV, K+V+2]
    rescue_leaf: torch.Tensor,  # [B] int, -1 = no rescue
) -> torch.Tensor:
    """After the walk: if request b rescued leaf l at depth j, slot j :=
    update(slot j-1, stash[l]).  Every other slot is untouched.  Returns the
    new chain_states.  After this, num_accepted = j + 1 reads slot j."""
    out = chain_states.clone()
    HV, V, K = chain_states.shape[2:]
    for b in range(out.shape[0]):
        l = int(rescue_leaf[b])
        if l < 0:
            continue
        j = spec.leaf_depth[l]
        s = leaf_stash[b, l].float()
        h = out[b, j - 1] * s[:, K + V, None, None]
        vn = (s[:, K : K + V] - torch.einsum("hvk,hk->hv", h, s[:, :K])) * s[
            :, K + V + 1, None
        ]
        out[b, j] = h + vn[:, :, None] * s[:, None, :K]
    return out


# ------------------------------------------------------------------ GDN conv
def conv_tree_ref(
    spec: TreeSpec,
    hist: torch.Tensor,  # [B, K-1, C] the K-1 conv inputs preceding x0 (time-major)
    x: torch.Tensor,  # [B, M, C] pre-conv inputs of every row
    weight: torch.Tensor,  # [C, Kw]
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> dict[str, torch.Tensor]:
    """Depthwise causal conv along each row's root path.

    Returns
      out        [B, M, C]
      window     [B, (Kw-2)+(D+1), C]  the stored conv state after verify
                 = hist[:, 1:] ++ x[:, chain rows]  (chain rows only; time-major;
                 a "column" below means an index on this time axis)
      leaf_x     [B, L, C]  leaf pre-conv inputs (the conv stash)
    """
    B, M, C = x.shape
    Kw = weight.shape[1]
    D = spec.D
    out = torch.empty(B, M, C)
    for b in range(B):
        for r in range(M):
            path = spec.path_rows(r)
            seq = torch.cat([hist[b].float(), x[b, list(path)].float()], 0)
            win = seq[-Kw:]  # [Kw, C]
            y = (win.t() * weight.float()).sum(-1)
            if bias is not None:
                y = y + bias.float()
            if activation in ("silu", "swish"):
                y = F.silu(y)
            out[b, r] = y
    window = torch.cat([hist[:, 1:].float(), x[:, : D + 1].float()], 1)
    leaf_x = x[:, list(spec.leaf_rows)].clone()
    return {"out": out, "window": window, "leaf_x": leaf_x}


def conv_commit_ref(
    spec: TreeSpec,
    window: torch.Tensor,  # [B, (Kw-2)+(D+1), C]
    leaf_x: torch.Tensor,  # [B, L, C]
    rescue_leaf: torch.Tensor,  # [B]
    Kw: int,
) -> torch.Tensor:
    """Rescue of leaf l at depth j overwrites window column (Kw-2)+j (chain
    token c_j's column, dead once c_j is rejected) with leaf_x[l]."""
    out = window.clone()
    for b in range(out.shape[0]):
        l = int(rescue_leaf[b])
        if l >= 0:
            out[b, (Kw - 2) + spec.leaf_depth[l]] = leaf_x[b, l].float()
    return out


def conv_next_hist_ref(window: torch.Tensor, num_accepted: torch.Tensor, Kw: int):
    """What the next step's conv reads: Kw-1 columns at offset
    num_accepted-1 (num_accepted = 1 + accepted drafts, as today)."""
    return torch.stack(
        [window[b, int(a) - 1 : int(a) - 1 + Kw - 1] for b, a in enumerate(num_accepted)]
    )


# ------------------------------------------------------------------ the walk
def tree_walk_ref(
    spec: TreeSpec,
    draft_token_ids: torch.Tensor,  # [B, M-1] int64; row r token at [:, r-1]; -1 = opt-out
    y: torch.Tensor,  # [B, M] int64 target sample at every row
) -> dict[str, torch.Tensor]:
    """Sample-and-match walk (lossless for deterministic drafts).

    cur = row 0; for d = 1..D: if y[cur] == c_d: cur = d; elif y[cur] equals a
    leaf token at depth d (rank order): emit leaf, emit y[leaf row], stop;
    else emit y[cur], stop.  Full chain: emit y[row D] (bonus).

    Returns
      output_token_ids [B, M] int64, path-ordered, -1 padded
                        (num non -1 = num_accepted = accepted drafts + 1 <= D+1)
      num_accepted      [B] int32
      rescue_leaf       [B] int32, -1 = none
      rescue_depth      [B] int32, 0 = none
      chain_view_rows   [B, D+1] int32, identity except [b, j] = leaf row
    """
    B = y.shape[0]
    D, M = spec.D, spec.M
    out = torch.full((B, M), -1, dtype=torch.int64)
    num_acc = torch.zeros(B, dtype=torch.int32)
    rleaf = torch.full((B,), -1, dtype=torch.int32)
    rdepth = torch.zeros(B, dtype=torch.int32)
    cvr = torch.arange(D + 1, dtype=torch.int32).repeat(B, 1)
    for b in range(B):
        emitted: list[int] = []
        cur = 0
        for d in range(1, D + 1):
            t = int(y[b, cur])
            if t == int(draft_token_ids[b, d - 1]) and t >= 0:
                emitted.append(t)
                cur = d
                continue
            hit = -1
            for l in spec.leaves_at_depth(d):
                lt = int(draft_token_ids[b, spec.leaf_rows[l] - 1])
                if lt >= 0 and lt == t:
                    hit = l
                    break
            if hit >= 0:
                row = spec.leaf_rows[hit]
                emitted += [t, int(y[b, row])]
                rleaf[b] = hit
                rdepth[b] = d
                cvr[b, d] = row
            else:
                emitted.append(t)
            break
        else:
            emitted.append(int(y[b, cur]))
        out[b, : len(emitted)] = torch.tensor(emitted)
        num_acc[b] = len(emitted)
    return {
        "output_token_ids": out,
        "num_accepted": num_acc,
        "rescue_leaf": rleaf,
        "rescue_depth": rdepth,
        "chain_view_rows": cvr,
    }


def tree_emit_distribution_ref(
    spec: TreeSpec,
    draft_tokens: list[int],  # [M-1] one request
    row_probs: torch.Tensor,  # [M, V] target distribution at every row
) -> dict[tuple[int, ...], float]:
    """Exact distribution of the emitted sequence for one request, by
    enumerating every y in V^M (tests only; keep V^M small)."""
    M, V = row_probs.shape
    dist: dict[tuple[int, ...], float] = {}
    drafts = torch.tensor([draft_tokens], dtype=torch.int64)
    for ys in itertools.product(range(V), repeat=M):
        p = 1.0
        for r, t in enumerate(ys):
            p *= float(row_probs[r, t])
        if p == 0.0:
            continue
        res = tree_walk_ref(spec, drafts, torch.tensor([ys], dtype=torch.int64))
        n = int(res["num_accepted"][0])
        key = tuple(int(t) for t in res["output_token_ids"][0, :n])
        dist[key] = dist.get(key, 0.0) + p
    return dist
