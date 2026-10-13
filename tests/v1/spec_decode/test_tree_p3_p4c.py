# SPDX-License-Identifier: Apache-2.0
"""CPU (TRITON_INTERPRET=1) tests for E16 P3 (TurboQuant verifier ancestor
mask) and P4c (causal conv1d tree update) against tree_reference.py."""

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
    causal_conv1d_update_tree,
)
from vllm.v1.attention.ops.triton_turboquant_decode import (
    triton_turboquant_spec_continuation_merge,
)
from vllm.v1.spec_decode import tree_reference as R
from vllm.v1.spec_decode.tree_spec import TreeSpec

SPECS = {
    "m7": TreeSpec.chain_plus_leaves(3),
    "w3": TreeSpec.chain_plus_leaves(3, width=3),
    "d2": TreeSpec.chain_plus_leaves(2),
}


def _prefix_state(q, pk, pv, scale):
    """(out, lse) of every query row over a dense prefix, torch fp32."""
    M, Hq, Dh = q.shape
    G = Hq // pk.shape[1]
    k = pk.float().repeat_interleave(G, 1)
    v = pv.float().repeat_interleave(G, 1)
    s = torch.einsum("mhd,nhd->hmn", q.float(), k) * scale
    return (
        torch.einsum("hmn,nhd->mhd", torch.softmax(s, -1), v),
        torch.logsumexp(s, -1).t().contiguous(),
    )


@pytest.mark.parametrize("name", list(SPECS))
def test_p3_tree_merge_matches_reference(name):
    torch.manual_seed(0)
    spec = SPECS[name]
    M, Hq, Hk, Dh, B = spec.M, 4, 2, 16, 2
    scale = Dh**-0.5
    anc = torch.tensor(spec.anc_mask, dtype=torch.int32)
    q = torch.randn(B, M, Hq, Dh)
    k = torch.randn(B, M, Hk, Dh)
    v = torch.randn(B, M, Hk, Dh)
    pk = [torch.randn(P, Hk, Dh) for P in (9, 5)]
    pv = [torch.randn(P, Hk, Dh) for P in (9, 5)]
    ref = R.tree_attention_ref(spec, q, k, v, pk, pv, scale)
    for b in range(B):
        po, pl = _prefix_state(q[b], pk[b], pv[b], scale)
        out = triton_turboquant_spec_continuation_merge(
            q[b], k[b], v[b], po.clone(), pl, scale, anc_mask=anc
        )
        torch.testing.assert_close(out, ref[b], atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("name", list(SPECS))
def test_p3_equals_causal_attention_along_each_path(name):
    """Leaf row == plain causal attention over (prefix, path rows)."""
    torch.manual_seed(1)
    spec = SPECS[name]
    M, Hq, Hk, Dh = spec.M, 4, 2, 16
    scale = Dh**-0.5
    anc = torch.tensor(spec.anc_mask, dtype=torch.int32)
    q = torch.randn(M, Hq, Dh)
    k = torch.randn(M, Hk, Dh)
    v = torch.randn(M, Hk, Dh)
    pk, pv = torch.randn(7, Hk, Dh), torch.randn(7, Hk, Dh)
    po, pl = _prefix_state(q, pk, pv, scale)
    out = triton_turboquant_spec_continuation_merge(
        q, k, v, po.clone(), pl, scale, anc_mask=anc
    )
    for r in range(M):
        path = list(spec.path_rows(r))
        T = len(path)
        # chain kernel on the gathered path: its last row is row r
        po_p, pl_p = _prefix_state(q[path], pk, pv, scale)
        o_path = triton_turboquant_spec_continuation_merge(
            q[path], k[path], v[path], po_p.clone(), pl_p, scale
        )
        torch.testing.assert_close(out[r], o_path[T - 1], atol=1e-5, rtol=1e-5)


def test_p3_chain_mask_is_bit_identical_and_off_unchanged():
    torch.manual_seed(2)
    M, Hq, Hk, Dh = 4, 4, 2, 16
    scale = Dh**-0.5
    q, k, v = torch.randn(M, Hq, Dh), torch.randn(M, Hk, Dh), torch.randn(M, Hk, Dh)
    po, pl = _prefix_state(q, torch.randn(6, Hk, Dh), torch.randn(6, Hk, Dh), scale)
    off = triton_turboquant_spec_continuation_merge(q, k, v, po.clone(), pl, scale)
    causal = torch.tensor([(1 << (r + 1)) - 1 for r in range(M)], dtype=torch.int32)
    on = triton_turboquant_spec_continuation_merge(
        q, k, v, po.clone(), pl, scale, anc_mask=causal
    )
    assert torch.equal(off, on)
    # chain-first truncation of a tree mask == chain output on those rows
    spec = SPECS["m7"]
    anc = torch.tensor(spec.anc_mask, dtype=torch.int32)
    q7, k7, v7 = (
        torch.randn(spec.M, Hq, Dh),
        torch.randn(spec.M, Hk, Dh),
        torch.randn(spec.M, Hk, Dh),
    )
    po7, pl7 = _prefix_state(q7, torch.randn(6, Hk, Dh), torch.randn(6, Hk, Dh), scale)
    tree = triton_turboquant_spec_continuation_merge(
        q7, k7, v7, po7.clone(), pl7, scale, anc_mask=anc
    )
    n = spec.D + 1
    chain = triton_turboquant_spec_continuation_merge(
        q7[:n], k7[:n], v7[:n], po7[:n].clone(), pl7[:n].contiguous(), scale
    )
    # different CHUNK_LEN -> different BLOCK_T tiling, so fp-reorder only
    torch.testing.assert_close(tree[:n], chain, atol=1e-6, rtol=1e-6)


def test_p3_anc_mask_validation():
    q = torch.randn(4, 4, 16)
    k = torch.randn(4, 2, 16)
    with pytest.raises(ValueError):
        triton_turboquant_spec_continuation_merge(
            q, k, k, torch.zeros_like(q), torch.zeros(4, 4), 1.0,
            anc_mask=torch.ones(4, dtype=torch.int64),
        )


# ---------------------------------------------------------------- P4c conv
def _conv_case(spec, dtype, B=3, C=24, Kw=4, silu=True, bias=True, nacc=None):
    g = torch.Generator().manual_seed(3)
    D, M, L = spec.D, spec.M, spec.L
    state_len = Kw - 1 + D
    weight = torch.randn(C, Kw, generator=g).to(dtype)
    bvec = torch.randn(C, generator=g).to(dtype) if bias else None
    x = torch.randn(B, M, C, generator=g).to(dtype)
    # SD layout: [slot, state_len, dim]; the kernel sees a [slot, dim, state_len] view
    n_slots = 1 + B * (D + 1)
    store = torch.randn(n_slots, state_len, C, generator=g).to(dtype)
    idx = (1 + torch.arange(B * (D + 1), dtype=torch.int32)).view(B, D + 1)
    na = torch.tensor(nacc if nacc is not None else [1, 2, D + 1][:B], dtype=torch.int32)
    return D, M, L, state_len, weight, bvec, x, store, idx, na


@pytest.mark.parametrize("name", list(SPECS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_p4c_conv_along_paths(name, dtype):
    spec = SPECS[name]
    Kw, C = 4, 24
    D, M, L, state_len, weight, bvec, x, store, idx, na = _conv_case(
        spec, dtype, Kw=Kw, C=C
    )
    B = x.shape[0]
    hist = torch.stack(
        [store[int(idx[b, 0]), int(na[b]) - 1 : int(na[b]) - 1 + Kw - 1] for b in range(B)]
    )
    ref = R.conv_tree_ref(spec, hist.float(), x.float(), weight.float(), bvec.float() if bvec is not None else None)
    stash = torch.zeros(B + 1, L, C, dtype=dtype)
    t = spec.device_tensors(torch.device("cpu"))
    state_before = store.clone()
    xin = x.reshape(B * M, C).clone()  # kernel is in place on x
    out = causal_conv1d_update_tree(
        xin, store.transpose(1, 2), weight, bvec, "silu", idx, na, t, stash, B
    )
    tol = 1e-4 if dtype == torch.float32 else 3e-2
    torch.testing.assert_close(out.view(B, M, C).float(), ref["out"], atol=tol, rtol=tol)
    torch.testing.assert_close(stash[:B].float(), ref["leaf_x"].float(), atol=0, rtol=0)
    # chain-only window store: slot idx[b,0] now holds hist[1:] ++ x[chain rows]
    for b in range(B):
        got = store[int(idx[b, 0])]
        exp = torch.cat([hist[b][1:], x[b, : D + 1]], 0)
        # the chain kernel shifts by na-1 first; compare as the next-step history
        torch.testing.assert_close(
            got[: D + Kw - 1].float(), exp.float(), atol=0, rtol=0
        )
        # untouched slots of other rows stay untouched
    touched = set(int(idx[b, 0]) for b in range(B))
    for s in range(store.shape[0]):
        if s not in touched:
            assert torch.equal(store[s], state_before[s])
    # next-step history after accepting a drafts equals conv_next_hist_ref
    win = torch.stack([store[int(idx[b, 0])] for b in range(B)]).float()
    for a in (1, 2, D + 1):
        nh = R.conv_next_hist_ref(win, torch.full((B,), a), Kw)
        assert nh.shape == (B, Kw - 1, C)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_p4c_chain_rows_bit_identical_to_chain_kernel(dtype):
    spec = SPECS["m7"]
    Kw, C = 4, 24
    D, M, L, state_len, weight, bvec, x, store, idx, na = _conv_case(spec, dtype, Kw=Kw, C=C)
    B = x.shape[0]
    s1, s2 = store.clone(), store.clone()
    stash = torch.zeros(B, L, C, dtype=dtype)
    t = spec.device_tensors(torch.device("cpu"))
    out_t = causal_conv1d_update_tree(
        x.reshape(B * M, C).clone(), s1.transpose(1, 2), weight, bvec, True, idx, na, t, stash, B
    ).view(B, M, C)
    xc = x[:, : D + 1].transpose(1, 2).contiguous()  # [B, C, D+1]
    out_c = causal_conv1d_update(
        xc, s2.transpose(1, 2), weight, bvec, True,
        conv_state_indices=idx, num_accepted_tokens=na,
    )
    assert torch.equal(out_t[:, : D + 1], out_c.transpose(1, 2))
    assert torch.equal(s1, s2)  # chain-only window store: identical state bytes


def test_p4c_leaf_equals_chain_when_same_input():
    """A leaf fed the chain token's own input must reproduce that chain row's
    output bit for bit (same taps, same order)."""
    spec = SPECS["m7"]
    Kw, C = 4, 24
    D, M, L, state_len, weight, bvec, x, store, idx, na = _conv_case(spec, torch.float32, Kw=Kw, C=C)
    B = x.shape[0]
    for l, (lr, d) in enumerate(zip(spec.leaf_rows, spec.leaf_depth)):
        x[:, lr] = x[:, d]  # leaf at depth d takes chain token c_d's input
    stash = torch.zeros(B, L, C)
    t = spec.device_tensors(torch.device("cpu"))
    out = causal_conv1d_update_tree(
        x.reshape(B * M, C).clone(), store.clone().transpose(1, 2), weight, bvec, True, idx, na, t, stash, B
    ).view(B, M, C)
    for lr, d in zip(spec.leaf_rows, spec.leaf_depth):
        assert torch.equal(out[:, lr], out[:, d])


def test_p4c_invalid_num_accepted_and_null_slot():
    spec = SPECS["m7"]
    Kw, C = 4, 24
    D, M, L, state_len, weight, bvec, x, store, idx, na = _conv_case(spec, torch.float32, Kw=Kw, C=C)
    B = x.shape[0]
    na = torch.tensor([0, 2, D + 2], dtype=torch.int32)
    idx[1, 0] = 0  # null block
    stash = torch.zeros(B, L, C)
    t = spec.device_tensors(torch.device("cpu"))
    out = causal_conv1d_update_tree(
        x.reshape(B * M, C).clone(), store.transpose(1, 2), weight, bvec, True, idx, na, t, stash, B
    ).view(B, M, C)
    for r in spec.leaf_rows:
        assert torch.count_nonzero(out[0, r]) == 0 and torch.count_nonzero(out[2, r]) == 0
