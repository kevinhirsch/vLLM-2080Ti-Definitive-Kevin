# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the E16 tree contract: TreeSpec, gating, and the torch
reference semantics every tree package is tested against."""

import random
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.v1.spec_decode import tree_reference as R
from vllm.v1.spec_decode.tree_spec import (
    TREE_MAX_ROWS,
    TreeSpec,
    TreeStash,
    decide_tree_step,
    opt_out_leaves,
    resolve_tree_spec,
    truncate_drafts_to_chain,
)

UPSTREAM_M7 = "[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1)]"


# ----------------------------------------------------------------- TreeSpec
def test_default_m7_tables():
    s = TreeSpec.chain_plus_leaves(3)
    assert (s.D, s.M, s.L) == (3, 7, 3)
    assert s.rows_parent == (-1, 0, 1, 2, 0, 1, 2)
    assert s.rows_depth == (0, 1, 2, 3, 1, 2, 3)
    assert s.rows_rank == (0, 0, 0, 0, 1, 1, 1)
    assert s.anc_mask == (0b1, 0b11, 0b111, 0b1111, 0b10001, 0b100011, 0b1000111)
    assert s.leaf_rows == (4, 5, 6)
    assert s.leaf_depth == (1, 2, 3)
    assert s.leaf_row_of_depth == (-1, 4, 5, 6)
    assert s.num_drafts == 6
    assert s == TreeSpec.from_choices(UPSTREAM_M7)


def test_order_insensitive_and_roundtrip():
    shuffled = "[(0,0,1),(0,),(0,1),(0,0,0),(1,),(0,0)]"
    a = TreeSpec.from_choices(shuffled)
    b = TreeSpec.from_choices(UPSTREAM_M7)
    assert a == b
    assert TreeSpec.from_choices(a.to_choices()) == a


def test_chain_only_is_causal_and_chain_first_truncation():
    for D in range(1, 6):
        c = TreeSpec.from_choices([(0,) * d for d in range(1, D + 1)])
        assert c.M == D + 1 and c.L == 0
        assert c.anc_mask == tuple((1 << (r + 1)) - 1 for r in range(D + 1))
        t = TreeSpec.chain_plus_leaves(D, width=3)
        n = D + 1
        assert t.rows_parent[:n] == c.rows_parent
        assert t.rows_depth[:n] == c.rows_depth
        assert t.anc_mask[:n] == c.anc_mask


def _random_spec(rng):
    D = rng.randint(1, 5)
    choices = [(0,) * d for d in range(1, D + 1)]
    for d in range(1, D + 1):
        for r in range(1, rng.randint(1, 3) + 1):
            if rng.random() < 0.5 and len(choices) + 1 <= TREE_MAX_ROWS - 1:
                choices.append((0,) * (d - 1) + (r,))
    rng.shuffle(choices)
    return TreeSpec.from_choices(choices, B_tree=rng.randint(1, 8))


def test_property_random_trees():
    rng = random.Random(0)
    for _ in range(300):
        s = _random_spec(rng)
        s.validate()
        for r in range(s.M):
            path = s.path_rows(r)
            assert path[0] == 0 and path[-1] == r
            assert len(path) == s.rows_depth[r] + 1
            assert s.anc_mask[r] == sum(1 << x for x in path)
        for l, row in enumerate(s.leaf_rows):
            assert row in [s.leaf_row_of_depth[s.leaf_depth[l]]] or s.leaf_rank[l] > 1
        assert s.max_leaves_per_depth <= 3


@pytest.mark.parametrize(
    "bad,exc",
    [
        ("[(0,),(1,1)]", NotImplementedError),  # non-chain parent
        ("[(0,),(0,1),(1,0)]", ValueError),  # (1,0) parent not chain
        ("[(0,0),(1,)]", ValueError),  # chain incomplete
        ("[(0,),(0,),(1,)]", ValueError),  # duplicate
        ("[(0,),(0,0,1)]", ValueError),  # missing (0,0)
        ("[(1,)]", ValueError),  # no chain
    ],
)
def test_rejects(bad, exc):
    with pytest.raises((exc, ValueError, NotImplementedError)):
        TreeSpec.from_choices(bad)


def test_rejects_too_many_rows():
    with pytest.raises(ValueError):
        TreeSpec.chain_plus_leaves(5, width=4)  # 1 + 5 + 15 = 21 > 16


def test_device_tensors_cpu():
    s = TreeSpec.chain_plus_leaves(3)
    t = s.device_tensors("cpu")
    assert t.anc_mask.dtype == torch.int32 and t.anc_mask.tolist() == list(s.anc_mask)
    assert t.rows_depth_i64.dtype == torch.int64
    assert t.leaf_parent_row.tolist() == [0, 1, 2]
    assert t.chain_rows.tolist() == [0, 1, 2, 3]
    assert t.leaf_row_of_depth.tolist() == [-1, 4, 5, 6]


def test_stash_allocate_shapes():
    s = TreeSpec.chain_plus_leaves(3, B_tree=4)
    st = TreeStash.allocate(s, ["a", "b"], 24, 128, 128, 5120, torch.float16, "cpu")
    assert st.gdn_leaf.shape == (2, 4, 3, 24, 258)
    assert st.conv_leaf_x.shape == (2, 4, 3, 5120)
    assert st.gdn_view("b").shape == (4, 3, 24, 258)
    assert st.conv_view("a").dtype == torch.float16


# ------------------------------------------------------------------- gating
def _cfg(**kw):
    base = dict(
        num_speculative_tokens=3,
        method="mtp",
        speculative_token_tree=None,
        speculative_tree_max_batch=None,
        parallel_drafting=False,
        num_speculative_tokens_per_batch_size=None,
        use_local_argmax_reduction=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_resolve_default_off():
    assert resolve_tree_spec(None, {}) is None
    assert resolve_tree_spec(_cfg(), {}) is None
    assert resolve_tree_spec(_cfg(), {"VLLM_SPEC_TREE": "0"}) is None
    assert resolve_tree_spec(_cfg(), {"VLLM_SPEC_TREE": ""}) is None


def test_resolve_env_and_config():
    s = resolve_tree_spec(_cfg(), {"VLLM_SPEC_TREE": "1"})
    assert s == TreeSpec.chain_plus_leaves(3) and s.B_tree == 4
    s = resolve_tree_spec(_cfg(), {"VLLM_SPEC_TREE": UPSTREAM_M7, "VLLM_SPEC_TREE_BMAX": "2"})
    assert s.M == 7 and s.B_tree == 2
    s = resolve_tree_spec(_cfg(speculative_token_tree="1", speculative_tree_max_batch=3), {})
    assert s.M == 7 and s.B_tree == 3
    # env off overrides config on
    assert resolve_tree_spec(_cfg(speculative_token_tree="1"), {"VLLM_SPEC_TREE": "0"}) is None
    s = resolve_tree_spec(_cfg(), {"VLLM_SPEC_TREE": "[(0,),(1,),(0,0),(0,0,0)]"})
    assert s.M == 5 and s.leaf_depth == (1,)


@pytest.mark.parametrize(
    "kw",
    [
        dict(num_speculative_tokens=4),  # chain depth mismatch
        dict(method="eagle"),
        dict(parallel_drafting=True),
        dict(use_local_argmax_reduction=True),
        dict(num_speculative_tokens_per_batch_size=[(1, 4, 3)]),
    ],
)
def test_resolve_rejects(kw):
    with pytest.raises(ValueError):
        resolve_tree_spec(_cfg(**kw), {"VLLM_SPEC_TREE": UPSTREAM_M7})


def test_speculative_config_has_fields_default_off():
    from vllm.config.speculative import SpeculativeConfig

    f = SpeculativeConfig.__dataclass_fields__
    assert f["speculative_token_tree"].default is None
    assert f["speculative_tree_max_batch"].default is None


def test_decide_tree_step():
    s = TreeSpec.chain_plus_leaves(3, B_tree=4)
    assert decide_tree_step(s, 1, True)
    assert decide_tree_step(s, 4, True)
    assert not decide_tree_step(s, 5, True)
    assert not decide_tree_step(s, 0, True)
    assert not decide_tree_step(s, 2, False)
    assert not decide_tree_step(s, 2, True, has_new_or_resumed_reqs=True)
    assert not decide_tree_step(s, 2, True, all_full_drafts=False)
    assert not decide_tree_step(None, 1, True)


def test_draft_list_helpers():
    s = TreeSpec.chain_plus_leaves(3)
    d = [11, 12, 13, 21, 22, 23]
    assert truncate_drafts_to_chain(s, d) == [11, 12, 13]
    assert opt_out_leaves(s, d) == [11, 12, 13, -1, -1, -1]
    assert opt_out_leaves(s, [11]) == [11]


# --------------------------------------------------------------- reference
def _dense_causal_last(q_last, k_seq, v_seq, scale, G):
    k_seq = k_seq.float().repeat_interleave(G, 1)
    v_seq = v_seq.float().repeat_interleave(G, 1)
    s = torch.einsum("hd,nhd->hn", q_last.float(), k_seq) * scale
    return torch.einsum("hn,nhd->hd", torch.softmax(s, -1), v_seq)


def test_tree_attention_equals_path_attention():
    torch.manual_seed(0)
    s = TreeSpec.chain_plus_leaves(3)
    B, Hq, Hkv, Dh = 2, 4, 2, 16
    q = torch.randn(B, s.M, Hq, Dh)
    k = torch.randn(B, s.M, Hkv, Dh)
    v = torch.randn(B, s.M, Hkv, Dh)
    pk = [torch.randn(5, Hkv, Dh), torch.randn(9, Hkv, Dh)]
    pv = [torch.randn(5, Hkv, Dh), torch.randn(9, Hkv, Dh)]
    out = R.tree_attention_ref(s, q, k, v, pk, pv, 0.25)
    for b in range(B):
        for r in range(s.M):
            path = list(s.path_rows(r))
            ks = torch.cat([pk[b], k[b, path]])
            vs = torch.cat([pv[b], v[b, path]])
            ref = _dense_causal_last(q[b, r], ks, vs, 0.25, Hq // Hkv)
            torch.testing.assert_close(out[b, r], ref, atol=1e-5, rtol=1e-5)


def test_chain_attention_equals_sdpa_causal():
    torch.manual_seed(1)
    s = TreeSpec.from_choices([(0,), (0, 0), (0, 0, 0)])
    B, H, Dh, P = 1, 2, 8, 6
    q = torch.randn(B, s.M, H, Dh)
    k = torch.randn(B, s.M, H, Dh)
    v = torch.randn(B, s.M, H, Dh)
    pk, pv = [torch.randn(P, H, Dh)], [torch.randn(P, H, Dh)]
    out = R.tree_attention_ref(s, q, k, v, pk, pv, Dh**-0.5)
    kk = torch.cat([pk[0], k[0]]).transpose(0, 1)
    vv = torch.cat([pv[0], v[0]]).transpose(0, 1)
    mask = torch.ones(s.M, P + s.M, dtype=torch.bool).tril(diagonal=P)
    ref = F.scaled_dot_product_attention(q[0].transpose(0, 1), kk, vv, attn_mask=mask)
    torch.testing.assert_close(out[0], ref.transpose(0, 1), atol=1e-5, rtol=1e-5)


def test_positions_and_offsets():
    s = TreeSpec.chain_plus_leaves(3)
    pos = R.tree_positions(s, torch.tensor([100, 7]))
    assert pos.tolist() == [[100, 101, 102, 103, 101, 102, 103], [7, 8, 9, 10, 8, 9, 10]]
    assert R.tree_token_offsets(s).tolist() == list(range(7))


def _gdn_inputs(s, B=2, H=2, HV=4, K=8, V=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        h0=torch.randn(B, HV, V, K, generator=g) * 0.1,
        q=torch.randn(B, s.M, H, K, generator=g),
        k=torch.randn(B, s.M, H, K, generator=g),
        v=torch.randn(B, s.M, HV, V, generator=g),
        g=-torch.rand(B, s.M, HV, generator=g),
        beta=torch.rand(B, s.M, HV, generator=g),
    )


def test_gdn_tree_equals_path_sequence_and_commit():
    s = TreeSpec.chain_plus_leaves(3)
    x = _gdn_inputs(s)
    K = x["q"].shape[-1]
    res = R.gdn_tree_ref(s, **x, scale=K**-0.5)
    for b in range(2):
        for r in range(s.M):
            path = list(s.path_rows(r))
            o, st = R.gdn_sequence_ref(
                x["h0"][b], x["q"][b, path], x["k"][b, path], x["v"][b, path],
                x["g"][b, path], x["beta"][b, path], K**-0.5,
            )
            torch.testing.assert_close(res["o"][b, r], o[-1], atol=1e-5, rtol=1e-5)
            if r <= s.D:
                torch.testing.assert_close(res["chain_states"][b, r], st[-1])
            else:
                torch.testing.assert_close(res["leaf_states"][b, r - s.D - 1], st[-1])
    # commit: rescue leaf 1 (depth 2) for b=0, none for b=1
    rl = torch.tensor([1, -1])
    new = R.gdn_commit_ref(s, res["chain_states"], res["leaf_stash"], rl)
    torch.testing.assert_close(new[0, 2], res["leaf_states"][0, 1], atol=1e-6, rtol=1e-5)
    assert torch.equal(new[0, :2], res["chain_states"][0, :2])
    assert torch.equal(new[1], res["chain_states"][1])


def test_gdn_chain_spec_is_todays_chain():
    s = TreeSpec.from_choices([(0,), (0, 0), (0, 0, 0)])
    x = _gdn_inputs(s, seed=3)
    res = R.gdn_tree_ref(s, **x, scale=0.3)
    for b in range(2):
        o, st = R.gdn_sequence_ref(
            x["h0"][b], x["q"][b], x["k"][b], x["v"][b], x["g"][b], x["beta"][b], 0.3
        )
        assert torch.equal(res["o"][b], o) and torch.equal(res["chain_states"][b], st)
    assert res["leaf_stash"].shape[1] == 0


def _conv_seq(seq, w, bias):
    y = (seq[-w.shape[1]:].t() * w).sum(-1) + bias
    return F.silu(y)


def test_conv_tree_path_and_commit_roll():
    torch.manual_seed(2)
    s = TreeSpec.chain_plus_leaves(3)
    B, C, Kw = 2, 6, 4
    hist = torch.randn(B, Kw - 1, C)
    x = torch.randn(B, s.M, C)
    w, bias = torch.randn(C, Kw), torch.randn(C)
    res = R.conv_tree_ref(s, hist, x, w, bias)
    assert res["window"].shape == (B, (Kw - 2) + s.D + 1, C)
    for b in range(B):
        for r in range(s.M):
            seq = torch.cat([hist[b], x[b, list(s.path_rows(r))]])
            torch.testing.assert_close(res["out"][b, r], _conv_seq(seq, w, bias))
    # rescue at depth j for every leaf: committed window rolled by num_accepted
    # equals the last Kw-1 inputs of the accepted path.
    for l, j in enumerate(s.leaf_depth):
        rl = torch.tensor([l, -1])
        win = R.conv_commit_ref(s, res["window"], res["leaf_x"], rl, Kw)
        acc = torch.tensor([j + 1, j])  # b=1: plain reject at depth j
        nh = R.conv_next_hist_ref(win, acc, Kw)
        path0 = list(s.path_rows(s.leaf_rows[l]))
        seq0 = torch.cat([hist[0], x[0, path0]])
        torch.testing.assert_close(nh[0], seq0[-(Kw - 1):])
        seq1 = torch.cat([hist[1], x[1, list(range(j))]])
        torch.testing.assert_close(nh[1], seq1[-(Kw - 1):])


def _drafts(c, s_):
    return torch.tensor([c + s_], dtype=torch.int64)


def test_walk_cases():
    s = TreeSpec.chain_plus_leaves(3)
    d = _drafts([11, 12, 13], [21, 22, 23])
    # y per row: rows [x0, c1, c2, c3, s1, s2, s3]
    cases = [
        # full accept -> c1 c2 c3 + bonus(y[3])
        ([11, 12, 13, 99, 0, 0, 0], [11, 12, 13, 99], -1, 0, [0, 1, 2, 3]),
        # reject at depth 1, rescue s1, bonus y[4]
        ([21, 0, 0, 0, 55, 0, 0], [21, 55], 0, 1, [0, 4, 2, 3]),
        # accept c1, rescue s2 at depth 2, bonus y[5]
        ([11, 22, 0, 0, 0, 66, 0], [11, 22, 66], 1, 2, [0, 1, 5, 3]),
        # accept c1 c2, rescue s3, bonus y[6]
        ([11, 12, 23, 0, 0, 0, 77], [11, 12, 23, 77], 2, 3, [0, 1, 2, 6]),
        # reject at depth 2 with no rescue -> recovered token
        ([11, 5, 0, 0, 0, 0, 0], [11, 5], -1, 0, [0, 1, 2, 3]),
        # leaf of depth 2 token appears at depth 1: no match
        ([22, 0, 0, 0, 0, 0, 0], [22], -1, 0, [0, 1, 2, 3]),
    ]
    for y, emitted, leaf, depth, cvr in cases:
        r = R.tree_walk_ref(s, d, torch.tensor([y]))
        n = int(r["num_accepted"][0])
        assert r["output_token_ids"][0, :n].tolist() == emitted
        assert (r["output_token_ids"][0, n:] == -1).all()
        assert int(r["rescue_leaf"][0]) == leaf and int(r["rescue_depth"][0]) == depth
        assert r["chain_view_rows"][0].tolist() == cvr
        assert n <= s.D + 1


def test_walk_opt_out_leaf_never_matches():
    s = TreeSpec.chain_plus_leaves(3)
    d = _drafts([11, 12, 13], [-1, -1, -1])
    r = R.tree_walk_ref(s, d, torch.tensor([[0, 0, 0, 0, 0, 0, 0]]))
    assert r["output_token_ids"][0, :1].tolist() == [0] and int(r["rescue_leaf"][0]) == -1


def test_walk_chain_spec_matches_greedy_chain_acceptance():
    rng = random.Random(4)
    s = TreeSpec.from_choices([(0,), (0, 0), (0, 0, 0)])
    for _ in range(200):
        drafts = [rng.randint(0, 3) for _ in range(3)]
        y = [rng.randint(0, 3) for _ in range(4)]
        # today's greedy chain: accept while target argmax == draft, then emit
        exp = []
        for i in range(3):
            exp.append(y[i])
            if y[i] != drafts[i]:
                break
        else:
            exp.append(y[3])
        r = R.tree_walk_ref(s, torch.tensor([drafts]), torch.tensor([y]))
        n = int(r["num_accepted"][0])
        assert r["output_token_ids"][0, :n].tolist() == exp


def test_walk_with_gather_chain_view_gives_committed_tokens():
    s = TreeSpec.chain_plus_leaves(3)
    d = _drafts([11, 12, 13], [21, 22, 23])
    tokens = torch.tensor([[7, 11, 12, 13, 21, 22, 23]])  # row tokens (x0=7)
    r = R.tree_walk_ref(s, d, torch.tensor([[11, 22, 0, 0, 0, 66, 0]]))
    cv = R.gather_chain_view(tokens, r["chain_view_rows"])
    n = int(r["num_accepted"][0])
    # committed sequence after x0 = chain view rows 1..n-1, then the bonus
    assert cv[0, 1:n].tolist() == r["output_token_ids"][0, : n - 1].tolist()


def test_walk_is_lossless_exact_enumeration():
    """Each emitted token is an exact draw from the target conditional at the
    node reached (V=3, D=2, a leaf at both depths: M=5, 3^5 outcomes)."""
    s = TreeSpec.chain_plus_leaves(2)  # rows [x0, c1, c2, s1, s2]
    g = torch.Generator().manual_seed(5)
    P = torch.softmax(torch.randn(s.M, 3, generator=g, dtype=torch.float64), -1)
    c1, c2, s1, s2 = 0, 1, 2, 2
    dist = R.tree_emit_distribution_ref(s, [c1, c2, s1, s2], P)
    assert abs(sum(dist.values()) - 1) < 1e-9
    first = torch.zeros(3, dtype=torch.float64)
    for seq, p in dist.items():
        first[seq[0]] += p
    torch.testing.assert_close(first, P[0].double(), atol=1e-9, rtol=0)

    def cond(prefix, nxt_row):
        tot = sum(p for q, p in dist.items() if q[: len(prefix)] == prefix)
        got = torch.zeros(3, dtype=torch.float64)
        for q, p in dist.items():
            if q[: len(prefix)] == prefix and len(q) > len(prefix):
                got[q[len(prefix)]] += p
        torch.testing.assert_close(got / tot, P[nxt_row].double(), atol=1e-9, rtol=0)

    cond((c1,), 1)  # after c1 the next token ~ p(row c1)
    cond((s1,), 3)  # after rescued s1 the bonus ~ p(row s1)
    cond((c1, c2), 2)  # after c1 c2 the bonus ~ p(row c2)
    cond((c1, s2), 4)  # after c1 + rescued s2 the bonus ~ p(row s2)
