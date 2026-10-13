# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the E16 P6 tree sampler (walk kernel + per-row sampling).

Run with TRITON_INTERPRET=1 (the walk is a Triton kernel)."""

import itertools
import random

import pytest
import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID, RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode import tree_reference as R
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.tree_sampler import (
    TreeSampler,
    apply_leaf_opt_out,
    tree_path_draft_cols,
    tree_row_contexts,
    tree_target_rows,
    tree_walk,
)
from vllm.v1.spec_decode.tree_spec import TreeSpec

DEV = "cpu"


@pytest.fixture(autouse=True)
def _no_flashinfer_sampler(monkeypatch):
    # CPU tests: the bonus row goes through Sampler; keep it off FlashInfer.
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

SPECS = {
    "m7": TreeSpec.chain_plus_leaves(3),
    "w3": TreeSpec.chain_plus_leaves(3, width=3),  # 2 leaves/depth, M=10
    "d13": TreeSpec.chain_plus_leaves(3, leaf_depths=(1, 3)),
    "chain3": TreeSpec.from_choices([(0,), (0, 0), (0, 0, 0)]),
    "d1": TreeSpec.chain_plus_leaves(1),
    "d2": TreeSpec.chain_plus_leaves(2),
    "gap": TreeSpec.from_choices([(0,), (0, 0), (0, 2), (0, 0, 0)]),  # rank 2 only
}


def _leaf_tables(spec):
    lr = torch.tensor(list(spec.leaf_rows) or [0], dtype=torch.int32)
    ld = torch.tensor(list(spec.leaf_depth) or [0], dtype=torch.int32)
    return lr, ld


def _md(
    B,
    *,
    greedy=True,
    temperature=None,
    top_k=None,
    top_p=None,
    output_token_ids=None,
    spec_token_ids=None,
    presence=None,
    prompt=None,
    bad_words=None,
    max_num_logprobs=None,
    generators=None,
    logitsprocs=None,
):
    if presence is not None:
        no_pen = False
        pres = torch.tensor(presence, dtype=torch.float32)
        freq = torch.zeros(B)
        rep = torch.ones(B)
    else:
        no_pen = True
        pres = freq = rep = torch.tensor([])
    if greedy:
        temperature = None
    return SamplingMetadata(
        temperature=temperature,
        all_greedy=greedy,
        all_random=not greedy and bool((temperature > 0).all()),
        top_p=top_p,
        top_k=top_k,
        generators=generators or {},
        max_num_logprobs=max_num_logprobs,
        no_penalties=no_pen,
        prompt_token_ids=prompt,
        frequency_penalties=freq,
        presence_penalties=pres,
        repetition_penalties=rep,
        output_token_ids=output_token_ids or [[] for _ in range(B)],
        allowed_token_ids_mask=None,
        bad_words_token_ids=bad_words or {},
        logitsprocs=logitsprocs or LogitsProcessors(),
        spec_token_ids=spec_token_ids,
    )


# --------------------------------------------------------------- tables
def test_target_rows_and_path_cols():
    s = SPECS["m7"]
    assert tree_target_rows(s) == (0, 1, 2, 4, 5, 6)
    assert tree_path_draft_cols(s) == ((), (0,), (0, 1), (0, 1, 2), (3,), (0, 4), (0, 1, 5))
    ctx = tree_row_contexts(s, [[7, 8]], [[11, 12, 13, 21, 22, 23]])
    assert ctx == [
        [7, 8],
        [7, 8, 11],
        [7, 8, 11, 12],
        [7, 8, 21],
        [7, 8, 11, 22],
        [7, 8, 11, 12, 23],
    ]


def test_leaf_opt_out():
    s = SPECS["m7"]
    d = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    o = apply_leaf_opt_out(s, d, [1])
    assert o.tolist() == [[1, 2, 3, 4, 5, 6], [1, 2, 3, -1, -1, -1]]
    assert apply_leaf_opt_out(s, d, []) is d
    assert d.tolist()[1] == [1, 2, 3, 4, 5, 6]  # input untouched


# ----------------------------------------------------------------- walk
def _check_walk(spec, drafts, y):
    lr, ld = _leaf_tables(spec)
    got = tree_walk(spec, drafts, y, lr, ld)
    ref = R.tree_walk_ref(spec, drafts, y)
    assert torch.equal(got.output_token_ids, ref["output_token_ids"])
    assert torch.equal(got.num_accepted, ref["num_accepted"])
    assert torch.equal(got.rescue_leaf, ref["rescue_leaf"])
    assert torch.equal(got.rescue_depth, ref["rescue_depth"])
    assert torch.equal(got.chain_view_rows, ref["chain_view_rows"])
    nvalid = (got.output_token_ids != PLACEHOLDER_TOKEN_ID).sum(1).to(torch.int32)
    assert torch.equal(nvalid, got.num_accepted)


@pytest.mark.parametrize("name", list(SPECS))
def test_walk_kernel_equals_reference_random(name):
    spec = SPECS[name]
    rng = random.Random(hash(name) & 0xFFFF)
    B = 64
    V = 4  # small vocab: many matches, rescues and corrections
    drafts = torch.tensor(
        [[rng.randrange(V) for _ in range(spec.M - 1)] for _ in range(B)]
    )
    # some opt-out leaves
    for b in range(0, B, 5):
        drafts[b, spec.D :] = -1
    y = torch.tensor([[rng.randrange(V) for _ in range(spec.M)] for _ in range(B)])
    _check_walk(spec, drafts, y)


def test_walk_kernel_exhaustive_m5():
    """Every y in V^M for a fixed draft (D=2, leaf at both depths)."""
    spec = SPECS["d2"]  # rows [x0, c1, c2, s1, s2]
    V = 3
    ys = torch.tensor(list(itertools.product(range(V), repeat=spec.M)))
    for drafts in ([0, 1, 2, 0], [0, 0, 1, 1], [2, 1, -1, 0]):
        d = torch.tensor([drafts] * ys.shape[0])
        _check_walk(spec, d, ys)


def test_walk_persistent_buffers_views():
    spec = SPECS["m7"]
    Bmax = 4
    from vllm.v1.spec_decode.tree_spec import TreeStepOutputs

    buf = TreeStepOutputs(
        output_token_ids=torch.full((Bmax, spec.M), 99, dtype=torch.int64),
        num_accepted=torch.zeros(Bmax, dtype=torch.int32),
        rescue_leaf=torch.zeros(Bmax, dtype=torch.int32),
        rescue_depth=torch.zeros(Bmax, dtype=torch.int32),
        chain_view_rows=torch.zeros((Bmax, spec.D + 1), dtype=torch.int32),
    )
    lr, ld = _leaf_tables(spec)
    d = torch.tensor([[11, 12, 13, 21, 22, 23], [11, 12, 13, 21, 22, 23]])
    y = torch.tensor([[11, 22, 0, 0, 0, 5, 0], [11, 12, 13, 9, 0, 0, 0]])
    res = tree_walk(spec, d, y, lr, ld, buf)
    assert res.output_token_ids.data_ptr() == buf.output_token_ids.data_ptr()
    assert res.output_token_ids.tolist() == [
        [11, 22, 5, -1, -1, -1, -1],
        [11, 12, 13, 9, -1, -1, -1],
    ]
    assert res.rescue_leaf.tolist() == [1, -1]
    assert res.rescue_depth.tolist() == [2, 0]
    assert res.chain_view_rows.tolist() == [[0, 1, 5, 3], [0, 1, 2, 3]]
    assert buf.output_token_ids[2].tolist() == [99] * spec.M  # untouched rows


# ------------------------------------------------------- exact lossless
def _kernel_emit_distribution(spec, draft, row_probs):
    """Exact emitted distribution computed with the *kernel* walk over every
    y in V^M (the sampler draws the rows independently)."""
    M, V = row_probs.shape
    ys = torch.tensor(list(itertools.product(range(V), repeat=M)))
    p = torch.ones(ys.shape[0], dtype=torch.float64)
    for r in range(M):
        p *= row_probs[r, ys[:, r]].double()
    lr, ld = _leaf_tables(spec)
    res = tree_walk(spec, torch.tensor([draft] * ys.shape[0]), ys, lr, ld)
    dist = {}
    for i in range(ys.shape[0]):
        if p[i] == 0:
            continue
        n = int(res.num_accepted[i])
        key = tuple(res.output_token_ids[i, :n].tolist())
        dist[key] = dist.get(key, 0.0) + float(p[i])
    return dist


@pytest.mark.parametrize(
    "draft",
    [[0, 1, 1, 2], [0, 0, 1, 1], [2, 1, 0, 0], [1, 2, -1, 0]],
)
def test_walk_kernel_is_exactly_lossless(draft):
    """Kernel walk == tree_emit_distribution_ref exactly, and each emitted
    token is an exact draw from the target conditional at its node."""
    spec = SPECS["d2"]  # rows [x0, c1, c2, s1, s2]
    torch.manual_seed(sum(draft) + 7)
    P = torch.rand(spec.M, 3, dtype=torch.float64) + 0.05
    P /= P.sum(-1, keepdim=True)
    got = _kernel_emit_distribution(spec, draft, P)
    ref = R.tree_emit_distribution_ref(spec, draft, P)
    assert set(got) == set(ref)
    for k in ref:
        assert abs(got[k] - ref[k]) < 1e-12
    # Autoregressive check: P(first token = t) = P[row 0, t]; conditional of
    # the second token given the first is the row of the node reached.
    c1, c2, s1, s2 = draft
    first = torch.zeros(3, dtype=torch.float64)
    for seq, p in got.items():
        first[seq[0]] += p
    assert torch.allclose(first, P[0], atol=1e-12)
    for t0, node in ((c1, 1), (s1, 3)):
        if t0 < 0 or (node == 3 and t0 == c1):
            continue
        cond = torch.zeros(3, dtype=torch.float64)
        for seq, p in got.items():
            if seq[0] == t0:
                assert len(seq) >= 2
                cond[seq[1]] += p
        assert torch.allclose(cond / cond.sum(), P[node], atol=1e-12)


def test_sampler_random_matches_exact_distribution():
    """End to end: TreeSampler (random, temperature 1) over many independent
    requests; the emitted-sequence frequencies match the exact enumeration
    within sampling noise (total variation)."""
    spec = SPECS["d2"]
    V = 3
    torch.manual_seed(0)
    P = torch.rand(spec.M, V) + 0.1
    P /= P.sum(-1, keepdim=True)
    draft = [0, 1, 1, 2]
    B = 1024
    reps = 6
    ts = TreeSampler(Sampler(), spec, DEV, max_batch=B)
    logits = P.log().repeat(B, 1)  # rows request-major
    drafts = torch.tensor([draft] * B)
    md = _md(B, greedy=False, temperature=torch.ones(B))
    counts = {}
    for _ in range(reps):
        _, res = ts(logits.clone(), drafts, md)
        for row, n in zip(res.output_token_ids.tolist(), res.num_accepted.tolist()):
            k = tuple(row[:n])
            counts[k] = counts.get(k, 0) + 1
    ref = R.tree_emit_distribution_ref(spec, draft, P.double())
    N = B * reps
    tv = 0.5 * sum(abs(counts.get(k, 0) / N - ref.get(k, 0)) for k in set(ref) | set(counts))
    assert set(counts) <= set(ref)
    assert tv < 0.03, tv


# ------------------------------------------------------ greedy / chain
def _onehot_logits(tokens_per_row, V=40):
    rows = [t for req in tokens_per_row for t in req]
    lg = torch.full((len(rows), V), -10.0)
    for i, t in enumerate(rows):
        lg[i, t] = 10.0
    return lg


def test_greedy_equals_reference_walk_of_argmax():
    spec = SPECS["m7"]
    rng = random.Random(3)
    B, V = 4, 6
    ts = TreeSampler(Sampler(), spec, DEV)
    for _ in range(20):
        logits = torch.randn(B * spec.M, V)
        drafts = torch.tensor([[rng.randrange(V) for _ in range(spec.M - 1)] for _ in range(B)])
        out, res = ts(logits, drafts, _md(B))
        ref = R.tree_walk_ref(spec, drafts, logits.argmax(-1).view(B, spec.M))
        assert torch.equal(res.output_token_ids, ref["output_token_ids"])
        assert torch.equal(res.chain_view_rows, ref["chain_view_rows"])
        assert out.sampled_token_ids.dtype == torch.int32
        assert torch.equal(out.sampled_token_ids.long(), ref["output_token_ids"])


def test_greedy_rescue_cases():
    spec = SPECS["m7"]
    ts = TreeSampler(Sampler(), spec, DEV)
    drafts = torch.tensor([[11, 12, 13, 21, 22, 23]] * 3)
    # y rows [x0, c1, c2, c3, s1, s2, s3]
    y = [
        [11, 12, 13, 14, 0, 0, 0],  # full chain + bonus 14
        [21, 0, 0, 0, 31, 0, 0],  # rescue at depth 1: s1, y[s1 row]=31
        [11, 12, 23, 0, 0, 0, 33],  # rescue at depth 3: c1 c2 s3 + 33
    ]
    out, res = ts(_onehot_logits(y), drafts, _md(3))
    assert res.output_token_ids.tolist() == [
        [11, 12, 13, 14, -1, -1, -1],
        [21, 31, -1, -1, -1, -1, -1],
        [11, 12, 23, 33, -1, -1, -1],
    ]
    assert res.rescue_depth.tolist() == [0, 1, 3]
    assert res.rescue_leaf.tolist() == [-1, 0, 2]
    assert res.chain_view_rows.tolist() == [[0, 1, 2, 3], [0, 4, 2, 3], [0, 1, 2, 6]]
    assert res.num_accepted.tolist() == [4, 2, 4]
    # opt-out of request 1: its leaf can no longer be accepted
    _, res = ts(_onehot_logits(y), drafts, _md(3), opt_out_reqs=[1])
    assert res.output_token_ids[1].tolist() == [21, -1, -1, -1, -1, -1, -1]
    assert res.rescue_leaf.tolist() == [-1, -1, 2]


def test_chain_spec_equals_rejection_sampler_greedy():
    """Chain-only spec: TreeSampler greedy == today's RejectionSampler."""
    spec = SPECS["chain3"]
    rng = random.Random(9)
    B, V, D = 3, 8, 3
    sampler = Sampler()
    ts = TreeSampler(sampler, spec, DEV)
    rs = RejectionSampler(sampler)
    for _ in range(25):
        logits = torch.randn(B * (D + 1), V)
        drafts = [[rng.randrange(V) for _ in range(D)] for _ in range(B)]
        md = _md(B)
        sd = SpecDecodeMetadata.make_dummy(drafts, device=torch.device(DEV))
        sd.target_logits_indices = torch.tensor(
            [b * (D + 1) + i for b in range(B) for i in range(D)], dtype=torch.int32
        )
        sd.bonus_logits_indices = torch.tensor(
            [b * (D + 1) + D for b in range(B)], dtype=torch.int32
        )
        want = rs(sd, None, logits.clone(), md).sampled_token_ids
        got, _ = ts(logits.clone(), torch.tensor(drafts), md)
        assert torch.equal(got.sampled_token_ids, want.to(torch.int32))


def test_presence_penalty_uses_path_context():
    """Row r is penalised by the drafts on ITS path only.  Leaf row s1
    (path x0->s1) must not see c1; chain row c1 must not see s1."""
    spec = SPECS["m7"]
    V = 8
    ts = TreeSampler(Sampler(), spec, DEV)
    # drafts: c1=1 c2=2 c3=3 s1=4 s2=5 s3=6
    drafts = [[1, 2, 3, 4, 5, 6]]
    # Every row prefers token 1 slightly over token 4.
    logits = torch.zeros(spec.M, V)
    logits[:, 1] = 1.0
    logits[:, 4] = 0.5
    md = _md(
        1,
        output_token_ids=[[]],
        spec_token_ids=drafts,
        presence=[2.0],
        prompt=torch.full((1, 1), V, dtype=torch.int64),  # pad-only prompt
    )
    # x0 row: no context -> argmax 1 = c1 -> continue; c1 row: context [1]
    # -> token 1 penalised -> 4; c2=2 != 4 -> leaf s2=5 != 4 -> emit 4, stop.
    _, res = ts(logits.clone(), torch.tensor(drafts), md)
    assert res.output_token_ids[0].tolist()[:3] == [1, 4, -1]
    # Make x0's row prefer s1 (=4); s1 row context is [4] (not c1=1):
    logits2 = logits.clone()
    logits2[0, 4] = 2.0
    _, res = ts(logits2, torch.tensor(drafts), md)
    # x0 -> 4 = s1 (rescue depth 1); s1 row: context [4], 4 penalised -> 1.
    assert res.output_token_ids[0].tolist()[:3] == [4, 1, -1]
    assert res.rescue_depth.tolist() == [1]


def test_bad_words_use_path_context():
    spec = SPECS["m7"]
    V = 8
    ts = TreeSampler(Sampler(), spec, DEV)
    drafts = [[1, 2, 3, 4, 5, 6]]
    logits = torch.zeros(spec.M, V)
    logits[:, 7] = 1.0  # everyone prefers 7
    logits[0, 4] = 5.0  # root -> s1
    # bad word [4, 7]: 7 is banned only after 4, i.e. on the s1 row's path.
    md = _md(1, output_token_ids=[[]], spec_token_ids=drafts, bad_words={0: [[4, 7]]})
    _, res = ts(logits, torch.tensor(drafts), md)
    out = res.output_token_ids[0].tolist()
    assert out[0] == 4 and out[1] != 7 and res.rescue_depth.tolist() == [1]


def test_top_k_respected_random(monkeypatch):
    # CPU: use the torch top-k/top-p path (the Triton one queries the GPU).
    import vllm.v1.sample.ops.topk_topp_sampler as tks

    monkeypatch.setattr(tks, "HAS_TRITON", False)
    spec = SPECS["m7"]
    V, B = 16, 8
    ts = TreeSampler(Sampler(), spec, DEV, max_batch=B)
    torch.manual_seed(1)
    logits = torch.randn(B * spec.M, V)
    drafts = torch.randint(0, V, (B, spec.M - 1))
    md = _md(
        B,
        greedy=False,
        temperature=torch.full((B,), 0.6),
        top_k=torch.full((B,), 2, dtype=torch.int64),
    )
    top2 = logits.topk(2, dim=-1).indices.view(B, spec.M, 2)
    for _ in range(10):
        _, res = ts(logits.clone(), drafts, md)
        # Every emitted token is in the top-2 of its source row.
        for b in range(B):
            n = int(res.num_accepted[b])
            for k in range(n):
                row = int(res.chain_view_rows[b, k])
                assert int(res.output_token_ids[b, k]) in top2[b, row].tolist()


def test_logprobs_gathered_along_path():
    spec = SPECS["m7"]
    V = 10
    ts = TreeSampler(Sampler(), spec, DEV)
    torch.manual_seed(5)
    drafts = torch.tensor([[1, 2, 3, 4, 5, 6]])
    logits = torch.randn(spec.M, V)
    logits[0, 4] = 9.0  # root -> s1 rescue
    logits[4, 8] = 9.0  # s1 row -> 8
    out, res = ts(logits.clone(), drafts, _md(1, max_num_logprobs=0))
    assert res.output_token_ids[0].tolist()[:2] == [4, 8]
    lp = out.logprobs_tensors.logprobs[:, 0]
    want = logits.log_softmax(-1)
    assert torch.allclose(lp[0], want[0, 4])
    assert torch.allclose(lp[1], want[4, 8])  # from the LEAF row, not row 1
    ids, lps = RejectionSampler.parse_output(
        out.sampled_token_ids, V, logprobs_tensors=out.logprobs_tensors
    )
    assert ids == [[4, 8]]
    assert len(lps.logprob_token_ids) == 2


def test_tree_sampler_does_not_mutate_input_logits():
    spec = SPECS["m7"]
    ts = TreeSampler(Sampler(), spec, DEV)
    logits = torch.randn(spec.M, 12)
    keep = logits.clone()
    ts(
        logits,
        torch.randint(0, 12, (1, spec.M - 1)),
        _md(1, greedy=False, temperature=torch.full((1,), 0.6)),
    )
    assert torch.equal(logits, keep)


def test_production_gencfg_mixed_batch(monkeypatch):
    """Production gencfg (T=0.6, top-k 20, top-p 0.95, presence penalty)
    with one greedy request in the batch: the greedy request equals the
    argmax walk of its penalised logits; random requests only emit tokens
    from the top-20 of their source row."""
    import vllm.v1.sample.ops.topk_topp_sampler as tks

    monkeypatch.setattr(tks, "HAS_TRITON", False)
    spec = SPECS["m7"]
    V, B = 64, 4
    ts = TreeSampler(Sampler(), spec, DEV)
    torch.manual_seed(11)
    logits = torch.randn(B * spec.M, V) * 3
    drafts = torch.randint(0, V, (B, spec.M - 1))
    temp = torch.tensor([0.0, 0.6, 0.6, 0.6])
    md = _md(
        B,
        greedy=False,
        temperature=temp,
        top_k=torch.full((B,), 20, dtype=torch.int64),
        top_p=torch.full((B,), 0.95),
        output_token_ids=[[] for _ in range(B)],
        spec_token_ids=drafts.tolist(),
        presence=[0.0] * B,  # penalties path active, zero strength
        prompt=torch.full((B, 1), V, dtype=torch.int64),
    )
    assert not md.all_greedy and not md.all_random
    top20 = logits.topk(20, dim=-1).indices.view(B, spec.M, 20)
    for _ in range(5):
        _, res = ts(logits.clone(), drafts, md)
        ref = R.tree_walk_ref(spec, drafts[:1], logits[: spec.M].argmax(-1)[None])
        assert torch.equal(res.output_token_ids[:1], ref["output_token_ids"])
        for b in range(1, B):
            for k in range(int(res.num_accepted[b])):
                row = int(res.chain_view_rows[b, k])
                assert int(res.output_token_ids[b, k]) in top20[b, row].tolist()
