# SPDX-License-Identifier: Apache-2.0
"""CPU tests for E16 P7: drafter leaves (topk per depth) and the chain-view
adapter.  Run with TRITON_INTERPRET=1 (prepare_inputs_padded is Triton)."""

import contextlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.spec_decode.llm_base_proposer as lbp
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode import tree_reference as R
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.tree_drafter import (
    TreeChainViewBuffers,
    assemble_tree_drafts,
    tree_chain_view,
    tree_leaf_ranks_by_depth,
    tree_leaf_tokens,
)
from vllm.v1.spec_decode.tree_spec import TreeSpec

M7 = TreeSpec.chain_plus_leaves(3)
W3 = TreeSpec.chain_plus_leaves(3, width=3)
GAP = TreeSpec.from_choices([(0,), (0, 0), (0, 2), (0, 0, 0)])


# ------------------------------------------------------------------ leaves
def test_leaf_ranks_by_depth():
    assert tree_leaf_ranks_by_depth(M7) == {1: (1,), 2: (1,), 3: (1,)}
    assert tree_leaf_ranks_by_depth(W3) == {1: (1, 2), 2: (1, 2), 3: (1, 2)}
    assert tree_leaf_ranks_by_depth(GAP) == {2: (2,)}


def test_leaf_tokens_exclude_chain_and_rank():
    logits = torch.tensor([[0.0, 5.0, 3.0, 4.0, 1.0], [9.0, 1.0, 2.0, 8.0, 7.0]])
    chain = logits.argmax(-1)
    assert tree_leaf_tokens(logits, chain, (1,)).tolist() == [[3], [3]]
    assert tree_leaf_tokens(logits, chain, (1, 2)).tolist() == [[3, 2], [3, 4]]
    assert tree_leaf_tokens(logits, chain, (2,)).tolist() == [[2], [4]]
    # A non-argmax chain token (probabilistic drafting) is still excluded.
    assert tree_leaf_tokens(logits, torch.tensor([3, 4]), (1,)).tolist() == [[1], [0]]
    keep = logits.clone()
    tree_leaf_tokens(logits, chain, (1, 2))
    assert torch.equal(logits, keep)


def test_leaf_tokens_ties_never_equal_chain():
    logits = torch.zeros(4, 6)
    logits[:, 2] = 1.0
    logits[:, 4] = 1.0  # tie with 2
    chain = logits.argmax(-1)
    leaf = tree_leaf_tokens(logits, chain, (1,))
    assert (leaf.view(-1) != chain).all()


def test_assemble_order_and_opt_out():
    chain = torch.tensor([[11, 12, 13]])
    leaves = {1: torch.tensor([[21, 31]]), 2: torch.tensor([[22, 32]]), 3: torch.tensor([[23, 33]])}
    out = assemble_tree_drafts(W3, chain, leaves)
    # leaf order = (depth, rank): s1r1 s1r2 s2r1 s2r2 s3r1 s3r2
    assert out.tolist() == [[11, 12, 13, 21, 31, 22, 32, 23, 33]]
    assert out.dtype == torch.int64
    assert assemble_tree_drafts(M7, chain, None).tolist() == [[11, 12, 13, -1, -1, -1]]


# -------------------------------------------------------------- chain view
def _tree_cad(spec, base, block_table=None, slot_base=1000):
    B, M = len(base), spec.M
    qsl = torch.arange(B + 1, dtype=torch.int32) * M
    seq_lens = torch.tensor([b + M for b in base], dtype=torch.int32)
    slots = torch.tensor(
        [slot_base * (i + 1) + b + r for i, b in enumerate(base) for r in range(M)],
        dtype=torch.int64,
    )
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.clone(),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.clone(),
        num_reqs=B,
        num_actual_tokens=B * M,
        max_query_len=M,
        max_seq_len=int(seq_lens.max()),
        block_table_tensor=block_table
        if block_table is not None
        else torch.zeros(B, 4, dtype=torch.int32),
        slot_mapping=slots,
    )


def _walk_case(spec, ys):
    """Drafts fixed; one request per y row -> walk outputs."""
    B = len(ys)
    drafts = torch.tensor([[11, 12, 13, 21, 22, 23]] * B)
    return drafts, R.tree_walk_ref(spec, drafts, torch.tensor(ys))


@pytest.mark.parametrize("H", [8])
def test_chain_view_no_rescue_and_rescue_at_each_depth(H):
    spec = M7
    D, M = spec.D, spec.M
    # rows [x0, c1, c2, c3, s1, s2, s3]; drafts c=11,12,13 s=21,22,23
    ys = [
        [11, 12, 13, 99, 0, 0, 0],  # full chain + bonus: num_acc 4
        [21, 0, 0, 0, 41, 0, 0],  # rescue@1
        [11, 22, 0, 0, 0, 42, 0],  # rescue@2
        [11, 12, 23, 0, 0, 0, 43],  # rescue@3
        [5, 0, 0, 0, 0, 0, 0],  # reject at 1, no leaf: num_acc 1
    ]
    drafts, walk = _walk_case(spec, ys)
    B = len(ys)
    base = [100, 200, 300, 400, 500]
    cad = _tree_cad(spec, base)
    x0 = [7, 8, 9, 10, 6]
    tok = torch.cat([torch.tensor([x0[b]] + drafts[b].tolist()) for b in range(B)])
    pos = torch.tensor([base[b] + spec.rows_depth[r] for b in range(B) for r in range(M)])
    hid = torch.randn(B * M, H)
    valid = walk["num_accepted"].clone()
    buf = TreeChainViewBuffers(spec, 8, "cpu")
    cv = tree_chain_view(spec, cad, walk["chain_view_rows"], valid, tok, pos, hid, buf)

    cvr = walk["chain_view_rows"]
    # tokens / positions / hidden gathered by chain_view_rows (reference op)
    assert torch.equal(cv.target_token_ids, R.gather_chain_view(tok.view(B, M), cvr).reshape(-1))
    assert torch.equal(cv.target_hidden_states, R.gather_chain_view(hid.view(B, M, H), cvr).reshape(-1, H))
    # positions are depth = base + k
    assert cv.target_positions.tolist() == [base[b] + k for b in range(B) for k in range(D + 1)]
    # KV slots: sequence index base + k (rows 0..D), never the leaf row's slot
    want_slots = [1000 * (b + 1) + base[b] + k for b in range(B) for k in range(D + 1)]
    assert cv.common_attn_metadata.slot_mapping.tolist() == want_slots
    # the accepted path is a prefix of the view tokens shifted by one
    for b in range(B):
        n = int(walk["num_accepted"][b])
        emitted = walk["output_token_ids"][b, :n].tolist()
        view = cv.target_token_ids.view(B, D + 1)[b].tolist()
        assert view[1:n] == emitted[: n - 1]
    # rescue rows really come from the leaf
    assert cv.target_token_ids.view(B, D + 1)[1].tolist()[1] == 21
    assert cv.target_token_ids.view(B, D + 1)[2].tolist()[2] == 22
    assert cv.target_token_ids.view(B, D + 1)[3].tolist()[3] == 23
    # CAD shape
    c = cv.common_attn_metadata
    assert c.query_start_loc.tolist() == [(D + 1) * i for i in range(B + 1)]
    assert c.query_start_loc_cpu.tolist() == c.query_start_loc.tolist()
    assert c.seq_lens.tolist() == [b + D + 1 for b in base]
    assert c.seq_lens_cpu_upper_bound.tolist() == [b + D + 1 for b in base]
    assert (c.num_reqs, c.num_actual_tokens, c.max_query_len) == (B, B * (D + 1), D + 1)
    assert c.max_seq_len == max(base) + D + 1
    assert c.block_table_tensor is cad.block_table_tensor
    # rejected / sampled index
    assert cv.num_rejected_tokens_gpu.tolist() == [D + 1 - int(n) for n in valid]
    assert cv.token_indices_to_sample.tolist() == [
        b * (D + 1) + int(valid[b]) - 1 for b in range(B)
    ]


def test_chain_view_matches_prepare_inputs_padded_on_equivalent_chain():
    """The chain view's CAD / sampled index / rejected count equal what
    prepare_inputs_padded gives a chain step whose drafts were the accepted
    path (same base, same valid counts, incl. a discarded request)."""
    spec = M7
    D, M = spec.D, spec.M
    ys = [[11, 12, 13, 99, 0, 0, 0], [11, 22, 0, 0, 0, 42, 0], [21, 0, 0, 0, 41, 0, 0]]
    drafts, walk = _walk_case(spec, ys)
    B = len(ys)
    base = [10, 20, 30]
    valid = walk["num_accepted"].clone()
    valid[2] = 0  # discarded request (backup token): valid count 0
    cad = _tree_cad(spec, base)
    tok = torch.arange(B * M)
    pos = torch.arange(B * M)
    hid = torch.randn(B * M, 4)
    cv = tree_chain_view(spec, cad, walk["chain_view_rows"], valid, tok, pos, hid)

    chain = TreeSpec.from_choices([(0,), (0, 0), (0, 0, 0)])
    chain_cad = _tree_cad(chain, base)
    sd = SpecDecodeMetadata.make_dummy([[1] * D] * B, device=torch.device("cpu"))
    want_cad, want_idx, want_rej = lbp.SpecDecodeBaseProposer.prepare_inputs_padded(
        None, chain_cad, sd, valid.to(torch.int32)
    )
    got = cv.common_attn_metadata
    assert torch.equal(cv.token_indices_to_sample, want_idx)
    assert torch.equal(cv.num_rejected_tokens_gpu, want_rej)
    for f in ("query_start_loc", "query_start_loc_cpu", "seq_lens", "seq_lens_cpu_upper_bound"):
        assert torch.equal(getattr(got, f), getattr(want_cad, f)), f
    for f in ("num_reqs", "num_actual_tokens", "max_query_len", "max_seq_len"):
        assert getattr(got, f) == getattr(want_cad, f), f


def test_chain_view_mrope_positions():
    spec = M7
    B, M, D = 1, spec.M, spec.D
    drafts, walk = _walk_case(spec, [[11, 22, 0, 0, 0, 42, 0]])
    cad = _tree_cad(spec, [50])
    pos = torch.stack([torch.arange(M) + 50] * 3)
    pos[:, D + 1 :] = torch.tensor(spec.rows_depth[D + 1 :]) + 50
    cv = tree_chain_view(
        spec, cad, walk["chain_view_rows"], walk["num_accepted"], torch.arange(M), pos, torch.zeros(M, 2)
    )
    assert cv.target_positions.shape == (3, D + 1)
    assert cv.target_positions[0].tolist() == [50, 51, 52, 53]


# ------------------------------------------------- proposer harness (propose)
class _FakeMTP:
    """Deterministic stand-in for the MTP module: hidden = h + emb[ids];
    logits = hidden @ W.  Records every compute_logits input."""

    def __init__(self, V=50, H=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.randn(V, H, generator=g)
        self.W = torch.randn(H, V, generator=g)
        self.logits_calls: list[torch.Tensor] = []

    def __call__(self, input_ids=None, positions=None, inputs_embeds=None, hidden_states=None):
        return hidden_states + self.emb[input_ids.long()]

    def compute_logits(self, h):
        lg = h @ self.W
        self.logits_calls.append(lg.clone())
        return lg


def _make_proposer(monkeypatch, cls, spec, D, B_max=16, H=16):
    p = cls.__new__(cls)
    p.method = "mtp"
    p.tree_spec = spec
    p._tree_leaf_ranks = tree_leaf_ranks_by_depth(spec) if spec else {}
    p._tree_cv_buffers = None
    p.parallel_drafting = False
    p._share_mtp_indices = False
    p.eplb_state = None
    p.uses_mrope = False
    p.constant_draft_positions = False
    p._enable_probabilistic_draft_probs = False
    p.use_local_argmax_reduction = False
    p.use_heterogeneous_vocab = False
    p.allowed_attn_types = None
    p.supports_mm_inputs = False
    p.pass_hidden_states_to_model = True
    p.needs_extra_input_slots = False
    p.block_size = 16
    p.vllm_config = None
    p.max_model_len = 10_000
    n = 256
    p.input_ids = torch.zeros(n, dtype=torch.int32)
    p.positions = torch.zeros(n, dtype=torch.int64)
    p.hidden_states = torch.zeros(n, H)
    p.arange = torch.arange(n, dtype=torch.int32)
    p.token_arange_np = np.arange(n, dtype=np.int32)
    p.model = _FakeMTP(H=H)
    p.num_speculative_tokens = D
    p._last_draft_probs = None
    p._draft_attn_layer_names = set()
    monkeypatch.setattr(p, "build_per_group_and_layer_attn_metadata", lambda *a, **k: ([], {}), raising=False)
    monkeypatch.setattr(p, "_determine_batch_execution_and_padding", lambda n: (None, n, None), raising=False)
    monkeypatch.setattr(p, "_get_slot_mapping", lambda *a, **k: {}, raising=False)
    monkeypatch.setattr(p, "model_returns_tuple", lambda: False, raising=False)
    monkeypatch.setattr(
        p, "_set_positions", lambda n, pos: p.positions[:n].copy_(pos), raising=False
    )
    monkeypatch.setattr(
        p, "_update_positions_dependent_metadata", lambda pos, *a, **k: pos + 1, raising=False
    )
    return p


def _run_propose(p, mod, D, B=3, T=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    N = B * T
    qsl = torch.arange(B + 1, dtype=torch.int32) * T
    cad = CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.clone(),
        seq_lens=torch.full((B,), 40, dtype=torch.int32),
        num_reqs=B,
        num_actual_tokens=N,
        max_query_len=T,
        max_seq_len=40,
        block_table_tensor=torch.zeros(B, 4, dtype=torch.int32),
        slot_mapping=torch.arange(N, dtype=torch.int64),
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod, "set_forward_context", lambda *a, **k: contextlib.nullcontext())
        return p.propose(
            num_speculative_tokens=D,
            target_token_ids=torch.randint(0, 50, (N,), generator=g),
            target_positions=torch.arange(N),
            target_hidden_states=torch.randn(N, p.hidden_states.shape[1], generator=g),
            next_token_ids=torch.randint(0, 50, (B,), dtype=torch.int32, generator=g),
            token_indices_to_sample=None,
            common_attn_metadata=cad,
            sampling_metadata=SimpleNamespace(all_greedy=True),
        )


@pytest.mark.parametrize("D,spec", [(3, M7), (3, W3), (3, GAP), (1, TreeSpec.chain_plus_leaves(1))])
def test_propose_tree_leaves_and_chain_unchanged(monkeypatch, D, spec):
    off = _make_proposer(monkeypatch, EagleProposer, None, D)
    chain = _run_propose(off, lbp, D)
    assert chain.shape == (3, D)

    on = _make_proposer(monkeypatch, EagleProposer, spec, D)
    out = _run_propose(on, lbp, D)
    assert out.shape == (3, spec.M - 1) and out.dtype == torch.int64
    # chain columns bit-identical to the tree-off drafter
    assert torch.equal(out[:, :D], chain)
    # leaves = topk of the same logits with the chain token removed
    calls = on.model.logits_calls
    assert len(calls) == D
    for l, (d, r) in enumerate(zip(spec.leaf_depth, spec.leaf_rank)):
        lg = calls[d - 1].clone()
        lg.scatter_(-1, out[:, d - 1 : d], float("-inf"))
        want = lg.topk(r, dim=-1).indices[:, r - 1]
        assert torch.equal(out[:, D + l], want)
        assert (out[:, D + l] != out[:, d - 1]).all()


def test_propose_over_b_tree_emits_minus_one_leaves(monkeypatch):
    spec = TreeSpec.chain_plus_leaves(3, B_tree=2)
    on = _make_proposer(monkeypatch, EagleProposer, spec, 3)
    out = _run_propose(on, lbp, 3, B=3)
    assert out.shape == (3, spec.M - 1)
    assert (out[:, 3:] == -1).all()
    off = _make_proposer(monkeypatch, EagleProposer, None, 3)
    assert torch.equal(out[:, :3], _run_propose(off, lbp, 3, B=3))


def _load_base_proposer_module():
    """llm_base_proposer.py as of the P1 base commit (before P7)."""
    root = Path(lbp.__file__).resolve().parents[3]
    try:
        src = subprocess.run(
            ["git", "-C", str(root), "show", "372ec8ec19:vllm/v1/spec_decode/llm_base_proposer.py"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except Exception as e:  # pragma: no cover
        pytest.skip(f"base source unavailable: {e}")
    name = "_lbp_base_p1"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__name__"] = name
    sys.modules[name] = mod
    exec(compile(src, f"<{name}>", "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("D", [1, 3])
def test_tree_off_propose_byte_identical_to_base(monkeypatch, D):
    """Tree off: the new propose() == the pre-P7 propose() (same outputs,
    same compute_logits inputs, same number of passes)."""
    base = _load_base_proposer_module()

    class BaseEagle(base.SpecDecodeBaseProposer):
        pass

    old = _make_proposer(monkeypatch, BaseEagle, None, D)
    new = _make_proposer(monkeypatch, EagleProposer, None, D)
    a = _run_propose(old, base, D)
    b = _run_propose(new, lbp, D)
    assert torch.equal(a, b) and a.dtype == b.dtype
    assert len(old.model.logits_calls) == len(new.model.logits_calls) == D
    for x, y in zip(old.model.logits_calls, new.model.logits_calls):
        assert torch.equal(x, y)


def test_proposer_prepare_inputs_tree_chain_view(monkeypatch):
    spec = M7
    p = _make_proposer(monkeypatch, EagleProposer, spec, 3)
    p._tree_cv_buffers = TreeChainViewBuffers(spec, spec.B_tree, "cpu")
    drafts, walk = _walk_case(spec, [[11, 22, 0, 0, 0, 42, 0]])
    from vllm.v1.spec_decode.tree_spec import TreeStepOutputs

    outs = TreeStepOutputs(
        output_token_ids=walk["output_token_ids"],
        num_accepted=walk["num_accepted"],
        rescue_leaf=walk["rescue_leaf"],
        rescue_depth=walk["rescue_depth"],
        chain_view_rows=walk["chain_view_rows"],
    )
    cad = _tree_cad(spec, [100])
    tok = torch.tensor([7] + drafts[0].tolist())
    cv = p.prepare_inputs_tree_chain_view(
        cad, outs, walk["num_accepted"], tok, torch.arange(spec.M), torch.zeros(spec.M, 2)
    )
    assert cv.target_token_ids.tolist() == [7, 11, 22, 13]
    assert cv.token_indices_to_sample.tolist() == [2]
    assert cv.num_rejected_tokens_gpu.tolist() == [1]
