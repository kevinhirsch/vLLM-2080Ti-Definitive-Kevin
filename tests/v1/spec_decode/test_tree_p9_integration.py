# SPDX-License-Identifier: Apache-2.0
"""E16 tree drafting, package P9: integration, CPU only.

Contract: /home/kevin/projects/lanes/coexec/results/tree-contract.md (§3-§6,
§10, §11).  Run with TRITON_INTERPRET=1 (the P4c conv and the P6 walk are
Triton kernels).

The end-to-end test drives several real tree steps:

    real Scheduler (tree decision, rollback, rescue metrics)
      -> real GPUModelRunner._update_states / _prepare_inputs (P2 inputs)
      -> GDN layer: the real K5 glue + the real P4c Triton tree conv
         (``_k5_tree_conv``, P9) + the P4a K5 kernel emulation (CUDA stand-in)
      -> TurboQuant KV store stand-in (packed bytes per (token, position))
      -> real runner._sample on a tree step (P6 TreeSampler + Triton walk)
      -> real runner hook -> real TreeCommit tables + the P5 kernel emulation
         (CUDA stand-in), BEFORE the (recorded) mamba align postprocess
      -> real runner._bookkeeping_sync -> scheduler.update_from_output
      -> P7 chain view (real tree_chain_view on the runner's tensors)
      -> P7 leaves (real tree_leaf_tokens / assemble_tree_drafts) on a
         synthetic drafter -> scheduler.update_draft_token_ids

with a deterministic synthetic target (greedy).  It checks the emitted
tokens equal plain greedy decoding (lossless), and after every step that the
GDN state the next step reads, the conv window and the TQ KV rows equal a
from-scratch chain run over the accepted history.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput, SamplerOutput
from vllm.v1.spec_decode import tree_reference as R
from vllm.v1.spec_decode.tree_drafter import (
    assemble_tree_drafts,
    tree_chain_view,
    tree_leaf_ranks_by_depth,
    tree_leaf_tokens,
)
from vllm.v1.spec_decode.tree_spec import TreeSpec, TreeStash

from tests.v1.core.utils import create_requests, create_scheduler
from tests.v1.spec_decode import test_tree_p2_p8 as P28
from tests.v1.spec_decode import test_tree_p4a_p5 as P45

SPEC = P28.SPEC  # chain_plus_leaves(3), B_tree 4, M = 7
D, M, W, L = SPEC.D, SPEC.M, SPEC.D + 1, SPEC.L
KD = VD = 128
H, HV = 1, 2  # GDN heads per rank in the stand-in layer
C = 2 * H * KD + HV * VD  # conv channels (q | k | v)
KW = 4  # conv width
CONV_LEN = (KW - 1) + D
NSLOTS = 16
TQ_BLOCKS, TQ_N, TQ_HK, TQ_C = 64, 16, 1, 8
PREFIX = "model.layers.0.linear_attn"


# ------------------------------------------------------------ synthetic models
def f_target(ctx: list[int]) -> int:
    """Greedy target: next token of a context."""
    return 100 + (ctx[-1] * 7 + len(ctx) * 3) % 997


def g_draft(ctx: list[int]) -> tuple[int, int]:
    """Drafter top-1 / top-2: right, rescuable (top-2 right) or a miss."""
    t = f_target(ctx)
    h = (ctx[-1] * 5 + len(ctx)) % 4
    if h in (0, 1):
        return t, t + 1
    if h == 2:
        return t + 1, t
    return t + 1, t + 2


def propose(ctxs: list[list[int]], V: int = 2048) -> torch.Tensor:
    """Synthetic MTP drafter using the real P7 leaf code: chain = argmax at
    every depth, leaves = tree_leaf_tokens of the same logits."""
    ranks = tree_leaf_ranks_by_depth(SPEC)
    cur = [list(c) for c in ctxs]
    chain, leaves = [], {}
    for d in range(1, D + 1):
        logits = torch.zeros(len(ctxs), V)
        for b, c in enumerate(cur):
            t1, t2 = g_draft(c)
            logits[b, t1], logits[b, t2] = 10.0, 9.0
        c_d = logits.argmax(-1)
        chain.append(c_d)
        if d in ranks:
            leaves[d] = tree_leaf_tokens(logits, c_d, ranks[d])
        for b in range(len(cur)):
            cur[b].append(int(c_d[b]))
    return assemble_tree_drafts(SPEC, torch.stack(chain, 1), leaves)


def emb(tok: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-conv GDN input row, a and b of a token (position independent)."""
    g = torch.Generator().manual_seed(max(tok, 0) + 17)
    x = (torch.randn(C, generator=g) * 0.6).half()
    a = torch.randn(HV, generator=g).half()
    b = torch.randn(HV, generator=g).half()
    return x, a, b


def tq_pack(tok: int, pos: int) -> torch.Tensor:
    """The packed TQ row the KV store writes for (token, RoPE position)."""
    return torch.tensor(
        [(tok * 31 + pos * 7 + 3 * h + c) % 256 for h in range(TQ_HK) for c in range(TQ_C)],
        dtype=torch.uint8,
    ).view(TQ_HK, TQ_C)


# ------------------------------------------------------------- GDN stand-in
class World:
    """One GDN layer (real glue, P4c Triton conv, K5 emulation) + one TQ
    layer's cache, for B requests on fixed slots (mamba 'none' semantics:
    the next step reads idx[b, num_accepted - 1])."""

    def __init__(self, monkeypatch, B: int, dim_first: bool, prompts: list[list[int]]):
        from vllm.model_executor.layers.mamba.gdn import gdn_sm75_cuda as k5
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as Q
        from vllm.v1.spec_decode.tree_commit import GDNCommitLayer, TreeCommit

        gen = torch.Generator().manual_seed(1)
        self.B = B
        self.dim_first = dim_first
        monkeypatch.setattr(Q, "is_conv_state_dim_first", lambda: dim_first)
        monkeypatch.setattr(k5, "eligible", lambda **kw: True)
        self.preflight_rows = []
        monkeypatch.setattr(
            k5,
            "inputs_supported",
            lambda **kw: self.preflight_rows.append(kw["rows_per_request"]) or True,
        )
        monkeypatch.setattr(k5, "gdn_mtp_tree", self._k5_tree)
        monkeypatch.setattr(
            k5, "gdn_mtp", lambda *a: pytest.fail("chain K5 launch on a tree step")
        )

        # ---- the layer (real class, stand-in submodules)
        lay = Q.QwenGatedDeltaNetAttention.__new__(Q.QwenGatedDeltaNetAttention)
        nn.Module.__init__(lay)
        lay.prefix = PREFIX
        lay.tp_size, lay.tp_rank = 1, 0
        lay.num_k_heads, lay.num_v_heads = H, HV
        lay.head_k_dim, lay.head_v_dim = KD, VD
        lay.key_dim, lay.value_dim = H * KD, HV * VD
        lay.hidden_size = HV * VD
        lay.activation = "silu"
        lay.gqa_interleaved_layout = False
        lay.disable_tp_for_ba_proj = False
        lay.A_log = torch.randn(HV, generator=gen) * 0.5
        lay.dt_bias = (torch.randn(HV, generator=gen) * 0.3).half()
        lay.conv1d = SimpleNamespace(
            weight=(torch.randn(C, 1, KW, generator=gen) * 0.4).half(),
            bias=(torch.randn(C, generator=gen) * 0.1).half(),
        )
        lay.norm = SimpleNamespace(
            weight=(1 + 0.1 * torch.randn(VD, generator=gen)).half(),
            eps=1e-6,
            activation="swish",
            norm_before_gate=False,
            group_size=None,
        )
        lay.out_proj = lambda x: (x, None)
        ssm = torch.randn(NSLOTS, HV, VD, KD, generator=gen) * 0.05
        if dim_first:
            conv_raw = torch.zeros(NSLOTS, C, CONV_LEN).half()
            conv_view = conv_raw
        else:
            conv_raw = torch.zeros(NSLOTS, CONV_LEN, C).half()
            conv_view = conv_raw.transpose(-1, -2)
        lay.kv_cache = (conv_raw, ssm)
        self.layer, self.ssm, self.conv = lay, ssm, conv_view

        # ---- runner-owned objects (as _init_tree_runtime builds them)
        self.stash = TreeStash.allocate(
            SPEC, [PREFIX], HV, KD, VD, C, torch.float16, "cpu"
        )
        lay.tree_stash = self.stash
        self.builder = P45._builder(SPEC, monkeypatch)
        self.sidx = torch.arange(1, 1 + B * W, dtype=torch.int32).view(B, W)
        self.tq = P45._tq_cache(TQ_BLOCKS, TQ_HK, TQ_N, TQ_C, gen)
        self.tq.zero_()
        self.commit = TreeCommit(
            SPEC,
            self.stash,
            [GDNCommitLayer(PREFIX, ssm, conv_view, self.builder.spec_state_indices_tensor)],
            [self.tq],
            "cpu",
        )
        self.commits = 0
        from vllm.v1.spec_decode import tree_commit as TC

        monkeypatch.setattr(TC, "_load", lambda: SimpleNamespace(tree_commit=self._commit_ext))

        # ---- initial state: h0 at idx[b, 0]; conv window = last K-1 prompt x
        self.h0 = [ssm[int(self.sidx[b, 0])].clone() for b in range(B)]
        for b in range(B):
            for col, tok in enumerate(prompts[b][-(KW - 1):]):
                self.conv[int(self.sidx[b, 0]), :, col] = emb(tok)[0]
        self.nacc = torch.ones(B, dtype=torch.int32)

    # CUDA stand-ins -----------------------------------------------------
    def _k5_tree(self, mixed, a, b, A_log, dt_bias, sidx, cu, nacc, state, z, norm_w, out,
                 scale, eps, null_id, sigmoid, leaf_rows, leaf_depth, stash):
        P45.emulate_k5(mixed, a, b, A_log, dt_bias, sidx, cu, nacc, state, z, norm_w, out,
                       scale, eps, null_id, sigmoid, leaf_rows, leaf_depth, stash)

    def _commit_ext(self, rescue_leaf, gdn_tab, gdn_leaf, conv_leaf_x, tq_tab, tq_slot_mapping,
                    *rest):
        assert gdn_tab is self.commit.gdn_tab and tq_tab is self.commit.tq_tab
        assert gdn_leaf is self.stash.gdn_leaf and conv_leaf_x is self.stash.conv_leaf_x
        P45.emulate_commit(self.commit, rescue_leaf, tq_slot_mapping)
        self.commits += 1

    # one verify forward ----------------------------------------------------
    def forward(self, toks: torch.Tensor, pos: torch.Tensor, slots: torch.Tensor):
        B, n = self.B, self.B * M
        rows = [emb(int(t)) for t in toks.tolist()]
        x = torch.stack([r[0] for r in rows])
        a = torch.stack([r[1] for r in rows])
        bb = torch.stack([r[2] for r in rows])
        z = torch.randn(n, HV * VD, generator=torch.Generator().manual_seed(9)).half()
        mixed_qkvz = torch.cat([x, z], 1)
        ba = torch.cat([bb, a], 1)  # split_ba: b, a = chunk(2)
        cam = SimpleNamespace(
            query_start_loc=(torch.arange(B + 1) * M).int(),
            query_start_loc_cpu=(torch.arange(B + 1) * M).int(),
            block_table_tensor=torch.cat(
                [self.sidx, torch.full((B, 4), -1, dtype=torch.int32)], 1
            ),
            seq_lens=torch.full((B,), 50),
            num_reqs=B,
            num_actual_tokens=n,
            slot_mapping=None,
            tree_step=True,
        )
        md = self.builder.build(0, cam, self.nacc.clone(), torch.full((B,), M - 1))
        assert md.tree_rows == M and md.tree_spec is SPEC
        self.pre_conv = x.clone()
        out = torch.zeros(n, HV * VD).half()
        assert self.layer._k5_spec_decode_md(md, mixed_qkvz, ba, out)
        assert self.preflight_rows[-1] == M
        self.post_conv = mixed_qkvz[:, :C].clone()  # the glue converts in place
        # TurboQuant store stand-in: row-indexed slots, RoPE (depth) positions.
        for r in range(n):
            s = int(slots[r])
            self.tq[s // TQ_N, :, s % TQ_N] = tq_pack(int(toks[r]), int(pos[r]))

    # references --------------------------------------------------------------
    def ref_post_conv(self, seq: list[int], P: int) -> torch.Tensor:
        """Post-conv rows of seq[P:] over the true history (fp32 conv,
        fp16 rounded like the kernel output)."""
        w = self.layer.conv1d.weight.view(C, KW).float()
        bias = self.layer.conv1d.bias.float()
        xs = torch.stack([emb(t)[0] for t in seq]).float()
        outs = []
        for i in range(P, len(seq)):
            win = xs[i - KW + 1 : i + 1]
            outs.append(F.silu((win.t() * w).sum(-1) + bias).half())
        return torch.stack(outs)

    def ref_state(self, b: int, seq: list[int], P: int) -> torch.Tensor:
        post = self.ref_post_conv(seq, P).float()
        q = post[:, : H * KD].view(-1, H, KD)
        k = post[:, H * KD : 2 * H * KD].view(-1, H, KD)
        v = post[:, 2 * H * KD :].view(-1, HV, VD)
        ab = [emb(t) for t in seq[P:]]
        a = torch.stack([r[1] for r in ab])
        bb = torch.stack([r[2] for r in ab])
        g, beta = P45.k5_gating(a, bb, self.layer.A_log, self.layer.dt_bias)
        _, states = R.gdn_sequence_ref(self.h0[b], q, k, v, g, beta, KD**-0.5)
        return states[-1]


# --------------------------------------------------------------- the runner
def _runner(monkeypatch):
    runner = P28._runner_with_tree(monkeypatch)
    assert runner.tree_sampler is not None and runner.tree_spec is SPEC
    # CPU: the D2H transfer event of _to_list is a no-op stand-in.
    runner.transfer_event = SimpleNamespace(record=lambda *a: None, synchronize=lambda: None)
    return runner


def _runner_bookkeep(runner, out, sampler_output, n):
    res = runner._bookkeeping_sync(out, sampler_output, None, torch.zeros(n, 1), n)
    return res[3]  # valid_sampled_token_ids


def _mro(runner, valid):
    ids = list(runner.input_batch.req_ids)
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={r: i for i, r in enumerate(ids)},
        sampled_token_ids=valid,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _hook_patches(monkeypatch, runner, world, order):
    """Let _update_states_after_model_execute run its hybrid/align branch on
    the CPU runner; the align postprocess is recorded, not run."""
    from vllm.v1.worker import gpu_model_runner as G

    monkeypatch.setattr(runner, "speculative_config", SimpleNamespace(), raising=False)
    monkeypatch.setattr(
        runner, "model_config", SimpleNamespace(is_hybrid=True, architecture="stub")
    )
    monkeypatch.setattr(runner.cache_config, "mamba_cache_mode", "align")
    monkeypatch.setattr(runner, "_get_mamba_bufs", lambda: None)
    monkeypatch.setattr(runner, "_get_mamba_state_copy_funcs", lambda: None)
    monkeypatch.setattr(
        runner, "num_accepted_tokens_event", SimpleNamespace(record=lambda: None),
        raising=False,
    )

    def post(**kw):
        order.append(("postprocess", world.commits))

    monkeypatch.setattr(G.mamba_utils, "postprocess_mamba_align_gpu", post)


@pytest.mark.parametrize("dim_first", [False, True], ids=["SD", "DS"])
def test_tree_steps_end_to_end(monkeypatch, dim_first):
    from vllm.v1.spec_decode import metrics as MET

    B = 2
    runner = _runner(monkeypatch)
    V = runner.input_batch.vocab_size
    sched = create_scheduler(num_speculative_tokens=D, max_num_seqs=16, num_blocks=TQ_BLOCKS)
    sched.tree_spec = SPEC
    sched.log_stats = True
    observed = []
    orig_observe = MET.SpecDecodingStats.observe_tree
    monkeypatch.setattr(
        MET.SpecDecodingStats,
        "observe_tree",
        lambda self, d: (observed.append(d), orig_observe(self, d))[1],
    )
    reqs = create_requests(num_requests=B, num_tokens=10, max_tokens=64)
    for r in reqs:
        r.sampling_params.temperature = 0.0
        sched.add_request(r)
    rid_b = {r.request_id: i for i, r in enumerate(reqs)}
    seq = [list(r.prompt_token_ids) for r in reqs]
    P = len(seq[0])

    # ---- prefill step (chain, not tree): runner inputs + bookkeeping only.
    out = sched.schedule()
    assert not out.tree_step
    runner._update_states(out)
    ids = list(runner.input_batch.req_ids)
    nst = np.array([out.num_scheduled_tokens[r] for r in ids], dtype=np.int32)
    runner._prepare_inputs(out, nst)
    t0 = [[f_target(seq[rid_b[r]])] for r in ids]
    valid = _runner_bookkeep(
        runner, out, SamplerOutput(torch.tensor(t0, dtype=torch.int32), None), int(nst.sum())
    )
    assert valid == t0
    sched.update_from_output(out, _mro(runner, valid))
    for r, v in zip(ids, valid):
        seq[rid_b[r]] += v
    drafts0 = propose([seq[rid_b[r]] for r in ids])
    sched.update_draft_token_ids(DraftTokenIds(ids, drafts0.tolist()))

    world = World(monkeypatch, B, dim_first, [seq[rid_b[r]][:P] for r in ids])
    runner.tree_commit = world.commit
    order: list = []

    rescues = []
    for step in range(5):
        out = sched.schedule()
        assert out.tree_step, step
        assert all(len(v) == M - 1 for v in out.scheduled_spec_decode_tokens.values())
        runner._update_states(out)
        ids = list(runner.input_batch.req_ids)
        nst = np.array([out.num_scheduled_tokens[r] for r in ids], dtype=np.int32)
        assert nst.tolist() == [M] * B
        _, sdm, _ = runner._prepare_inputs(out, nst)
        n = B * M
        slots = runner.input_batch.block_table[0].slot_mapping.gpu[:n].clone()
        pos = runner.positions[:n].clone()
        toks = runner.input_ids.gpu[:n].clone()
        drafts = runner.tree_draft_token_ids[:B].clone()
        base = [len(seq[rid_b[r]]) - 1 for r in ids]
        assert pos.view(B, M).tolist() == [[bb + d for d in SPEC.rows_depth] for bb in base]

        # ---- forward: GDN layer (glue + conv + K5 stand-in) and TQ store
        world.forward(toks, pos, slots)

        # ---- target logits: row r sees committed seq + its path's drafts
        logits = torch.zeros(n, V)
        for i, r in enumerate(ids):
            ctx0 = seq[rid_b[r]]
            for row in range(M):
                path = [int(drafts[i, q - 1]) for q in SPEC.path_rows(row)[1:]]
                logits[i * M + row, f_target(ctx0 + path)] = 10.0
        sampler_output = runner._sample(logits, sdm, tree_step=True)
        outs = runner._tree_step_outputs
        walk = R.tree_walk_ref(SPEC, drafts, logits.argmax(-1).view(B, M))
        assert torch.equal(outs.output_token_ids[:B], walk["output_token_ids"])
        assert torch.equal(outs.rescue_leaf[:B].long(), walk["rescue_leaf"].long())
        assert torch.equal(outs.chain_view_rows[:B].long(), walk["chain_view_rows"].long())
        assert torch.equal(sampler_output.sampled_token_ids.long(), walk["output_token_ids"])

        # ---- commit (runner hook) before the align postprocess
        runner._tree_tq_slot_mapping = slots
        before = world.commits
        with monkeypatch.context() as mp:
            _hook_patches(mp, runner, world, order)
            runner._update_states_after_model_execute(
                sampler_output.sampled_token_ids, out
            )
        assert world.commits == before + 1
        assert order[-1] == ("postprocess", before + 1)  # commit ran first
        nacc = runner.num_accepted_tokens.gpu[:B].clone()
        assert torch.equal(nacc.long(), walk["num_accepted"].long())

        # ---- bookkeeping -> scheduler
        valid = _runner_bookkeep(runner, out, sampler_output, n)
        for i, r in enumerate(ids):
            want = []
            ctx = list(seq[rid_b[r]])
            for _ in range(len(valid[i])):  # lossless: plain greedy decoding
                want.append(f_target(ctx))
                ctx.append(want[-1])
            assert valid[i] == want, (step, r)
        sched.update_from_output(out, _mro(runner, valid))
        rescues += [int(x) for x in outs.rescue_depth[:B]]
        assert observed[-B:] == [int(x) for x in outs.rescue_depth[:B]]
        for i, r in enumerate(ids):
            assert sched.requests[r].num_computed_tokens == base[i] + int(nacc[i])

        # ---- the next step's view of GDN / conv / TQ == chain run on history
        for i, r in enumerate(ids):
            b = rid_b[r]
            k = int(nacc[i])
            processed = seq[b][-1:] + valid[i][:-1]  # [x0, accepted path]
            seq[b] += valid[i]
            hist = seq[b][:-1]  # every token the layers have consumed
            src = int(world.sidx[i, k - 1])
            torch.testing.assert_close(
                world.ssm[src], world.ref_state(i, hist, P), atol=2e-3, rtol=2e-3
            )
            win = world.conv[int(world.sidx[i, 0])][:, k - 1 : k - 1 + KW - 1].t()
            want_win = torch.stack([emb(t)[0] for t in hist[-(KW - 1):]])
            assert torch.equal(win, want_win), (step, r)
            for kk in range(k):
                s = int(slots[i * M + kk])
                got = world.tq[s // TQ_N, :, s % TQ_N]
                assert torch.equal(got, tq_pack(processed[kk], base[i] + kk)), (step, r, kk)
            # the conv outputs of the accepted rows equal a conv over history
            cvr = outs.chain_view_rows[i, :k].long()
            got_post = world.post_conv[i * M + cvr].float()
            want_post = world.ref_post_conv(hist, len(hist) - k).float()
            torch.testing.assert_close(got_post, want_post, atol=2e-3, rtol=2e-3)
        world.nacc = nacc.clone()

        # ---- P7 chain view on the runner's real step tensors
        cad = CommonAttentionMetadata(
            query_start_loc=runner.query_start_loc.gpu[: B + 1],
            query_start_loc_cpu=runner.query_start_loc.cpu[: B + 1],
            seq_lens=runner.seq_lens[:B],
            num_reqs=B,
            num_actual_tokens=n,
            max_query_len=M,
            max_seq_len=int(runner.seq_lens[:B].max()),
            block_table_tensor=runner.input_batch.block_table[0].get_device_tensor(B),
            slot_mapping=slots,
        )
        hidden = torch.arange(n, dtype=torch.float32)[:, None]
        cv = tree_chain_view(
            SPEC, cad, outs.chain_view_rows[:B], nacc, toks, pos, hidden
        )
        for i, r in enumerate(ids):
            k = int(nacc[i])
            view_tok = cv.target_token_ids.view(B, D + 1)[i, :k].tolist()
            assert view_tok == seq[rid_b[r]][-k - 1 : -1]
            assert cv.target_positions.view(B, D + 1)[i].tolist() == [
                base[i] + j for j in range(D + 1)
            ]
            assert int(cv.token_indices_to_sample[i]) == i * (D + 1) + k - 1
        assert torch.equal(
            cv.common_attn_metadata.slot_mapping,
            slots.view(B, M)[:, : D + 1].reshape(-1),
        )
        assert cv.common_attn_metadata.seq_lens.tolist() == [bb + D + 1 for bb in base]

        # ---- next drafts (M - 1 per request, chain-first)
        nxt = propose([seq[rid_b[r]] for r in ids])
        sched.update_draft_token_ids(DraftTokenIds(ids, nxt.tolist()))

    assert any(d > 0 for d in rescues), rescues  # at least one leaf rescue
    assert any(d == 0 for d in rescues), rescues


# ------------------------------------------------------------ runtime wiring
def test_init_tree_runtime_builds_stash_and_commit(monkeypatch):
    """_init_tree_runtime: stash once (shape per contract §6/§11), the same
    object on every GDN layer, commit tables over the layer's builder buffer
    and the single TQ group's target layers (drafter layers excluded)."""
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as Q
    from vllm.v1.attention.backends import gdn_attn as G
    from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionBackend
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first",
        lambda: False,
    )
    names = [f"model.layers.{i}.linear_attn" for i in (0, 1, 2, 10)]
    layers = {}
    for nm in names:
        lay = Q.QwenGatedDeltaNetAttention.__new__(Q.QwenGatedDeltaNetAttention)
        nn.Module.__init__(lay)
        lay.prefix = nm
        lay.tp_size, lay.num_v_heads, lay.head_k_dim, lay.head_v_dim = 1, HV, KD, VD
        lay.conv1d = SimpleNamespace(weight=torch.zeros(C, 1, KW).half())
        lay.kv_cache = (
            torch.zeros(NSLOTS, CONV_LEN, C).half(),
            torch.zeros(NSLOTS, HV, VD, KD),
        )
        layers[nm] = lay
    tq_layers = {
        "model.layers.3.self_attn.attn": SimpleNamespace(kv_cache=P45._tq_cache(4, 2, 16, 8, None)),
        "mtp.layers.0.self_attn.attn": SimpleNamespace(kv_cache=P45._tq_cache(4, 2, 16, 8, None)),
    }
    bld = G.GDNAttentionMetadataBuilder.__new__(G.GDNAttentionMetadataBuilder)
    bld.spec_state_indices_tensor = torch.zeros(64, W, dtype=torch.int32)
    tq_bld = object()
    groups = [
        SimpleNamespace(
            layer_names=names[::-1], backend=None, kv_cache_group_id=1,
            get_metadata_builder=lambda: bld,
        ),
        SimpleNamespace(
            layer_names=list(tq_layers), backend=TurboQuantAttentionBackend,
            kv_cache_group_id=0, get_metadata_builder=lambda: tq_bld,
        ),
    ]
    ctx = {**layers, **tq_layers}
    fake = SimpleNamespace(
        tree_spec=SPEC,
        tree_tensors=SPEC.device_tensors("cpu"),
        tree_sampler=None,
        tree_stash=None,
        device=torch.device("cpu"),
        compilation_config=SimpleNamespace(static_forward_context=ctx),
        drafter=SimpleNamespace(_draft_attn_layer_names={"mtp.layers.0.self_attn.attn"}),
        _attn_group_iterator=lambda: iter(groups),
    )
    fake._tree_gdn_layers = lambda: GPUModelRunner._tree_gdn_layers(fake)
    GPUModelRunner._init_tree_runtime(fake)
    st = fake.tree_stash
    assert list(st.layer_index) == names  # natural layer order (10 after 2)
    assert tuple(st.gdn_leaf.shape) == (4, SPEC.B_tree, L, HV, KD + VD + 2)
    assert tuple(st.conv_leaf_x.shape) == (4, SPEC.B_tree, L, C)
    assert st.conv_leaf_x.dtype == torch.float16 and st.gdn_leaf.is_contiguous()
    assert all(lay.tree_stash is st for lay in layers.values())
    assert Q.QwenGatedDeltaNetAttention.tree_stash is None  # class default untouched
    tc = fake.tree_commit
    assert fake._tree_tq_gid == 0
    assert tc.gdn_tab.shape == (4, 8) and tc.tq_tab.shape == (1, 6)  # MTP layer excluded
    assert int(tc.gdn_tab[0, 7]) == bld.spec_state_indices_tensor.data_ptr()
    assert int(tc.tq_tab[0, 0]) == tq_layers["model.layers.3.self_attn.attn"].kv_cache.data_ptr()
    # SD layout: the commit sees the glue's [slots, C, len] view
    assert int(tc.gdn_tab[0, 5]) == 1 and int(tc.gdn_tab[0, 6]) == C
    # re-initialization (profiling KV cache -> real one) keeps the stash
    GPUModelRunner._init_tree_runtime(fake)
    assert fake.tree_stash is st and fake.tree_commit is not tc
    # two TurboQuant groups for target layers: refuse
    groups[1].layer_names = ["model.layers.3.self_attn.attn"]
    groups.append(
        SimpleNamespace(
            layer_names=["model.layers.7.self_attn.attn"], backend=TurboQuantAttentionBackend,
            kv_cache_group_id=2, get_metadata_builder=lambda: tq_bld,
        )
    )
    ctx["model.layers.7.self_attn.attn"] = SimpleNamespace(kv_cache=P45._tq_cache(4, 2, 16, 8, None))
    with pytest.raises(ValueError, match="exactly one TurboQuant"):
        GPUModelRunner._init_tree_runtime(fake)


def test_engine_tree_spec_is_the_single_source():
    """Builder, drafter and runner size B_tree from resolve_engine_tree_spec
    (clamped), so they agree with the scheduler and the dispatcher."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    for rel in (
        "vllm/v1/attention/backends/gdn_attn.py",
        "vllm/v1/spec_decode/llm_base_proposer.py",
        "vllm/v1/worker/gpu_model_runner.py",
        "vllm/v1/core/sched/scheduler.py",
    ):
        src = (root / rel).read_text()
        assert "resolve_engine_tree_spec(" in src, rel
        assert "resolve_tree_spec(" not in src.replace("resolve_engine_tree_spec(", ""), rel


# ------------------------------------------------------------------ metrics
def test_rescue_metrics_per_depth():
    from vllm.v1.spec_decode.metrics import (
        SpecDecodingLogging,
        SpecDecodingStats,
        tree_rescue_depth,
    )

    drafts = [11, 12, 13, 21, 22, 23]
    assert tree_rescue_depth(drafts, [11, 12, 13, 99], 3) == 0  # full chain
    assert tree_rescue_depth(drafts, [11, 22, 77], 2) == 2  # rescue at depth 2
    assert tree_rescue_depth(drafts, [21, 5], 1) == 1
    assert tree_rescue_depth(drafts, [5], 0) == 0  # nothing accepted
    s = SpecDecodingStats.new(3)
    for depth, acc in ((2, 2), (0, 3), (1, 1)):
        s.observe_draft(num_draft_tokens=6, num_accepted_tokens=acc)
        s.observe_tree(depth)
    assert s.num_tree_drafts == 3 and s.num_rescued_per_depth == [1, 1, 0]
    assert s.num_draft_tokens_per_pos == [3, 3, 3]  # capped at D
    lines = []
    lg = SpecDecodingLogging()
    lg.observe(s)
    lg.log(log_fn=lambda fmt, *a: lines.append(fmt % a))
    assert len(lines) == 2 and lines[1].startswith("SpecDecoding tree: tree steps 3")
    assert "per-depth rescue rate: 0.333, 0.333, 0.000" in lines[1]


def test_metrics_tree_off_unchanged(monkeypatch):
    """Tree off: no tree log line, no tree Prometheus counters."""
    import prometheus_client

    from vllm.v1.spec_decode import metrics as MET

    monkeypatch.delenv("VLLM_SPEC_TREE", raising=False)
    s = MET.SpecDecodingStats.new(3)
    s.observe_draft(num_draft_tokens=3, num_accepted_tokens=2)
    assert s.num_tree_drafts == 0 and s.num_rescued_per_depth == []
    lines = []
    lg = MET.SpecDecodingLogging()
    lg.observe(s)
    lg.log(log_fn=lambda fmt, *a: lines.append(fmt % a))
    assert len(lines) == 1 and "tree" not in lines[0]

    reg = prometheus_client.CollectorRegistry()
    monkeypatch.setattr(
        MET.SpecDecodingProm,
        "_counter_cls",
        staticmethod(lambda **kw: prometheus_client.Counter(registry=reg, **kw)),
    )
    cfg = SimpleNamespace(
        num_speculative_tokens=3, method="mtp", speculative_token_tree=None,
        speculative_tree_max_batch=None,
    )
    prom = MET.SpecDecodingProm(cfg, ["engine"], {0: ["0"]})
    assert not hasattr(prom, "counter_spec_decode_num_tree_drafts")
    prom.observe(s, 0)
    monkeypatch.setenv("VLLM_SPEC_TREE", "1")
    reg2 = prometheus_client.CollectorRegistry()
    monkeypatch.setattr(
        MET.SpecDecodingProm,
        "_counter_cls",
        staticmethod(lambda **kw: prometheus_client.Counter(registry=reg2, **kw)),
    )
    prom2 = MET.SpecDecodingProm(cfg, ["engine"], {0: ["0"]})
    s2 = MET.SpecDecodingStats.new(3)
    s2.observe_draft(num_draft_tokens=6, num_accepted_tokens=2)
    s2.observe_tree(2)
    prom2.observe(s2, 0)
    assert reg2.get_sample_value(
        "vllm:spec_decode_num_rescued_per_depth_total", {"engine": "0", "depth": "2"}
    ) == 1.0
    assert reg2.get_sample_value(
        "vllm:spec_decode_num_tree_drafts_total", {"engine": "0"}
    ) == 1.0


# ---------------------------------------------------------------- tree off
def test_runner_tree_off_sample_path_unchanged(monkeypatch):
    """Tree off: _sample keeps today's call (rejection sampler), no tree
    objects exist, and sample_tokens passes no tree arguments."""
    import inspect

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    fake = SimpleNamespace(
        vllm_config=P28._fake_vllm_config(), device=torch.device("cpu")
    )
    fake.vllm_config.speculative_config.speculative_token_tree = None
    GPUModelRunner._init_tree_state(fake)
    assert fake.tree_spec is None and fake.tree_sampler is None
    assert fake.tree_commit is None and fake.tree_stash is None
    src = inspect.getsource(GPUModelRunner.sample_tokens)
    assert "sampler_output = self._sample(logits, spec_decode_metadata)\n" in src
    src = inspect.getsource(GPUModelRunner.initialize_kv_cache)
    assert "if self.tree_spec is not None:\n            self._init_tree_runtime()" in src


# ------------------------------------------------------------------ warmup
def test_tree_kernels_warm_up_eagerly_at_init(monkeypatch):
    """Coordinator rule (K5 capture invalidation, tree gate 'Ninja is
    required' on the first commit inside capture): every tree kernel gets its
    first launch in _init_tree_runtime, before any capture."""
    import inspect

    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as Q
    from vllm.v1.spec_decode import tree_commit as TC
    from vllm.v1.spec_decode.tree_sampler import TreeSampler
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    src = inspect.getsource(GPUModelRunner._init_tree_runtime)
    for call in ("first.tree_warmup(", "self.tree_commit.warmup()", "self.tree_sampler.warmup()"):
        assert call in src, call
    wsrc = inspect.getsource(Q.QwenGatedDeltaNetAttention.tree_warmup)
    for call in ("_k5_gdn._load()", "causal_conv1d_update_tree(", "_k5_gdn.gdn_mtp_tree(", "PAD_SLOT_ID"):
        assert call in wsrc, call
    # the sampler warmup launches the walk and only touches its own buffers
    from vllm.v1.sample.sampler import Sampler

    ts = TreeSampler(Sampler(), SPEC, "cpu")
    ts.warmup()
    assert ts.outputs.num_accepted[0].item() == 1  # walk ran: no draft matched
    # a prebuilt extension .so is imported without ninja when fresh
    called = []
    monkeypatch.setattr(TC, "_EXT", None)
    monkeypatch.setattr(TC, "_import_prebuilt", lambda n, bd, src: called.append(n) or "MOD")
    assert TC._load() == "MOD" and called == ["tree_commit_sm75"]


def test_import_prebuilt_requires_fresh_so(tmp_path):
    import os

    from vllm.v1.spec_decode import tree_commit as TC

    src = tmp_path / "x.cu"
    src.write_text("//")
    assert TC._import_prebuilt("nope", str(tmp_path), str(src)) is None  # no .so
    so = tmp_path / "nope.so"
    so.write_bytes(b"not an elf")
    os.utime(so, (1, 1))  # older than the source: stale
    assert TC._import_prebuilt("nope", str(tmp_path), str(src)) is None
    os.utime(so, None)
    assert TC._import_prebuilt("nope", str(tmp_path), str(src)) is None  # bad ELF
