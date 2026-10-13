# SPDX-License-Identifier: Apache-2.0
"""E16 tree drafting, packages P2 (runner inputs) + P8 (scheduler decision,
dispatcher, graph keys).  CPU only.

Contract: /home/kevin/projects/lanes/coexec/results/tree-contract.md
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.spec_decode import tree_reference as ref
from vllm.v1.spec_decode.tree_spec import (
    TreeSpec,
    clamp_tree_spec,
    resolve_engine_tree_spec,
    tree_cudagraph_sizes,
)

from tests.v1.core.utils import create_requests, create_scheduler

D = 3
SPEC = TreeSpec.chain_plus_leaves(D, B_tree=4)  # M = 7
M = SPEC.M


# ----------------------------------------------------------------- helpers
def _chain(i: int) -> list[int]:
    return [1000 + 10 * i + d for d in range(1, D + 1)]


def _tree_drafts(i: int) -> list[int]:
    # chain-first: c1..cD then leaves (one -1 leaf to prove it is kept)
    return _chain(i) + [2000 + 10 * i + 1, -1, 2000 + 10 * i + 3]


def _sched(tree: bool, max_num_seqs: int = 16):
    s = create_scheduler(num_speculative_tokens=D, max_num_seqs=max_num_seqs)
    assert s.tree_spec is None  # ngram config, env unset: tree off by default
    if tree:
        s.tree_spec = SPEC
    return s


def _to_decode(sched, n: int, drafts) -> list:
    """Admit n requests, run their prefill, hand them drafts."""
    reqs = create_requests(num_requests=n, num_tokens=10, max_tokens=64)
    for r in reqs:
        sched.add_request(r)
    out = sched.schedule()
    assert len(out.scheduled_new_reqs) == n
    assert not out.tree_step
    _finish(sched, out, [[7] for _ in reqs])
    sched.update_draft_token_ids(
        DraftTokenIds([r.request_id for r in reqs], [drafts(i) for i in range(n)])
    )
    return reqs


def _finish(sched, out: SchedulerOutput, sampled: list[list[int]]):
    ids = list(out.num_scheduled_tokens)
    mro = ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={r: i for i, r in enumerate(ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    return sched.update_from_output(out, mro)


def _summary(out: SchedulerOutput):
    return (
        dict(out.num_scheduled_tokens),
        {k: list(v) for k, v in out.scheduled_spec_decode_tokens.items()},
        out.total_num_scheduled_tokens,
        out.tree_step,
    )


# ----------------------------------------------------------- clamp / gating
def test_clamp_tree_spec_bounds():
    # production: max_num_seqs 16, capture max 64 -> B_tree 4 unchanged
    assert clamp_tree_spec(SPEC, 16, 64) is SPEC
    # 16*4//7 = 9 ; 64//7 = 9 -> a larger B_tree is cut to 9
    big = TreeSpec.chain_plus_leaves(D, B_tree=12)
    assert clamp_tree_spec(big, 16, 64).B_tree == 9
    # buffer clamp: 2*4//7 = 1
    assert clamp_tree_spec(SPEC, 2, 64).B_tree == 1
    # graph clamp: 14//7 = 2
    assert clamp_tree_spec(SPEC, 16, 14).B_tree == 2
    # nothing fits -> tree off
    assert clamp_tree_spec(SPEC, 1, 64) is None
    assert clamp_tree_spec(SPEC, 16, 6) is None
    # no capture size known (cudagraphs off) -> only the buffer clamp
    assert clamp_tree_spec(SPEC, 16, None).B_tree == 4
    assert clamp_tree_spec(None, 16, 64) is None


def _fake_vllm_config(async_scheduling=False, max_num_seqs=16, max_cap=64):
    spec_cfg = SimpleNamespace(
        num_speculative_tokens=D,
        method="mtp",
        speculative_token_tree="1",
        speculative_tree_max_batch=None,
        parallel_drafting=False,
        num_speculative_tokens_per_batch_size=None,
        use_local_argmax_reduction=False,
    )
    return SimpleNamespace(
        speculative_config=spec_cfg,
        scheduler_config=SimpleNamespace(
            async_scheduling=async_scheduling, max_num_seqs=max_num_seqs
        ),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=max_cap),
    )


def test_resolve_engine_tree_spec():
    spec = resolve_engine_tree_spec(_fake_vllm_config(), environ={})
    assert spec == SPEC and spec.B_tree == 4
    assert resolve_engine_tree_spec(
        _fake_vllm_config(), environ={"VLLM_SPEC_TREE": "0"}
    ) is None
    assert (
        resolve_engine_tree_spec(_fake_vllm_config(max_num_seqs=2), environ={}).B_tree
        == 1
    )
    with pytest.raises(ValueError, match="no-async-scheduling"):
        resolve_engine_tree_spec(_fake_vllm_config(async_scheduling=True), environ={})
    assert tree_cudagraph_sizes(SPEC) == [7, 14, 21, 28]
    assert tree_cudagraph_sizes(None) == []


# --------------------------------------------------------- scheduler (P8)
def test_scheduler_tree_step_two_requests():
    s = _sched(tree=True)
    reqs = _to_decode(s, 2, _tree_drafts)
    out = s.schedule()
    assert out.tree_step is True
    for i, r in enumerate(reqs):
        assert out.num_scheduled_tokens[r.request_id] == M
        assert out.scheduled_spec_decode_tokens[r.request_id] == _tree_drafts(i)
    assert out.total_num_scheduled_tokens == 2 * M


def test_scheduler_tree_rollback_arithmetic():
    """num_computed += 1 + accepted after a tree step (contract section 3)."""
    s = _sched(tree=True)
    reqs = _to_decode(s, 2, _tree_drafts)
    base = [r.num_computed_tokens for r in reqs]
    out = s.schedule()
    assert out.tree_step
    # req0: path of 2 chain tokens + bonus (3 tokens); req1: rescue at
    # depth 1 -> leaf + bonus (2 tokens).  Sampler output is path-ordered.
    _finish(s, out, [[_chain(0)[0], _chain(0)[1], 55], [2011, 66]])
    assert reqs[0].num_computed_tokens == base[0] + 3
    assert reqs[1].num_computed_tokens == base[1] + 2
    assert list(reqs[0].output_token_ids)[-3:] == [_chain(0)[0], _chain(0)[1], 55]
    assert list(reqs[1].output_token_ids)[-2:] == [2011, 66]


def test_scheduler_too_many_requests_is_chain_step():
    """B > B_tree: drafts cut to the chain before scheduling; the step equals
    the tree-off step for the same (chain) drafts, bit for bit."""
    s_tree = _sched(tree=True)
    _to_decode(s_tree, 5, _tree_drafts)
    s_off = _sched(tree=False)
    _to_decode(s_off, 5, _chain)
    out_tree, out_off = s_tree.schedule(), s_off.schedule()
    assert not out_tree.tree_step
    assert _summary(out_tree) == _summary(out_off)
    assert all(n == D + 1 for n in out_tree.num_scheduled_tokens.values())


def test_scheduler_demotes_when_new_request_arrives():
    """Candidate pre-pass, then a new request joins: demote to the chain."""
    s = _sched(tree=True)
    reqs = _to_decode(s, 2, _tree_drafts)
    (new,) = create_requests(num_requests=1, num_tokens=10, req_ids=["new"])
    s.add_request(new)
    base = [r.num_computed_tokens for r in reqs]
    out = s.schedule()
    assert not out.tree_step
    assert [r.req_id for r in out.scheduled_new_reqs] == ["new"]
    for i, r in enumerate(reqs):
        assert out.num_scheduled_tokens[r.request_id] == D + 1
        assert out.scheduled_spec_decode_tokens[r.request_id] == _chain(i)
    assert out.total_num_scheduled_tokens == sum(out.num_scheduled_tokens.values())
    assert out.total_num_scheduled_tokens == 2 * (D + 1) + 10
    # bookkeeping after a demoted step is the chain's
    _finish(s, out, [[_chain(0)[0], 9], [8], [5]])
    assert reqs[0].num_computed_tokens == base[0] + 2
    assert reqs[1].num_computed_tokens == base[1] + 1


def test_scheduler_short_drafts_force_chain():
    """A request with only the chain (e.g. grammar truncated) -> chain step,
    and the other request's leaves are dropped."""
    s = _sched(tree=True)
    reqs = _to_decode(
        s, 2, lambda i: _tree_drafts(i) if i == 0 else _chain(i)
    )
    out = s.schedule()
    assert not out.tree_step
    assert out.scheduled_spec_decode_tokens[reqs[0].request_id] == _chain(0)
    assert out.scheduled_spec_decode_tokens[reqs[1].request_id] == _chain(1)
    assert all(n == D + 1 for n in out.num_scheduled_tokens.values())


def test_scheduler_bad_words_request_stores_chain_only():
    s = _sched(tree=True)
    reqs = create_requests(num_requests=1, num_tokens=10)
    reqs[0].sampling_params._bad_words_token_ids = [[123]]
    s.add_request(reqs[0])
    out = s.schedule()
    _finish(s, out, [[7]])
    s.update_draft_token_ids(DraftTokenIds([reqs[0].request_id], [_tree_drafts(0)]))
    assert reqs[0].spec_token_ids == _chain(0)
    assert not s.schedule().tree_step


def test_scheduler_tree_off_byte_identical():
    """Tree off: the step for M-1-wide drafts is unchanged code (no pre-pass),
    and chain drafts schedule as before."""
    s = _sched(tree=False)
    reqs = _to_decode(s, 2, _chain)
    out = s.schedule()
    assert not out.tree_step
    assert all(n == D + 1 for n in out.num_scheduled_tokens.values())
    assert out.scheduled_spec_decode_tokens[reqs[1].request_id] == _chain(1)


def test_scheduler_prefill_running_blocks_tree():
    s = _sched(tree=True)
    reqs = _to_decode(s, 1, _tree_drafts)
    # a second, long request mid-prefill in the running queue
    (long_req,) = create_requests(num_requests=1, num_tokens=40, req_ids=["long"])
    s.add_request(long_req)
    s.max_num_scheduled_tokens = 30  # forces a chunk
    out = s.schedule()
    assert not out.tree_step
    _finish(s, out, [[_chain(0)[0], 9], [3]])
    assert long_req.status == RequestStatus.RUNNING
    assert long_req.num_computed_tokens < long_req.num_prompt_tokens
    s.update_draft_token_ids(DraftTokenIds([reqs[0].request_id], [_tree_drafts(0)]))
    out = s.schedule()
    assert not out.tree_step  # running prefill -> not a candidate
    assert out.scheduled_spec_decode_tokens[reqs[0].request_id] == _chain(0)


# ------------------------------------------------------- dispatcher (P8)
def _dispatch_config(mode: str, comp_mode, max_num_seqs=16):
    from tests.v1.cudagraph.test_cudagraph_dispatch import _create_vllm_config

    comp = CompilationConfig(
        cudagraph_mode=mode,
        mode=comp_mode,
        cudagraph_capture_sizes=[4 * i for i in range(1, 17)],
    )
    cfg = _create_vllm_config(comp, max_num_seqs=max_num_seqs)
    cfg.num_speculative_tokens = D
    # runner path: chain sizes rounded to multiples of D + 1 (no-op here)
    comp.adjust_cudagraph_sizes_for_spec_decode(D + 1, 1)
    return cfg, comp


@pytest.mark.parametrize(
    "mode,comp_mode",
    [
        ("FULL_DECODE_ONLY", CompilationMode.NONE),
        ("FULL_AND_PIECEWISE", CompilationMode.VLLM_COMPILE),
    ],
)
def test_dispatcher_tree_family(mode, comp_mode):
    cfg, comp = _dispatch_config(mode, comp_mode)
    off = CudagraphDispatcher(cfg)
    off.initialize_cudagraph_keys(comp.cudagraph_mode, D + 1)
    on = CudagraphDispatcher(cfg)
    on.initialize_cudagraph_keys(comp.cudagraph_mode, D + 1, tree_spec=SPEC)

    chain_full = {d for d in on.cudagraph_keys[CUDAGraphMode.FULL] if not d.tree}
    tree_full = {d for d in on.cudagraph_keys[CUDAGraphMode.FULL] if d.tree}
    # tree off: the chain keys are exactly today's keys
    assert chain_full == off.cudagraph_keys[CUDAGraphMode.FULL]
    assert (
        on.cudagraph_keys[CUDAGraphMode.PIECEWISE]
        == off.cudagraph_keys[CUDAGraphMode.PIECEWISE]
    )
    assert sorted(d.num_tokens for d in chain_full) == [4 * i for i in range(1, 17)]
    # tree family: un-rounded B*M, B <= B_tree
    assert sorted((d.num_tokens, d.num_reqs) for d in tree_full) == [
        (7, 1),
        (14, 2),
        (21, 3),
        (28, 4),
    ]
    assert all(d.uniform for d in tree_full)
    assert comp.cudagraph_capture_sizes == [4 * i for i in range(1, 17)]

    # 28 tokens: chain = 7 requests, tree = 4 requests
    m, d = on.dispatch(28, uniform_decode=True)
    assert m == CUDAGraphMode.FULL
    assert d == BatchDescriptor(28, num_reqs=7, uniform=True)
    m, d = on.dispatch(28, uniform_decode=True, tree_step=True)
    assert m == CUDAGraphMode.FULL
    assert d == BatchDescriptor(28, num_reqs=4, uniform=True, tree=True)
    for b in (1, 2, 3):
        m, d = on.dispatch(b * M, uniform_decode=True, tree_step=True)
        assert (m, d.num_tokens, d.num_reqs, d.tree) == (
            CUDAGraphMode.FULL,
            b * M,
            b,
            True,
        )
    # chain dispatch is unchanged by the tree family (7 tokens pads to 8)
    for n in (1, 4, 7, 12, 28, 64):
        for uni in (False, True):
            assert on.dispatch(n, uniform_decode=uni) == off.dispatch(
                n, uniform_decode=uni
            )
    # eager when FULL is excluded
    m, d = on.dispatch(14, uniform_decode=True, tree_step=True,
                       invalid_modes={CUDAGraphMode.FULL})
    assert m == CUDAGraphMode.NONE and d.tree
    # capture list carries the tree descs
    full = dict(on.get_capture_descs())[CUDAGraphMode.FULL]
    assert {d for d in full if d.tree} == tree_full


def test_dispatcher_tree_sizes_capped_by_max_capture():
    cfg, comp = _dispatch_config("FULL_DECODE_ONLY", CompilationMode.NONE)
    comp.max_cudagraph_capture_size = 16
    comp.cudagraph_capture_sizes = [4, 8, 12, 16]
    on = CudagraphDispatcher(cfg)
    on.initialize_cudagraph_keys(comp.cudagraph_mode, D + 1, tree_spec=SPEC)
    assert on.tree_capture_sizes == [7, 14]
    assert on.dispatch(14, uniform_decode=True, tree_step=True)[0] == CUDAGraphMode.FULL
    # 21 > max size -> eager, never a chain graph
    m, d = on.dispatch(21, uniform_decode=True, tree_step=True)
    assert m == CUDAGraphMode.NONE and d.tree


def test_drafter_style_dispatcher_has_no_tree_keys():
    cfg, comp = _dispatch_config("FULL_DECODE_ONLY", CompilationMode.NONE)
    disp = CudagraphDispatcher(cfg)
    disp.initialize_cudagraph_keys(comp.cudagraph_mode, D + 1)
    assert not any(d.tree for d in disp.cudagraph_keys[CUDAGraphMode.FULL])
    with pytest.raises(AssertionError):
        disp.dispatch(14, uniform_decode=True, tree_step=True)


# ------------------------------------------- CommonAttentionMetadata flag
def _cad(query_lens, tree_step):
    qsl = torch.tensor(np.concatenate([[0], np.cumsum(query_lens)]), dtype=torch.int32)
    n = int(qsl[-1])
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor(query_lens, dtype=torch.int32) + 100,
        num_reqs=len(query_lens),
        num_actual_tokens=n,
        max_query_len=max(query_lens),
        max_seq_len=200,
        block_table_tensor=torch.zeros(len(query_lens), 4, dtype=torch.int32),
        slot_mapping=torch.zeros(n, dtype=torch.int64),
        tree_step=tree_step,
    )


def test_split_decodes_and_prefills_tree_bypass():
    # threshold stays 1 + D = 4; M = 7 rows would be "prefills" without the flag
    assert split_decodes_and_prefills(_cad([7, 7], False), D + 1) == (0, 2, 0, 14)
    assert split_decodes_and_prefills(_cad([7, 7], True), D + 1) == (2, 0, 14, 0)
    # chain steps unchanged
    assert split_decodes_and_prefills(_cad([4, 4, 9], False), D + 1) == (2, 1, 8, 9)
    assert CommonAttentionMetadata.__dataclass_fields__["tree_step"].default is False


# ------------------------------------------------------ runner inputs (P2)
def _runner_with_tree(monkeypatch):
    """A real GPUModelRunner on the CPU (opt-125m, one attention layer) with
    the tree state built by its own _init_tree_state()."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from tests.v1.worker.test_gpu_model_runner import (
        get_vllm_config,
        initialize_kv_cache,
    )

    cfg = get_vllm_config()
    cfg.scheduler_config.async_scheduling = False  # tree needs sync scheduling
    with set_current_vllm_config(cfg):
        mc = cfg.model_config
        cfg.compilation_config.static_forward_context["layer.0"] = Attention(
            mc.get_num_kv_heads(cfg.parallel_config), mc.get_head_size(), 0.1
        )
        monkeypatch.setattr(
            GPUModelRunner,
            "_init_device_properties",
            lambda self: setattr(self, "num_sms", 1),
        )
        runner = GPUModelRunner(cfg, torch.device("cpu"))
        initialize_kv_cache(runner)
    assert runner.tree_spec is None  # off by default
    monkeypatch.setattr(
        "vllm.v1.spec_decode.tree_spec.resolve_engine_tree_spec",
        lambda vllm_config, environ=None: SPEC,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True, is_first_rank=True),
    )
    runner._init_tree_state()
    assert runner.tree_spec is SPEC
    return runner


def _new_req_output(prompts: dict[str, list[int]], blocks: dict[str, list[int]]):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData

    new = [
        NewRequestData(
            req_id=rid,
            prompt_token_ids=p,
            mm_features=[],
            sampling_params=SamplingParams(),
            pooling_params=None,
            block_ids=(blocks[rid],),
            num_computed_tokens=0,
            lora_request=None,
        )
        for rid, p in prompts.items()
    ]
    nst = {rid: len(p) - 1 for rid, p in prompts.items()}
    return SchedulerOutput(
        scheduled_new_reqs=new,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=nst,
        total_num_scheduled_tokens=sum(nst.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _verify_output(prompts, drafts, tree_step):
    from vllm.v1.core.sched.output import CachedRequestData

    ids = list(prompts)
    nst = {rid: 1 + len(drafts[rid]) for rid in ids}
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=ids,
            resumed_req_ids=set(),
            new_token_ids=[[] for _ in ids],
            all_token_ids={},
            new_block_ids=[None] * len(ids),
            num_computed_tokens=[len(prompts[r]) - 1 for r in ids],
            num_output_tokens=[0 for _ in ids],
        ),
        num_scheduled_tokens=nst,
        total_num_scheduled_tokens=sum(nst.values()),
        scheduled_spec_decode_tokens=dict(drafts),
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        tree_step=tree_step,
    )


@pytest.mark.parametrize("tree_step", [True, False])
def test_runner_prepare_inputs_two_request_tree_step(monkeypatch, tree_step):
    """2-request step: input ids and slot_mapping row-indexed, positions at
    depth (tree) or row (chain), logits over all rows, -1 drafts kept for the
    sampler and clamped only at the embedding."""
    runner = _runner_with_tree(monkeypatch)
    BS = 16
    prompts = {"a": list(range(11, 16)), "b": list(range(21, 41))}  # 5, 20
    blocks = {"a": [3], "b": [5, 6]}
    runner._update_states(_new_req_output(prompts, blocks))

    if tree_step:
        drafts = {"a": [101, 102, 103, 111, -1, 113], "b": [201, 202, 203, 211, 212, -1]}
    else:
        drafts = {"a": [101, 102, 103], "b": [201, 202, 203]}
    out = _verify_output(prompts, drafts, tree_step)
    runner._update_states(out)
    ids = runner.input_batch.req_ids
    nst = np.array([out.num_scheduled_tokens[r] for r in ids], dtype=np.int32)
    logits_indices, sdm, max_sampled = runner._prepare_inputs(out, nst)

    rows = M if tree_step else D + 1
    N = 2 * rows
    base = {r: len(prompts[r]) - 1 for r in ids}
    exp_ids, exp_pos, exp_slot = [], [], []
    for r in ids:
        toks = [prompts[r][-1]] + drafts[r]
        exp_ids += toks
        b = base[r]
        if tree_step:
            depth = ref.tree_positions(SPEC, torch.tensor([b]))[0].tolist()
        else:
            depth = [b + k for k in range(rows)]
        exp_pos += depth
        for k in range(rows):  # row-indexed slot: sequence index b + k
            seq = b + k
            exp_slot.append(blocks[r][seq // BS] * BS + seq % BS)

    assert runner.input_ids.cpu[:N].tolist() == exp_ids  # -1 kept here
    assert runner.input_ids.gpu[:N].tolist() == exp_ids
    assert runner.positions[:N].tolist() == exp_pos
    slot = runner.input_batch.block_table[0].slot_mapping.gpu[:N].tolist()
    assert slot == exp_slot
    assert logits_indices.tolist() == list(range(N))
    assert sdm.num_draft_tokens == [rows - 1] * 2
    assert max_sampled == rows
    assert runner.seq_lens[:2].tolist() == [base[r] + rows for r in ids]
    if tree_step:
        exp_drafts = [drafts[r] for r in ids]
        assert runner.tree_draft_token_ids[:2].tolist() == exp_drafts
        # positions == contract formula num_computed + rows_depth[r]
        assert exp_pos[:M] == [base[ids[0]] + d for d in SPEC.rows_depth]
    else:
        # chain step: the tree runner's inputs equal plain arange positions
        assert runner.tree_draft_token_ids.eq(-1).all()


def test_runner_tree_step_rejects_bad_batch(monkeypatch):
    runner = _runner_with_tree(monkeypatch)
    with pytest.raises(AssertionError):
        runner._check_tree_step(2, np.array([7, 4], dtype=np.int32))
    with pytest.raises(AssertionError):
        runner._check_tree_step(5, np.array([7] * 5, dtype=np.int32))
    runner._check_tree_step(4, np.array([7] * 4, dtype=np.int32))


def test_runner_tree_off_has_no_tree_state():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    fake = SimpleNamespace(vllm_config=_fake_vllm_config(), device=torch.device("cpu"))
    fake.vllm_config.speculative_config.speculative_token_tree = None
    GPUModelRunner._init_tree_state(fake)
    assert fake.tree_spec is None and fake.tree_draft_token_ids is None
    assert fake._tree_depth_delta is None


def test_runner_mrope_depth_offset():
    """M-RoPE: every section gets the same depth offset (decode text)."""
    delta = np.asarray(SPEC.rows_depth) - np.arange(M)
    mrope = np.stack([np.arange(50, 50 + 2 * M)] * 3)  # [3, B*M], decode text
    qp = np.tile(np.arange(M), 2)
    mrope = mrope + delta[qp]
    exp = ref.tree_positions(SPEC, torch.tensor([50, 50 + M])).reshape(-1)
    for sec in range(3):
        assert mrope[sec].tolist() == exp.tolist()
