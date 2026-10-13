# SPDX-License-Identifier: Apache-2.0
"""[FORK][LANE CR] Copy-mode GDN prompt-tail reuse at production shapes.

Drives the REAL Scheduler (priority policy, MTP-style spec decode with three
speculative Mamba blocks, mamba "align", sparse retention 3x block, block
3568, prefix_match_unit 16, one attention + three GDN groups) through the
window probe's traffic shape (growing sessions, concurrent, fresh tail each
round, sibling bursts) and replays every SchedulerOutput on a worker model
that follows the GPU runner's contract step for step:

  * zero ``new_block_ids_to_zero``, then run ``kv_cache_block_copies``;
  * the worker block table is append-only (new reqs get the full table, cached
    reqs append ``new_block_ids``);
  * ``preprocess_mamba`` moves the running state to the last
    (1 + num_speculative_blocks) column when the step crosses a block;
  * GDN reads/writes the state slot gathered at ``(seq_len - 1) // B`` (non-spec
    rows: column 0; spec rows: columns 0..num_spec, initial state at
    ``num_accepted - 1``);
  * the fused postprocess copies the block-boundary state after a spec step.

Every forward asserts that the Mamba state it starts from is exactly the state
at the request's num_computed_tokens and that every attention KV slot below
num_computed_tokens holds the request's own token. A copy-mode defect that
publishes or restores a wrong state therefore fails the first forward that
consumes it, with the request, step and slot named.
"""

import os
import random
from collections import defaultdict

import numpy as np
import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ObservabilityConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

MODEL = os.environ.get("VLLM_SCHED_TEST_MODEL", "facebook/opt-125m")
B = 3568  # production attention == mamba block (16 x 223)
NUM_SPEC = 3  # MTP3
NUM_MAMBA_GROUPS = 3
EOS = 50256
ATTN = 0
Z = "zero"


def _scheduler(unit: int, num_blocks: int, max_model_len: int = 200_000) -> Scheduler:
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    model_config = ModelConfig(
        model=MODEL,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=not os.path.isdir(MODEL),
        max_model_len=max_model_len,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=3632,
        max_model_len=max_model_len,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=0,
        watermark=0.0,
        is_encoder_decoder=False,
        policy="priority",
    )
    cache_config = CacheConfig(
        block_size=B,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        observability_config=ObservabilityConfig(),
    )
    groups = [
        KVCacheGroupSpec(
            ["attn"],
            FullAttentionSpec(
                block_size=B, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        )
    ]
    for g in range(NUM_MAMBA_GROUPS):
        groups.append(
            KVCacheGroupSpec(
                [f"gdn{g}"],
                MambaSpec(
                    block_size=B,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                    num_speculative_blocks=NUM_SPEC,
                    supports_fine_grained_prefix_cache=True,
                ),
            )
        )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=3 * B,
    )
    cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    sched = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=B,
        hash_block_size=unit,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    # MTP (single module): one prefill lookahead, three drafts per step, the
    # eagle trailing-block drop disabled exactly like production.
    sched.use_eagle = True
    sched.use_eagle_block_drop = False
    sched.num_prefill_lookahead = 1
    sched.num_spec_tokens = NUM_SPEC
    sched.num_lookahead_tokens = NUM_SPEC
    return sched


_INIT = [False]


def _req(rid: str, toks: list[int], unit: int, max_tokens: int, prio: int = 0):
    if not _INIT[0]:
        init_none_hash(sha256)
        _INIT[0] = True
    sp = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    sp.update_from_generation_config({}, EOS)
    return Request(
        request_id=rid,
        prompt_token_ids=toks,
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(unit, sha256),
        priority=prio,
    )


class Worker:
    """GPU runner model: state slots carry the token position they hold."""

    def __init__(self, sched: Scheduler, num_blocks: int, seed: int = 0):
        self.s = sched
        self.rnd = random.Random(seed)
        self.mamba: list[dict[int, object]] = [
            {} for _ in range(NUM_MAMBA_GROUPS)
        ]  # per group: block id -> position (or Z / None)
        self.kv = np.full((num_blocks, B), -1, dtype=np.int64)  # attention KV
        self.bt: dict[str, tuple[list[int], ...]] = {}
        self.state_idx: dict[str, int] = {}  # preprocess_mamba's mamba_state_idx
        self.num_accepted: dict[str, int] = defaultdict(lambda: 1)
        self.drafts: dict[str, list[int]] = {}
        self.step_no = 0
        self.hits: list[tuple[str, int]] = []
        self.who: dict[tuple[int, int], str] = {}  # (group, block) -> last writer

    def _w(self, g: int, b: int, val, why: str) -> None:
        self.mamba[g][b] = val
        self.who[(g, b)] = f"{why} @step{self.step_no}"

    # -- scheduler-output application ------------------------------------
    def _apply_tables(self, so) -> None:
        for nr in so.scheduled_new_reqs:
            self.bt[nr.req_id] = tuple(list(ids) for ids in nr.block_ids)
            self.state_idx.pop(nr.req_id, None)
            self.num_accepted[nr.req_id] = 1
            if nr.num_computed_tokens:
                self.hits.append((nr.req_id, nr.num_computed_tokens))
        cr = so.scheduled_cached_reqs
        for i, rid in enumerate(cr.req_ids):
            new = cr.new_block_ids[i]
            if rid in cr.resumed_req_ids:
                self.bt[rid] = tuple(list(ids) for ids in new)
                self.state_idx.pop(rid, None)
                self.num_accepted[rid] = 1
            elif new is not None:
                for g, ids in enumerate(new):
                    self.bt[rid][g].extend(ids)
        for rid in so.finished_req_ids:
            self.bt.pop(rid, None)
            self.state_idx.pop(rid, None)
        for rid in so.preempted_req_ids or ():
            self.state_idx.pop(rid, None)

    def _zero_and_copy(self, so) -> None:
        z = so.new_block_ids_to_zero
        if z:
            ids = (
                [b for v in z.values() for b in v] if isinstance(z, dict) else z
            )
            for b in ids:
                self.kv[b] = -1
                for g in range(NUM_MAMBA_GROUPS):
                    self._w(g, b, Z, "zero")
        copies = so.kv_cache_block_copies or []
        if copies:
            kv_snap = {c.src_block_id: self.kv[c.src_block_id].copy() for c in copies}
            m_snap = [
                {c.src_block_id: self.mamba[g].get(c.src_block_id) for c in copies}
                for g in range(NUM_MAMBA_GROUPS)
            ]
            for c in copies:
                assert c.group_id is None
                self.kv[c.dst_block_id] = kv_snap[c.src_block_id]
                for g in range(NUM_MAMBA_GROUPS):
                    self._w(g, c.dst_block_id, m_snap[g][c.src_block_id], f"copy<-{c.src_block_id}")

    # -- one engine step --------------------------------------------------
    def step(self) -> bool:
        so = self.s.schedule()
        self.step_no += 1
        self._check_refs("schedule")
        self._apply_tables(so)
        self._zero_and_copy(so)
        if not so.num_scheduled_tokens:
            return False
        sampled: dict[str, list[int]] = {}
        rows = []
        new_ids = {nr.req_id: nr for nr in so.scheduled_new_reqs}
        cr = so.scheduled_cached_reqs
        cached_nc = dict(zip(cr.req_ids, cr.num_computed_tokens))
        # preprocess_mamba (scalar path) over the batch
        for rid, n in so.num_scheduled_tokens.items():
            nc = new_ids[rid].num_computed_tokens if rid in new_ids else cached_nc[rid]
            prev = self.state_idx.get(rid)
            if prev is None:
                prev = (nc - 1) // B
            curr = cdiv(nc + n, B) - 1
            self.state_idx[rid] = curr
            if prev != -1 and prev != curr:
                bias = self.num_accepted[rid] - 1
                for g in range(NUM_MAMBA_GROUPS):
                    t = self.bt[rid][1 + g]
                    self._w(g, t[curr], self.mamba[g].get(t[prev + bias]), f"pre {rid} col{prev}+{bias}->{curr}")
                self.num_accepted[rid] = 1
            rows.append((rid, nc, n))
        # forward
        for rid, nc, n in rows:
            req = self.s.requests[rid]
            drafts = so.scheduled_spec_decode_tokens.get(rid, [])
            toks = list(req.all_token_ids[nc : nc + n - len(drafts)]) + list(drafts)
            assert len(toks) == n, (rid, nc, n, len(drafts))
            self._check_attention(rid, req, nc)
            start = (nc + n - 1) // B
            for g in range(NUM_MAMBA_GROUPS):
                t = self.bt[rid][1 + g]
                if drafts:
                    slot = t[start + self.num_accepted[rid] - 1]
                else:
                    slot = t[start]
                got = self.mamba[g].get(slot)
                want = nc if nc > 0 else got
                assert got == want, (
                    f"step {self.step_no}: {rid} group {g} starts at {nc} from "
                    f"block {slot} (col {start}) holding state@{got}; "
                    f"table {t[: start + NUM_SPEC + 1]}; last writer "
                    f"{self.who.get((g, slot))}; {self._holders(g, slot)}; mgr table "
                    f"{[x.block_id for x in self.s.kv_cache_manager.coordinator.single_type_managers[1 + g].req_to_blocks[rid]]}"
                    f" worker {t}; hits {[h for h in self.hits if h[0] == rid]}"
                )
                if drafts:
                    for i in range(n):
                        self._w(g, t[start + i], nc + i + 1, f"spec {rid} col{start + i}")
                else:
                    self._w(g, t[start], nc + n, f"fwd {rid} col{start}")
            # attention KV write
            a = self.bt[rid][ATTN]
            for i, tok in enumerate(toks):
                p = nc + i
                self.kv[a[p // B], p % B] = tok
            # sampling
            if drafts:
                acc = self.rnd.randrange(0, len(drafts) + 1)
                out = list(drafts[:acc]) + [self.rnd.randrange(1, 40000)]
                self.num_accepted[rid] = acc + 1
            elif nc + n >= req.num_tokens:
                out = [self.rnd.randrange(1, 40000)]
                self.num_accepted[rid] = 1
            else:
                out = []
                self.num_accepted[rid] = 1
            sampled[rid] = out
        # fused postprocess (spec + align)
        for rid, nc, n in rows:
            drafts = so.scheduled_spec_decode_tokens.get(rid, [])
            acc = self.num_accepted[rid]
            running = nc + n - len(drafts)
            new_nc = running + acc - 1
            aligned = new_nc // B * B
            if aligned >= running and aligned > 0:
                bias = aligned - running
                dest = aligned // B - 1
                src = self.state_idx[rid]
                for g in range(NUM_MAMBA_GROUPS):
                    t = self.bt[rid][1 + g]
                    self._w(g, t[dest], self.mamba[g].get(t[src + bias]), f"post {rid} col{src}+{bias}->{dest}")
                if src == dest:
                    self.num_accepted[rid] = 1
        req_ids = list(so.num_scheduled_tokens)
        out = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=[sampled[r] for r in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        self.s.update_from_output(so, out)
        self._check_refs("update")
        # drafts for the next step
        d_ids, d_toks = [], []
        for r in req_ids:
            req = self.s.requests.get(r)
            if req is None or req.is_finished():
                continue
            d_ids.append(r)
            d_toks.append([self.rnd.randrange(1, 40000) for _ in range(NUM_SPEC)])
        if d_ids:
            self.s.update_draft_token_ids(DraftTokenIds(d_ids, d_toks))
        return True

    def _holders(self, g: int, b: int) -> str:
        m = self.s.kv_cache_manager.coordinator.single_type_managers[1 + g]
        blk = self.s.kv_cache_manager.block_pool.blocks[b]
        mh = [(r, i) for r, bl in m.req_to_blocks.items() for i, x in enumerate(bl) if x.block_id == b]
        wh = [(r, i) for r, t in self.bt.items() for i, x in enumerate(t[1 + g]) if x == b]
        return (f"block {b}: ref {blk.ref_cnt} hash_tokens {blk.block_hash_num_tokens} "
                f"mgr-holders {mh} worker-holders {wh}")

    def _check_refs(self, where: str) -> None:
        """Every block a live request's table holds must be referenced."""
        mgr = self.s.kv_cache_manager
        for g, m in enumerate(mgr.coordinator.single_type_managers):
            for rid, blocks in m.req_to_blocks.items():
                for i, b in enumerate(blocks):
                    assert b.is_null or b.ref_cnt > 0, (
                        f"step {self.step_no} ({where}): group {g} {rid} col {i} "
                        f"block {b.block_id} has ref_cnt {b.ref_cnt} (in free queue)"
                    )
        # The worker keeps an append-only table: the state column it will read
        # next must still be owned by the request on the manager side.
        for rid, col in self.state_idx.items():
            if rid not in self.bt or rid not in self.s.requests:
                continue
            for g in range(NUM_MAMBA_GROUPS):
                m = mgr.coordinator.single_type_managers[1 + g]
                mb = m.req_to_blocks.get(rid)
                if not mb or col >= len(mb):
                    continue
                wb = self.bt[rid][1 + g][col]
                assert mb[col].block_id == wb, (
                    f"step {self.step_no} ({where}): {rid} group {g} running col "
                    f"{col}: worker uses block {wb} but the manager holds "
                    f"{mb[col].block_id} there (block {wb} ref "
                    f"{self.s.kv_cache_manager.block_pool.blocks[wb].ref_cnt})"
                )

    def _check_attention(self, rid: str, req: Request, nc: int) -> None:
        if nc == 0:
            return
        a = self.bt[rid][ATTN]
        want = np.asarray(req.all_token_ids[:nc], dtype=np.int64)
        nb = cdiv(nc, B)
        got = np.concatenate([self.kv[a[j]] for j in range(nb)])[:nc]
        bad = np.nonzero(got != want)[0]
        assert bad.size == 0, (
            f"step {self.step_no}: {rid} attention KV wrong at {bad.size} "
            f"positions, first {int(bad[0])} (block col {int(bad[0]) // B}, "
            f"id {a[int(bad[0]) // B]}): got {int(got[bad[0]])} "
            f"want {int(want[bad[0]])}; nc={nc}"
        )

    def run_until_done(self, reqs: list[Request], max_steps: int = 5000) -> None:
        for _ in range(max_steps):
            self.step()
            if all(r.is_finished() for r in reqs):
                return
        raise AssertionError("requests did not finish")


def _toks(seed: int, n: int) -> list[int]:
    r = random.Random(seed)
    return [r.randrange(1, 40000) for _ in range(n)]


def _probe(unit: int, num_blocks: int, sizes=(20000, 35000, 50000), rounds=8,
           seed=0, siblings=4):
    sched = _scheduler(unit, num_blocks)
    w = Worker(sched, num_blocks, seed)
    rnd = random.Random(seed)
    prompts = [_toks(100 + i, s) for i, s in enumerate(sizes)]
    # round 0: the registry prompt, then `rounds` continuations, all sessions
    # concurrent (one request in flight per session)
    for rd in range(rounds + 1):
        reqs = []
        for si, p in enumerate(prompts):
            r = _req(f"s{si}r{rd}", list(p), unit, 8 if rd == 0 else rnd.randrange(8, 48))
            sched.add_request(r)
            reqs.append(r)
        w.run_until_done(reqs)
        for si, r in enumerate(reqs):
            prompts[si] = (
                list(r.prompt_token_ids)
                + list(r.output_token_ids)
                + _toks(1000 * rd + si, rnd.randrange(20, 400))
            )
    # sibling burst on the largest session, twice
    shared = prompts[-1][: len(prompts[-1]) - 300]
    for burst in range(2):
        reqs = []
        for j in range(siblings):
            r = _req(f"sib{burst}{j}", shared + _toks(77 + j, 25), unit, 16)
            sched.add_request(r)
            reqs.append(r)
        w.run_until_done(reqs)
    return w, sched


@pytest.fixture
def copy_mode(monkeypatch):
    monkeypatch.setenv("VLLM_GDN_TAIL_PUBLISH", "copy")


def test_production_block_aligned_baseline_is_consistent(monkeypatch):
    monkeypatch.delenv("VLLM_GDN_TAIL_PUBLISH", raising=False)
    w, _ = _probe(unit=B, num_blocks=1024, rounds=4)
    assert all(h % B == 0 for _, h in w.hits)


@pytest.mark.parametrize("unit", [16, 892])
def test_copy_mode_tail_reuse_resumes_exact_state(copy_mode, unit):
    w, sched = _probe(unit=unit, num_blocks=1024, rounds=6)
    tails = [h for _, h in w.hits if h % B]
    assert tails, "no sub-block tail hit happened"


@pytest.mark.parametrize("unit", [16])
def test_copy_mode_under_block_pressure(copy_mode, unit):
    """A tight pool forces evictions (and preemptions) between publish and use."""
    w, sched = _probe(unit=unit, num_blocks=150, rounds=6, seed=3)
    tails = [h for _, h in w.hits if h % B]
    assert tails
