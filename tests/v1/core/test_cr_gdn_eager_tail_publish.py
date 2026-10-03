# SPDX-License-Identifier: Apache-2.0
"""[FORK][LANE CR] Copy-based prompt-tail publication for GDN (VLLM_GDN_TAIL_PUBLISH=copy).

A tool-loop continuation re-sends its predecessor's prompt plus a tool result. In mamba
"align" mode the recurrent state only exists at whole-block boundaries, so the continuation
recomputes everything past the predecessor's last boundary (avg B/2 = ~928 tokens at
B=1856). Sub-block ("fine-grained") reuse was gated off for GDN (weicj #241) because the
producer published its tail state from the LIVE running slot under a lazy CoW, which could
advertise a state no kernel wrote (#240). Copy mode never hashes a live slot: it pins the
slot, and one step later (after the forward that wrote the state, before any forward that
advances it) copies it into a dedicated block and registers the hash there.

These tests drive the REAL KVCacheManager + the REAL Scheduler._mamba_block_aligned_split
and a model of the GDN kernel/worker contract:
  * at the start of a step the worker runs the step's queued block copies (batched),
  * then the forward writes the chunk-end state into the running slot (n-1)//B,
and check that every hit a consumer takes resumes from exactly the state its hash claims.
"""

import random
from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

B = 64  # attention == mamba block (production: 1856)
U = 16  # prefix_match_unit (production: 64)
MAMBA = 1


def _manager(fine_grained: bool, num_blocks: int = 400) -> KVCacheManager:
    cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attn"],
                FullAttentionSpec(
                    block_size=B, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["gdn"],
                MambaSpec(
                    block_size=B,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                    supports_fine_grained_prefix_cache=fine_grained,
                ),
            ),
        ],
    )
    return KVCacheManager(
        cfg,
        max_model_len=100000,
        enable_caching=True,
        hash_block_size=U if fine_grained else B,
        scheduler_block_size=B,
    )


_INIT = [False]


def _req(rid: str, toks: list[int], unit: int) -> Request:
    if not _INIT[0]:
        init_none_hash(sha256)
        _INIT[0] = True
    sp = SamplingParams(max_tokens=64)
    sp.update_from_generation_config({}, eos_token_id=100000)
    return Request(
        request_id=rid,
        prompt_token_ids=toks,
        mm_features=None,
        sampling_params=sp,
        pooling_params=None,
        lora_request=None,
        cache_salt=None,
        block_hasher=get_request_block_hasher(unit, sha256),
    )


class World:
    """Scheduler-step model: copies at step start, then each forward."""

    def __init__(self, mgr: KVCacheManager, partial: bool):
        self.mgr = mgr
        self.partial = partial
        self.state_at: dict[int, int] = {}  # mamba block id -> state offset it holds
        self.mamba = mgr.coordinator.single_type_managers[MAMBA]
        self.stub = SimpleNamespace(
            block_size=B,
            cache_config=SimpleNamespace(block_size=B),
            use_eagle_block_drop=False,
            max_num_scheduled_tokens=4 * B,
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
            mamba_partial_cache_hit=partial,
            mamba_fine_grained_prefix_cache=False,
            hash_block_size=U if partial else B,
            mamba_has_prefill_checkpoint_blocks=False,
            mamba_prefill_checkpoint_alignment=None,
        )
        self.resumed_from: dict[str, int] = {}

    def step(self, work: list[tuple[Request, int]]) -> None:
        """One engine step: ``work`` = [(request, num_new_tokens)] already split."""
        self.mgr.new_step_starts()
        for req, n in work:
            assert self.mgr.allocate_slots(req, n) is not None
        copies, retained = self.mgr.take_kv_cache_block_copies()
        # worker: batched copies first (gather then scatter, like index_copy)
        snap = {c.src_block_id: self.state_at.get(c.src_block_id) for c in copies}
        for c in copies:
            self.state_at[c.dst_block_id] = snap[c.src_block_id]
        # forward: chunk end state into the running slot
        for req, n in work:
            req.num_computed_tokens += n
            blocks = self.mamba.req_to_blocks[req.request_id]
            run = cdiv(req.num_computed_tokens, B) - 1
            self.state_at[blocks[run].block_id] = req.num_computed_tokens
        # scheduler: fenced free after the step that ran the copies
        self.mgr.block_pool.free_blocks(retained)

    def admit(self, req: Request) -> int:
        blocks, hit = self.mgr.get_computed_blocks(req)[:2]
        if hit:
            # The consumer must resume from exactly the state its hit claims.
            mb = blocks.blocks[MAMBA]
            src = [b for b in mb if not b.is_null][-1]
            assert self.state_at.get(src.block_id) == hit, (
                f"{req.request_id}: hit {hit} resumes from state@"
                f"{self.state_at.get(src.block_id)}"
            )
        n = Scheduler._mamba_block_aligned_split(
            self.stub, req, req.num_prompt_tokens - hit, hit, 0
        )
        self.mgr.new_step_starts()
        assert (
            self.mgr.allocate_slots(
                req, n, num_new_computed_tokens=hit, new_computed_blocks=blocks
            )
            is not None
        )
        copies, retained = self.mgr.take_kv_cache_block_copies()
        snap = {c.src_block_id: self.state_at.get(c.src_block_id) for c in copies}
        for c in copies:
            self.state_at[c.dst_block_id] = snap[c.src_block_id]
        req.num_computed_tokens = hit + n
        bl = self.mamba.req_to_blocks[req.request_id]
        self.state_at[bl[cdiv(req.num_computed_tokens, B) - 1].block_id] = (
            req.num_computed_tokens
        )
        self.mgr.block_pool.free_blocks(retained)
        return hit

    def finish_prefill(self, req: Request) -> None:
        while req.num_computed_tokens < req.num_prompt_tokens:
            n = Scheduler._mamba_block_aligned_split(
                self.stub, req, req.num_prompt_tokens - req.num_computed_tokens
            )
            self.step([(req, n)])

    def decode(self, req: Request, k: int) -> None:
        for t in range(k):
            req.append_output_token_ids(90000 + t)
            self.step([(req, 1)])

    def run(self, req: Request, decode: int = 5) -> int:
        hit = self.admit(req)
        self.finish_prefill(req)
        self.decode(req, decode)
        self.mgr.free(req)
        return hit


def _toks(seed: int, n: int) -> list[int]:
    r = random.Random(seed)
    return [r.randrange(1, 50000) for _ in range(n)]


@pytest.fixture
def copy_mode(monkeypatch):
    monkeypatch.setenv("VLLM_GDN_TAIL_PUBLISH", "copy")


def _chain(world: World, unit: int, turns: int = 6, seed: int = 0):
    """A tool loop: each turn re-sends the previous prompt + a tool result."""
    rnd = random.Random(seed)
    base = _toks(seed, 5 * B + 23)
    hits, expected = [], []
    prompt = base
    for t in range(turns):
        req = _req(f"t{t}", list(prompt), unit)
        hits.append(world.run(req, decode=rnd.randrange(1, 9)))
        prev = len(prompt)
        expected.append(prev)
        prompt = prompt + _toks(1000 + t, rnd.randrange(5, 3 * B))
    return hits, expected


def test_copy_mode_reuses_the_prompt_tail(copy_mode):
    world = World(_manager(fine_grained=True), partial=True)
    assert world.mamba.eager_tail_publish
    hits, prompts = _chain(world, U)
    # turn t>0 resumes at the predecessor's last U boundary, not its last B one
    for t in range(1, len(hits)):
        prev = prompts[t - 1]
        assert hits[t] == (prev // U) * U, (t, hits[t], prev)
    partial_tails = sum(1 for n in prompts if (n // U) * U % B)
    assert world.mamba.num_eager_tail_published == partial_tails


def test_block_aligned_baseline_loses_the_tail():
    world = World(_manager(fine_grained=False), partial=False)
    hits, prompts = _chain(world, B)
    for t in range(1, len(hits)):
        assert hits[t] == (prompts[t - 1] // B) * B


def test_copy_mode_never_hashes_a_live_slot(copy_mode):
    """No hashed mamba block may be a slot a running request still advances."""
    world = World(_manager(fine_grained=True), partial=True)
    req = _req("p", _toks(7, 3 * B + 40), U)
    world.admit(req)
    while req.num_computed_tokens < req.num_prompt_tokens:
        n = Scheduler._mamba_block_aligned_split(
            world.stub, req, req.num_prompt_tokens - req.num_computed_tokens
        )
        world.step([(req, n)])
        live = world.mamba.req_to_blocks["p"][cdiv(req.num_computed_tokens, B) - 1]
        if req.num_computed_tokens % B:
            assert live.block_hash is None, "live running slot carries a hash"
    world.decode(req, 3)
    world.mgr.free(req)


def test_sibling_hits_while_producer_still_decoding(copy_mode):
    """A consumer admitted while the producer keeps decoding (advancing its slot)
    must still resume from the published copy, not the advanced live slot."""
    world = World(_manager(fine_grained=True), partial=True)
    p = _req("p", _toks(3, 4 * B + 50), U)
    world.admit(p)
    world.finish_prefill(p)
    world.decode(p, 4)  # producer slot now holds state@(prompt+4)
    c = _req("c", list(p.prompt_token_ids) + _toks(9, 30), U)
    hit = world.admit(c)  # admit() asserts the resumed state matches
    assert hit == (p.num_prompt_tokens // U) * U


def test_publish_dropped_when_pool_is_exhausted(copy_mode):
    mgr = _manager(fine_grained=True, num_blocks=12)
    world = World(mgr, partial=True)
    p = _req("p", _toks(5, 2 * B + 40), U)
    world.admit(p)
    while p.num_computed_tokens < p.num_prompt_tokens:
        n = Scheduler._mamba_block_aligned_split(
            world.stub, p, p.num_prompt_tokens - p.num_computed_tokens
        )
        if world.mamba._eager_tail_intents:
            # exhaust the pool right before the publish step
            hog = mgr.block_pool.get_new_blocks(mgr.block_pool.get_num_free_blocks())
            world.step([(p, n)])
            mgr.block_pool.free_blocks(hog)
        else:
            world.step([(p, n)])
    assert world.mamba.num_eager_tail_dropped == 1
    assert world.mamba.num_eager_tail_published == 0
    world.mgr.free(p)
    # nothing leaked: every block is back in the pool
    assert mgr.block_pool.get_num_free_blocks() == 12 - 1


def test_env_off_keeps_gdn_gated(monkeypatch):
    monkeypatch.delenv("VLLM_GDN_TAIL_PUBLISH", raising=False)
    from vllm.v1.core.single_type_kv_cache_manager import eager_tail_publish_enabled

    assert not eager_tail_publish_enabled()
    world = World(_manager(fine_grained=True), partial=True)
    assert not world.mamba.eager_tail_publish
