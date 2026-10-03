"""Lane R2: drive the REAL KVCacheManager (CPU only) with a hybrid 1-full + 3-mamba(align) layout
that mirrors Qwen3.8-27B on our build, scaled so 1 block = 16 tokens (real: 3568).

Used to measure (a) pages a request actually holds in the shared pool, (b) cyclic-replay hit
behaviour vs pool size, (c) any eviction-policy change, with the engine's own code paths.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch  # noqa: E402

from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec)
from vllm.v1.request import Request  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402

init_none_hash(sha256)

BS = 16  # tokens per block (stands for 3568)


def make_manager(num_blocks: int, n_mamba: int = 3, retention_interval=None, spec_blocks: int = 0,
                 enable_caching: bool = True):
    groups = [KVCacheGroupSpec(["a0"], FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=1,
                                                         dtype=torch.float32))]
    for i in range(n_mamba):
        groups.append(KVCacheGroupSpec([f"m{i}"], MambaSpec(block_size=BS, shapes=((1, 1),),
                                                            dtypes=(torch.float32,), mamba_cache_mode="align",
                                                            num_speculative_blocks=spec_blocks)))
    cfg = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups,
                        prefix_cache_retention_interval=retention_interval)
    return KVCacheManager(cfg, max_model_len=100000, enable_caching=enable_caching, hash_block_size=BS,
                          scheduler_block_size=BS)


def make_req(rid: str, toks: list[int], max_tokens: int = 4) -> Request:
    sp = SamplingParams(max_tokens=max_tokens)
    sp.update_from_generation_config({}, eos_token_id=100)
    return Request(request_id=rid, prompt_token_ids=toks, mm_features=None, sampling_params=sp,
                   pooling_params=None, lora_request=None, cache_salt=None,
                   block_hasher=get_request_block_hasher(BS, sha256), session_id=None)


@dataclass
class Result:
    hit_tokens: int
    prompt_tokens: int
    held_blocks: int
    ok: bool


TICK = [0]


def run_request(mgr: KVCacheManager, req: Request, decode: int = 3, lookahead: int = 0, no_store: bool = False) -> Result:
    """Prefill (chunk = one block) + a few decode steps + free. Returns hit and peak held blocks."""
    TICK[0] += 1
    n = req.num_prompt_tokens
    computed_blocks, hit, _extra = mgr.get_computed_blocks(req)
    pool = mgr.block_pool
    first = True
    pos = hit
    peak = 0
    while pos < n:
        step = min(BS - (pos % BS) if pos % BS else BS, n - pos) if False else min(BS, n - pos)
        if first:
            nb = mgr.allocate_slots(req, step, num_new_computed_tokens=hit, new_computed_blocks=computed_blocks,
                                    num_lookahead_tokens=lookahead)
            first = False
        else:
            nb = mgr.allocate_slots(req, step, num_lookahead_tokens=lookahead)
        if nb is None:
            mgr.free(req)
            return Result(hit, n, peak, False)
        req.num_computed_tokens = pos + step
        pos += step
        mgr.new_step_starts()
        peak = max(peak, pool.num_gpu_blocks - 1 - pool.get_num_free_blocks())
    for t in range(decode):
        req.append_output_token_ids(1000 + t)
        nb = mgr.allocate_slots(req, 1, num_lookahead_tokens=lookahead)
        if nb is None:
            mgr.free(req)
            return Result(hit, n, peak, False)
        req.num_computed_tokens += 1
        mgr.new_step_starts()
    peak = max(peak, pool.num_gpu_blocks - 1 - pool.get_num_free_blocks())
    if no_store:
        mine = [b for m in mgr.coordinator.single_type_managers for b in m.req_to_blocks.get(req.request_id, [])]
        new_blocks = [b for b in mine if not b.is_null and b.ref_cnt == 1]   # not shared with another reader
    mgr.free(req)
    if no_store:
        for b in new_blocks:
            if b.ref_cnt == 0 and b.block_hash is not None and b.prev_free_block is not None:
                pool._maybe_evict_cached_block(b)
                pool.free_block_queue.remove(b)
                pool.free_block_queue.prepend_n([b])
    return Result(hit, n, peak, True)


def body(seed: int, nblocks_f: float) -> list[int]:
    rnd = random.Random(seed)
    return [rnd.randrange(1, 90000) for _ in range(int(nblocks_f * BS))]


if __name__ == "__main__":
    # sanity: one body alone: pages it leaves in the pool, and a replay hit
    for ri in (None, 0, 2):
        mgr = make_manager(300, retention_interval=None if ri is None else ri * BS)
        t = body(1, 8.6)
        free0 = mgr.block_pool.get_num_free_blocks()
        r = run_request(mgr, make_req("a", t))
        cached_resident = sum(1 for b in mgr.block_pool.blocks if b.block_hash is not None)
        r2 = run_request(mgr, make_req("b", t))
        print(f"retention={ri} first: hit={r.hit_tokens}/{r.prompt_tokens} peak_held={r.held_blocks} "
              f"hashed_blocks_after={cached_resident}  replay hit={r2.hit_tokens} ({r2.hit_tokens/BS:.1f} blocks)")
