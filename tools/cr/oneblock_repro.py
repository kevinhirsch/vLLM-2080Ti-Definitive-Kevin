"""Lane CR (f): reproduce 'continuation lands one block short' on the REAL KVCacheManager (CPU).
Predecessor A computes ACROSS its last block boundary E (hit at pca < E), then decodes with MTP
lookahead; continuation B = A.prompt + growth.  Expected hit = E."""
import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.request import Request
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
init_none_hash(sha256)
BS = 16

def mk(num_blocks=400, spec=3, n_mamba=3):
    g = [KVCacheGroupSpec(["a0"], FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=1, dtype=torch.float32))]
    for i in range(n_mamba):
        g.append(KVCacheGroupSpec([f"m{i}"], MambaSpec(block_size=BS, shapes=((1, 1),), dtypes=(torch.float32,),
                                                      mamba_cache_mode="align", num_speculative_blocks=spec)))
    cfg = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=g)
    return KVCacheManager(cfg, max_model_len=100000, enable_caching=True, hash_block_size=BS, scheduler_block_size=BS,
                          use_eagle=False, num_prefill_lookahead=0)

def rq(rid, toks):
    sp = SamplingParams(max_tokens=64); sp.update_from_generation_config({}, eos_token_id=100)
    return Request(request_id=rid, prompt_token_ids=toks, mm_features=None, sampling_params=sp, pooling_params=None,
                   lora_request=None, cache_salt=None, block_hasher=get_request_block_hasher(BS, sha256), session_id=None)

def run(mgr, req, chunks_fn, decode=int(os.environ.get("DEC", 3)), la=3):
    blocks, hit, *_ = mgr.get_computed_blocks(req)
    n = req.num_prompt_tokens; pos = hit; first = True
    for end in chunks_fn(hit, n):
        step = end - pos
        kw = dict(num_lookahead_tokens=la if (end == n or os.environ.get("LA_ALL")) else 0)
        if first:
            r = mgr.allocate_slots(req, step, num_new_computed_tokens=hit, new_computed_blocks=blocks, **kw); first = False
        else:
            r = mgr.allocate_slots(req, step, **kw)
        assert r is not None
        req.num_computed_tokens = end; pos = end; mgr.new_step_starts()
    for t in range(decode):
        req.append_output_token_ids(1000 + t)
        assert mgr.allocate_slots(req, 1, num_lookahead_tokens=la) is not None
        req.num_computed_tokens += 1; mgr.new_step_starts()
    mgr.free(req)
    return hit

def aligned(hit, n):          # what _mamba_block_aligned_split does: stop at every block boundary + last boundary
    out = []; p = hit
    while p < n:
        nxt = min((p // BS + 1) * BS, n); out.append(nxt); p = nxt
    return out

def one_shot(hit, n):         # a single step from hit to n (no stop at E)
    return [n]

def stop_at_E(hit, n):
    E = n - n % BS
    return [E, n] if hit < E < n else [n]

rnd = random.Random(1)
base = [rnd.randrange(1, 9000) for _ in range(200)]
for name, fn in (("aligned", aligned), ("stop_at_E", stop_at_E), ("one_shot", one_shot)):
    for spec, la in ((0, 0), (3, 3)):
        for tail in (3, 7, 12):
            mgr = mk(spec=spec)
            p0 = base[:5 * BS + 5]                        # grand-predecessor: hit E0 = 5 blocks
            run(mgr, rq("g", p0), aligned, la=la)
            pa = base[:7 * BS + tail]                     # A: hit at 80, computes across 96 and 112
            ha = run(mgr, rq("a", pa), fn, la=la)
            pb = pa + base[150:150 + 20]
            hb = run(mgr, rq("b", pb), aligned, la=la)
            E = (len(pa) // BS) * BS
            print(f"{name:9s} spec={spec} tailA={tail:2d}: A hit {ha}  B hit {hb}  expected {E}  {'OK' if hb >= E else 'SHORT by %d' % (E - hb)}")
