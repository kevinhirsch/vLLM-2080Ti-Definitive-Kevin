import os, sys
CH=sys.argv[1]
os.environ["VLLM_R2_CHAIN_AWARE_EVICT"]=CH
from pool_harness import *
import pool_harness as ph, workload
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id as mk
from collections import Counter
stats=Counter()
prev={}   # first-block-hash-of-session -> n_prev tokens  (approximate session id by first 2 blocks? use sid via toks prefix)
orig_run=ph.run_request
last_len={}
def run2(mgr, req, **kw):
    pool=mgr.block_pool
    bh=req.block_hashes
    # find predecessor: longest previous prompt whose block hashes are a prefix of ours (by 4th block hash identity)
    key=bh[3] if len(bh)>3 else None
    # session identity: last known request sharing block hash index 10 (beyond shared head)
    sk=bh[9] if len(bh)>9 else None
    if sk is not None and sk in last_len:
        b_prev=last_len[sk]
        stats['followups']+=1
        # oracle availability
        snap_ok=all(pool.cached_block_hash_to_block.get_one_block(mk(bh[b_prev//BS-1],g)) is not None for g in (1,2,3)) if b_prev//BS-1 < len(bh) else False
        attn_ok=all(pool.cached_block_hash_to_block.get_one_block(mk(bh[j],0)) is not None for j in range(b_prev//BS)) if b_prev//BS<=len(bh) else False
        stats[f'snap_ok={snap_ok},attn_ok={attn_ok}']+=1
    r=orig_run(mgr, req, **kw)
    if sk is not None:
        last_len[sk]=((req.num_prompt_tokens-1)//BS)*BS
    return r
workload.run_request=run2
tot=0
for s in range(3):
    res=workload.simulate(285, seed=s, n_req=1500)
    tot+=res['hit']/3
print("chain",CH,"hit",round(tot,3),dict(stats))
