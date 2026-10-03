import os
os.environ["VLLM_R2_CHAIN_AWARE_EVICT"]="1"
from pool_harness import *
import pool_harness as ph, workload
from vllm.v1.core.kv_cache_utils import get_block_hash
from collections import Counter
c=Counter()
orig_run=ph.run_request
def run2(mgr, req, **kw):
    cb, hit, _ = mgr.get_computed_blocks(req)
    ci=mgr.block_pool.chain_index
    if hit>0:
        sb=cb.blocks[1][-1]
        h=get_block_hash(sb.block_hash) if sb.block_hash else None
        c['hit']+=1
        c['snap_in_anc' if h in ci.anc else 'snap_NOT_in_anc']+=1
        if h in ci.anc: c['alive' if ci._snapshot_alive(h) else 'registered_but_dead']+=1
        # are attention pages of hit all in desc[h]?
    return orig_run(mgr, req, **kw)
workload.run_request=run2
r=workload.simulate(285, seed=0, n_req=1500)
print(dict(c))
