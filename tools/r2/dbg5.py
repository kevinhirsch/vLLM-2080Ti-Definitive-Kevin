import os
os.environ["VLLM_R2_CHAIN_AWARE_EVICT"]="1"
from pool_harness import *
import pool_harness as ph, workload
from vllm.v1.core.hybrid_chain_index import HybridChainIndex
from collections import Counter
demoted_keys={}   # key -> reason context
orig=HybridChainIndex._demote
def dem(self, blk):
    if self.dry_run: return
    if blk is not None and not blk.is_null and blk.ref_cnt==0 and blk.prev_free_block is not None:
        demoted_keys[blk.block_hash]=self.ctx
    return orig(self, blk)
HybridChainIndex._demote=dem
od=HybridChainIndex._drop_snapshot
def drop(self,s,demote_snap_pages):
    self.ctx='drop'
    return od(self,s,demote_snap_pages)
HybridChainIndex._drop_snapshot=drop
oe=HybridChainIndex.on_evicted
def ev(self,removed):
    from vllm.v1.core.kv_cache_utils import get_group_id
    self.ctx='snapevict' if any(get_group_id(k) in self.snap_gids for k in removed) else 'attnevict'
    return oe(self,removed)
HybridChainIndex.on_evicted=ev
hits=Counter(); 
orig_run=ph.run_request
def run2(mgr, req, **kw):
    cb, hit, _ = mgr.get_computed_blocks(req)
    for gi, blks in enumerate(cb.blocks):
        for b in blks:
            if b.block_hash in demoted_keys: hits[(gi>0, demoted_keys[b.block_hash])]+=1
    return orig_run(mgr, req, **kw)
workload.run_request=run2
r=workload.simulate(285, seed=0, n_req=1500)
print("hit",r['hit'],"demoted distinct",len(demoted_keys),"hits on demoted pages (is_snapshot, cause):",dict(hits))
