import os
os.environ["VLLM_R2_CHAIN_AWARE_EVICT"]="1"
from pool_harness import *
import workload
from vllm.v1.core.hybrid_chain_index import HybridChainIndex
demoted=[]
orig=HybridChainIndex._demote
def dem(self, blk):
    if blk is not None and not blk.is_null and blk.ref_cnt==0 and blk.prev_free_block is not None:
        demoted.append(blk.block_hash)
    return orig(self, blk)
HybridChainIndex._demote=dem
# instrument hits
import vllm.v1.core.block_pool as bp
hitkeys=[]
og=bp.BlockPool.get_cached_block
res=workload.simulate(285, seed=0, n_req=1500)
mgr=res['mgr']
ci=mgr.block_pool.chain_index
print("demoted",ci.demoted,"killed",ci.snapshots_killed, "anc",len(ci.anc),"desc",len(ci.desc))
print("hit",res['hit'],res['hit_agent'])

# second pass: count hits landing on previously-demoted pages, and lost hits
import pool_harness as ph
from collections import Counter
demoted_set=set(); hits_on_demoted=Counter()
orig_run=ph.run_request
def run2(mgr, req, **kw):
    cb, hit, _ = mgr.get_computed_blocks(req)
    for gi, blks in enumerate(cb):
        for b in blks:
            if b.block_hash in demoted_set: hits_on_demoted[gi]+=1
    return orig_run(mgr, req, **kw)
demoted.clear()
def dem2(self, blk):
    if blk is not None and not blk.is_null and blk.ref_cnt==0 and blk.prev_free_block is not None:
        demoted_set.add(blk.block_hash)
    return orig(self, blk)
HybridChainIndex._demote=dem2
workload.run_request=run2
res=workload.simulate(285, seed=0, n_req=1500)
print("hits landing on demoted pages by group:", dict(hits_on_demoted), "demoted distinct", len(demoted_set))
