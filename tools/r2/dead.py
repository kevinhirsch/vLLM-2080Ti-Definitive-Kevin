import os
os.environ["VLLM_R2_CHAIN_AWARE_EVICT"]="1"
from pool_harness import *
import workload
from collections import Counter
def mk(N):
    m=make_manager(N, retention_interval=0); m.block_pool.chain_index.dry_run=True; return m
for N in (120,200,285,400):
    agg=Counter(); hit=0
    for s in range(3):
        r=workload.simulate(N, seed=s, n_req=1500, mgr_factory=mk)
        d=r['mgr'].block_pool.chain_index.dead_weight(); agg.update(d); hit+=r['hit']/3
    tot=agg['attn']+agg['snap']
    print(f"N={N} hit={hit:.3f} hashed pages/seed={tot/3:.0f} attn_dead={agg['attn_dead']/max(agg['attn'],1):.2f} snap_dead={agg['snap_dead']/max(agg['snap'],1):.2f} dead_total_frac={(agg['attn_dead']+agg['snap_dead'])/tot:.2f} live_snapshots/seed={agg['live_snapshots']/3:.0f}")
