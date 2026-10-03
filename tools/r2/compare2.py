import os, sys, time
from pool_harness import *
import workload
def mkm(n, ret, grace):
    m = make_manager(n, retention_interval=ret)
    ci = m.block_pool.chain_index
    if ci is not None:
        ci.clock = lambda: float(TICK[0]); ci.supersede_grace_s = grace
    return m


def run(N, chain, sup, ret=0, seeds=3, n_req=1500, grace=0.0, **kw):
    os.environ["VLLM_R2_CHAIN_AWARE_EVICT"]=str(chain); os.environ["VLLM_R2_SUPERSEDE"]=str(sup)
    res=[workload.simulate(N, seed=s, n_req=n_req, mgr_factory=lambda n: mkm(n, ret, grace), **kw) for s in range(seeds)]
    return sum(r['hit'] for r in res)/seeds, sum(r['hit_agent'] for r in res)/seeds
if __name__=="__main__":
    for N in (120,200,285,400,800):
        a=run(N,0,0); b=run(N,1,0); c=run(N,1,1)
        print(f"N={N}: LRU {a[0]:.3f}/{a[1]:.3f} | dead-recycle {b[0]:.3f}/{b[1]:.3f} | +supersede {c[0]:.3f}/{c[1]:.3f}")
