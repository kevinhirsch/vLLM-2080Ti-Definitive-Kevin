import os, sys, time
from pool_harness import *
import workload, cyclic

def mk(N, chain):
    def f(n):
        if chain: os.environ["VLLM_R2_CHAIN_AWARE_EVICT"] = "1"
        else: os.environ["VLLM_R2_CHAIN_AWARE_EVICT"] = "0"
        return make_manager(n, retention_interval=RET)
    return f
RET = 0
if __name__ == "__main__":
    for RET in (0, None):
        print("retention", RET)
        for N in (120, 200, 285, 400):
            row = []
            for chain in (0, 1):
                t = time.time()
                res = [workload.simulate(N, seed=s, n_req=1500, mgr_factory=mk(N, chain)) for s in range(3)]
                row.append((sum(r['hit'] for r in res)/3, sum(r['hit_agent'] for r in res)/3, sum(r['fails'] for r in res), time.time()-t))
            print(f"  N={N}: LRU hit={row[0][0]:.3f}/agent {row[0][1]:.3f} ({row[0][3]:.0f}s) | chain hit={row[1][0]:.3f}/agent {row[1][1]:.3f} ({row[1][3]:.0f}s) | fails {row[0][2]},{row[1][2]}")
