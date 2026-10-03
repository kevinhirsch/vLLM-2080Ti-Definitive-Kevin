import workload
from pool_harness import BS
from compare2 import run
import sys
for label,kw in (("base",{}),("turn x2",{"TS":2.0}),("sessions24",{"n_sessions":24}),("noise.05",{"noise_frac":0.05}),("fork.15",{"fork_frac":0.15})):
    kw=dict(kw); workload.TURN_SCALE=kw.pop("TS",1.0)
    out=[]
    for ri in (0,3,4,5,6,8,12):
        h=run(285,0,0,ret=ri*BS,seeds=6,n_req=1200,**kw); out.append(f"ri{ri}:{h[0]:.3f}")
    print(label," | ".join(out))
