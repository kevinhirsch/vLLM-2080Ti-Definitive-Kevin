from pool_harness import BS
from compare2 import run
for N in (200,285,400):
    row=[]
    for ri in (0,2,3,4,6,None):
        ret = None if ri is None else ri*BS
        h=run(N,0,0,ret=ret,n_req=1500)
        row.append(f"ri={ri}: {h[0]:.3f}")
    print(f"N={N}: "+" | ".join(row))
