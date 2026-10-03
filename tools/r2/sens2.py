import workload
from compare2 import run
for ff in (0.0, 0.05, 0.15, 0.3):
    a=run(285,0,0,n_req=1500,fork_frac=ff); c=run(285,1,1,n_req=1500,fork_frac=ff); d=run(285,1,0,n_req=1500,fork_frac=ff)
    print(f"fork_frac={ff}: LRU {a[0]:.3f} dead-recycle {d[0]:.3f} supersede {c[0]:.3f} ({100*((1-c[0])/(1-a[0])-1):+.0f}% computed)")
