import workload
from compare2 import run
for ts in (0.5, 1.0, 2.0, 4.0):
    workload.TURN_SCALE=ts
    for ns,nf in ((14,0.25),(8,0.25),(24,0.25),(14,0.5)):
        a=run(285,0,0,n_req=1200,n_sessions=ns,noise_frac=nf); c=run(285,1,1,n_req=1200,n_sessions=ns,noise_frac=nf)
        print(f"turn x{ts} sessions={ns} noise={nf}: LRU {a[0]:.3f} -> supersede {c[0]:.3f}  (miss {1-a[0]:.3f} -> {1-c[0]:.3f}, {100*((1-c[0])/(1-a[0])-1):+.0f}% computed)")
