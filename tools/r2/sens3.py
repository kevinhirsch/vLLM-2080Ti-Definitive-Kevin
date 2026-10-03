from compare2 import run
for ff in (0.0, 0.15, 0.3):
    a=run(285,0,0,n_req=1500,fork_frac=ff)
    row=[f"LRU {a[0]:.3f}"]
    for g in (0, 3, 6, 12, 25):
        c=run(285,1,1,n_req=1500,fork_frac=ff,grace=float(g))
        row.append(f"grace{g}: {c[0]:.3f} ({100*((1-c[0])/(1-a[0])-1):+.0f}%)")
    print(f"fork_frac={ff}: "+" | ".join(row))
