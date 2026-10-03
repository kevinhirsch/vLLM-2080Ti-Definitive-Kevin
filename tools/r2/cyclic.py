import sys
from pool_harness import *

def cyc(N, ri, nb=12, passes=4, blocks=8.6):
    mgr = make_manager(N, retention_interval=None if ri is None else ri * BS)
    bodies = [body(i, blocks) for i in range(nb)]
    out = []
    for p in range(passes):
        hit = tot = 0
        for i, t in enumerate(bodies):
            r = run_request(mgr, make_req(f"p{p}b{i}", t))
            if not r.ok: hit = -10**9
            hit += r.hit_tokens; tot += r.prompt_tokens
        out.append(hit / tot)
    return out

if __name__ == "__main__":
    for ri in (None, 0, 2):
        print("retention", ri)
        for N in (100, 150, 200, 250, 300, 360, 400, 450):
            o = cyc(N, ri)
            print(f"  N={N:4d} ({N/32:4.1f} bodies-dense) hit by pass: " + " ".join(f"{x:5.2f}" for x in o))
