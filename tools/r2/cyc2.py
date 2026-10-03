from pool_harness import BS
import cyclic
for ri in (0,2,3,6):
    print("ri",ri,[ (N,[round(x,2) for x in cyclic.cyc(N, ri if ri==0 else ri)][1]) for N in (120,150,200,250)])
