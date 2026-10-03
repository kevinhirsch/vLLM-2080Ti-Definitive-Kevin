import os
from pool_harness import *
import cyclic
for chain in ("0","1"):
    os.environ["VLLM_R2_CHAIN_AWARE_EVICT"]=chain
    for N in (60, 90, 120, 150):
        print("chain",chain,"N",N, [round(x,2) for x in cyclic.cyc(N, 0)])
