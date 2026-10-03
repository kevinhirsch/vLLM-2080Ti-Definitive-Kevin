import os, traceback
os.environ["VLLM_R2_CHAIN_AWARE_EVICT"]="1"
from pool_harness import *
import workload
import vllm.v1.core.block_pool as bp
from vllm.v1.core.kv_cache_utils import get_group_id
from collections import Counter
c=Counter()
oi=bp.BlockPool._insert_block_hash
def ins(self,key,block,num_tokens):
    if get_group_id(key)==1:
        st=[f.name for f in traceback.extract_stack()[-4:-1]]
        c[tuple(st)]+=1
    return oi(self,key,block,num_tokens)
bp.BlockPool._insert_block_hash=ins
r=workload.simulate(285, seed=0, n_req=500)
for k,v in c.items(): print(v,k)
