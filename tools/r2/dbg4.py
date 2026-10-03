import os
os.environ["VLLM_R2_CHAIN_AWARE_EVICT"]="1"
from pool_harness import *
mgr=make_manager(60, retention_interval=0)
ci=mgr.block_pool.chain_index
t=body(1,8.6)
r=run_request(mgr, make_req("a",t))
print("anc",len(ci.anc),"alive",[ci._snapshot_alive(s) for s in ci.anc], "desc",len(ci.desc), [len(v) for v in ci.desc.values()])
t2=t+body(2,2.5)
r=run_request(mgr, make_req("b",t2)); print("hit",r.hit_tokens)
print("anc",len(ci.anc),"alive",[ci._snapshot_alive(s) for s in ci.anc],"desc",len(ci.desc))
# queue order dump
q=mgr.block_pool.free_block_queue
b=q.fake_free_list_head.next_free_block; order=[]
while b is not q.fake_free_list_tail:
    order.append((b.block_id, None if b.block_hash is None else (b.block_hash[-1]))); b=b.next_free_block
print(order[:60])
