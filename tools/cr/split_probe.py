"""Lane CR (f): drive the REAL Scheduler._mamba_block_aligned_split with production geometry
(B=1856, step budget 3632, MTP3, no eagle block drop) over fragmented budgets and report every
chunk sequence that crosses the prompt's last boundary E WITHOUT ending a chunk on it."""
import os, sys, random, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from types import SimpleNamespace
from vllm.v1.core.sched.scheduler import Scheduler
from tests.v1.core.utils import create_requests
B = int(os.environ.get("B", 1856)); MAXT = int(os.environ.get("MAXT", 3632))
stub = SimpleNamespace(block_size=B, cache_config=SimpleNamespace(block_size=B), use_eagle_block_drop=False,
                       max_num_scheduled_tokens=MAXT, scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
                       mamba_partial_cache_hit=False, mamba_fine_grained_prefix_cache=False, hash_block_size=B,
                       mamba_has_prefill_checkpoint_blocks=False, mamba_prefill_checkpoint_alignment=None)
rnd = random.Random(0); res = collections.Counter(); ex = []
for trial in range(3000):
    nb0 = rnd.randrange(3, 20); tail = rnd.randrange(5, B - 5); grow = rnd.randrange(1, 3) 
    pp = (nb0 + grow) * B + tail; pca = nb0 * B
    (req,) = create_requests(1, num_tokens=pp, block_size=16)
    req.num_computed_tokens = pca
    E = pp - pp % B; ends = []
    for step in range(200):
        c = req.num_computed_tokens
        if c >= pp: break
        budget = rnd.choice([MAXT, MAXT - 4 * rnd.randrange(1, 9), rnd.randrange(1, MAXT)])
        n = Scheduler._mamba_block_aligned_split(stub, req, min(pp - c, budget))
        if n == 0: continue
        req.num_computed_tokens = c + n; ends.append(c + n)
    ok = E in ends or E <= pca
    res["stops_at_E" if ok else "CROSSES_E"] += 1
    if not ok and len(ex) < 5: ex.append((pca, pp, E, ends))
print(dict(res)); [print(e) for e in ex]
