"""Multi-tenant agent workload on the real hybrid KVCacheManager (scaled: 1 block = 16 tok = 3568 real)."""
import random, sys
from pool_harness import *

HEADS = 2           # shared system+tools heads (Halo/pi style)
HEAD_BLOCKS = 7.8   # ~28K tokens
TURN_BLOCKS = 0.5   # ~1.8K tokens per tool-loop turn
MAX_BLOCKS = 28     # session reset ~100K
TURN_SCALE = 1.0

class Session:
    def __init__(s, sid, rnd, heads, kind):
        s.sid, s.rnd, s.kind = sid, rnd, kind
        s.toks = list(heads[rnd.randrange(len(heads))]) if kind == "agent" else []
        s.toks = s.toks + [rnd.randrange(1, 90000) for _ in range(int(rnd.uniform(0.2, 1.0) * BS))]
        s.turn = 0
    def extend(s):
        s.toks = s.toks + [s.rnd.randrange(1, 90000) for _ in range(int(s.rnd.uniform(0.3, 0.8) * TURN_SCALE * BS))]
        s.turn += 1
    def done(s):
        return len(s.toks) > MAX_BLOCKS * BS

def simulate(N, seed=0, n_req=3000, n_sessions=14, mgr_factory=None, noise_frac=0.25, retention=0, fork_frac=0.0, noise_no_store=False):
    rnd = random.Random(seed)
    heads = [body(900 + i, HEAD_BLOCKS) for i in range(HEADS)]
    mgr = mgr_factory(N) if mgr_factory else make_manager(N, retention_interval=retention)
    sess = [Session(i, rnd, heads, "agent") for i in range(n_sessions)]
    nsid = n_sessions
    hit = tot = 0
    hit_agent = tot_agent = 0
    fails = 0
    # zipf-ish activity: some sessions are hot
    weights = [1.0 / (i + 1) ** 0.6 for i in range(n_sessions)]
    for k in range(n_req):
        if rnd.random() < noise_frac:       # one-shot big prompt (e.g. card-repair / overflow), never reused
            t = body(10_000_000 + k, rnd.uniform(1.0, 6.0)); kind = "noise"
        else:
            i = rnd.choices(range(n_sessions), weights)[0]
            s = sess[i]
            if fork_frac and s.turn > 1 and rnd.random() < fork_frac:
                # sub-agent style fork: a new session that starts from this session's current context
                j = rnd.randrange(n_sessions)
                ns = Session(nsid, rnd, heads, "agent"); nsid += 1
                ns.toks = list(s.toks); ns.turn = 1
                sess[j] = ns
                if j == i: s = ns
            if s.turn > 0:
                s.extend()
            else:
                s.turn = 1
            t = list(s.toks); kind = "agent"
            if s.done():
                sess[i] = Session(nsid, rnd, heads, "agent"); nsid += 1
        r = run_request(mgr, make_req(f"r{k}", t), no_store=(noise_no_store and kind == "noise"))
        if not r.ok:
            fails += 1; continue
        hit += r.hit_tokens; tot += r.prompt_tokens
        if kind == "agent":
            hit_agent += r.hit_tokens; tot_agent += r.prompt_tokens
    return dict(hit=hit / tot, hit_agent=hit_agent / tot_agent, computed=tot - hit, fails=fails, mgr=mgr)

if __name__ == "__main__":
    for N in (120, 200, 285, 400):
        res = [simulate(N, seed=s, n_req=1500) for s in range(3)]
        print(f"N={N}: hit(all)={sum(r['hit'] for r in res)/3:.3f} hit(agent)={sum(r['hit_agent'] for r in res)/3:.3f} fails={sum(r['fails'] for r in res)}")
