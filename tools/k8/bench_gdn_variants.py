import sys, itertools, torch
sys.path.insert(0, "/home/kevin/Desktop/wt-k8/tools/k8")
import gdn_variants as G
H, HV, K, V, TS = 8, 24, 128, 128, 4
dev = "cuda"
def mk(N, dtype=torch.float16, seed=0):
    g = torch.Generator(device=dev); g.manual_seed(seed)
    r = lambda *s, dt=torch.float16: torch.randn(*s, device=dev, generator=g).to(dt)
    d = dict(st=(r(N * TS + 2, HV, V, K) * 0.1).to(dtype), q=r(1, N * TS, H, K), k=r(1, N * TS, H, K), v=r(1, N * TS, HV, V),
             a=r(N * TS, HV), b=r(N * TS, HV), A_log=r(HV, dt=torch.float32), dtb=r(HV, dt=torch.float32),
             cu=torch.arange(N + 1, device=dev, dtype=torch.int32) * TS,
             idx=(torch.arange(N * TS, device=dev, dtype=torch.int32) + 1).reshape(N, TS),
             nacc=torch.randint(1, TS + 1, (N,), device=dev, dtype=torch.int32, generator=g))
    return d
def call(d, st, **kw):
    return G.gdn_var_update(A_log=d["A_log"], a=d["a"], b=d["b"], dt_bias=d["dtb"], q=d["q"], k=d["k"], v=d["v"], initial_state=st,
                            inplace_final_state=True, cu_seqlens=d["cu"], ssm_state_indices=d["idx"], num_accepted_tokens=d["nacc"],
                            use_qk_l2norm_in_kernel=True, null_block_id=-1, **kw)
def timeit(d, reps=80, **kw):
    st = d["st"].clone()
    for _ in range(8): call(d, st, **kw)
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); call(d, st, **kw); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1000)
    ts.sort(); return ts[len(ts) // 2], ts[0]
if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    d = mk(N)
    s0 = d["st"].clone(); o0, _ = call(d, s0)  # reference (default config)
    print("cfg                      median  min   (us)   maxerr_o  maxerr_state")
    for bv, nw, sm in [(32, 4, 0), (32, 4, 1), (32, 4, 2), (32, 1, 0), (32, 2, 0), (32, 8, 0), (16, 1, 0), (16, 2, 0), (16, 4, 0), (8, 1, 0), (8, 2, 0), (64, 4, 0), (64, 8, 0), (128, 4, 0), (128, 8, 0)]:
        try:
            med, mn = timeit(d, BV_OVR=bv, NWARPS=nw, STORE_MODE=sm)
            err = ("", "")
            if sm == 0:
                s1 = d["st"].clone(); o1, _ = call(d, s1, BV_OVR=bv, NWARPS=nw)
                err = ((o1.float() - o0.float()).abs().max().item(), (s1.float() - s0.float()).abs().max().item())
            print(f"BV={bv:3d} warps={nw} store={sm}  {med:7.1f} {mn:7.1f}   {err[0]}  {err[1]}", flush=True)
        except Exception as e:
            print(f"BV={bv} warps={nw} store={sm} FAIL {str(e)[:80]}")
    print("peak MiB", torch.cuda.max_memory_allocated() / 2**20)
