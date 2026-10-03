"""K7 quick kernel A/B (interleaved): CUTLASS per-token W4A4 vs hand-written group-scaled W4A4, a few shapes, M=2048."""
import os, sys, statistics, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k7ext
E = k7ext.ext()
def tm(fn):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record(); fn(); e.record(); e.synchronize(); return s.elapsed_time(e)
M = int(os.environ.get("KB_M", 2048))
for (N, K, cfg) in [(8704, 5120, 2), (5120, 8704, 0), (5120, 3072, 2)]:
    A = torch.randint(-128, 128, (M, K // 2), dtype=torch.int8, device="cuda"); B = torch.randint(-128, 128, (N, K // 2), dtype=torch.int8, device="cuda")
    sa = torch.rand(M, device="cuda"); sb = torch.rand(N, device="cuda"); SI = torch.randint(1, 2048, (K // 128, M), dtype=torch.int32, device="cuda")
    f = {"cutlass_pt": lambda: E.w4a4_gemm(A, B, sa, sb, cfg), "hand_g3": lambda: E.w4a4g_gemm(A, B, SI, sa, sb, 11, 3), "probe_nofold": lambda: E.w4a4g_gemm(A, B, SI, sa, sb, 11, 4)}
    for fn in f.values(): fn(); fn()
    ts = {k: [] for k in f}
    for _ in range(20):
        for k, fn in f.items(): ts[k].append(tm(fn))
    print(f"N={N} K={K} M={M}: " + "  ".join(f"{k} {min(v):.3f} ms ({2*M*N*K/min(v)/1e9:.0f} TOPS)" for k, v in ts.items()), flush=True)
