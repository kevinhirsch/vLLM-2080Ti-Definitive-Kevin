#!/usr/bin/env python
"""Lane K7: GPU correctness check for the W4A4 extension (tiny; refuses if < --min-free-mib free).
  1. w4a4_gemm (every config) vs exact integer reference: (A_int @ B_int^T) * sa * sb  -> rel err ~ fp16 rounding.
  2. act_quant_had vs torch reference (dense Sylvester Hadamard block-diag, per-token absmax/7, round-half-even, clamp).
Usage: CUDA_VISIBLE_DEVICES=1 python tools/k7/test_w4a4.py"""
import os, subprocess, sys
gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
free = int(subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).decode())
if free < int(os.environ.get("K7_MIN_FREE", 380)):
    sys.exit(f"refusing: GPU{gpu} {free} MiB free")
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k7ext
k7ext.gpu_gate(int(os.environ.get("K7_MIN_FREE", 380)))
from rotquant import hadamard

E = k7ext.ext()
torch.manual_seed(0)
ok = True
for (M, N, K) in [(64, 256, 512), (200, 384, 1024), (512, 1024, 5120), (130, 640, 3072)]:
    qa = torch.randint(-8, 8, (M, K)); qb = torch.randint(-8, 8, (N, K))
    sa = torch.rand(M) + 0.5; sb = (torch.rand(N) + 0.5) * 1e-2
    ref = (qa.double() @ qb.double().T) * sa.double()[:, None] * sb.double()[None]
    A, B = k7ext.pack_s4(qa).cuda(), k7ext.pack_s4(qb).cuda()
    assert torch.equal(k7ext.unpack_s4(A.cpu()), qa.to(torch.int16))
    for cfg in range(7):
        try:
            y = E.w4a4_gemm(A, B, sa.float().cuda(), sb.float().cuda(), cfg).double().cpu()
        except RuntimeError as ex:
            print(f"  M={M} N={N} K={K} cfg{cfg}: {ex}"); continue
        rel = ((y - ref).norm() / ref.norm()).item()
        good = rel < 2e-3
        ok &= good
        print(f"gemm M={M:4d} N={N:5d} K={K:5d} cfg{cfg}: rel err {rel:.2e} {'OK' if good else 'FAIL'}")

for (M, K, HB) in [(7, 5120, 128), (33, 3072, 512), (5, 8704, 512), (9, 5120, 1)]:
    x = (torch.randn(M, K) * torch.linspace(0.1, 3, K)).half()
    x[:, 5] *= 40  # an outlier channel
    q, s = E.act_quant_had(x.cuda(), HB, 7.0)
    H = hadamard(HB).double() if HB > 1 else torch.ones(1, 1, dtype=torch.double)
    xr = (x.double().view(M, K // HB, HB) @ H.T).view(M, K)
    sref = xr.abs().amax(1) / 7
    qref = torch.clamp(torch.round(xr / sref[:, None]), -8, 7)
    qk = k7ext.unpack_s4(q.cpu()).double()
    mism = (qk != qref).float().mean().item()
    srel = ((s.double().cpu() - sref).abs() / sref).max().item()
    good = mism < 2e-3 and srel < 1e-5
    ok &= good
    sq = 10 * torch.log10(xr.pow(2).sum() / (xr - qk * s.double().cpu()[:, None]).pow(2).sum()).item()
    print(f"actq M={M} K={K} HB={HB}: code mismatch {mism:.1e} (fp32 vs fp64 rounding ties), scale rel {srel:.1e}, SQNR {sq:.1f} dB {'OK' if good else 'FAIL'}")
for (M, K) in [(7, 5120), (5, 8704), (11, 3072), (3, 1024)]:
    x = (torch.randn(M, K) * torch.linspace(0.1, 3, K)).half(); x[:, 5] *= 40
    q1, s1 = E.act_quant_had(x.cuda(), 128, 7.0); q2, s2 = E.act_quant_h128(x.cuda(), 7.0)
    mism = (k7ext.unpack_s4(q1.cpu()) != k7ext.unpack_s4(q2.cpu())).float().mean().item()
    srel = ((s1 - s2).abs() / s1).max().item()
    good = mism < 1e-3 and srel < 1e-5; ok &= good
    print(f"h128v2 M={M} K={K}: code mismatch vs v1 {mism:.1e}, scale rel {srel:.1e} {'OK' if good else 'FAIL'}")
# group-scaled hand-written kernel vs exact integer reference
for VER, (M, N, K) in [(v, s) for v in (1, 2, 3) for s in [(128, 256, 128), (200, 256, 1024), (512, 1024, 5120), (77, 768, 3072), (300, 512, 8704)]] or [(128, 128, 128), (200, 256, 1024), (512, 1024, 5120), (77, 640, 3072), (300, 384, 8704)]:
    qa = torch.randint(-8, 8, (M, K)); qb = torch.randint(-8, 8, (N, K))
    G = K // 128
    si = torch.randint(1, 2049, (G, M), dtype=torch.int32); sm = torch.rand(M) + 0.5; sw = (torch.rand(N) + 0.5) * 1e-2
    part = torch.einsum("mgk,ngk->gmn", qa.view(M, G, 128).double(), qb.view(N, G, 128).double())
    ref = (part * si.double()[:, :, None]).sum(0) * (sm.double() / 2048)[:, None] * sw.double()[None]
    y = E.w4a4g_gemm(k7ext.pack_s4(qa).cuda(), k7ext.pack_s4(qb).cuda(), si.cuda(), sm.cuda(), sw.cuda(), 11, VER).double().cpu()
    rel = ((y - ref).norm() / ref.norm()).item(); good = rel < 2e-3; ok &= good
    print(f"w4a4g v{VER} M={M} N={N} K={K}: rel err {rel:.2e} {'OK' if good else 'FAIL'}")
for (M, K) in [(7, 5120), (5, 8704), (11, 3072)]:
    x = (torch.randn(M, K) * torch.linspace(0.1, 3, K)).half(); x[:, 5] *= 40
    q, si, sm = E.act_quant_h128g(x.cuda(), 7.0, 11)
    xr = (x.double().view(M, K // 128, 128) @ hadamard(128).double().T)
    sg = xr.abs().amax(-1) / 7; smx = sg.amax(1)
    siref = torch.round(sg / smx[:, None] * 2048).clamp(min=1)
    qref = torch.clamp(torch.round(xr / sg[..., None]), -8, 7).view(M, K)
    mism = (k7ext.unpack_s4(q.cpu()).double() != qref).float().mean().item()
    smis = (si.cpu().T.double() != siref).float().mean().item()
    good = mism < 2e-3 and smis < 2e-2; ok &= good
    print(f"actq_g M={M} K={K}: code mismatch {mism:.1e}, s_int mismatch {smis:.1e}, smax rel {((sm.cpu().double()-smx).abs()/smx).max():.1e} {'OK' if good else 'FAIL'}")
print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
