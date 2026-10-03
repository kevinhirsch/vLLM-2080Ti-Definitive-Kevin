"""Lane K9: what does Triton emit for tl.dot on sm_75? fp16 inputs, out_dtype fp32 vs fp16; count HMMA/FFMA/HFMA2 in SASS."""
import os, re, subprocess, collections, tempfile
os.environ.setdefault("TRITON_CACHE_DIR", "/home/kevin/projects/lanes/k9/triton_cache_dot")
import torch, triton, triton.language as tl
@triton.jit
def k(a, b, c, OUT16: tl.constexpr):
    r = tl.arange(0, 64)
    x = tl.load(a + r[:, None] * 64 + r[None, :]); y = tl.load(b + r[:, None] * 64 + r[None, :])
    if OUT16:
        z = tl.dot(x, y, out_dtype=tl.float16)
    else:
        z = tl.dot(x, y, out_dtype=tl.float32)
    tl.store(c + r[:, None] * 64 + r[None, :], z.to(tl.float16))
A = torch.randn(64, 64, device="cuda", dtype=torch.half); B = torch.randn(64, 64, device="cuda", dtype=torch.half); C = torch.empty_like(A)
print("triton", triton.__version__)
for o16 in (False, True):
    h = k[(1,)](A, B, C, OUT16=o16)
    cub = h.asm["cubin"]; f = tempfile.NamedTemporaryFile(suffix=".cubin", delete=False); f.write(cub); f.close()
    sass = subprocess.run(["/usr/local/cuda-13/bin/cuobjdump", "-sass", f.name], capture_output=True, text=True).stdout
    cnt = collections.Counter(re.findall(r"\b(HMMA\.[0-9A-Z.]+|FFMA|HFMA2(?:\.MMA)?)\b", sass))
    err = ((C.float() - (A.float() @ B.float())).norm() / (A.float() @ B.float()).norm()).item()
    print("out_dtype", "fp16" if o16 else "fp32", dict(cnt), "ttgir dot layout:", "mma" if "nvidia_mma" in h.asm["ttgir"] else "blocked/FMA", f"rel_err {err:.2e}")
