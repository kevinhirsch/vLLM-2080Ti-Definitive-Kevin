"""Lane K1 shared helpers: GPU safety guard, FlashInfer reference wrapper, fp32 reference attention."""

from __future__ import annotations

import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "vllm", "v1", "attention", "ops"))

import fa75_prefill as K1  # noqa: E402

# Never let this process hold more than this many bytes of tensors (the live engine shares the card).
CAP_BYTES = int(os.getenv("K1_CAP_MB", "260")) * 1024 * 1024
MIN_FREE_MB = int(os.getenv("K1_MIN_FREE_MB", "420"))


def preflight() -> None:
    """No CUDA call before this. The shared estate gate (~/projects/lanes/windows/gpuok.sh): no GPU-busy signal
    (windows, boots), no planned offline, engine healthy, enough free VRAM on the physical GPU we will use."""
    phys = (os.getenv("CUDA_VISIBLE_DEVICES") or "0").split(",")[0] or "0"
    need = int(os.getenv("K1_CAP_MB", "260")) + 250  # tensors + CUDA context
    gate = os.path.expanduser("~/projects/lanes/windows/gpuok.sh")
    r = subprocess.run([gate, phys, str(need)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"K1 guard (gpuok.sh {phys} {need}): stay off the GPU: {r.stdout.strip()} {r.stderr.strip()}")


def gpu_guard(dev: int = 0) -> None:
    """Refuse to run unless the card has headroom and no engine window/boot is in progress; cap the allocator."""
    preflight()
    free, total = torch.cuda.mem_get_info(dev)
    if free < MIN_FREE_MB * 1024 * 1024:
        raise SystemExit(f"K1 guard: only {free / 2**20:.0f} MiB free on cuda:{dev}, need {MIN_FREE_MB}")
    torch.cuda.set_per_process_memory_fraction(CAP_BYTES / total, dev)


def smi_free() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,clocks.sm,utilization.gpu", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip().replace("\n", " | ")


_FI_WS = None
_FI_W = None


def flashinfer_run(q, k, v, scale, causal, return_lse=False):
    """The production call: BatchPrefillWithRaggedKVCacheWrapper fa2, single request."""
    global _FI_WS, _FI_W
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

    if _FI_WS is None:
        _FI_WS = torch.empty(int(os.getenv("K1_FI_WS_MB", "48")) * 1024 * 1024, dtype=torch.uint8, device=q.device)
        _FI_W = BatchPrefillWithRaggedKVCacheWrapper(_FI_WS, "NHD", backend="fa2")
    Tq, Hq, D = q.shape
    Tkv, Hk, _ = k.shape
    w = _FI_W
    qo = torch.tensor([0, Tq], dtype=torch.int32)
    kvp = torch.tensor([0, Tkv], dtype=torch.int32)
    w.plan(
        qo,
        kvp,
        Hq,
        Hk,
        D,
        causal=causal,
        sm_scale=scale,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    return w, (lambda: w.run(q, k, v, return_lse=return_lse))


def ref_attention(q, k, v, scale, causal, chunk_heads=1):
    """fp32 reference, per head (bounded memory). Returns out fp32 [Tq,Hq,D], lse fp32 [Tq,Hq] (natural log)."""
    Tq, Hq, D = q.shape
    Tkv, Hk, _ = k.shape
    g = Hq // Hk
    out = torch.empty(Tq, Hq, D, dtype=torch.float32, device=q.device)
    lse = torch.empty(Tq, Hq, dtype=torch.float32, device=q.device)
    cols = torch.arange(Tkv, device=q.device)[None, :]
    RC = max(1, (4 * 1024 * 1024) // (4 * Tkv))  # rows per chunk: ~4 MB of scores
    for h in range(Hq):
        kh = k[:, h // g].float()
        vh = v[:, h // g].float()
        for r0 in range(0, Tq, RC):
            r1 = min(Tq, r0 + RC)
            s = (q[r0:r1, h].float() @ kh.T) * scale
            if causal:
                rows = torch.arange(r0, r1, device=q.device)[:, None] + (Tkv - Tq)
                s.masked_fill_(cols > rows, float("-inf"))
            lse[r0:r1, h] = torch.logsumexp(s, dim=-1)
            out[r0:r1, h] = torch.softmax(s, dim=-1) @ vh
            del s
    return out, lse


if not os.getenv("K1_NO_PREFLIGHT"):
    preflight()


def ref_attention_long(q, k, v, scale, causal, kc=16384):
    """fp32 reference streaming over key chunks (online softmax): bounded memory at 128K+ keys."""
    Tq, Hq, D = q.shape
    Tkv, Hk, _ = k.shape
    g = Hq // Hk
    out = torch.empty(Tq, Hq, D, dtype=torch.float32, device=q.device)
    lse = torch.empty(Tq, Hq, dtype=torch.float32, device=q.device)
    rows = torch.arange(Tq, device=q.device)[:, None] + (Tkv - Tq)
    for h in range(Hq):
        qh = q[:, h].float()
        m = torch.full((Tq,), float("-inf"), device=q.device)
        l = torch.zeros(Tq, device=q.device)
        acc = torch.zeros(Tq, D, device=q.device)
        for c0 in range(0, Tkv, kc):
            c1 = min(Tkv, c0 + kc)
            s = (qh @ k[c0:c1, h // g].float().T) * scale
            if causal:
                s.masked_fill_(torch.arange(c0, c1, device=q.device)[None, :] > rows, float("-inf"))
            mn = torch.maximum(m, s.max(-1).values)
            a = torch.exp(m - mn)
            p = torch.exp(s - mn[:, None])
            l = l * a + p.sum(-1)
            acc = acc * a[:, None] + p @ v[c0:c1, h // g].float()
            m = mn
            del s, p
        out[:, h] = acc / l[:, None]
        lse[:, h] = m + torch.log(l)
    return out, lse
