# SPDX-License-Identifier: Apache-2.0
"""Lane LP W8X: large-M prefill GEMM = int8 IMMA W8A8 (vLLM CUTLASS sm75 c2x, ~2x Marlin W4A16 per K7) on an int8 weight
scratch expanded just-in-time from the resident Marlin-layout W4 (asym, g128) weights.

Why it is safe: group scales of this checkpoint vary little inside an output channel (max/min group scale: median 1.3-1.4,
p99 2-3), so a per-output-channel int8 grid represents the W4 grid with rms error 0.026-0.031 of an int4 step (int4 RTN's own
error is 0.289 step) -> +~1% error variance (measured, layers 0/3/40). Activations: per-token int8 (vLLM scaled_int8_quant);
scoped by default to the layers whose inputs quantize well per token (mlp.gate_up_proj, GDN in_proj_*: 28-36 dB) - NOT
down_proj / out_proj / o_proj (15-25 dB). Decode (M < threshold) stays on Marlin W4A16: zero decode change.
Env: VLLM_LP_W8X=1, VLLM_LP_W8X_MIN_M (2048), VLLM_LP_W8X_ONLY (regex, default 'gate_up_proj|in_proj_qkvz|in_proj_qkv|in_proj_z').
Scratch: one int8 [N, K] buffer per device, sized to the largest enabled layer (gate_up per rank: 17408 x 5120 = 89 MB).
"""
import os
import re

import numpy as np
import torch
import triton
import triton.language as tl


def enabled() -> bool:
    return os.environ.get("VLLM_LP_W8X", "0") == "1"


def min_m() -> int:
    return int(os.environ.get("VLLM_LP_W8X_MIN_M", "2048"))


def allowed(prefix: str | None) -> bool:
    skip = os.environ.get("VLLM_LP_W8X_SKIP", r"mtp|lm_head|embed|visual")
    if not prefix or (skip and re.search(skip, prefix)):
        return False
    pat = os.environ.get("VLLM_LP_W8X_ONLY", r"gate_up_proj|in_proj_qkvz|in_proj_qkv|in_proj_z")
    return re.search(pat, prefix) is not None


_INVPERM = {}


def _invperm(device):
    if device not in _INVPERM:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import get_weight_perm

        perm = get_weight_perm(4).to(torch.int64)          # new[j] = old[perm[j]]
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel())              # inv[old] = j
        _INVPERM[device] = inv.to(torch.int32).to(device)
    return _INVPERM[device]


def _unpack_rows_int32(packed):
    p = packed.to(torch.int64) & 0xFFFFFFFF
    return torch.stack([(p >> (4 * i)) & 0xF for i in range(8)], dim=-1).reshape(packed.shape[0], packed.shape[1] * 8)


def unpermute_scales_zp(marlin_s, marlin_zp, size_k, size_n, group_size=128):
    """(tiny, once per layer) Marlin-permuted fp16 scales [G,N] and packed zero points [G,N/8] -> dense [G,N] fp32 / uint8."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils import get_scale_perms

    scale_perm, _ = get_scale_perms()
    inv = torch.from_numpy(np.argsort(np.array(scale_perm))).to(marlin_s.device)
    s = marlin_s.reshape(-1, len(scale_perm))[:, inv].reshape(-1, size_n).float()
    zp = _unpack_rows_int32(marlin_zp)
    undo = torch.from_numpy(np.argsort(np.array([0, 2, 4, 6, 1, 3, 5, 7]))).to(zp.device)
    zp = zp.reshape(-1, 8)[:, undo].reshape(size_k // group_size, size_n)
    zp = zp.reshape(-1, len(scale_perm))[:, inv].reshape(size_k // group_size, size_n)
    return s.contiguous(), zp.to(torch.uint8).contiguous()


@triton.jit
def _expand_kernel(q_ptr, ms_ptr, mzp_ptr, sch_ptr, inv_ptr, st_ptr, zt_ptr, out_ptr, K, N, BK: tl.constexpr):
    # one program = BK(=128 = one group) x 64 output channels, straight from the resident Marlin tensors:
    #   codes: Marlin weight layout (16x16 tiles, 1024-code permutation over 4 n-tiles), inv_ptr = inverse weight perm
    #   scale: Marlin-permuted fp16 [G, N]; s[g, n] = ms[g*N + (n//64)*64 + st[n%64]]
    #   zero point: Marlin-permuted packed int32 [G, N/8]; nibble idx = (n//64)*64 + zt[n%64] of row g
    # out = int8 [N, K] (K-contiguous = CUTLASS column-major B), value = round((code - zp) * s / s_ch[n])
    pk = tl.program_id(0)
    pn = tl.program_id(1)
    kk = tl.arange(0, BK)
    nn = tl.arange(0, 64)
    kt = pk * (BK // 16) + kk // 16
    k_in = kk % 16
    old = (nn[:, None] // 16) * 256 + k_in[None, :] * 16 + (nn[:, None] % 16)
    j = tl.load(inv_ptr + old)
    n_tiles = N // 64
    word = tl.load(q_ptr + (kt[None, :] * n_tiles + pn) * 128 + j // 8)
    code = (word >> ((j % 8) * 4)) & 0xF
    n = pn * 64 + nn
    g = pk
    sidx = g * N + pn * 64 + tl.load(st_ptr + nn)
    sv = tl.load(ms_ptr + sidx).to(tl.float32)
    zi = pn * 64 + tl.load(zt_ptr + nn)
    zw = tl.load(mzp_ptr + g * (N // 8) + zi // 8)
    z = ((zw >> ((zi % 8) * 4)) & 0xF).to(tl.float32)
    r = sv / tl.load(sch_ptr + n)
    w = (code.to(tl.float32) - z[:, None]) * r[:, None]
    w = tl.where(w >= 0, tl.floor(w + 0.5), tl.ceil(w - 0.5))   # round half away from zero
    w = tl.minimum(tl.maximum(w, -127.0), 127.0)
    k = pk * BK + kk
    tl.store(out_ptr + n[:, None] * K + k[None, :], w.to(tl.int8))


_TABLES = {}


def _tables(device):
    if device not in _TABLES:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import get_scale_perms

        scale_perm, _ = get_scale_perms()
        inv = np.argsort(np.array(scale_perm))                    # 64
        undo = np.argsort(np.array([0, 2, 4, 6, 1, 3, 5, 7]))
        zt = 8 * (inv // 8) + undo[inv % 8]
        _TABLES[device] = (torch.tensor(inv, dtype=torch.int32, device=device), torch.tensor(zt, dtype=torch.int32, device=device))
    return _TABLES[device]


class W8XState:
    """Per-layer: only s_ch [N] fp32 is new memory (~70 KB for gate_up per rank); everything else is read from the Marlin tensors."""

    def __init__(self, marlin_q, marlin_s, marlin_zp, size_k, size_n):
        s, _ = unpermute_scales_zp(marlin_s, marlin_zp, size_k, size_n)
        self.s_ch = (s.amax(0) * 15.0 / 127.0).reshape(1, -1).contiguous()   # [1, N] fp32 (CUTLASS scale_b)
        self.q, self.ms, self.mzp = marlin_q, marlin_s, marlin_zp
        self.K, self.N = size_k, size_n
        # reference-only (tests): dense ratio / zp
        self._s_dense = None


_SCRATCH = {}


def scratch(device, numel):
    buf = _SCRATCH.get(device)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, dtype=torch.int8, device=device)
        _SCRATCH[device] = buf
    return buf[:numel]


def expand_raw(q, ms, mzp, s_ch, K, N, out):
    st_t, zt_t = _tables(q.device)
    _expand_kernel[(K // 128, N // 64)](q, ms, mzp, s_ch, _invperm(q.device), st_t, zt_t, out, K, N, BK=128)
    return out


def expand(st: W8XState, out=None):
    out = out if out is not None else torch.empty((st.N, st.K), dtype=torch.int8, device=st.q.device)
    return expand_raw(st.q, st.ms, st.mzp, st.s_ch, st.K, st.N, out)


def gemm(x: torch.Tensor, st: W8XState) -> torch.Tensor:
    return w8x_gemm_raw(x, st.q, st.ms, st.mzp, st.s_ch, st.K, st.N)


def w8x_gemm_raw(x, q, ms, mzp, s_ch, K, N):
    from vllm import _custom_ops as ops

    w8 = expand_raw(q, ms, mzp, s_ch, K, N, scratch(x.device, N * K).view(N, K))
    xq, xs, _ = ops.scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
    return ops.cutlass_scaled_mm(xq, w8.t(), xs, s_ch, torch.float16)


@torch.library.custom_op("lp::w4_prefill_gemm", mutates_args=())
def w4_prefill_gemm(x: torch.Tensor, q: torch.Tensor, ms: torch.Tensor, mzp: torch.Tensor, s_ch: torch.Tensor,
                    workspace: torch.Tensor, size_k: int, size_n: int, threshold: int) -> torch.Tensor:
    """M >= threshold: W8X (int8 scratch + CUTLASS int8). Else: the unchanged Marlin W4A16 call."""
    if x.shape[0] >= threshold:
        return w8x_gemm_raw(x, q, ms, mzp, s_ch, size_k, size_n)
    from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_gptq_marlin_linear
    from vllm.scalar_type import scalar_types

    return apply_gptq_marlin_linear(input=x, weight=q, weight_scale=ms, weight_zp=mzp, workspace=workspace,
                                    wtype=scalar_types.uint4, output_size_per_partition=size_n,
                                    input_size_per_partition=size_k, bias=None)


@w4_prefill_gemm.register_fake
def _(x, q, ms, mzp, s_ch, workspace, size_k, size_n, threshold):
    return x.new_empty((x.shape[0], size_n), dtype=torch.float16)
