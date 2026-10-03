"""Lane K7 (shared with Lane LP): rotation + symmetric re-quantization pipeline for Turing integer tensor-core GEMMs.

Pieces (CPU torch, no GPU needed):
  hadamard(n)                      orthonormal Sylvester Hadamard, n a power of 2
  block_had(x, hb)                 x @ blockdiag(H_hb)^T along the last dim (what k7 act_quant_had does online)
  rotate_weight(W, hb)             W @ blockdiag(H_hb)^T  so that (W H^T)(H x) == W x exactly
  sym_quant(W, bits, group, ...)   symmetric RTN with per-(row,group) MSE clip search -> (codes int8, scale fp32)
                                   group=0 -> per output channel (what the per-token x per-channel W4A4 kernel takes)
  act_quant(x, bits, group, hb)    emulation of the online activation path: block Hadamard, then symmetric per-token
                                   (group=0) or per-group absmax quant; returns the de-quantized fp32 tensor
  ct_pack_sym(codes, scale)        compressed-tensors pack-quantized layout for a SYMMETRIC int4 g128 weight
                                   (uint4b8: stored nibble = code + 8; no zero_point tensor) -> Marlin W4A16/W4A8 loadable
Dims of Qwen3.8-27B linear inputs: 5120 = 2^10*5, per-rank 8704 = 2^9*17, 3072 = 2^10*3 -> any power-of-2 block <= 512
divides every K, so block-Hadamard never needs a non-Sylvester factor.
"""
import math
import torch

_H = {}


def hadamard(n: int, dtype=torch.float32) -> torch.Tensor:
    assert n >= 1 and n & (n - 1) == 0
    if n not in _H:
        h = torch.ones(1, 1, dtype=torch.float64)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _H[n] = h / math.sqrt(n)
    return _H[n].to(dtype)


def block_had(x: torch.Tensor, hb: int) -> torch.Tensor:
    if hb <= 1:
        return x
    K = x.shape[-1]
    H = hadamard(hb, x.dtype).to(x.device)
    return (x.reshape(*x.shape[:-1], K // hb, hb) @ H.T).reshape(x.shape)


def rotate_weight(W: torch.Tensor, hb: int) -> torch.Tensor:
    """y = W x = (W H^T)(H x) for orthonormal block-diagonal H acting on the input (last) dim."""
    return block_had(W, hb)


def sym_quant(W: torch.Tensor, bits: int = 4, group: int = 0, clip_grid=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7),
              qneg: bool = True, chunk: int = 512):
    """Symmetric RTN.  scale = clip*absmax/qmax (clip picked per row/group by min squared error),
    codes in [-(qmax+1 if qneg else qmax), qmax].  Row-chunked for cache locality.
    Returns codes (int8, same shape as W) and scale (fp32, [rows] or [rows, K/group])."""
    qmax = 2 ** (bits - 1) - 1
    qmin = -(qmax + 1) if qneg else -qmax
    R, K = W.shape
    g = K if group == 0 else group
    codes = torch.empty(R, K, dtype=torch.int8, device=W.device)
    scale = torch.empty(R, K // g, device=W.device)
    for r0 in range(0, R, chunk):
        Wg = W[r0:r0 + chunk].float().reshape(-1, K // g, g)
        amax = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        best_err = None
        for c in clip_grid:
            s = amax * (c / qmax)
            q = torch.div(Wg, s).round_().clamp_(qmin, qmax)
            err = q.mul_(s).sub_(Wg).square_().sum(-1, keepdim=True)
            if best_err is None:
                best_err, best_s = err, s
            else:
                upd = err < best_err
                best_err = torch.where(upd, err, best_err); best_s = torch.where(upd, s, best_s)
        codes[r0:r0 + chunk] = torch.div(Wg, best_s).round_().clamp_(qmin, qmax).to(torch.int8).reshape(-1, K)
        scale[r0:r0 + chunk] = best_s.squeeze(-1)
    return codes, (scale.squeeze(-1) if group == 0 else scale)


def dequant(codes: torch.Tensor, scale: torch.Tensor, group: int = 0) -> torch.Tensor:
    scale = scale.to(codes.device)
    R, K = codes.shape
    if group == 0:
        return codes.float() * scale.float().view(R, 1)
    return (codes.float().view(R, K // group, group) * scale.float().unsqueeze(-1)).view(R, K)


def act_quant(x: torch.Tensor, bits: int = 4, group: int = 0, hb: int = 128, clip: float = 1.0) -> torch.Tensor:
    """Emulates the online path: block Hadamard (hb) then symmetric absmax quant per token (group=0) or per group.
    Returns the de-quantized fp32 activations IN THE ROTATED BASIS (pair with rotate_weight(W, hb))."""
    qmax = 2 ** (bits - 1) - 1
    xr = block_had(x.float(), hb)
    K = xr.shape[-1]
    g = K if group == 0 else group
    xg = xr.reshape(*xr.shape[:-1], K // g, g)
    s = (xg.abs().amax(-1, keepdim=True) * clip / qmax).clamp_min(1e-12)
    return (torch.clamp(torch.round(xg / s), -(qmax + 1), qmax) * s).reshape(xr.shape)


_SH = torch.arange(8, dtype=torch.int32) * 4


def ct_pack_sym(codes: torch.Tensor, scale: torch.Tensor):
    """codes int8 [N,K] in [-8,7], scale [N, K/g] -> compressed-tensors pack-quantized symmetric tensors
    {weight_packed int32 [N,K/8], weight_scale fp16 [N,K/g], weight_shape}. nibble = code + 8 (uint4b8)."""
    N, K = codes.shape
    u = (codes.to(torch.int32) + 8).reshape(N, K // 8, 8)
    packed = (u << _SH).sum(-1).to(torch.int32)  # disjoint nibbles -> sum == OR (int32 wraparound is bit-exact)
    return {"weight_packed": packed, "weight_scale": scale.to(torch.float16), "weight_shape": torch.tensor([N, K])}


def gptq_sym(W: torch.Tensor, H: torch.Tensor, bits: int = 4, group: int = 0, blocksize: int = 128,
             percdamp: float = 0.01, act_order: bool = True):
    """GPTQ (Frantar et al. 2022) with symmetric scales, for W [N,K] and Hessian H = X^T X [K,K] in the SAME basis as W
    (pass rotated W and rotated-activation H).  Per-channel (group=0) scales are fixed up front from an MSE-clip search
    on W; group scales (group>0, static groups in the original column order) are fixed per group from the error-updated
    weights when the group starts.  Returns (codes int8 [N,K], scale [N] or [N,K/group])."""
    qmax = 2 ** (bits - 1) - 1
    W = W.clone().float()
    N, K = W.shape
    H = H.clone().float()  # fp32 like reference GPTQ (fp64 tripled RAM: ~7 GB peak on down_proj)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    perm = torch.argsort(torch.diag(H), descending=True) if act_order else torch.arange(K, device=H.device)
    if group == 0:
        _, scale = sym_quant(W, bits, 0)
        sc_col = None
    else:
        scale = torch.zeros(N, K // group, device=W.device)
    W = W[:, perm]
    H = H[perm][:, perm]
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(K, dtype=H.dtype, device=H.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True).float()
    Q = torch.zeros_like(W)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = perm[i1 + i]
            w = W1[:, i]
            if group == 0:
                s = scale
            else:
                g = int(col) // group
                if scale[0, g] == 0:  # first column of this group seen: fix the group scale from current weights
                    cols = (perm[i1 + i:] // group == g).nonzero().squeeze(1) + i1 + i
                    cur = torch.cat([W1[:, i:], W[:, i2:]], 1)[:, cols - (i1 + i)] if True else None
                    amax = cur.abs().amax(1).clamp_min(1e-12)
                    best_s, best_e = amax / qmax, None
                    for c in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75):
                        st = amax * c / qmax
                        e = (torch.clamp(torch.round(cur / st[:, None]), -qmax - 1, qmax) * st[:, None] - cur).pow(2).sum(1)
                        if best_e is None:
                            best_e, best_s = e, st
                        else:
                            u = e < best_e; best_e = torch.where(u, e, best_e); best_s = torch.where(u, st, best_s)
                    scale[:, g] = best_s
                s = scale[:, g]
            q = torch.clamp(torch.round(w / s), -qmax - 1, qmax)
            Q1[:, i] = q
            err = (w - q * s) / Hi[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hi[i, i:].unsqueeze(0)
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    inv = torch.argsort(perm)
    return Q[:, inv].to(torch.int8), scale


def pack_s4(q: torch.Tensor) -> torch.Tensor:
    """codes [R,K] in [-8,7] -> int8 [R,K/2], low nibble = even column (cutlass int4b_t / k7 kernels)."""
    q = q.to(torch.int16)
    return ((q[:, 0::2] & 15) | ((q[:, 1::2] & 15) << 4)).to(torch.uint8).view(torch.int8)


def unpack_s4(p: torch.Tensor) -> torch.Tensor:
    u = p.view(torch.uint8).to(torch.int16)
    lo, hi = u & 15, (u >> 4) & 15
    lo = torch.where(lo > 7, lo - 16, lo); hi = torch.where(hi > 7, hi - 16, hi)
    return torch.stack([lo, hi], -1).reshape(p.shape[0], -1).to(torch.int8)
