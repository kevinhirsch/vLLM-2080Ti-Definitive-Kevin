"""CPU emulations of stock W4A8 (per-token, int16 scales) and LP_A8G (per-(row,128), fp16 scales) Marlin."""
import torch


def emulate(t, x, N, K):
    """CPU fp32 emulation of s8xu4(zp) Marlin: per-token int8 act, int16 group scales (per-layer max -> 4096)."""
    sh = torch.arange(8, dtype=torch.int32) * 4
    q = ((t["weight_packed"].unsqueeze(-1) >> sh) & 15).reshape(N, K // 128, 128)
    zp = ((t["weight_zero_point"].unsqueeze(1) >> sh.view(1, 8, 1)) & 15).reshape(N, -1)
    s = t["weight_scale"].half().float(); smax = s.max(); s_int = torch.round(s / smax * 4096)
    w = ((q - zp.unsqueeze(-1)).float() * (s_int * smax / 4096).unsqueeze(-1)).reshape(N, K)
    w16 = ((q - zp.unsqueeze(-1)).float() * s.unsqueeze(-1)).reshape(N, K)
    xf = x.float(); amax = xf.abs().amax(-1, keepdim=True); sc = amax / 127
    xq = torch.clamp(torch.round(xf / sc), -127, 127) * sc
    return xq @ w.T, xf @ w16.T


def emulate_g(t, x, N, K):
    """CPU emulation of the LP_A8G kernel: per-(row,128) int8 act, fp16 weight group scales."""
    sh = torch.arange(8, dtype=torch.int32) * 4
    q = ((t["weight_packed"].unsqueeze(-1) >> sh) & 15).reshape(N, K // 128, 128)
    zp = ((t["weight_zero_point"].unsqueeze(1) >> sh.view(1, 8, 1)) & 15).reshape(N, -1)
    w = ((q - zp.unsqueeze(-1)).float() * t["weight_scale"].half().float().unsqueeze(-1)).reshape(N, K)
    xg = x.float().view(x.shape[0], -1, 128); sc = xg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
    return (torch.clamp(torch.round(xg / sc), -127, 127) * sc).view(x.shape[0], K) @ w.T




def emulate_e(t, x, N, K, emax, level):
    """CPU emulation of LP_A8E: act int8 scale amax/127 * 2^-e[row,g] (e in [0,emax]), int weight scales with `level` levels."""
    sh = torch.arange(8, dtype=torch.int32) * 4
    q = ((t["weight_packed"].unsqueeze(-1) >> sh) & 15).reshape(N, K // 128, 128)
    zp = ((t["weight_zero_point"].unsqueeze(1) >> sh.view(1, 8, 1)) & 15).reshape(N, -1)
    s = t["weight_scale"].half().float(); smax = s.max(); s_int = torch.round(s / smax * level)
    w = ((q - zp.unsqueeze(-1)).float() * (s_int * smax / level).unsqueeze(-1)).reshape(N, K)
    xf = x.float(); M = xf.shape[0]; xg = xf.view(M, -1, 128)
    amax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-8); gmax = xg.abs().amax(-1).clamp(min=1e-30)
    e = torch.clamp(torch.floor(torch.log2(amax / gmax)), 0, emax)
    sc = (amax / 127) / torch.pow(2.0, e)
    return (torch.clamp(torch.round(xg / sc.unsqueeze(-1)), -127, 127) * sc.unsqueeze(-1)).view(M, K) @ w.T
