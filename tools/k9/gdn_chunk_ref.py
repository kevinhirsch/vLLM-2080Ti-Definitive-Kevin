#!/usr/bin/env python
"""Lane K9 (L102): reference implementations of the gated delta rule forward with the FlashQLA-legacy conventions.
  state M [K=dk, V=dv] (stored as [N, Hv, V, K]); per token: kv = M^T k; delta = beta (v - e^g kv); M = e^g M + k delta^T;
  o = scale * M^T q.  q, k: [T, Hk, D] (l2-normalized upstream), v: [T, Hv, D], g (log decay), beta: [T, Hv].
naive(): token recurrence.  chunked(): UT/WY chunk form (C=64), the algorithm the CUDA kernel implements:
  G = cumsum(g) in chunk; L_ji = beta_j e^{G_j-G_i} k_j.k_i (i<j); R = beta (V - e^G (K M0)); U = (I+L)^{-1} R (forward subst.)
  O = scale (e^G (Q M0) + (e^{G_t-G_i} q_t.k_i)_{i<=t} U);  M1 = e^{G_C} M0 + K^T (e^{G_C-G} U)."""
import torch


def naive(q, k, v, g, beta, scale, state):  # state [Hv, V, K]; single sequence
    T, Hv, D = v.shape; rep = Hv // q.shape[1]
    M = state.transpose(-1, -2).clone().double()  # [Hv, K, V]
    o = torch.empty(T, Hv, D, dtype=torch.float64)
    for t in range(T):
        kt = k[t].double().repeat_interleave(rep, 0); qt = q[t].double().repeat_interleave(rep, 0)  # [Hv, K]
        gt = g[t].double().exp()[:, None]; bt = beta[t].double()[:, None]
        kv = torch.einsum("hkv,hk->hv", M, kt)
        delta = bt * (v[t].double() - gt * kv)
        M = gt[..., None] * M + kt[:, :, None] * delta[:, None, :]
        o[t] = scale * torch.einsum("hkv,hk->hv", M, qt)
    return o, M.transpose(-1, -2)


def chunked(q, k, v, g, beta, scale, state, C=64):
    T, Hv, D = v.shape; rep = Hv // q.shape[1]
    M = state.transpose(-1, -2).clone().double()  # [Hv, K, V]
    o = torch.empty(T, Hv, D, dtype=torch.float64)
    for s in range(0, T, C):
        e = min(T, s + C); n = e - s
        K = k[s:e].double().repeat_interleave(rep, 1).transpose(0, 1)  # [Hv, n, K]
        Q = q[s:e].double().repeat_interleave(rep, 1).transpose(0, 1)
        V = v[s:e].double().transpose(0, 1)
        G = g[s:e].double().transpose(0, 1).cumsum(-1)  # [Hv, n]
        B = beta[s:e].double().transpose(0, 1)
        dG = G[:, :, None] - G[:, None, :]  # [Hv, j, i] = G_j - G_i
        tril = torch.tril(torch.ones(n, n, dtype=torch.bool), -1)
        L = torch.where(tril, B[:, :, None] * dG.exp() * (K @ K.transpose(-1, -2)), torch.zeros((), dtype=torch.float64))
        R = B[..., None] * (V - G.exp()[..., None] * (K @ M))
        U = torch.linalg.solve_triangular(torch.eye(n, dtype=torch.float64) + L, R, upper=False, unitriangular=True)
        P = torch.where(torch.tril(torch.ones(n, n, dtype=torch.bool)), dG.exp() * (Q @ K.transpose(-1, -2)), torch.zeros((), dtype=torch.float64))
        o[s:e] = (scale * (G.exp()[..., None] * (Q @ M) + P @ U)).transpose(0, 1)
        M = G[:, -1:].exp()[..., None] * M + K.transpose(-1, -2) @ ((G[:, -1:] - G).exp()[..., None] * U)
    return o, M.transpose(-1, -2)


if __name__ == "__main__":
    torch.manual_seed(0)
    T, Hk, Hv, D = 200, 2, 6, 128
    q = torch.nn.functional.normalize(torch.randn(T, Hk, D), dim=-1); k = torch.nn.functional.normalize(torch.randn(T, Hk, D), dim=-1)
    v = torch.randn(T, Hv, D); g = -torch.rand(T, Hv) * 0.2; beta = torch.rand(T, Hv); st = torch.randn(Hv, D, D) * 0.1
    o1, s1 = naive(q, k, v, g, beta, D ** -0.5, st); o2, s2 = chunked(q, k, v, g, beta, D ** -0.5, st)
    print("out rel", ((o1 - o2).norm() / o1.norm()).item(), "state rel", ((s1 - s2).norm() / s1.norm()).item())
