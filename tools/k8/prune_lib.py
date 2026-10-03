"""Lane K8 track H: Wanda / SparseGPT(2:4 + joint int4 on a FIXED grid) / GPTQ(intN) on CPU, layer-wise.
All weights W [out,in]; X [tokens,in].  Grid for the sparse+int4 case is the checkpoint's own (scale, zp) per
[row, 128-col group] so the pruned model keeps the deployed int4 lattice (kept values never leave it)."""
import math, torch

GROUP = 128


def hessian(X, damp=0.01):
    """H = 2/N X^T X (+ percdamp * mean diag on diagonal); X [N,n] float32."""
    n = X.shape[1]
    H = torch.zeros(n, n, dtype=torch.float32)
    for s in range(0, X.shape[0], 1024):
        xb = X[s:s + 1024].float()
        H.addmm_(xb.T, xb)
    H.mul_(2.0 / X.shape[0])
    return H


def _hinv_chol(H, damp=0.01):
    H = H.clone()
    n = H.shape[0]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    idx = torch.arange(n)
    H[idx, idx] += damp * torch.mean(torch.diag(H))
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    return torch.linalg.cholesky(Hinv, upper=True), dead


@torch.no_grad()
def sparsegpt(W0, H, n_keep=2, m=4, grid=None, nbits=4, blocksize=128, damp=0.01, prune=True):
    """Returns W' (dense tensor with zeros at pruned). grid: dict(scale [out,ng], zp [out,ng]) fixed int4 lattice
    (w = (q - zp)*scale, q in [0,2^nbits-1]); if None and nbits<16 the grid is fitted per group (min-max) from the
    ALREADY-compensated weights (GPTQ style). prune=False -> pure GPTQ (dense intN)."""
    W = W0.clone().float()
    rows, n = W.shape
    Hinv, dead = _hinv_chol(H, damp)
    W[:, dead] = 0
    qmax = 2 ** nbits - 1
    scale = zp = None
    if grid is not None:
        scale_g, zp_g = grid["scale"].float(), grid["zp"].float()
    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        if prune:
            mask1 = torch.zeros_like(W1, dtype=torch.bool)
        # one 128-col group == one block (blocksize == GROUP) so the grid is per-block
        if nbits < 16:
            if grid is not None:
                g = i1 // GROUP
                scale, zp = scale_g[:, g:g + 1], zp_g[:, g:g + 1]
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]
            if prune and i % m == 0:
                tmp = W1[:, i:i + m] ** 2 / (torch.diag(Hinv1)[i:i + m].reshape(1, -1)) ** 2
                kill = torch.topk(tmp, m - n_keep, dim=1, largest=False)[1]
                mask1[:, i:i + m].scatter_(1, kill, True)
            if nbits < 16:
                if grid is None and i == 0:  # GPTQ: fit grid on current (compensated) group
                    mn = W1.amin(1, keepdim=True).clamp(max=0) if False else W1.amin(1, keepdim=True)
                    mx = W1.amax(1, keepdim=True)
                    scale = ((mx - mn) / qmax).clamp_min(1e-9)
                    zp = torch.round(-mn / scale)
                q = torch.clamp(torch.round(w.unsqueeze(1) / scale + zp), 0, qmax)
                wq = ((q - zp) * scale).flatten()
            else:
                wq = w
            if prune:
                wq = torch.where(mask1[:, i], torch.zeros_like(wq), wq)
            Q1[:, i] = wq
            err = (w - wq) / d
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return W


@torch.no_grad()
def wanda(W0, X, n_keep=2, m=4):
    """keep the n_keep largest |w|*||x_j|| per group of m along the input dim; no weight update."""
    xn = X.float().pow(2).sum(0).sqrt()
    S = W0.abs() * xn.unsqueeze(0)
    rows, n = W0.shape
    Sg = S.reshape(rows, n // m, m)
    kill = torch.topk(Sg, m - n_keep, dim=2, largest=False)[1]
    mask = torch.zeros_like(Sg, dtype=torch.bool)
    mask.scatter_(2, kill, True)
    return torch.where(mask.reshape(rows, n), torch.zeros_like(W0), W0)


@torch.no_grad()
def rtn(W0, nbits):
    rows, n = W0.shape
    Wg = W0.reshape(rows, n // GROUP, GROUP)
    mn, mx = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
    qmax = 2 ** nbits - 1
    scale = ((mx - mn) / qmax).clamp_min(1e-9)
    zp = torch.round(-mn / scale)
    q = torch.clamp(torch.round(Wg / scale + zp), 0, qmax)
    return ((q - zp) * scale).reshape(rows, n)


def rel_err(Wq, W0, X):
    """||(Wq-W0) X^T|| / ||W0 X^T|| over tokens."""
    num = 0.0; den = 0.0
    for s in range(0, X.shape[0], 2048):
        xb = X[s:s + 2048].float()
        d = xb @ (Wq - W0).T
        r = xb @ W0.T
        num += d.pow(2).sum().item(); den += r.pow(2).sum().item()
    return math.sqrt(num / den)
