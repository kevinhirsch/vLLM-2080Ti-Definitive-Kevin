#!/usr/bin/env python
"""Lane K8 track G: how much precision does the Gated-DeltaNet SSM state need when STORED (cache / checkpoint page)
while the recurrence math stays fp32?  Teacher-forced on captured q,k,v,g,beta of a real 6.4K-token flightrec prefix
(capture.py).  Compares storage formats (fp16 / e4m3 / e5m2 / intN with per-row-or-col scale) x rounding (RNE / SR)
x regime (checkpoint-only requant every P tokens = chunked prefill / align blocks;  every-token requant = decode).

State S[h,K,V]:  S <- S*exp(g);  d = beta*(v - k^T S);  S <- S + k d^T;  o = (q*K^-.5)^T S.
Metric: relative L2 error of o vs the fp32 run, per token bin; also state rel err.
Usage: python state_quant.py --tag a --layers 0,1,2,6 --out res.json
"""
import os, sys, json, time, argparse, math
import torch

CAP = "/home/kevin/projects/lanes/k8/cap"
dev = os.environ.get("K8_DEV", "cpu")  # GPUs are full (prod trial): CPU only
torch.set_num_threads(int(os.environ.get("K8_THREADS", "6")))


def load_layer(tag, L, T=None):
    d = torch.load(f"{CAP}/{tag}_gdn_L{L:02d}.pt")
    q, k, v = d["q"].float(), d["k"].float(), d["v"].float()      # q,k: [1,T,16,128] v: [1,T,48,128]
    q, k = q.repeat_interleave(3, dim=2), k.repeat_interleave(3, dim=2)
    l2 = lambda x: x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)
    q, k = l2(q) * (128 ** -0.5), l2(k)
    return [t[0, :T].to(dev) for t in (q, k, v, d["g"].float(), d["beta"].float())]


# ---- storage formats -------------------------------------------------------------------------------------------
def _round(x, step, sr, gen):
    """round x to multiples of step (tensor) : RNE (ties-to-even via torch.round) or stochastic (unbiased)."""
    y = x / step
    if sr:
        u = torch.rand(y.shape, device=y.device, generator=gen)
        return torch.floor(y + u) * step
    return torch.round(y) * step


def make_quantizer(fmt, axis, sr, gen):
    """returns f(S [H,K,V] fp32) -> dequantized fp32 after storing in `fmt`. axis: 'row' scale per k-row (over V),
    'col' per v-col (over K), 'g16' per 16-element block along V."""
    if fmt == "fp32":
        return lambda S: S
    if fmt == "fp16":
        if not sr:
            return lambda S: S.half().float()
        def f(S):  # S4/wt-s4sr definition: add 13 random mantissa bits to the fp32 pattern, truncate, then (exact) cast
            r = torch.randint(0, 8192, S.shape, device=S.device, generator=gen, dtype=torch.int32)
            b = (S.contiguous().view(torch.int32) + r) & ~0x1FFF
            return b.view(torch.float32).half().float()
        return f
    if fmt == "bf16":
        def f(S):
            e = torch.floor(torch.log2(S.abs().clamp_min(2.0 ** -126)))
            return _round(S, torch.exp2(e - 7), sr, gen)
        return f

    def scale_of(S, qmax):
        if axis == "row":
            a = S.abs().amax(-1, keepdim=True)
        elif axis == "col":
            a = S.abs().amax(-2, keepdim=True)
        elif axis.startswith("g"):
            n = int(axis[1:]); H, K, V = S.shape
            a = S.reshape(H, K, V // n, n).abs().amax(-1, keepdim=True).expand(H, K, V // n, n).reshape(H, K, V)
        else:
            raise ValueError(axis)
        s = (a / qmax).clamp_min(1e-20)
        return s.to(torch.bfloat16).float()  # scale stored as bf16 (range-safe); values re-clamped to +-qmax*scale

    if fmt.startswith("int"):
        bits = int(fmt[3:]); qmax = 2 ** (bits - 1) - 1
        def f(S):
            s = scale_of(S, qmax)
            y = _round(S, s, sr, gen)
            return torch.clamp(y, -qmax * s, qmax * s)
        return f
    if fmt in ("e4m3", "e5m2", "e3m4", "e2m5"):
        mant, emin, fmax = {"e4m3": (3, -6, 448.0), "e5m2": (2, -14, 57344.0), "e3m4": (4, -2, 31.0), "e2m5": (5, 0, 7.75)}[fmt]
        def f(S):
            s = scale_of(S, fmax)
            z = S / s
            e = torch.floor(torch.log2(z.abs().clamp_min(2.0 ** emin)))
            y = _round(z, torch.exp2(e - mant), sr, gen)
            return torch.clamp(y, -fmax, fmax) * s
        return f
    raise ValueError(fmt)


# ---- recurrence --------------------------------------------------------------------------------------------------
@torch.no_grad()
def run(layers, variants, T0, ckpt_tag, log=print, Tmax=None, seed=0):
    """layers: list of loaded (q,k,v,g,beta) tuples; variants: list of dict(name,fmt,axis,sr,P) ; T0 = first decode step.
    Returns per variant: rel err of o in bins."""
    nL = len(layers)
    q = torch.stack([l[0] for l in layers], 1)  # [T, nL, H, K]
    k = torch.stack([l[1] for l in layers], 1)
    v = torch.stack([l[2] for l in layers], 1)
    g = torch.stack([l[3] for l in layers], 1).exp()  # [T,nL,H]
    beta = torch.stack([l[4] for l in layers], 1)
    T = q.shape[0]
    H, K, V = q.shape[2], q.shape[3], v.shape[3]
    gens = {}
    qs, states = [], []
    for vr in variants:
        gen = torch.Generator(device=dev); gen.manual_seed(seed + 1)
        qs.append(make_quantizer(vr["fmt"], vr.get("axis", "row"), vr.get("sr", False), gen))
        states.append(torch.zeros(nL * H, K, V, device=dev))
    def _cad(c, D):
        if c == 1: return set(range(D))
        out, t, pat, i = set(), 0, [3, 2], 0
        while t < D:
            t += pat[i % 2]; out.add(min(t, D) - 1); i += 1
        return out
    decsets = [_cad(vr.get("cad", 1), T - T0) for vr in variants]
    Sref = torch.zeros(nL * H, K, V, device=dev)
    nv = len(variants)
    err_num = [torch.zeros(T, nL, device=dev) for _ in range(nv)]
    ref_den = torch.zeros(T, nL, device=dev)
    sterr = [torch.zeros(T, nL, device=dev) for _ in range(nv)]
    stden = torch.zeros(T, nL, device=dev)

    def step(S, qt, kt, vt, gt, bt):
        S.mul_(gt.reshape(-1, 1, 1))
        kS = torch.bmm(kt.unsqueeze(1), S).squeeze(1)
        d = bt.reshape(-1, 1) * (vt - kS)
        S.baddbmm_(kt.unsqueeze(2), d.unsqueeze(1))
        o = torch.bmm(qt.unsqueeze(1), S).squeeze(1)
        return S, o

    t0 = time.time()
    for t in range(T):
        args = [x[t].reshape(nL * H, -1) if x.dim() == 4 else x[t].reshape(nL * H) for x in (q, k, v, g, beta)]
        Sref, oref = step(Sref, *args)
        ref_den[t] = oref.pow(2).sum(-1).reshape(nL, H).sum(-1)
        stden[t] = Sref.pow(2).sum((-1, -2)).reshape(nL, H).sum(-1)
        for i, vr in enumerate(variants):
            S, o = step(states[i], *args)
            err_num[i][t] = (o - oref).pow(2).sum(-1).reshape(nL, H).sum(-1)
            sterr[i][t] = (S - Sref).pow(2).sum((-1, -2)).reshape(nL, H).sum(-1)  # before requant (drift incl.)
            P = vr.get("P", 1856)
            if (t >= T0 and (t - T0) in decsets[i]) or (t < T0 and (t + 1) % P == 0):
                S = qs[i](S)
            states[i] = S
        if t % 1000 == 999:
            log(f"  t={t+1} {time.time()-t0:.0f}s")
    out = {}
    edges = sorted(set([0, 1856, 3712, 5568, T0, T]))
    edges = [e for e in edges if e <= T]
    for i, vr in enumerate(variants):
        rows = {}
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            e = (err_num[i][a:b].sum(0) / ref_den[a:b].sum(0)).sqrt()  # per layer
            s = (sterr[i][a:b].sum(0) / stden[a:b].sum(0)).sqrt()
            rows[f"{a}-{b}"] = {"o_rel": e.cpu().tolist(), "S_rel": s.cpu().tolist()}
        out[vr["name"]] = rows
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a")
    ap.add_argument("--layers", default="0,1,2,6")
    ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--decode", type=int, default=512)
    ap.add_argument("--sweep", default="main")
    ap.add_argument("--out", default="/home/kevin/projects/lanes/k8/state_quant_res.json")
    a = ap.parse_args()
    Ls = [int(x) for x in a.layers.split(",")]
    lay = [load_layer(a.tag, L, a.T or None) for L in Ls]
    T = lay[0][0].shape[0]
    T0 = T - a.decode
    sweeps = {
        "main": [dict(name="fp16-RNE", fmt="fp16", P=1856), dict(name="fp16-RNE-step", fmt="fp16", P=1856, cad="step"),
                 dict(name="e4m3-row-RNE", fmt="e4m3", axis="row", P=928), dict(name="e4m3-row-RNE-step", fmt="e4m3", axis="row", P=928, cad="step"),
                 dict(name="e3m4-row-RNE", fmt="e3m4", axis="row", P=928), dict(name="e3m4-row-RNE-step", fmt="e3m4", axis="row", P=928, cad="step"),
                 dict(name="e2m5-row-RNE-step", fmt="e2m5", axis="row", P=928, cad="step"),
                 dict(name="int8-row-RNE-step", fmt="int8", axis="row", P=928, cad="step"),
                 dict(name="e3m4-row-SR-step", fmt="e3m4", axis="row", sr=True, P=928, cad="step"),
                 dict(name="e3m4-col-RNE-step", fmt="e3m4", axis="col", P=928, cad="step"),
                 dict(name="e3m4-g16-RNE-step", fmt="e3m4", axis="g16", P=928, cad="step")],
    }
    res = run(lay, sweeps[a.sweep], T0, a.tag)
    res["_meta"] = {"layers": Ls, "T": T, "T0": T0}
    json.dump(res, open(a.out, "w"), indent=1)
    for n, rows in res.items():
        if n.startswith("_"): continue
        print(n.ljust(16), "  ".join(f"{k}: o={sum(r['o_rel'])/len(r['o_rel']):.4f} S={sum(r['S_rel'])/len(r['S_rel']):.4f}" for k, r in rows.items()))
