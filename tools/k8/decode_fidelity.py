#!/usr/bin/env python
"""Lane K8 track G end-to-end test: "restore from a stored checkpoint, then decode D tokens" with the GDN recurrent
state held in a candidate STORAGE format.  fp32 CPU reference (U2b harness math), all 64 layers, V variants batched.

For every GDN layer: S0 = Q_variant(S_tp)  (the cached checkpoint, quantized once), then the D decode tokens run the
exact recurrence in fp32 but the state is requantized (stored) every `cad` tokens (cad=1: every token = worst case;
'step': after 3,2,3,2.. tokens = MTP3 accepted-length cadence).  Attention layers use the fp32 prefix K/V from capture.py.
Compare next-token distributions per position with the fp32-state variant (variant 0).
Usage: python decode_fidelity.py --tag a --variants fp32,fp16-RNE-1,... --out res.json [--layers-limit N]"""
import os, sys, json, time, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/home/kevin/projects/lanes/u2-quant"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
import ref_dump as R
import gpu_linear; gpu_linear.install()
from state_quant import make_quantizer

CAP = "/home/kevin/projects/lanes/k8/cap"


def parse_variant(name):
    """fmt-axis-RNE|SR-cad   e.g. fp32 | fp16-RNE-1 | e4m3-row-RNE-step | int8-row-SR-1"""
    if name == "fp32":
        return dict(name=name, fmt="fp32", axis="row", sr=False, cad="1")
    p = name.split("-")
    if p[0] in ("fp16", "bf16"):
        fmt, axis, rnd, cad = p[0], "row", p[1], p[2]
    else:
        fmt, axis, rnd, cad = p[0], p[1], p[2], p[3]
    return dict(name=name, fmt=fmt, axis=axis, sr=(rnd == "SR"), cad=cad)


def cadence(cad, D):
    """set of decode step indices (0-based, after finishing token t) at which the state is stored."""
    if cad == "1":
        return set(range(D))
    if cad == "none":      # restore-only: quantize the cached checkpoint once, keep fp32 afterwards
        return set()
    if cad == "step":
        out, t, pat, i = set(), 0, [3, 2], 0
        while t < D:
            t += pat[i % 2]; out.add(min(t, D) - 1); i += 1
        return out
    raise ValueError(cad)


def gdn_dec(h, W, Q, S0, tail, cads, B, D):
    """h [B*D,5120] normed. S0 [48,128,128] fp32 prefix state; tail [conv_dim,3]. Q: per-variant quantizer fn."""
    key_dim, value_dim = R.LK_HEADS * R.LK_DIM, R.LV_HEADS * R.LV_DIM
    qkv = F.linear(h, W["in_proj_qkv"]).view(B, D, -1).transpose(1, 2)           # [B,conv,D]
    z = F.linear(h, W["in_proj_z"]).view(B, D, R.LV_HEADS, R.LV_DIM)
    b = F.linear(h, W["in_proj_b"]).view(B, D, R.LV_HEADS)
    a = F.linear(h, W["in_proj_a"]).view(B, D, R.LV_HEADS)
    conv_w = W["conv1d"].squeeze(1).unsqueeze(1)
    inp = torch.cat([tail.unsqueeze(0).expand(B, -1, -1), qkv], 2)               # [B,conv,3+D]
    qkv = F.silu(F.conv1d(inp, conv_w, None, padding=0, groups=inp.shape[1]))     # [B,conv,D]
    qkv = qkv.transpose(1, 2)
    q, k, v = torch.split(qkv, [key_dim, key_dim, value_dim], dim=-1)
    q = q.reshape(B, D, R.LK_HEADS, R.LK_DIM).repeat_interleave(3, dim=2)
    k = k.reshape(B, D, R.LK_HEADS, R.LK_DIM).repeat_interleave(3, dim=2)
    v = v.reshape(B, D, R.LV_HEADS, R.LV_DIM)
    l2 = lambda x: x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)
    q, k = l2(q) * (R.LK_DIM ** -0.5), l2(k)
    beta = b.sigmoid()
    g = (-W["A_log"].float().exp() * F.softplus(a.float() + W["dt_bias"].float())).exp()   # decay
    outs = []
    dv = gpu_linear._dev
    for bi in range(B):
        S = Q[bi](S0.clone().to(dv)).contiguous()                                 # restore from stored checkpoint
        kb, qb, vb, gb, bb = (x[bi].to(dv) for x in (k, q, v, g, beta))
        ob = []
        for t in range(D):
            S.mul_(gb[t].reshape(-1, 1, 1))
            kS = torch.bmm(kb[t].unsqueeze(1), S).squeeze(1)
            d = bb[t].reshape(-1, 1) * (vb[t] - kS)
            S.baddbmm_(kb[t].unsqueeze(2), d.unsqueeze(1))
            ob.append(torch.bmm(qb[t].unsqueeze(1), S).squeeze(1))
            if t in cads[bi]:
                S = Q[bi](S).contiguous()
        outs.append(torch.stack(ob, 0).cpu())                                     # [D,48,128]
    core = torch.stack(outs, 0).reshape(-1, R.LV_DIM)
    zz = z.reshape(-1, R.LV_DIM)
    var = core.pow(2).mean(-1, keepdim=True)
    core = W["norm"].float() * (core * torch.rsqrt(var + R.EPS)) * F.silu(zz)
    return F.linear(core.reshape(B * D, value_dim), W["out_proj"])


def attn_dec(h, W, kp, vp, TP, B, D):
    q = F.linear(h, W["q_proj"]).view(B, D, R.N_HEAD, R.HEAD_DIM * 2)
    q, gate = q[..., :R.HEAD_DIM], q[..., R.HEAD_DIM:]
    gate = gate.reshape(B * D, R.N_HEAD * R.HEAD_DIM)
    k = F.linear(h, W["k_proj"]).view(B, D, R.N_KV, R.HEAD_DIM)
    v = F.linear(h, W["v_proj"]).view(B, D, R.N_KV, R.HEAD_DIM)
    q = R.rms1p(q, W["q_norm"]).transpose(1, 2)
    k = R.rms1p(k, W["k_norm"]).transpose(1, 2)
    v = v.transpose(1, 2)
    cos, sin = R.rope_cos_sin(TP + D)
    cos, sin = cos[TP:], sin[TP:]
    q, k = R.apply_rope(q, cos, sin), R.apply_rope(k, cos, sin)
    K_ = torch.cat([kp.expand(B, -1, -1, -1), k], 2).repeat_interleave(R.N_HEAD // R.N_KV, dim=1)
    V_ = torch.cat([vp.expand(B, -1, -1, -1), v], 2).repeat_interleave(R.N_HEAD // R.N_KV, dim=1)
    mask = torch.ones(D, TP + D, dtype=torch.bool).tril(TP)
    o = F.scaled_dot_product_attention(q, K_, V_, attn_mask=mask, scale=R.HEAD_DIM ** -0.5)
    o = o.transpose(1, 2).reshape(B * D, R.N_HEAD * R.HEAD_DIM) * torch.sigmoid(gate)
    return F.linear(o, W["o_proj"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a")
    ap.add_argument("--variants", default="fp32,fp16-RNE-1")
    ap.add_argument("--dec", type=int, default=256)
    ap.add_argument("--layers-limit", type=int, default=64)
    ap.add_argument("--out", default="/home/kevin/projects/lanes/k8/decode_fidelity.json")
    a = ap.parse_args()
    import capture as C
    ids = C.get_ids(6400)[0]
    D = a.dec; T = ids.shape[0]; TP = T - D
    vs = [parse_variant(n) for n in a.variants.split(",")]
    B = len(vs)
    Q, cads = [], []
    for i, v in enumerate(vs):
        gen = torch.Generator(device=gpu_linear._dev); gen.manual_seed(100 + i)
        Q.append(make_quantizer(v["fmt"], v["axis"], v["sr"], gen)); cads.append(cadence(v["cad"], D))
    x = R.load_embed()[ids[TP:].unsqueeze(0).expand(B, -1).reshape(-1)].float()
    t0 = time.time()
    ck = f"{CAP}/resume_{abs(sum(map(ord, a.variants))) % 10**8}.pt"
    start = 0
    if os.path.exists(ck):
        c = torch.load(ck)
        if c["variants"] == a.variants:
            x, start = c["x"], c["i"] + 1
            print("resumed at layer", start, flush=True)
    for i in range(start, min(a.layers_limit, R.N_LAYERS)):
        tl = time.time()
        W = R.load_layer_weights(i)
        fn = f"{CAP}/{a.tag}_dec_L{i:02d}.pt"
        while not os.path.exists(fn):   # run as a pipeline behind capture.py
            time.sleep(15)
        time.sleep(3)
        dec = torch.load(fn)
        with torch.no_grad():
            h = R.rms1p(x, W["input_layernorm"])
            if R.LAYER_TYPES[i] == "full_attention":
                mix = attn_dec(h, W, dec["k"], dec["v"], TP, B, D)
            else:
                mix = gdn_dec(h, W, Q, dec["S_tp"][0].contiguous(), dec["conv_tail"][0], cads, B, D)
            x = x + mix
            h2 = R.rms1p(x, W["post_attention_layernorm"])
            x = x + R.mlp(h2, W)
        torch.save({"x": x, "i": i, "variants": a.variants}, ck + ".tmp"); os.replace(ck + ".tmp", ck)
        print(f"layer {i:2d} {R.LAYER_TYPES[i][:4]} {time.time()-tl:6.1f}s elapsed {time.time()-t0:6.0f}s  rel|x-x0| per variant "
              + " ".join(f"{(x.view(B,D,-1)[j]-x.view(B,D,-1)[0]).norm()/x.view(B,D,-1)[0].norm():.4f}" for j in range(B)), flush=True)
    H = R.rms1p(x, R._t(R.MAIN_ST, f"{R.PFX}.norm.weight")).view(B, D, -1)
    Wl = R.load_lm_head()
    res = {v["name"]: {"KL": [], "top1": []} for v in vs}
    lp0 = None
    for p0 in range(0, D, 32):
        lps = []
        for j in range(B):
            lg = torch.zeros(min(32, D - p0), Wl.shape[0])
            for v0 in range(0, Wl.shape[0], 16384):
                lg[:, v0:v0 + 16384] = H[j, p0:p0 + 32] @ Wl[v0:v0 + 16384].float().T
            lps.append(torch.log_softmax(lg, -1))
        for j in range(B):
            kl = (lps[0].exp() * (lps[0] - lps[j])).sum(-1)
            res[vs[j]["name"]]["KL"] += kl.tolist()
            res[vs[j]["name"]]["top1"] += (lps[0].argmax(-1) == lps[j].argmax(-1)).float().tolist()
    out = {}
    for n, r in res.items():
        kl = torch.tensor(r["KL"]); t1 = torch.tensor(r["top1"])
        out[n] = {"KL_mean": kl.mean().item(), "KL_p99": kl.quantile(0.99).item(), "top1_agree": t1.mean().item(),
                  "KL_first64": kl[:64].mean().item(), "KL_last64": kl[-64:].mean().item()}
        print(n.ljust(24), json.dumps(out[n]))
    json.dump({"per_pos": res, "summary": out, "meta": {"D": D, "TP": TP}}, open(a.out, "w"))


if __name__ == "__main__":
    main()
