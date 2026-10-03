#!/usr/bin/env python
"""Lane K9 (L102): REAL GDN-layer kernel inputs (TP rank-0 shard: k heads 0-7, v heads 0-23) from captured in_proj
inputs (U2 CPU fp32 reference): q, k (l2-normalised), v after the causal conv + SiLU, g = -e^{A_log} softplus(a + dt_bias),
beta = sigmoid(b). Windows of 320 contiguous tokens. Saves acts/gdn_real.pt {layer: {q,k,v (fp16) [W,320,H,128], g, beta}}"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
import torch.nn.functional as F
sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
import ref_dump as R
out = {}
for path in sys.argv[1:]:
    caps = torch.load(path)["caps"]
    for name, c in caps.items():
        if not name.endswith("linear_attn.in_proj_qkv"): continue
        i = int(name.split("layers.")[1].split(".")[0]); p = name[: -len(".in_proj_qkv")]
        h = c["x"]; Wn = 2; T = h.shape[0] // Wn
        sf = R._sf(R.MAIN_ST)
        qkv = F.linear(h, R.dequant_linear(sf, p + ".in_proj_qkv")).view(Wn, T, -1).transpose(1, 2)
        a = F.linear(h, sf.get_tensor(p + ".in_proj_a.weight").float()).view(Wn, T, 48)
        b = F.linear(h, sf.get_tensor(p + ".in_proj_b.weight").float()).view(Wn, T, 48)
        cw = sf.get_tensor(p + ".conv1d.weight").float().squeeze(1)
        qkv = F.silu(F.conv1d(qkv, cw.unsqueeze(1), None, padding=3, groups=qkv.shape[1])[:, :, :T]).transpose(1, 2)
        q, k, v = torch.split(qkv, [2048, 2048, 6144], dim=-1)
        q = R._l2norm(q.reshape(Wn, T, 16, 128))[:, :, :8]; k = R._l2norm(k.reshape(Wn, T, 16, 128))[:, :, :8]
        v = v.reshape(Wn, T, 48, 128)[:, :, :24]
        g = (-sf.get_tensor(p + ".A_log").float().exp() * F.softplus(a + sf.get_tensor(p + ".dt_bias").float()))[:, :, :24]
        beta = b.sigmoid()[:, :, :24]
        out[i] = {"q": q.half(), "k": k.half(), "v": v.half(), "g": g, "beta": beta}
        print(i, f"g min {g.min():.2f} median {g.median():.3f} | beta median {beta.median():.2f} | v absmax {v.abs().max():.1f}", flush=True)
dst = "/home/kevin/projects/lanes/k9/acts/gdn_real.pt"
if os.path.exists(dst):
    old = torch.load(dst); old.update(out); out = old
torch.save(out, dst)
