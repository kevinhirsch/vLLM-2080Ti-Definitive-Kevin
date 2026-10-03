#!/usr/bin/env python
"""Lane K9: REAL post-norm, pre-RoPE q/k and v of full-attention layers from the captured q/k/v_proj inputs
(U2 CPU fp32 reference). Saves {layer: {q [N,24,256], k [N,4,256], v [N,4,256], pos [N]}} + logit-magnitude stats."""
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
import ref_dump as R
out, stats = {}, {}
for path in sys.argv[1:]:
    caps = torch.load(path)["caps"]
    for name in caps:
        if not name.endswith("self_attn.q_proj"): continue
        i = int(name.split("layers.")[1].split(".")[0]); p = name[: -len(".q_proj")]
        x = caps[name]["x"]; N = x.shape[0]
        h = R._sf(R.MAIN_ST)
        q = torch.nn.functional.linear(x, R.dequant_linear(h, p + ".q_proj")).view(N, 24, 512)[..., :256]
        k = torch.nn.functional.linear(x, R.dequant_linear(h, p + ".k_proj")).view(N, 4, 256)
        v = torch.nn.functional.linear(x, R.dequant_linear(h, p + ".v_proj")).view(N, 4, 256)
        q = R.rms1p(q, h.get_tensor(p + ".q_norm.weight")); k = R.rms1p(k, h.get_tensor(p + ".k_norm.weight"))
        pos = torch.arange(N) % 320
        out[i] = {"q": q.half(), "k": k.half(), "v": v.half(), "pos": pos}
        # worst-case logit bound per (q head, kv head): max|q| * max|k| (Cauchy-Schwarz), and realized max |q.k| (no RoPE)
        qn, kn = q.norm(dim=-1), k.norm(dim=-1)  # [N,24], [N,4]
        kh = torch.arange(24) // 6
        bound = (qn.max(0).values * kn.max(0).values[kh]) / 16.0
        real = torch.einsum("qhd,khd->hqk", q, k[:, kh]).abs().amax((1, 2)) / 16.0
        stats[i] = {"q_norm_max": qn.max().item(), "k_norm_max": kn.max().item(), "scaled_logit_bound_max": bound.max().item(),
                    "scaled_logit_realized_max": real.max().item(), "q_absmax": q.abs().max().item(), "k_absmax": k.abs().max().item(),
                    "v_absmax": v.abs().max().item(), "head_bound": [round(b, 1) for b in bound.tolist()]}
        print(i, json.dumps({k_: (round(v_, 2) if isinstance(v_, float) else v_) for k_, v_ in stats[i].items()}), flush=True)
dst = "/home/kevin/projects/lanes/k9/acts/qkv_real.pt"
if os.path.exists(dst):
    old = torch.load(dst); old["qkv"].update(out); old["stats"].update(stats); out, stats = old["qkv"], old["stats"]
torch.save({"qkv": out, "stats": stats}, dst)
