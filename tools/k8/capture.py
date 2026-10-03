#!/usr/bin/env python
"""Lane K8 capture pass: fp32 CPU reference forward over ONE contiguous flightrec prefix (default 6400 tokens),
saving (a) linear-layer inputs for sampled layers (Hessians / pruning fidelity, track H) and
(b) post-conv q,k,v,g,beta of sampled GDN layers (SSM-state precision, track G).
Reuses /home/kevin/projects/lanes/u2-quant/ref_dump.py (the U2b harness; imports only, no edits).
Run: CUDA_VISIBLE_DEVICES="" nice -n 10 python capture.py [--ntok 6400]"""
import os, sys, json, time, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/home/kevin/projects/lanes/u2-quant")
import torch, torch.nn.functional as F
import ref_dump as R
import gpu_linear; gpu_linear.install()

OUT = "/home/kevin/projects/lanes/k8/cap"
H_LAYERS = [1, 6, 15, 29, 38, 47, 58, 62]          # linear-input capture (6 GDN + 2 full attention)
S_LAYERS = [0, 1, 2, 6, 10, 14, 18, 22, 29, 38, 50, 58, 62]  # GDN recurrence-input capture


def get_ids(ntok, body="1790999015_23730tok.json"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(R.MODEL_DIR)
    tpl = open(R.CHAT_TEMPLATE).read()
    d = json.load(open(f"{R.FLIGHTREC}/{body}"))
    s = tok.apply_chat_template(d["messages"], tools=d.get("tools"), tokenize=False, add_generation_prompt=True,
                                chat_template=tpl, **(d.get("chat_template_kwargs") or {}))
    return torch.tensor([tok(s, add_special_tokens=False)["input_ids"][:ntok]], dtype=torch.long)


def gdn_cap(h, W, B, T, li, cap):
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    key_dim, value_dim = R.LK_HEADS * R.LK_DIM, R.LV_HEADS * R.LV_DIM
    qkv = F.linear(h, W["in_proj_qkv"]).view(B, T, -1).transpose(1, 2)
    z = F.linear(h, W["in_proj_z"]).view(B, T, R.LV_HEADS, R.LV_DIM)
    b = F.linear(h, W["in_proj_b"]).view(B, T, R.LV_HEADS)
    a = F.linear(h, W["in_proj_a"]).view(B, T, R.LV_HEADS)
    conv_w = W["conv1d"].squeeze(1)
    qkv_pre = qkv
    qkv = F.conv1d(qkv, conv_w.unsqueeze(1), None, padding=R.CONV_K - 1, groups=qkv.shape[1])[:, :, :T]
    qkv = F.silu(qkv).transpose(1, 2)
    q, k, v = torch.split(qkv, [key_dim, key_dim, value_dim], dim=-1)
    q = q.reshape(B, T, R.LK_HEADS, R.LK_DIM).repeat_interleave(R.LV_HEADS // R.LK_HEADS, dim=2)
    k = k.reshape(B, T, R.LK_HEADS, R.LK_DIM).repeat_interleave(R.LV_HEADS // R.LK_HEADS, dim=2)
    v = v.reshape(B, T, R.LV_HEADS, R.LV_DIM)
    beta = b.sigmoid()
    g = -W["A_log"].float().exp() * F.softplus(a.float() + W["dt_bias"].float())
    if li in S_LAYERS:  # store kv heads un-repeated (16) to save space
        cap["gdn"] = {"q": q[:, :, ::3].half().clone(), "k": k[:, :, ::3].half().clone(), "v": v.half().clone(),
                      "g": g.clone(), "beta": beta.clone()}
    TP = cap["TP"]
    core, S_tp = gpu_linear.gdn_core(torch_chunk_gated_delta_rule, q, k, v, g, beta, TP=TP)
    cap["dec"] = {"S_tp": S_tp.float().clone(), "conv_tail": qkv_pre[:, :, TP - 3:TP].clone()}
    core = core.reshape(-1, R.LV_DIM)
    zz = z.reshape(-1, R.LV_DIM)
    var = core.pow(2).mean(-1, keepdim=True)
    core = W["norm"].float() * (core * torch.rsqrt(var + R.EPS)) * F.silu(zz)
    core = core.reshape(B * T, value_dim)
    cap["core"] = core
    return F.linear(core, W["out_proj"])


def attn_cap(h, W, B, T, cap):
    q = F.linear(h, W["q_proj"]).view(B, T, R.N_HEAD, R.HEAD_DIM * 2)
    q, gate = q[..., :R.HEAD_DIM], q[..., R.HEAD_DIM:]
    gate = gate.reshape(B * T, R.N_HEAD * R.HEAD_DIM)
    k = F.linear(h, W["k_proj"]).view(B, T, R.N_KV, R.HEAD_DIM)
    v = F.linear(h, W["v_proj"]).view(B, T, R.N_KV, R.HEAD_DIM)
    q = R.rms1p(q, W["q_norm"]).transpose(1, 2)
    k = R.rms1p(k, W["k_norm"]).transpose(1, 2)
    v = v.transpose(1, 2)
    cos, sin = R.rope_cos_sin(T)
    q, k = R.apply_rope(q, cos, sin), R.apply_rope(k, cos, sin)
    rep = R.N_HEAD // R.N_KV
    TP = cap["TP"]
    cap["dec"] = {"k": k[:, :, :TP].clone(), "v": v[:, :, :TP].clone()}   # un-repeated [B,4,TP,256], post norm+rope
    k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
    o = gpu_linear.sdpa_causal(q, k, v, R.HEAD_DIM ** -0.5)
    o = o.transpose(1, 2).reshape(B * T, R.N_HEAD * R.HEAD_DIM) * torch.sigmoid(gate)
    cap["core"] = o
    return F.linear(o, W["o_proj"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntok", type=int, default=6400)
    ap.add_argument("--body", default="1790999015_23730tok.json")
    ap.add_argument("--tag", default="a")
    ap.add_argument("--dec", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ids = get_ids(a.ntok, a.body)
    B, T = ids.shape
    print("ids", ids.shape, flush=True)
    x = R.load_embed()[ids.reshape(-1)].float()
    t0 = time.time()
    for i in range(R.N_LAYERS):
        tl = time.time()
        W = R.load_layer_weights(i)
        cap = {"TP": T - a.dec}
        with torch.no_grad():
            h = R.rms1p(x, W["input_layernorm"])
            cap["h_attn"] = h
            if R.LAYER_TYPES[i] == "full_attention":
                mix = attn_cap(h, W, B, T, cap)
            else:
                mix = gdn_cap(h, W, B, T, i, cap)
            x = x + mix
            h2 = R.rms1p(x, W["post_attention_layernorm"])
            cap["h_mlp"] = h2
            act = F.silu(F.linear(h2, W["gate_proj"])) * F.linear(h2, W["up_proj"])
            cap["act"] = act
            x = x + F.linear(act, W["down_proj"])
        if i in H_LAYERS:
            d = {k: cap[k].half() for k in ("h_attn", "core", "h_mlp", "act")}
            for k, v in d.items():
                assert v.abs().max() < 6e4, (i, k)
            torch.save({"type": R.LAYER_TYPES[i], "ntok": T, **d}, f"{OUT}/{a.tag}_lin_L{i:02d}.pt")
        if i in S_LAYERS:
            torch.save({k: v.cpu() for k, v in cap["gdn"].items()}, f"{OUT}/{a.tag}_gdn_L{i:02d}.pt")
        torch.save({k: v.cpu() for k, v in cap["dec"].items()}, f"{OUT}/{a.tag}_dec_L{i:02d}.pt")
        del W, cap
        print(f"layer {i:2d} {R.LAYER_TYPES[i][:4]} {time.time()-tl:6.1f}s elapsed {time.time()-t0:6.0f}s |x|rms {x.pow(2).mean().sqrt():.3f}", flush=True)
    torch.save({"ids": ids}, f"{OUT}/{a.tag}_ids.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
