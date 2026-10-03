"""Lane DFT: pure-torch re-implementation of the Qwen3.5/3.8 MTP drafter block (the engine's `mtp.*` weights) for
distillation fine-tuning, plus helpers shared by train_mtp.py / eval_mtp.py.

Exactly mirrors vllm/model_executor/models/qwen3_5_mtp.py + Qwen3NextAttention:
  x   = fc(cat[pre_fc_norm_embedding(embed(tok_{t+1})), pre_fc_norm_hidden(h_t)])         (embeds FIRST)
  res = x ; y = (1+w)-RMSNorm(x)                                                         (GemmaRMSNorm)
  q|gate = q_proj(y).view(nh, 2*hd).chunk(2) ; q,k = (1+w)-RMSNorm per head ; partial NeoX rope (64 of 256 dims, yarn x2 cache)
  attn (GQA 24/4, causal) * sigmoid(gate) -> o_proj ; res += . ; y = norm(res) ; res += down(silu(gate)*up)
  out = (1+w)-RMSNorm(res)  (mtp.norm)   -> lm_head (shared with the target, frozen)  AND fed back as hidden for the next draft step.
Draft step k (1-based) for origin t runs at position t+k-1 and attends: step-1 keys at positions <= t (written from the TARGET's
hidden states) plus its own chain keys of steps 2..k at origin t (written from the previous draft steps' outputs).
"""
import json, os, struct, math
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DIR = "/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven"
H, I, NH, NKV, HD, ROT, EPS, VOCAB = 5120, 17408, 24, 4, 256, 64, 1e-6, 248320
ROPE_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rope_cache.pt")
ROPE_PARAMS = {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 262144, "mrope_interleaved": True,
               "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000}


def gnorm(x, w, eps=EPS):
    """GemmaRMSNorm: x * rsqrt(mean(x^2)+eps) * (1+w), computed in fp32, returned in x.dtype."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * (1.0 + w.float())).to(x.dtype)


def build_rope_cache(max_pos=8192):
    """The engine's own cos_sin_cache (yarn factor 2, mscale folded into cos/sin) built with vLLM's get_rope on CPU."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope
    with set_current_vllm_config(VllmConfig()):
        r = get_rope(head_size=HD, max_position=max_pos, rope_parameters=ROPE_PARAMS)
    c = r.cos_sin_cache[:max_pos].clone().float()
    torch.save(c, ROPE_CACHE)
    return c


def load_rope_cache(device="cpu"):
    if not os.path.exists(ROPE_CACHE):
        build_rope_cache()
    return torch.load(ROPE_CACHE).to(device)


def apply_rope(x, pos, cache):
    """x: (L, n, HD); pos: (L,) long; cache: (P, ROT) = [cos(ROT/2) | sin(ROT/2)]; NeoX layout on the first ROT dims."""
    cs = cache[pos]
    cos, sin = cs[:, : ROT // 2].unsqueeze(1).to(x.dtype), cs[:, ROT // 2:].unsqueeze(1).to(x.dtype)
    r = x[..., :ROT]
    x1, x2 = r[..., : ROT // 2], r[..., ROT // 2:]
    o = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([o, x[..., ROT:]], dim=-1)


class MTPBlock(nn.Module):
    def __init__(self, h=H, i=I, nh=NH, nkv=NKV, hd=HD, rot=ROT):
        super().__init__()
        self.h, self.nh, self.nkv, self.hd, self.rot = h, nh, nkv, hd, rot
        L = lambda a, b: nn.Linear(a, b, bias=False)
        self.fc = L(2 * h, h)
        self.q_proj, self.k_proj, self.v_proj = L(h, nh * 2 * hd), L(h, nkv * hd), L(h, nkv * hd)
        self.o_proj = L(nh * hd, h)
        self.gate_proj, self.up_proj, self.down_proj = L(h, i), L(h, i), L(i, h)
        P = lambda n: nn.Parameter(torch.zeros(n))
        self.pre_fc_norm_hidden, self.pre_fc_norm_embedding = P(h), P(h)
        self.input_layernorm, self.post_attention_layernorm, self.norm = P(h), P(h), P(h)
        self.q_norm, self.k_norm = P(hd), P(hd)

    # ----- checkpoint <-> module names (the on-disk `mtp.*` layout of model-mtp.safetensors)
    MAP = {
        "fc.weight": "fc.weight", "norm.weight": "norm", "pre_fc_norm_hidden.weight": "pre_fc_norm_hidden",
        "pre_fc_norm_embedding.weight": "pre_fc_norm_embedding",
        "layers.0.input_layernorm.weight": "input_layernorm", "layers.0.post_attention_layernorm.weight": "post_attention_layernorm",
        "layers.0.self_attn.q_proj.weight": "q_proj.weight", "layers.0.self_attn.k_proj.weight": "k_proj.weight",
        "layers.0.self_attn.v_proj.weight": "v_proj.weight", "layers.0.self_attn.o_proj.weight": "o_proj.weight",
        "layers.0.self_attn.q_norm.weight": "q_norm", "layers.0.self_attn.k_norm.weight": "k_norm",
        "layers.0.mlp.gate_proj.weight": "gate_proj.weight", "layers.0.mlp.up_proj.weight": "up_proj.weight",
        "layers.0.mlp.down_proj.weight": "down_proj.weight",
    }

    def load_ckpt(self, path):
        from safetensors import safe_open
        sd = self.state_dict()
        with safe_open(path, "pt") as f:
            keys = set(f.keys())
            for ck, mk in self.MAP.items():
                t = f.get_tensor("mtp." + ck)
                assert sd[mk].shape == t.shape, (ck, sd[mk].shape, t.shape)
                sd[mk].copy_(t.float())
            extra = keys - {"mtp." + k for k in self.MAP}
            assert not extra, extra
        return self

    def export_ckpt(self, path, dtype=torch.bfloat16):
        from safetensors.torch import save_file
        sd = self.state_dict()
        out = {"mtp." + ck: sd[mk].detach().to("cpu", dtype).contiguous() for ck, mk in self.MAP.items()}
        save_file(out, path, metadata={"format": "pt"})

    # ----- one draft step, front half: returns roped q, k, v, gate and the residual stream
    def front(self, hid, emb, pos, cache):
        x = self.fc(torch.cat([gnorm(emb, self.pre_fc_norm_embedding), gnorm(hid, self.pre_fc_norm_hidden)], dim=-1))
        res = x
        y = gnorm(x, self.input_layernorm)
        L = y.shape[0]
        qg = self.q_proj(y).view(L, self.nh, 2 * self.hd)
        q, gate = qg[..., : self.hd], qg[..., self.hd:]
        k = self.k_proj(y).view(L, self.nkv, self.hd)
        v = self.v_proj(y).view(L, self.nkv, self.hd)
        q, k = gnorm(q, self.q_norm), gnorm(k, self.k_norm)
        return apply_rope(q, pos, cache), apply_rope(k, pos, cache), v, gate, res

    def back(self, attn, gate, res):
        a = (attn * torch.sigmoid(gate)).reshape(attn.shape[0], -1)
        res = res + self.o_proj(a)
        y = gnorm(res, self.post_attention_layernorm)
        res = res + self.down_proj(F.silu(self.gate_proj(y)) * self.up_proj(y))
        return gnorm(res, self.norm)


def chain_forward(blk, Hs, emb, cache, K=3):
    """Vectorised draft chain for every origin of one window.
    Hs: (T, h) target post-norm hidden states; emb: (T, h) embeddings of the window's tokens X[0..T-1].
    Returns [m_1..m_K], m_k of shape (T-k, h): the drafter's post-norm output for origin t=0..T-1-k at draft step k
    (its logits are compared with the target's distribution at position t+k)."""
    T = Hs.shape[0]
    outs, kv1, chain = [], None, []   # chain: list of (k_j, v_j) for steps 2..
    prev = None
    for step in range(1, K + 1):
        L = T - step
        if L <= 0:
            break
        hid = Hs[:L] if step == 1 else prev[:L]
        e = emb[step: step + L]
        pos = torch.arange(L, device=Hs.device) + (step - 1)
        q, k, v, gate, res = blk.front(hid, e, pos, cache)
        if step == 1:
            kv1 = (k, v)
            Kc, Vc = k, v
            S = L
            mask = torch.ones(L, L, dtype=torch.bool, device=Hs.device).tril()
        else:
            chain.append((k, v))
            Kc = torch.cat([kv1[0][:L]] + [c[0][:L] for c in chain], dim=0)
            Vc = torch.cat([kv1[1][:L]] + [c[1][:L] for c in chain], dim=0)
            S = Kc.shape[0]
            mask = torch.zeros(L, S, dtype=torch.bool, device=Hs.device)
            mask[:, :L] = torch.ones(L, L, dtype=torch.bool, device=Hs.device).tril()
            ar = torch.arange(L, device=Hs.device)
            for j in range(len(chain)):
                mask[ar, L + j * L + ar] = True
        g = blk.nh // blk.nkv
        qh = q.transpose(0, 1).unsqueeze(0)                                  # 1,nh,L,hd
        kh = Kc.transpose(0, 1).repeat_interleave(g, 0).unsqueeze(0)         # 1,nh,S,hd
        vh = Vc.transpose(0, 1).repeat_interleave(g, 0).unsqueeze(0)
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)       # scale = hd**-0.5
        o = o.squeeze(0).transpose(0, 1)                                    # L,nh,hd
        prev = blk.back(o, gate, res)
        outs.append(prev)
    return outs


def teacher_topk(logits_fn, Hs, k=32, temp=0.6, top_k=20, top_p=0.95, chunk=512):
    """Target next-token SAMPLING distribution at every position (production gencfg: T=0.6, top_k=20, top_p=0.95).
    logits_fn(h) -> (n, V) logits.  Returns idx (T,k) long, p (T,k) float (renormalised; zero outside the sampling support)."""
    idxs, ps = [], []
    for s in range(0, Hs.shape[0], chunk):
        lg = logits_fn(Hs[s: s + chunk]).float()
        v, i = lg.topk(k, dim=-1)
        pr = torch.softmax(v / temp, dim=-1)
        keep = torch.arange(k, device=v.device)[None, :] < top_k
        pr = pr * keep
        pr = pr / pr.sum(-1, keepdim=True)
        cs = pr.cumsum(-1)
        keep = (cs - pr) < top_p
        pr = pr * keep
        pr = pr / pr.sum(-1, keepdim=True)
        idxs.append(i); ps.append(pr)
    return torch.cat(idxs), torch.cat(ps)


def read_tensor(path, name):
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        return f.get_tensor(name)


def int4_sim_(blk):
    """Replace the 8 MTP linears by their U2b int4 (RTN + MSE-clip, g128, asymmetric) dequantised values, in place (what VLLM_U2_INT4_MTP=1 serves)."""
    from vllm.model_executor.layers.quantization import u2_headquant as Q
    with torch.no_grad():
        for m in (blk.fc, blk.q_proj, blk.k_proj, blk.v_proj, blk.o_proj, blk.gate_proj, blk.up_proj, blk.down_proj):
            w = m.weight
            t = Q.quantize_linear(w.detach().float().cpu(), device=w.device if w.is_cuda else "cpu")
            m.weight.copy_(Q.dequantize_linear(t).to(w.device, w.dtype))
    return blk
