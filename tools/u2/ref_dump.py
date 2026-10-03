#!/usr/bin/env python
"""Offline CPU-only fp32 reference harness for Qwen3.8-27B (HauhauCS W4A16 compressed-tensors).

Run:   CUDA_VISIBLE_DEVICES="" nice -n 10 /home/kevin/Desktop/wt-integrate/.venv/bin/python ref_dump.py [--windows 12 --win 320]
Output: ref.pt (see README.txt) + sanity-gate numbers.

Import API (no heavy work at import time):
    build_prompt_set(n_windows, win, seed) -> (ids LongTensor[B,T], manifest list[dict])
    run_main(ids, ckpt_dir=None) -> H FloatTensor[B,T,5120]   (post-final-norm hidden)
    dequant_linear(f, prefix) -> FloatTensor[out,in]           (int4 -> fp32)
    load_embed() / load_lm_head() -> bf16 [V,5120]
    lm_head_stats(H2d, targets, W=None, topk=0) -> dict         (chunked; W overridable)
    load_mtp_bf16() -> dict[str, bf16 Tensor]                  (names 'mtp.*')
    mtp_forward(H, next_tok_ids, weights, embed_w=None) -> G    (post mtp.norm, pre-logits)
    load_ref(path) -> dict
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, json, glob, math, time, hashlib, random, argparse
import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.set_num_threads(int(os.environ.get("REF_THREADS", "8")))

MODEL_DIR = "/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven"
MAIN_ST = f"{MODEL_DIR}/model.safetensors"
MTP_ST = f"{MODEL_DIR}/model-mtp.safetensors"
FLIGHTREC = os.environ.get("REF_FLIGHTREC", "/home/kevin/projects/lanes/u2-quant/frozen_flightrec")
CHAT_TEMPLATE = "/home/kevin/.local/share/vllm-qwen27b/chat_template-froggeric-v22-official.jinja"
CORPUS = "/home/kevin/Desktop/qwen38-evalkit/corpus/corpus_128000tok.txt"
PROSE_FILES = [f"{MODEL_DIR}/README.md", "/home/kevin/Obsidian/Memory/Frontier Working Framework.md"]
OUT_DIR = "/home/kevin/projects/lanes/u2-quant"

# --- text config (config.json: text_config) ---
D = 5120
N_LAYERS = 64
FFN = 17408
EPS = 1e-6
N_HEAD, N_KV, HEAD_DIM = 24, 4, 256
ROT_DIM = int(HEAD_DIM * 0.25)  # 64
ROPE_BASE = 1e7
ROPE_YARN_FACTOR = 2.0
ROPE_ORIG_MAX = 262144
LAYER_TYPES = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(N_LAYERS)]
LK_HEADS, LV_HEADS, LK_DIM, LV_DIM, CONV_K = 16, 48, 128, 128, 4
GROUP = 128
PFX = "model.language_model"

_files = {}


def _sf(path):
    if path not in _files:
        _files[path] = safe_open(path, "pt")
    return _files[path]


def _t(path, name):
    return _sf(path).get_tensor(name)


# ----------------------------------------------------------------------------------------------
# int4 (compressed-tensors pack-quantized, group128, asymmetric) -> fp32
# ----------------------------------------------------------------------------------------------
_SHIFTS = (torch.arange(8, dtype=torch.int32) * 4)


def dequant_linear(f, prefix, dtype=torch.float32):
    """f: safe_open handle / path; prefix like 'model.language_model.layers.0.mlp.down_proj'.
    packed[out, in/8] int32 (nibble k of word j = column j*8+k), scale[out, in/128] bf16,
    zero_point[out/8, in/128] int32 packed along dim 0 (nibble k of row r = out row r*8+k).
    signed value = nibble-8 for both q and zp; w = (q - zp) * scale.  (verified vs compressed_tensors)."""
    h = _sf(f) if isinstance(f, str) else f
    packed = h.get_tensor(prefix + ".weight_packed")
    scale = h.get_tensor(prefix + ".weight_scale").to(torch.float32)
    zpp = h.get_tensor(prefix + ".weight_zero_point")
    shape = h.get_tensor(prefix + ".weight_shape")
    out_f, in_f = int(shape[0]), int(shape[1])
    q = ((packed.unsqueeze(-1) >> _SHIFTS) & 15).reshape(out_f, in_f)  # int32 nibble 0..15
    zp = ((zpp.unsqueeze(1) >> _SHIFTS.view(1, 8, 1)) & 15).reshape(out_f, -1)  # [out, ngroups]
    ng = in_f // GROUP
    w = (q.reshape(out_f, ng, GROUP) - zp.unsqueeze(-1)).to(torch.float32)
    w.mul_(scale.unsqueeze(-1))
    return w.reshape(out_f, in_f).to(dtype)


# ----------------------------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------------------------
def rms1p(x, w, eps=EPS):
    """Qwen3_5 / GemmaRMSNorm: x/rms * (1 + w), fp32 math."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())


def yarn_inv_freq_and_mscale():
    """Mirror of vllm YaRNScalingRotaryEmbedding (MRotaryEmbedding with scaling_factor)."""
    dim, base, orig, factor = ROT_DIM, ROPE_BASE, ROPE_ORIG_MAX, ROPE_YARN_FACTOR
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
    extra = 1.0 / pos_freqs
    inter = 1.0 / (factor * pos_freqs)

    def corr_dim(n_rot):
        return (dim * math.log(orig / (n_rot * 2 * math.pi))) / (2 * math.log(base))
    low = max(math.floor(corr_dim(32)), 0)
    high = min(math.ceil(corr_dim(1)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float) - low) / (high - low), 0, 1)
    mask = 1 - ramp
    inv = inter * (1 - mask) + extra * mask
    mscale = 0.1 * math.log(factor) + 1.0
    return inv, mscale


_rope_cache = {}


def rope_cos_sin(T):
    """text-only: all 3 mrope rows identical -> plain neox rope with yarn freqs. [T, ROT_DIM] cos, sin (fp32)."""
    if T not in _rope_cache:
        inv, ms = yarn_inv_freq_and_mscale()
        fr = torch.outer(torch.arange(T, dtype=torch.float32), inv)
        emb = torch.cat([fr, fr], -1)
        _rope_cache[T] = (emb.cos() * ms, emb.sin() * ms)
    return _rope_cache[T]


def _rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def apply_rope(x, cos, sin):
    """x [B,H,T,hd]; rotate first ROT_DIM dims (neox style)."""
    xr, xp = x[..., :ROT_DIM], x[..., ROT_DIM:]
    xr = xr * cos + _rot_half(xr) * sin
    return torch.cat([xr, xp], -1)


def full_attention(h, W, B, T):
    """h [B*T, D] (already input-normed). W: dict with q_proj,k_proj,v_proj,o_proj ([out,in] fp32), q_norm,k_norm."""
    q = F.linear(h, W["q_proj"]).view(B, T, N_HEAD, HEAD_DIM * 2)
    q, gate = q[..., :HEAD_DIM], q[..., HEAD_DIM:]
    gate = gate.reshape(B * T, N_HEAD * HEAD_DIM)
    k = F.linear(h, W["k_proj"]).view(B, T, N_KV, HEAD_DIM)
    v = F.linear(h, W["v_proj"]).view(B, T, N_KV, HEAD_DIM)
    q = rms1p(q, W["q_norm"]).transpose(1, 2)  # [B,H,T,hd]
    k = rms1p(k, W["k_norm"]).transpose(1, 2)
    v = v.transpose(1, 2)
    cos, sin = rope_cos_sin(T)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    rep = N_HEAD // N_KV
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=HEAD_DIM ** -0.5)  # [B,H,T,hd]
    o = o.transpose(1, 2).reshape(B * T, N_HEAD * HEAD_DIM)
    o = o * torch.sigmoid(gate)
    return F.linear(o, W["o_proj"])


def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def gdn(h, W, B, T):
    """Gated DeltaNet (linear attention) layer, prefill from zero state. h [B*T, D]."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    key_dim, value_dim = LK_HEADS * LK_DIM, LV_HEADS * LV_DIM
    qkv = F.linear(h, W["in_proj_qkv"]).view(B, T, -1).transpose(1, 2)  # [B, conv_dim, T]
    z = F.linear(h, W["in_proj_z"]).view(B, T, LV_HEADS, LV_DIM)
    b = F.linear(h, W["in_proj_b"]).view(B, T, LV_HEADS)
    a = F.linear(h, W["in_proj_a"]).view(B, T, LV_HEADS)
    conv_w = W["conv1d"].squeeze(1)  # [conv_dim, K]
    qkv = F.conv1d(qkv, conv_w.unsqueeze(1), None, padding=CONV_K - 1, groups=qkv.shape[1])[:, :, :T]
    qkv = F.silu(qkv).transpose(1, 2)  # [B,T,conv_dim]
    q, k, v = torch.split(qkv, [key_dim, key_dim, value_dim], dim=-1)
    q = q.reshape(B, T, LK_HEADS, LK_DIM).repeat_interleave(LV_HEADS // LK_HEADS, dim=2)
    k = k.reshape(B, T, LK_HEADS, LK_DIM).repeat_interleave(LV_HEADS // LK_HEADS, dim=2)
    v = v.reshape(B, T, LV_HEADS, LV_DIM)
    beta = b.sigmoid()
    g = -W["A_log"].float().exp() * F.softplus(a.float() + W["dt_bias"].float())
    core, _ = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None,
                                           output_final_state=False, use_qk_l2norm_in_kernel=True)
    core = core.reshape(-1, LV_DIM)
    zz = z.reshape(-1, LV_DIM)
    # RMSNormGated: weight * rmsnorm(x) * silu(z)   (weight NOT offset by 1)
    var = core.pow(2).mean(-1, keepdim=True)
    core = W["norm"].float() * (core * torch.rsqrt(var + EPS)) * F.silu(zz)
    return F.linear(core.reshape(B * T, value_dim), W["out_proj"])


def mlp(h, W):
    return F.linear(F.silu(F.linear(h, W["gate_proj"])) * F.linear(h, W["up_proj"]), W["down_proj"])


def load_layer_weights(i):
    """Dequantize layer i of the main model to fp32."""
    h = _sf(MAIN_ST)
    p = f"{PFX}.layers.{i}"
    W = {"input_layernorm": h.get_tensor(p + ".input_layernorm.weight"),
         "post_attention_layernorm": h.get_tensor(p + ".post_attention_layernorm.weight")}
    for n in ("gate_proj", "up_proj", "down_proj"):
        W[n] = dequant_linear(h, f"{p}.mlp.{n}")
    if LAYER_TYPES[i] == "full_attention":
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            W[n] = dequant_linear(h, f"{p}.self_attn.{n}")
        W["q_norm"] = h.get_tensor(p + ".self_attn.q_norm.weight")
        W["k_norm"] = h.get_tensor(p + ".self_attn.k_norm.weight")
    else:
        for n in ("in_proj_qkv", "in_proj_z", "out_proj"):
            W[n] = dequant_linear(h, f"{p}.linear_attn.{n}")
        for n in ("in_proj_a", "in_proj_b"):  # bf16, not quantized
            W[n] = h.get_tensor(f"{p}.linear_attn.{n}.weight").float()
        for n in ("conv1d", "norm", "A_log", "dt_bias"):
            nm = f"{p}.linear_attn.{n}" + (".weight" if n in ("conv1d", "norm") else "")
            W[n] = h.get_tensor(nm).float()
    return W


def decoder_layer(x, i, W, B, T):
    """x [B*T, D] fp32 residual stream."""
    h = rms1p(x, W["input_layernorm"])
    mix = full_attention(h, W, B, T) if LAYER_TYPES[i] == "full_attention" else gdn(h, W, B, T)
    x = x + mix
    h = rms1p(x, W["post_attention_layernorm"])
    return x + mlp(h, W)


# ----------------------------------------------------------------------------------------------
# embeddings / lm_head / main model
# ----------------------------------------------------------------------------------------------
_embed = None


def load_embed():
    global _embed
    if _embed is None:
        _embed = _t(MAIN_ST, f"{PFX}.embed_tokens.weight")  # bf16 [V, D]
    return _embed


def load_lm_head():
    return _t(MAIN_ST, "lm_head.weight")  # bf16 [V, D]


def run_main(ids, ckpt_dir=None, log=print):
    """ids [B,T] -> post-final-norm hidden H [B,T,D] fp32. Loads/dequantizes each layer once for the whole batch."""
    B, T = ids.shape
    x = load_embed()[ids.reshape(-1)].float()  # [B*T, D]
    start = 0
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        cks = sorted(glob.glob(f"{ckpt_dir}/x_after_*.pt"))
        if cks:
            c = torch.load(cks[-1])
            if c["ids_hash"] == hashlib.md5(ids.numpy().tobytes()).hexdigest():
                x, start = c["x"], c["layer"] + 1
                log(f"resumed from {cks[-1]} -> starting at layer {start}")
    t0 = time.time()
    for i in range(start, N_LAYERS):
        tl = time.time()
        W = load_layer_weights(i)
        tload = time.time() - tl
        with torch.no_grad():
            x = decoder_layer(x, i, W, B, T)
        del W
        log(f"layer {i:2d} {LAYER_TYPES[i][:4]} load {tload:5.1f}s total {time.time()-tl:5.1f}s "
            f"elapsed {time.time()-t0:6.0f}s |x|rms {x.pow(2).mean().sqrt():.3f}")
        if ckpt_dir and (i % 8 == 7) and i < N_LAYERS - 1:
            torch.save({"x": x, "layer": i, "ids_hash": hashlib.md5(ids.numpy().tobytes()).hexdigest()},
                       f"{ckpt_dir}/x_after_{i:02d}.pt")
            for old in sorted(glob.glob(f"{ckpt_dir}/x_after_*.pt"))[:-1]:
                os.remove(old)
    H = rms1p(x, _t(MAIN_ST, f"{PFX}.norm.weight"))
    return H.view(B, T, D)


def lm_head_stats(H, targets=None, W=None, topk=0, chunk=8192):
    """H [P,D] fp32. W [V,D] (bf16/fp32; default = checkpoint bf16 lm_head). targets [P] long or None.
    Chunked over vocab. Returns dict: argmax[P], lse[P], tgt_logit[P] (if targets), topk_val/topk_idx [P,k] (if topk)."""
    if W is None:
        W = load_lm_head()
    P = H.shape[0]
    V = W.shape[0]
    best_v = torch.full((P,), -float("inf"))
    best_i = torch.zeros(P, dtype=torch.long)
    m = torch.full((P,), -float("inf"))
    s = torch.zeros(P)
    tgt = torch.zeros(P) if targets is not None else None
    tv = ti = None
    if topk:
        tv = torch.full((P, topk), -float("inf"))
        ti = torch.zeros(P, topk, dtype=torch.long)
    for v0 in range(0, V, chunk):
        Wc = W[v0:v0 + chunk].float()
        lg = H @ Wc.T  # [P, c]
        cv, ci = lg.max(1)
        upd = cv > best_v
        best_v = torch.where(upd, cv, best_v)
        best_i = torch.where(upd, ci + v0, best_i)
        nm = torch.maximum(m, cv)
        s = s * torch.exp(m - nm) + torch.exp(lg - nm[:, None]).sum(1)
        m = nm
        if targets is not None:
            inr = (targets >= v0) & (targets < v0 + lg.shape[1])
            if inr.any():
                tgt[inr] = lg[inr, targets[inr] - v0]
        if topk:
            cat_v = torch.cat([tv, lg], 1)
            cat_i = torch.cat([ti, torch.arange(v0, v0 + lg.shape[1]).expand(P, -1)], 1)
            tv, sel = cat_v.topk(topk, dim=1)
            ti = cat_i.gather(1, sel)
    out = {"argmax": best_i, "lse": m + torch.log(s)}
    if targets is not None:
        out["tgt_logit"] = tgt
    if topk:
        out["topk_val"], out["topk_idx"] = tv, ti
    return out


# ----------------------------------------------------------------------------------------------
# MTP (step 0 of speculative decoding), exact vLLM order (qwen3_5_mtp.py):
#   e = pre_fc_norm_embedding(embed(tok[t+1])); h = pre_fc_norm_hidden(H[t]);
#   x = fc(cat([e, h], -1))                       # embedding FIRST, hidden second
#   one full-attention decoder layer (mtp.layers.0), position = t (same as target position), residual=None at entry
#   G = mtp.norm(x_after_layer)                    # (1+w) norm, then logits = lm_head(G) with the MAIN lm_head
# ----------------------------------------------------------------------------------------------
MTP_NAMES = ["mtp.fc.weight", "mtp.norm.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.pre_fc_norm_embedding.weight",
             "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.post_attention_layernorm.weight",
             "mtp.layers.0.self_attn.q_proj.weight", "mtp.layers.0.self_attn.k_proj.weight",
             "mtp.layers.0.self_attn.v_proj.weight", "mtp.layers.0.self_attn.o_proj.weight",
             "mtp.layers.0.self_attn.q_norm.weight", "mtp.layers.0.self_attn.k_norm.weight",
             "mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.0.mlp.up_proj.weight", "mtp.layers.0.mlp.down_proj.weight"]


def load_mtp_bf16():
    """-> dict name -> bf16 tensor, names exactly as in model-mtp.safetensors ('mtp.fc.weight', 'mtp.layers.0....')."""
    h = _sf(MTP_ST)
    return {k: h.get_tensor(k) for k in h.keys()}


def mtp_forward(H, next_tok_ids, weights, embed_w=None):
    """H [B,L,D] (or [L,D]) = main-model post-final-norm hidden at positions 0..L-1 of each sequence;
    next_tok_ids [B,L] (or [L]) = token at position t+1 for each t (true token under teacher forcing).
    weights: dict of mtp.* tensors (any float dtype; bf16 from load_mtp_bf16() or your dequantized replacements).
    embed_w: optional [V,D] embedding table (default: checkpoint bf16 embed_tokens; MTP shares it).
    Returns G, same leading shape as H, fp32: post-mtp.norm, pre-logits."""
    squeeze = H.dim() == 2
    if squeeze:
        H, next_tok_ids = H[None], next_tok_ids[None]
    B, L, _ = H.shape
    w = {k: v.float() for k, v in weights.items()}
    E = (load_embed() if embed_w is None else embed_w)
    e = E[next_tok_ids.reshape(-1)].float()
    with torch.no_grad():
        e = rms1p(e, w["mtp.pre_fc_norm_embedding.weight"])
        hh = rms1p(H.reshape(B * L, D).float(), w["mtp.pre_fc_norm_hidden.weight"])
        x = F.linear(torch.cat([e, hh], -1), w["mtp.fc.weight"])
        p = "mtp.layers.0."
        W = {"q_proj": w[p + "self_attn.q_proj.weight"], "k_proj": w[p + "self_attn.k_proj.weight"],
             "v_proj": w[p + "self_attn.v_proj.weight"], "o_proj": w[p + "self_attn.o_proj.weight"],
             "q_norm": w[p + "self_attn.q_norm.weight"], "k_norm": w[p + "self_attn.k_norm.weight"]}
        r = x
        a = full_attention(rms1p(x, w[p + "input_layernorm.weight"]), W, B, L)
        x = r + a
        m = F.linear(F.silu(F.linear(rms1p(x, w[p + "post_attention_layernorm.weight"]), w[p + "mlp.gate_proj.weight"]))
                     * F.linear(rms1p(x, w[p + "post_attention_layernorm.weight"]), w[p + "mlp.up_proj.weight"]),
                     w[p + "mlp.down_proj.weight"])
        x = x + m
        G = rms1p(x, w["mtp.norm.weight"]).view(B, L, D)
    return G[0] if squeeze else G


# ----------------------------------------------------------------------------------------------
# prompt set
# ----------------------------------------------------------------------------------------------
def _ngram_hashes(ids, n=32):
    return {hash(tuple(ids[i:i + n])) for i in range(0, len(ids) - n + 1, 4)}


def build_prompt_set(n_windows=12, win=320, seed=1234, log=print):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tpl = open(CHAT_TEMPLATE).read()
    rng = random.Random(seed)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    bodies = []
    for f in sorted(glob.glob(f"{FLIGHTREC}/*.json")):
        d = json.load(open(f))
        try:
            s = tok.apply_chat_template(d["messages"], tools=d.get("tools"), tokenize=False, add_generation_prompt=True,
                                        chat_template=tpl, **(d.get("chat_template_kwargs") or {}))
        except Exception as ex:  # fallback: plain join
            s = "\n".join(str(m.get("content")) for m in d["messages"])
        ids = tok(s, add_special_tokens=False)["input_ids"]
        first_user = next((m["content"] for m in d["messages"] if m["role"] == "user"), "")
        series = hashlib.md5((str(first_user)[:400] + str(len(d.get("tools") or []))).encode()).hexdigest()[:8]
        sys_end = ids.index(im_end) if im_end in ids else 0
        bodies.append({"file": os.path.basename(f), "ids": ids, "sys_end": sys_end, "series": series})
    log(f"{len(bodies)} bodies, {len(set(b['series'] for b in bodies))} series")

    used = set()
    wins, manifest = [], []

    def add(ids_w, src, kind):
        hs = _ngram_hashes(ids_w)
        if used and len(hs & used) > 0.1 * max(1, len(hs)):
            return False
        used.update(hs)
        wins.append(ids_w)
        manifest.append({"src": src, "kind": kind, "preview": tok.decode(ids_w[:40]).replace("\n", "\\n")[:120]})
        return True

    n_corpus, n_prose = (2, 2) if n_windows >= 10 else (1, 1)
    n_conv = n_windows - 1 - n_corpus - n_prose  # + 1 system/tool-schema window
    # 1 window from the system/tool-schema region (most tools)
    sb = max(bodies, key=lambda b: b["sys_end"])
    off = rng.randrange(60, max(61, sb["sys_end"] - win))
    add(sb["ids"][off:off + win], f"{sb['file']}@{off}", "system_toolschema")
    # conversational windows: round-robin over series, offset after the system turn
    series_list = sorted(set(b["series"] for b in bodies))
    rng.shuffle(series_list)
    k = 0
    tries = 0
    while sum(1 for m in manifest if m["kind"] == "conversation") < n_conv and tries < 2000:
        tries += 1
        ser = series_list[k % len(series_list)]
        k += 1
        cand = [b for b in bodies if b["series"] == ser and len(b["ids"]) - b["sys_end"] > win + 50]
        if not cand:
            continue
        b = rng.choice(cand)
        off = rng.randrange(b["sys_end"] + 5, len(b["ids"]) - win)
        add(b["ids"][off:off + win], f"{b['file']}@{off}", "conversation")
    # corpus code windows
    txt = open(CORPUS).read()
    for _ in range(n_corpus):
        for _try in range(50):
            c0 = rng.randrange(0, len(txt) - 4000)
            ids_c = tok(txt[c0:c0 + 4000], add_special_tokens=False)["input_ids"]
            if len(ids_c) >= win + 20 and add(ids_c[10:10 + win], f"corpus_128000tok.txt@char{c0}", "corpus_code"):
                break
    # prose windows
    for pf in PROSE_FILES[:n_prose]:
        t = open(pf).read()
        ids_p = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids_p) < win + 20:
            continue
        off = rng.randrange(0, len(ids_p) - win)
        add(ids_p[off:off + win], f"{os.path.basename(pf)}@{off}", "prose")
    log(f"{len(wins)} windows")
    return torch.tensor(wins, dtype=torch.long), manifest


def load_ref(path=f"{OUT_DIR}/ref.pt"):
    return torch.load(path)


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--win", type=int, default=320)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ckpt", default=f"{OUT_DIR}/ckpt2")
    ap.add_argument("--out", default=f"{OUT_DIR}/ref.pt")
    a = ap.parse_args()
    T0 = time.time()
    log = lambda *s: print(f"[{time.time()-T0:7.0f}s]", *s, flush=True)

    ids, manifest = build_prompt_set(a.windows, a.win, a.seed, log=log)
    B, T = ids.shape
    log(f"ids {tuple(ids.shape)}; manifest kinds: {[m['kind'] for m in manifest]}")
    for i, m in enumerate(manifest):
        log(f"  win{i:2d} {m['kind']:18s} {m['src']:60s} {m['preview'][:70]}")

    t_main = time.time()
    H = run_main(ids, ckpt_dir=a.ckpt, log=log)  # [B,T,D]
    t_main = time.time() - t_main
    Hf = H.reshape(B * T, D)

    # ---- sanity gate: next-token prediction on own windows ----
    log("lm_head stats ...")
    tgt = torch.full((B, T), -100, dtype=torch.long)
    tgt[:, :-1] = ids[:, 1:]
    tflat = tgt.reshape(-1)
    valid = tflat >= 0
    st = lm_head_stats(Hf, tflat.clamp(min=0), topk=20)
    ce = st["lse"] - st["tgt_logit"]
    am = st["argmax"]
    hit = (am == tflat) & valid
    pos = torch.arange(T).repeat(B)
    late = valid & (pos >= 16)
    res = {
        "main_top1_acc": hit[valid].float().mean().item(),
        "main_top1_acc_pos>=16": hit[late].float().mean().item(),
        "main_mean_CE": ce[valid].mean().item(),
        "main_mean_CE_pos>=16": ce[late].mean().item(),
        "n_pos_scored": int(valid.sum()),
    }
    per_kind = {}
    for kind in sorted(set(m["kind"] for m in manifest)):
        idx = torch.tensor([i for i, m in enumerate(manifest) if m["kind"] == kind])
        msk = torch.zeros(B, T, dtype=torch.bool); msk[idx] = True
        msk = msk.reshape(-1) & late
        per_kind[kind] = {"top1": hit[msk].float().mean().item(), "CE": ce[msk].mean().item(), "n": int(msk.sum())}
    res["per_kind(pos>=16)"] = per_kind
    log("MAIN SANITY:", json.dumps(res, indent=1))

    # ---- MTP step 0 ----
    log("MTP ...")
    t_mtp = time.time()
    mw = load_mtp_bf16()
    L = T - 1
    G = mtp_forward(H[:, :L], ids[:, 1:], mw)  # [B,L,D]; position t uses H[t], tok[t+1]
    t_mtp = time.time() - t_mtp
    Gf = G.reshape(B * L, D)
    # true t+2 (exists for t<=T-3), main argmax at t+1 (exists for t<=T-2)
    am_bt = am.view(B, T)
    true2 = torch.full((B, L), -100, dtype=torch.long); true2[:, :-1] = ids[:, 2:]
    mst = lm_head_stats(Gf, None, topk=0)
    d_am = mst["argmax"].view(B, L)
    v_true = true2 >= 0
    acc_true = ((d_am == true2) & v_true)
    main_next = am_bt[:, 1:]  # target argmax at position t+1 (predicts token t+2)
    acc_main = (d_am == main_next)
    posL = torch.arange(L).expand(B, L)
    m16 = posL >= 16
    mres = {
        "mtp_top1_vs_true_t+2": acc_true[v_true].float().mean().item(),
        "mtp_top1_vs_true_t+2_pos>=16": acc_true[v_true & m16].float().mean().item(),
        "mtp_top1_vs_main_argmax_t+1 (acceptance)": acc_main.float().mean().item(),
        "mtp_top1_vs_main_argmax_t+1_pos>=16": acc_main[m16].float().mean().item(),
        "main_top1_on_same_rows_vs_true_t+2 (for reference)": hit.view(B, T)[:, 1:][v_true].float().mean().item(),
        "n_rows": int(B * L),
    }
    # mtp CE vs true t+2
    mtgt = true2.reshape(-1).clamp(min=0)
    mst2 = lm_head_stats(Gf, mtgt, topk=20)
    mce = (mst2["lse"] - mst2["tgt_logit"]).view(B, L)
    mres["mtp_mean_CE_vs_true_t+2"] = mce[v_true].mean().item()
    log("MTP:", json.dumps(mres, indent=1))

    torch.save({
        "ids": ids, "manifest": manifest, "H": H, "G": G,
        "next_tok_ids_for_mtp": ids[:, 1:].contiguous(),  # [B,L]
        "main_argmax": am.view(B, T), "main_lse": st["lse"].view(B, T), "main_tgt_logit": st["tgt_logit"].view(B, T),
        "main_topk_val": st["topk_val"].view(B, T, -1), "main_topk_idx": st["topk_idx"].view(B, T, -1),
        "mtp_argmax": d_am, "mtp_lse": mst2["lse"].view(B, L), "mtp_tgt_logit": mst2["tgt_logit"].view(B, L),
        "mtp_topk_val": mst2["topk_val"].view(B, L, -1), "mtp_topk_idx": mst2["topk_idx"].view(B, L, -1),
        "results": {"main": res, "mtp": mres},
        "meta": {"B": B, "T": T, "L": L, "D": D, "dtype": "fp32 CPU, bf16-stored scales/norms; dequantized int4 W in fp32",
                 "main_seconds": t_main, "mtp_seconds": t_mtp, "seed": a.seed,
                 "model_dir": MODEL_DIR},
    }, a.out)
    log(f"saved {a.out}  (main {t_main:.0f}s, mtp {t_mtp:.0f}s, total {time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
