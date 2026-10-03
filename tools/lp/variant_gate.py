#!/usr/bin/env python
"""Lane LP SHARED QUANT PIPELINE: CPU fidelity gate for any per-linear numeric variant of the shipped abliterated W4A16 model.
(K7 W4A4 / INT4 heads and K8 FP8-storage / 2:4-sparsity reuse this; add a variant = add one function to VARIANTS.)

Stages (all CPU, no GPU, original weights read-only):
  1. source   : shipped CT pack-quantized asym g128 W4 (ref_dump.dequant_linear), exact (q - zp) and scales kept per linear
  2. variant  : f(name, x, info) -> y   for every Marlin linear (mlp gate/up/down, attn q/k/v/o, GDN in_proj_qkv/z, out_proj)
  3. forward  : U2 layer-streamed fp32 reference over the U2 estate windows (ids from ref.pt), one residual stream per variant
  4. gate     : vs the W4A16 baseline in ref.pt: next-token top-1 agreement, KL(p_base||p_var) mean/p99/max, dCE, hidden cosine
Calibrated reading (U2b, same harness): int4 embeddings = top1 98.2%, KL 0.011, dCE +0.003 -> passed and shipped.
Built-in variants:
  fp16       linear inputs+weights rounded to fp16 (engine numerics floor vs this fp32 reference)
  w4a8       kernel-exact s8 x u4(zp) Marlin: per-token int8 act, int16 group scales (per-layer max -> 4096)
  w4a8nd     w4a8 except mlp.down_proj stays W4A16
  w4a8sm     w4a8 with SmoothQuant-style migration on down_proj only (alpha 0.5, weights re-quantized RTN asym g128 from the W4 dequant)
  w4a8gd     w4a8 but down_proj uses per-(row,128-group) activation scales (MX-style int8 epilogue)
  w4a8g      per-(row,128-group) activation scales on every linear
  w4a8hd     w4a8 but down_proj input block-Hadamard rotated (weights re-quantized RTN asym g128 from the W4 dequant)
  w4a4tok    naive per-token int4 activations (no rotation) on asym W4 -- the "no tricks" W4A4 floor
  w4a4had    QuaRot-style: block-Hadamard(128) on the K dim of act and weight, weights re-quantized RTN asym g128, per-token int4 act
  sp24       2:4 structured sparsity on the W4 weights by Wanda score (|w| * ||x_k||), kept values = original W4 values
  fp8w       weights stored FP8 E4M3 per-output-channel scale (from the W4 dequant) with fp16 act -- storage-format check (K8)
Usage: CUDA_VISIBLE_DEVICES= nice -n 15 python tools/lp/variant_gate.py --variants fp16,w4a4had,sp24 --out ~/projects/lanes/lp/gate_X.json
"""
import argparse, json, math, os, sys, time, types
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
import ref_dump as R

G = 128


def int4_rtn_asym(w):
    """RTN asym 4-bit g128 (memoryless minmax like the shipped recipe) -> dequantized fp32."""
    o, i = w.shape; wg = w.view(o, i // G, G)
    mn, mx = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
    s = ((mx - mn) / 15).clamp(min=1e-10); z = torch.round(-mn / s).clamp(0, 15)
    return ((torch.clamp(torch.round(wg / s) + z, 0, 15) - z) * s).view(o, i)


def hadamard(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


H128 = hadamard(G)


def blk_had(x):  # block-diagonal Hadamard on the last dim (QuaRot-style online rotation per 128-block)
    return (x.reshape(*x.shape[:-1], -1, G) @ H128).reshape(x.shape)


def q_tok(x, bits):
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / qmax
    return torch.clamp(torch.round(x / s), -qmax, qmax) * s


def int_scales(info):
    s16 = info["s"].half().float(); smax = s16.max(); s_int = torch.round(s16 / smax * 4096)
    return (info["qz"] * (s_int * smax / 4096).unsqueeze(-1)).reshape(info["w"].shape)


def v_fp16(n, x, info):
    return x.half().float() @ info["w"].half().float().T


def v_w4a8(n, x, info):
    if "wq8" not in info: info["wq8"] = int_scales(info)
    return q_tok(x.half().float(), 8) @ info["wq8"].T


def v_w4a8nd(n, x, info):
    return v_fp16(n, x, info) if n.endswith("down_proj") else v_w4a8(n, x, info)


def v_w4a8sm(n, x, info, alpha=0.5):
    if not n.endswith("down_proj"):
        return v_w4a8(n, x, info)
    if "sm" not in info:  # smoothing factors from this call's activations (calibration = first batch seen; one batch per layer here)
        ax = x.abs().amax(0).clamp(min=1e-5); aw = info["w"].abs().amax(0).clamp(min=1e-5)
        sm = (ax.pow(alpha) / aw.pow(1 - alpha)).clamp(min=1e-2, max=1e2)
        info["sm"] = sm; info["wsm"] = int4_rtn_asym(info["w"] * sm)
    return q_tok((x / info["sm"]).half().float(), 8) @ info["wsm"].T


def q_grp(x, bits, g=G):
    """per-(row, g-block) symmetric scales (MX-style int8 with a float scale per 128-K group: needs the s8 Marlin epilogue to apply
    a_scale[row, group] * w_scale[group, col] per group in fp32 instead of one per-row scale at the end)."""
    qmax = 2 ** (bits - 1) - 1
    xg = x.reshape(x.shape[0], -1, g)
    sc = xg.abs().amax(-1, keepdim=True).clamp(min=1e-12) / qmax
    return (torch.clamp(torch.round(xg / sc), -qmax, qmax) * sc).reshape(x.shape)


def v_w4a8gd(n, x, info):
    if not n.endswith("down_proj"):
        return v_w4a8(n, x, info)
    if "wq8" not in info: info["wq8"] = int_scales(info)
    return q_grp(x.half().float(), 8) @ info["wq8"].T


def v_w4a8g(n, x, info):
    """kernel-exact LP_A8G: per-(row,128) int8 act scales, REAL fp16 weight group scales (no int16 rounding)."""
    if "w16s" not in info: info["w16s"] = (info["qz"] * info["s"].half().float().unsqueeze(-1)).reshape(info["w"].shape)
    return q_grp(x.half().float(), 8) @ info["w16s"].T


def v_w4a8hd(n, x, info):
    if not n.endswith("down_proj"):
        return v_w4a8(n, x, info)
    if "wh" not in info: info["wh"] = int4_rtn_asym(blk_had(info["w"]))
    return q_tok(blk_had(x.half().float()), 8) @ info["wh"].T


def v_w4a4tok(n, x, info):
    return q_tok(x.half().float(), 4) @ info["w"].T


def v_w4a4had(n, x, info):
    if "wh" not in info: info["wh"] = int4_rtn_asym(blk_had(info["w"]))
    return q_tok(blk_had(x.half().float()), 4) @ info["wh"].T


def v_sp24(n, x, info):
    if "wsp" not in info:
        w = info["w"]; score = w.abs() * x.reshape(-1, x.shape[-1]).norm(dim=0).unsqueeze(0)
        sc = score.view(w.shape[0], -1, 4); keep = torch.zeros_like(sc, dtype=torch.bool)
        keep.scatter_(-1, sc.topk(2, dim=-1).indices, True)
        info["wsp"] = (w.view_as(sc) * keep).view_as(w)
    return x.half().float() @ info["wsp"].T


def v_fp8w(n, x, info):
    if "w8" not in info:
        w = info["w"]; s = w.abs().amax(1, keepdim=True).clamp(min=1e-12) / 448
        info["w8"] = (w / s).to(torch.float8_e4m3fn).float() * s
    return x.half().float() @ info["w8"].T


VARIANTS = {k[2:]: v for k, v in globals().items() if k.startswith("v_")}

# Plugins (K7 2026-10-03): VG_PLUGINS=/path/a.py,/path/b.py -> each file's register(VARIANTS) adds its variants.
for _pf in filter(None, os.environ.get("VG_PLUGINS", "").split(",")):
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location(os.path.splitext(os.path.basename(_pf))[0], _pf)
    _pm = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_pm); _pm.register(VARIANTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/home/kevin/projects/lanes/u2-quant/ref.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="fp16,w4a8")
    ap.add_argument("--layers", type=int, default=R.N_LAYERS)
    ap.add_argument("--windows", type=int, default=0, help="first N windows of ref.pt only (0 = all 12)")
    a = ap.parse_args()
    VARS = a.variants.split(",")
    for v in VARS: assert v in VARIANTS, f"unknown variant {v}; have {list(VARIANTS)}"
    T0 = time.time(); log = lambda *s: print(f"[{time.time()-T0:6.0f}s]", *s, flush=True)
    ref = torch.load(a.ref); ids, H0 = ref["ids"], ref["H"]
    if a.windows: ids, H0 = ids[:a.windows], H0[:a.windows]
    B, T = ids.shape; D = R.D
    QI, CUR = {}, {"v": None}

    def load(i):
        W = R.load_layer_weights(i); h = R._sf(R.MAIN_ST); p = f"{R.PFX}.layers.{i}"
        names = [("mlp", n) for n in ("gate_proj", "up_proj", "down_proj")]
        names += [("self_attn", n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")] if R.LAYER_TYPES[i] == "full_attention" else \
                 [("linear_attn", n) for n in ("in_proj_qkv", "in_proj_z", "out_proj")]
        for blk, n in names:
            s = h.get_tensor(f"{p}.{blk}.{n}.weight_scale").float(); w = W[n]
            qz = (w.view(w.shape[0], s.shape[1], -1) / s.unsqueeze(-1)).round()
            QI[id(w)] = {"name": f"L{i}.{n}", "w": w, "s": s, "qz": qz, "per_var": {}}
        return W

    _lin = torch.nn.functional.linear

    def lin(x, w, b=None):
        info = QI.get(id(w)); v = CUR["v"]
        if info is None or v is None:
            return _lin(x, w, b)
        pv = info["per_var"].setdefault(v, {"w": info["w"], "s": info["s"], "qz": info["qz"]})
        xs = x.reshape(-1, x.shape[-1])
        return VARIANTS[v](info["name"], xs, pv).reshape(*x.shape[:-1], -1)

    R.F = types.SimpleNamespace(**{k: getattr(torch.nn.functional, k) for k in dir(torch.nn.functional) if not k.startswith("__")}); R.F.linear = lin
    xs = {v: R.load_embed()[ids.reshape(-1)].float() for v in VARS}
    for i in range(a.layers):
        tl = time.time(); QI.clear(); W = load(i)
        with torch.no_grad():
            for v in VARS:
                CUR["v"] = v; xs[v] = R.decoder_layer(xs[v], i, W, B, T)
        CUR["v"] = None; del W
        log(f"layer {i:2d} {time.time()-tl:5.1f}s " + " ".join(f"{v}:{xs[v].pow(2).mean().sqrt():.3f}" for v in VARS))
    nw = R._t(R.MAIN_ST, f"{R.PFX}.norm.weight"); Wh = R.load_lm_head().float()
    tgt = torch.full((B, T), -1, dtype=torch.long); tgt[:, :-1] = ids[:, 1:]
    late = torch.arange(T).repeat(B) >= 16; hf0 = H0.reshape(-1, D); tf = tgt.reshape(-1)
    out = {"variants": {}, "meta": {"layers": a.layers, "rows_late": int(late.sum())}}
    for v in VARS:
        hf1 = R.rms1p(xs[v], nw).reshape(-1, D); cos = torch.nn.functional.cosine_similarity(hf0, hf1, dim=-1)
        ag, kl, c0, c1, ok = [], [], [], [], []
        for r0 in range(0, B * T, 192):
            z0, z1 = hf0[r0:r0 + 192] @ Wh.T, hf1[r0:r0 + 192] @ Wh.T; l0, l1 = torch.log_softmax(z0, -1), torch.log_softmax(z1, -1)
            ag.append((z0.argmax(-1) == z1.argmax(-1)).float()); kl.append((l0.exp() * (l0 - l1)).sum(-1))
            t = tf[r0:r0 + 192]; vv = t >= 0; tc = t.clamp(min=0)[:, None]
            c0.append(torch.where(vv, -l0.gather(1, tc)[:, 0], torch.zeros(()))); c1.append(torch.where(vv, -l1.gather(1, tc)[:, 0], torch.zeros(()))); ok.append(vv.float())
        ag, kl, c0, c1, ok = map(torch.cat, (ag, kl, c0, c1, ok)); m = late & (ok > 0)
        res = {"top1_agree": ag[late].mean().item(), "KL_mean_nats": kl[late].mean().item(), "KL_p99_nats": kl[late].quantile(0.99).item(),
               "KL_max_nats": kl[late].max().item(), "dCE_nats": (c1[m] - c0[m]).mean().item(), "CE_var": c1[m].mean().item(),
               "H_cosine_mean": cos.mean().item(), "H_cosine_p01": cos.quantile(0.01).item()}
        out["variants"][v] = res; log(v, json.dumps(res))
    out["meta"]["seconds"] = time.time() - T0
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
