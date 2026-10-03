#!/usr/bin/env python
"""Lane LP: CPU fidelity gate for W4A8-INT8 Marlin on the SHIPPED asymmetric W4 checkpoint (no re-quant).

Emulates exactly what the s8 x u4(zp) Marlin kernel computes for every Marlin linear of the main model
(mlp gate/up/down, attn q/k/v/o, GDN in_proj_qkv/in_proj_z/out_proj; in_proj_a/b stay fp16, lm_head/MTP untouched):
  * activations: per-token symmetric int8 (vLLM per_token_quant_int8: scale = absmax/127, round, clamp)
  * group scales: marlin_act_int8_process_scales -> s_int = round(s / s.max() * 4096) (int16), global = s.max()/4096
  * weights: (q - zp) exact in int8; accumulation exact (fp32 holds the int32 partials exactly, |partial| << 2^24)
Variants carried in ONE layer-streamed pass over the SAME ids as ref.pt (U2 harness, CPU fp32):
  fp16  : linear inputs rounded to fp16 (noise floor of what the engine already does vs this fp32 reference)
  w4a8  : kernel-exact int8 emulation above
  w4a8nd: w4a8 everywhere except mlp.down_proj kept W4A16 (down_proj input = silu(gate)*up has the heaviest outliers)
Compares each vs the fp32 W4A16 baseline in ref.pt: next-token top-1 agreement, KL(p_base||p_var), dCE, hidden cosine.
Also logs per-linear activation-quant SQNR and the max |int32 group partial * s_int| (overflow headroom) on a row sample.
Usage: CUDA_VISIBLE_DEVICES= nice -n 15 python tools/lp/w4a8_fidelity.py --out ~/projects/lanes/lp/w4a8_fidelity.json
"""
import argparse, json, os, sys, time, types
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
import ref_dump as R

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="/home/kevin/projects/lanes/u2-quant/ref.pt")
ap.add_argument("--out", default="/home/kevin/projects/lanes/lp/w4a8_fidelity.json")
ap.add_argument("--variants", default="fp16,w4a8")
ap.add_argument("--layers", type=int, default=R.N_LAYERS)
a = ap.parse_args()
VARS = a.variants.split(",")
T0 = time.time(); log = lambda *s: print(f"[{time.time()-T0:6.0f}s]", *s, flush=True)

ref = torch.load(a.ref); ids, H0 = ref["ids"], ref["H"]
B, T = ids.shape; D = R.D

QINFO = {}    # id(weight fp32 tensor) -> dict(name, s_eff-weight tensor)
STATS = {}    # name -> list of stats
CUR = {"var": None, "layer": -1}


def load_layer_weights_q(i):
    """Like R.load_layer_weights, but each Marlin linear also gets the int-scale weight (kernel-exact for w4a8)."""
    W = R.load_layer_weights(i)
    h = R._sf(R.MAIN_ST); p = f"{R.PFX}.layers.{i}"
    names = [("mlp", n) for n in ("gate_proj", "up_proj", "down_proj")]
    names += [("self_attn", n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")] if R.LAYER_TYPES[i] == "full_attention" else \
             [("linear_attn", n) for n in ("in_proj_qkv", "in_proj_z", "out_proj")]
    for blk, n in names:
        pre = f"{p}.{blk}.{n}"
        s = h.get_tensor(pre + ".weight_scale").to(torch.float32)          # [out, ng] (bf16 stored; engine casts to fp16)
        s16 = s.half().float()
        smax = s16.max()
        s_int = torch.round(s16 / smax * 4096)
        s_eff = s_int * (smax / 4096)
        w = W[n]                                                            # (q-zp)*s fp32 [out, in]
        ng = s.shape[1]
        qz = (w.view(w.shape[0], ng, -1) / s.unsqueeze(-1)).round()       # exact (q-zp) integers
        wq = (qz * s_eff.unsqueeze(-1)).reshape(w.shape)
        rel = ((wq - w).norm() / w.norm()).item()
        QINFO[id(w)] = {"name": f"L{i}.{n}", "wq": wq, "qz": qz, "s_int": s_int, "zero_groups": int((s_int == 0).sum())}
        STATS.setdefault(f"L{i}.{n}", {})["scale_int_rel_err"] = rel
        STATS[f"L{i}.{n}"]["zero_groups"] = int((s_int == 0).sum())
    return W


_lin = torch.nn.functional.linear


def lin(x, w, b=None):
    info = QINFO.get(id(w))
    v = CUR["var"]
    if info is None or v is None:
        return _lin(x, w, b)
    if v == "fp16" or (v == "w4a8nd" and info["name"].endswith("down_proj")):
        return _lin(x.half().float(), w.half().float(), b)
    # w4a8: per-token int8
    xs = x.reshape(-1, x.shape[-1])
    amax = xs.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    xh = xs.half().float()
    sc = (amax.half().float() / 127.0)
    xq = torch.clamp(torch.round(xh / sc), -127, 127)
    st = STATS.setdefault(info["name"], {})
    if "act_sqnr_db" not in st:
        err = (xq * sc - xs)
        st["act_sqnr_db"] = (10 * torch.log10(xs.pow(2).sum() / err.pow(2).sum().clamp(min=1e-30))).item()
        # overflow headroom: max |sum_g(xq*qz)| * s_int over a 64-row sample (int32 limit 2^31)
        rs = xq[:: max(1, xq.shape[0] // 64)][:64]
        qz, s_int = info["qz"], info["s_int"]                              # [out, ng, 128], [out, ng]
        ng = qz.shape[1]
        part = torch.einsum("rgk,ogk->rog", rs.view(rs.shape[0], ng, -1), qz)  # group partials
        acc = (part * s_int.unsqueeze(0)).cumsum(-1).abs().amax().item()
        st["max_abs_int32_acc"] = acc
        st["int32_headroom_x"] = 2**31 / max(acc, 1)
        st["act_absmax_over_rms"] = (amax / xs.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-8)).median().item()
    y = _lin(xq * sc, info["wq"], b)
    return y.reshape(*x.shape[:-1], -1)


R.F = types.SimpleNamespace(**{k: getattr(torch.nn.functional, k) for k in dir(torch.nn.functional) if not k.startswith("__")})
R.F.linear = lin

# ---- layer-streamed forward, one residual stream per variant ----
xs = {v: R.load_embed()[ids.reshape(-1)].float() for v in VARS}
for i in range(a.layers):
    tl = time.time()
    QINFO.clear()
    W = load_layer_weights_q(i)
    with torch.no_grad():
        for v in VARS:
            CUR["var"] = v
            xs[v] = R.decoder_layer(xs[v], i, W, B, T)
    CUR["var"] = None
    del W
    log(f"layer {i:2d} {R.LAYER_TYPES[i][:4]} {time.time()-tl:5.1f}s " + " ".join(f"{v}:rms {xs[v].pow(2).mean().sqrt():.3f}" for v in VARS))

nw = R._t(R.MAIN_ST, f"{R.PFX}.norm.weight")
Wh = R.load_lm_head().float()
tgt = torch.full((B, T), -1, dtype=torch.long); tgt[:, :-1] = ids[:, 1:]
pos = torch.arange(T).repeat(B); late = (pos >= 16)
hf0 = H0.reshape(-1, D); tf = tgt.reshape(-1)
out = {"meta": {"rows_late": int(late.sum()), "B": B, "T": T, "layers": a.layers, "seconds": None}, "variants": {}}
for v in VARS:
    hf1 = R.rms1p(xs[v], nw).reshape(-1, D)
    cos = torch.nn.functional.cosine_similarity(hf0, hf1, dim=-1)
    agree, kl, ce0, ce1, ok = [], [], [], [], []
    for r0 in range(0, B * T, 192):
        z0, z1 = hf0[r0:r0 + 192] @ Wh.T, hf1[r0:r0 + 192] @ Wh.T
        l0, l1 = torch.log_softmax(z0, -1), torch.log_softmax(z1, -1)
        agree.append((z0.argmax(-1) == z1.argmax(-1)).float()); kl.append((l0.exp() * (l0 - l1)).sum(-1))
        t = tf[r0:r0 + 192]; vv = t >= 0; tc = t.clamp(min=0)[:, None]
        ce0.append(torch.where(vv, -l0.gather(1, tc)[:, 0], torch.zeros(()))); ce1.append(torch.where(vv, -l1.gather(1, tc)[:, 0], torch.zeros(()))); ok.append(vv.float())
    agree, kl, ce0, ce1, ok = map(torch.cat, (agree, kl, ce0, ce1, ok))
    m = late & (ok > 0)
    res = {"top1_agree": agree[late].mean().item(), "KL_mean_nats": kl[late].mean().item(), "KL_p99_nats": kl[late].quantile(0.99).item(),
           "KL_max_nats": kl[late].max().item(), "CE_base": ce0[m].mean().item(), "CE_var": ce1[m].mean().item(),
           "dCE_nats": (ce1[m] - ce0[m]).mean().item(), "H_cosine_mean": cos.mean().item(), "H_cosine_p01": cos.quantile(0.01).item()}
    out["variants"][v] = res
    log(v, json.dumps(res))
lin_stats = {k: s for k, s in STATS.items()}
sq = [s["act_sqnr_db"] for s in lin_stats.values() if "act_sqnr_db" in s]
hr = [s["int32_headroom_x"] for s in lin_stats.values() if "int32_headroom_x" in s]
se = [s["scale_int_rel_err"] for s in lin_stats.values()]
out["linear_summary"] = {"act_sqnr_db_min": min(sq) if sq else None, "act_sqnr_db_median": sorted(sq)[len(sq)//2] if sq else None,
                         "int32_headroom_min_x": min(hr) if hr else None, "scale_int_rel_err_max": max(se), "scale_int_rel_err_median": sorted(se)[len(se)//2],
                         "zero_groups_total": sum(s.get("zero_groups", 0) for s in lin_stats.values())}
worst = sorted(((s.get("act_sqnr_db", 99), k) for k, s in lin_stats.items()))[:12]
out["worst_act_sqnr"] = worst
out["per_linear"] = lin_stats
out["meta"]["seconds"] = time.time() - T0
json.dump(out, open(a.out, "w"), indent=1)
log("SUMMARY", json.dumps(out["linear_summary"]), "worst", worst[:6])
