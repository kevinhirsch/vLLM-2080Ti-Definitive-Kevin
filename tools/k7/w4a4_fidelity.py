#!/usr/bin/env python
"""Lane K7: CPU fidelity gate for W4A4 (rotated) on the abliterated HauhauCS W4A16 checkpoint.  No GPU, no engine.

Same harness as U2 (tools/u2/ref_dump.py, fp32 layer-streamed forward over the recorded estate prompts in ref.pt) and the
same streaming pattern as Lane LP's w4a8_fidelity.py.  Every Marlin linear (mlp gate/up/down, attn q/k/v/o, GDN
in_proj_qkv/in_proj_z/out_proj) is replaced per variant; in_proj_a/b, norms, conv, GDN/attention internals, lm_head stay.
Rotation: y = W x = (W H^T)(H x), H = blockdiag of orthonormal Hadamard(hb) along the input dim.
Weights: the SHIPPED int4 is dequantized (we have no bf16 abliterated weights), rotated, then re-quantized symmetric int4
(RTN + per-row/group MSE clip search) -> this double quantization is part of the measured cost.
Variants (one residual stream each):
  a4x     : exact rotated weights (fp32 W H^T), int4 per-token activations           -> activation error alone
  w4r     : re-quantized per-channel rotated weights, fp32 activations              -> weight double-quant error alone
  pc      : W4A4 per-channel W x per-token A, every Marlin linear                    -> what the built kernel computes
  g128    : W4A4 g128 W x g128 A (needs a group-epilogue kernel)                     -> quality ceiling of group scales
  pc_in   : pc on the input projections only (gate/up, qkv/z, q/k/v); down/out/o stay W4A16
Metrics vs the fp32 W4A16 baseline H in ref.pt, rows pos>=16: top-1 agreement, KL(p_base||p_var), dCE, hidden cosine.
Usage: CUDA_VISIBLE_DEVICES= REF_THREADS=8 nice -n 15 python tools/k7/w4a4_fidelity.py --variants a4x,w4r,pc,g128,pc_in
"""
import argparse, json, os, sys, time, types
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
sys.path.insert(0, "/home/kevin/Desktop/wt-integrate/tools/u2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_dump as R
import rotquant as RQ

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="/home/kevin/projects/lanes/u2-quant/ref.pt")
ap.add_argument("--out", default="/home/kevin/projects/lanes/k7/w4a4_fidelity.json")
ap.add_argument("--variants", default="a4x,w4r,pc,g128,pc_in")
ap.add_argument("--hb", type=int, default=128)
ap.add_argument("--act-clip", type=float, default=1.0)
ap.add_argument("--layers", type=int, default=R.N_LAYERS)
ap.add_argument("--windows", type=int, default=0, help="use only the first N windows (0 = all)")
a = ap.parse_args()
VARS = a.variants.split(",")
T0 = time.time(); log = lambda *s: print(f"[{time.time()-T0:6.0f}s]", *s, flush=True)

ref = torch.load(a.ref); ids, H0 = ref["ids"], ref["H"]
if a.windows:
    ids, H0 = ids[:a.windows], H0[:a.windows]
B, T = ids.shape; D = R.D
IN_PROJ = ("gate_proj", "up_proj", "in_proj_qkv", "in_proj_z", "q_proj", "k_proj", "v_proj")
NEED = {"a4x": ("rx",), "w4r": ("pc",), "pc": ("pc",), "g128": ("g128",), "pc_in": ("pc",)}
need = set(k for v in VARS for k in NEED[v])
QINFO, STATS, CUR = {}, {}, {"var": None}


def load_layer_weights_rot(i):
    W = R.load_layer_weights(i)
    names = ["gate_proj", "up_proj", "down_proj"] + (["q_proj", "k_proj", "v_proj", "o_proj"] if R.LAYER_TYPES[i] == "full_attention"
                                                   else ["in_proj_qkv", "in_proj_z", "out_proj"])
    for n in names:
        w = W[n]
        wr = RQ.rotate_weight(w, a.hb)
        info = {"name": f"L{i}.{n}", "short": n}
        if "rx" in need:
            info["rx"] = wr
        if "pc" in need:
            c, s = RQ.sym_quant(wr, 4, 0)
            info["pc"] = RQ.dequant(c, s, 0)
            STATS.setdefault(info["name"], {})["w_pc_sqnr_db"] = (10 * torch.log10(wr.pow(2).sum() / (info["pc"] - wr).pow(2).sum())).item()
        if "g128" in need:
            c, s = RQ.sym_quant(wr, 4, 128)
            info["g128"] = RQ.dequant(c, s, 128)
            STATS.setdefault(info["name"], {})["w_g128_sqnr_db"] = (10 * torch.log10(wr.pow(2).sum() / (info["g128"] - wr).pow(2).sum())).item()
        QINFO[id(w)] = info
    return W


_lin = torch.nn.functional.linear


def sqnr(x, y):
    return (10 * torch.log10(x.pow(2).sum() / (x - y).pow(2).sum().clamp(min=1e-30))).item()


def lin(x, w, b=None):
    info = QINFO.get(id(w)); v = CUR["var"]
    if info is None or v is None:
        return _lin(x, w, b)
    xs = x.reshape(-1, x.shape[-1])
    if v == "pc_in" and info["short"] not in IN_PROJ:
        return _lin(x, w, b)
    if v == "w4r":
        xa, wq = RQ.block_had(xs, a.hb), info["pc"]
    else:
        grp = 128 if v == "g128" else 0
        xa = RQ.act_quant(xs, 4, grp, a.hb, a.act_clip)
        wq = info["rx"] if v == "a4x" else info["g128"] if v == "g128" else info["pc"]
        st = STATS.setdefault(info["name"], {})
        key = f"act_sqnr_db_{'g128' if grp else 'pt'}"
        if key not in st:
            st[key] = sqnr(RQ.block_had(xs, a.hb), xa)
            st["act_absmax_over_rms"] = (xs.abs().amax(-1) / xs.pow(2).mean(-1).sqrt().clamp(min=1e-8)).median().item()
            xr = RQ.block_had(xs, a.hb)
            st["act_absmax_over_rms_rot"] = (xr.abs().amax(-1) / xr.pow(2).mean(-1).sqrt().clamp(min=1e-8)).median().item()
    return _lin(xa, wq, b).reshape(*x.shape[:-1], -1)


R.F = types.SimpleNamespace(**{k: getattr(torch.nn.functional, k) for k in dir(torch.nn.functional) if not k.startswith("__")})
R.F.linear = lin

xs = {v: R.load_embed()[ids.reshape(-1)].float() for v in VARS}
for i in range(a.layers):
    tl = time.time()
    QINFO.clear()
    W = load_layer_weights_rot(i)
    tq = time.time() - tl
    with torch.no_grad():
        for v in VARS:
            CUR["var"] = v
            xs[v] = R.decoder_layer(xs[v], i, W, B, T)
    CUR["var"] = None
    del W
    log(f"layer {i:2d} {R.LAYER_TYPES[i][:4]} {time.time()-tl:5.1f}s (quant {tq:4.1f}s) " + " ".join(f"{v}:rms {xs[v].pow(2).mean().sqrt():.3f}" for v in VARS))

nw = R._t(R.MAIN_ST, f"{R.PFX}.norm.weight")
Wh = R.load_lm_head().float()
tgt = torch.full((B, T), -1, dtype=torch.long); tgt[:, :-1] = ids[:, 1:]
pos = torch.arange(T).repeat(B); late = (pos >= 16)
kinds = [m["kind"] for m in ref["manifest"]][:B]
kind_row = torch.tensor([sorted(set(kinds)).index(k) for k in kinds]).repeat_interleave(T)
hf0 = H0.reshape(-1, D); tf = tgt.reshape(-1)
out = {"meta": {"rows_late": int(late.sum()), "B": B, "T": T, "layers": a.layers, "hb": a.hb, "act_clip": a.act_clip}, "variants": {}}
for v in VARS:
    hf1 = R.rms1p(xs[v], nw).reshape(-1, D)
    cos = torch.nn.functional.cosine_similarity(hf0, hf1, dim=-1)
    agree, kl, ce0, ce1, ok = [], [], [], [], []
    for r0 in range(0, B * T, 192):
        z0, z1 = hf0[r0:r0 + 192] @ Wh.T, hf1[r0:r0 + 192] @ Wh.T
        l0, l1 = torch.log_softmax(z0, -1), torch.log_softmax(z1, -1)
        agree.append((z0.argmax(-1) == z1.argmax(-1)).float()); kl.append((l0.exp() * (l0 - l1)).sum(-1))
        t = tf[r0:r0 + 192]; vv = t >= 0; tc = t.clamp(min=0)[:, None]
        ce0.append(torch.where(vv, -l0.gather(1, tc)[:, 0], torch.zeros(()))); ce1.append(torch.where(vv, -l1.gather(1, tc)[:, 0], torch.zeros(())))
        ok.append(vv.float())
    agree, kl, ce0, ce1, ok = map(torch.cat, (agree, kl, ce0, ce1, ok))
    m = late & (ok > 0)
    res = {"top1_agree": agree[late].mean().item(), "KL_mean_nats": kl[late].mean().item(), "KL_p99_nats": kl[late].quantile(0.99).item(),
           "KL_max_nats": kl[late].max().item(), "CE_base": ce0[m].mean().item(), "CE_var": ce1[m].mean().item(),
           "dCE_nats": (ce1[m] - ce0[m]).mean().item(), "H_cosine_mean": cos.mean().item(), "H_cosine_p01": cos.quantile(0.01).item(),
           "per_kind_top1": {k: agree[late & (kind_row == j)].mean().item() for j, k in enumerate(sorted(set(kinds)))}}
    out["variants"][v] = res
    log(v, json.dumps(res))
S = STATS
def summ(key):
    vals = [s[key] for s in S.values() if key in s]
    return {"min": min(vals), "median": sorted(vals)[len(vals) // 2], "n": len(vals)} if vals else None
out["linear_summary"] = {k: summ(k) for k in ("w_pc_sqnr_db", "w_g128_sqnr_db", "act_sqnr_db_pt", "act_sqnr_db_g128", "act_absmax_over_rms", "act_absmax_over_rms_rot")}
by_type = {}
for k, s in S.items():
    t = k.split(".", 1)[1]
    for kk in ("act_sqnr_db_pt", "act_sqnr_db_g128", "w_pc_sqnr_db"):
        if kk in s:
            by_type.setdefault(t, {}).setdefault(kk, []).append(s[kk])
out["by_type_median"] = {t: {kk: sorted(v)[len(v) // 2] for kk, v in d.items()} for t, d in by_type.items()}
out["per_linear"] = S
out["meta"]["seconds"] = time.time() - T0
json.dump(out, open(a.out, "w"), indent=1)
log("SUMMARY", json.dumps(out["linear_summary"]))
log("BY TYPE", json.dumps(out["by_type_median"]))
