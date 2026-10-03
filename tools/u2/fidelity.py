#!/usr/bin/env python
"""Lane U2 fidelity gate for the int4 lm_head / int4 MTP block arms.  CPU only (no GPU, no engine).

Input : ref.pt written by tools/u2/ref_dump.py (fp32 CPU reference forward of the shipped W4A16 model over recorded
        estate prompts: post-final-norm hidden H, MTP step-0 output G, ids).
Output: JSON + printed table.  Variants (draft hidden, draft head, target head):
          base  = (bf16 MTP block, bf16 head, bf16 head)            -- what runs today
          head  = (bf16 MTP block, int4 head, int4 head)            -- VLLM_U2_INT4_HEAD=1  (MTP shares the target head)
          mtp   = (int4 MTP block, bf16 head, bf16 head)            -- VLLM_U2_INT4_MTP=1
          both  = (int4 MTP block, int4 head, int4 head)
Metrics, on rows with position >= 16 (the first tokens of a window have no context) unless stated:
  M1 target-head fidelity (bf16 head vs int4 head, same hidden state H):
       top1_agree, top5_contains (bf16 top-1 inside int4 top-5), KL(p_bf16 || p_int4) mean/p99/max (nats, T=1),
       TV (total variation), dCE (cross-entropy vs the true next token, int4 - bf16), flip_margin (median logit margin of flipped rows)
  M2 MTP step-0 acceptance (draft argmax == target argmax at the same position; target = the head of that variant):
       greedy_accept per variant and delta vs base; sampled_accept = E[sum_x min(p_target, q_draft)] at T=1 and T=0.6
Usage: CUDA_VISIBLE_DEVICES= python tools/u2/fidelity.py [--ref ~/projects/lanes/u2-quant/ref.pt] [--out fidelity.json]
"""
import argparse, importlib.util, json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ref_dump as R  # noqa: E402
spec = importlib.util.spec_from_file_location("u2hq", os.path.join(HERE, "../../vllm/model_executor/layers/quantization/u2_headquant.py"))
hq = importlib.util.module_from_spec(spec); spec.loader.exec_module(hq)

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="/home/kevin/projects/lanes/u2-quant/ref.pt")
ap.add_argument("--out", default=os.path.join(HERE, "fidelity.json"))
ap.add_argument("--block", type=int, default=192)
ap.add_argument("--model", default=R.MODEL_DIR)
a = ap.parse_args()
T0 = time.time()
log = lambda *s: print(f"[{time.time()-T0:6.0f}s]", *s, flush=True)


class L:
    info = staticmethod(lambda f, *x: log("u2:", f % x)); warning = info


ref = torch.load(a.ref)
ids, H, G, manifest = ref["ids"], ref["H"], ref["G"], ref["manifest"]
B, T, D = H.shape
Lq = T - 1
log(f"ref: B={B} T={T}  windows kinds={[m['kind'] for m in manifest]}")

# --- weights -------------------------------------------------------------------------------------
Wb_bf16 = R.load_lm_head()
q = hq.get_quantized(a.model, "lm_head.weight", Wb_bf16, L)
Wq = hq.dequantize_linear(q)  # fp32 [V, D]
Wb = Wb_bf16.float()
log(f"head: bf16 {tuple(Wb.shape)}  int4 reconstruction SQNR {10*torch.log10(Wb.pow(2).sum()/(Wb-Wq).pow(2).sum()).item():.2f} dB")

mw = R.load_mtp_bf16()
mwq = dict(mw)
for n in list(mw):
    if hq._MTP_LINEAR.match(n):
        t = hq.get_quantized(a.model, n, mw[n], L)
        mwq[n] = hq.dequantize_linear(t).to(torch.bfloat16)
Gq = R.mtp_forward(H[:, :Lq], ids[:, 1:], mwq)  # int4 MTP block hidden
log(f"MTP block int4 recomputed; rel diff G: {((Gq-G).norm()/G.norm()).item():.4f}")

# --- rows: r=(b,t), t<Lq.  target rows = H[:, 1:]; draft rows = G[:, :Lq]; true next token for target = ids[b, t+2]
Hn = H[:, 1:].reshape(-1, D)
Gb = G.reshape(-1, D)
Gqf = Gq.reshape(-1, D)
true = torch.full((B, Lq), -1, dtype=torch.long)
true[:, :-1] = ids[:, 2:]
true = true.reshape(-1)
pos = torch.arange(Lq).repeat(B)
kind = torch.tensor([sorted(set(m["kind"] for m in manifest)).index(m["kind"]) for m in manifest]).repeat_interleave(Lq)
kinds = sorted(set(m["kind"] for m in manifest))
P = Hn.shape[0]


def lsm(z, temp=1.0):
    return torch.log_softmax(z / temp, -1)


acc = {k: [] for k in ("top1", "top5", "kl", "tv", "ce_b", "ce_q", "margin_flip", "flip")}
var_names = ("base", "head", "mtp", "both")
dr = {v: {"greedy": [], "s10": [], "s06": []} for v in var_names}
for r0 in range(0, P, a.block):
    sl = slice(r0, min(P, r0 + a.block))
    h, g, gq = Hn[sl], Gb[sl], Gqf[sl]
    zt_b, zt_q = h @ Wb.T, h @ Wq.T
    zd = {"base": g @ Wb.T, "head": g @ Wq.T, "mtp": gq @ Wb.T, "both": gq @ Wq.T}
    tgt = {"base": zt_b, "head": zt_q, "mtp": zt_b, "both": zt_q}
    # M1
    lp_b, lp_q = lsm(zt_b), lsm(zt_q)
    pb = lp_b.exp()
    acc["kl"].append((pb * (lp_b - lp_q)).sum(-1))
    acc["tv"].append(0.5 * (pb - lp_q.exp()).abs().sum(-1))
    ab, aq = zt_b.argmax(-1), zt_q.argmax(-1)
    acc["top1"].append((ab == aq).float())
    acc["top5"].append((zt_q.topk(5, -1).indices == ab[:, None]).any(-1).float())
    t = true[sl]
    okr = t >= 0
    tt = t.clamp(min=0)[:, None]
    acc["ce_b"].append(torch.where(okr, -lp_b.gather(1, tt)[:, 0], torch.zeros(())))
    acc["ce_q"].append(torch.where(okr, -lp_q.gather(1, tt)[:, 0], torch.zeros(())))
    top2 = zt_b.topk(2, -1).values
    acc["margin_flip"].append(top2[:, 0] - top2[:, 1])
    acc["flip"].append((ab != aq).float())
    # M2
    for v in var_names:
        lq_, lt_ = lsm(zd[v]), lsm(tgt[v])
        dr[v]["greedy"].append((zd[v].argmax(-1) == tgt[v].argmax(-1)).float())
        dr[v]["s10"].append(torch.minimum(lq_.exp(), lt_.exp()).sum(-1))
        dr[v]["s06"].append(torch.minimum(lsm(zd[v], 0.6).exp(), lsm(tgt[v], 0.6).exp()).sum(-1))
    del zt_b, zt_q, zd, lp_b, lp_q
    log(f"rows {sl.stop}/{P}")

cat = lambda l: torch.cat(l)
A = {k: cat(v) for k, v in acc.items()}
Dr = {v: {k: cat(x) for k, x in d.items()} for v, d in dr.items()}
late = pos >= 16
valid_ce = (true >= 0) & late


def summarize(mask):
    out = {"n_rows": int(mask.sum())}
    out["M1"] = {
        "top1_agree": A["top1"][mask].mean().item(),
        "top5_contains": A["top5"][mask].mean().item(),
        "KL_mean_nats": A["kl"][mask].mean().item(),
        "KL_p99_nats": A["kl"][mask].quantile(0.99).item(),
        "KL_max_nats": A["kl"][mask].max().item(),
        "TV_mean": A["tv"][mask].mean().item(),
        "flipped_rows": int(A["flip"][mask].sum()),
        "flip_median_margin_logit": (A["margin_flip"][mask & (A["flip"] > 0)].median().item() if (A["flip"][mask] > 0).any() else None),
    }
    vm = mask & (true >= 0)
    out["M1"]["CE_bf16"] = A["ce_b"][vm].mean().item()
    out["M1"]["CE_int4"] = A["ce_q"][vm].mean().item()
    out["M1"]["dCE_nats"] = out["M1"]["CE_int4"] - out["M1"]["CE_bf16"]
    out["M2"] = {}
    for v in var_names:
        out["M2"][v] = {k: Dr[v][k][mask].mean().item() for k in ("greedy", "s10", "s06")}
    for v in var_names[1:]:
        out["M2"][v]["delta_greedy_vs_base"] = out["M2"][v]["greedy"] - out["M2"]["base"]["greedy"]
        out["M2"][v]["delta_s10_vs_base"] = out["M2"][v]["s10"] - out["M2"]["base"]["s10"]
        out["M2"][v]["delta_s06_vs_base"] = out["M2"][v]["s06"] - out["M2"]["base"]["s06"]
    return out


res = {"all_pos>=16": summarize(late)}
for i, k in enumerate(kinds):
    res[f"kind={k}"] = summarize(late & (kind == i))
res["meta"] = {"B": B, "T": T, "rows": P, "ref": a.ref, "grid": hq.CLIP_GRID, "group": hq.GROUP,
               "head_sqnr_db": 10 * torch.log10(Wb.pow(2).sum() / (Wb - Wq).pow(2).sum()).item()}
json.dump(res, open(a.out, "w"), indent=1)
s = res["all_pos>=16"]
log("RESULT (rows pos>=16, n=%d)" % s["n_rows"])
print(json.dumps(s, indent=1))
log(f"wrote {a.out}")
