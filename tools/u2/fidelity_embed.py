#!/usr/bin/env python
"""Lane U2 fidelity gate for int4 (symmetric g128) embeddings: re-runs the CPU fp32 reference forward (~13 min) with a dequantized embed table
over the SAME ids as ref.pt, then compares hidden state H and next-token distribution (bf16 head) against the baseline in ref.pt.
Output: top-1 agreement, KL(p_base||p_embed) mean/p99, dCE.  Usage: CUDA_VISIBLE_DEVICES= python tools/u2/fidelity_embed.py"""
import argparse, importlib.util, json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ref_dump as R
spec = importlib.util.spec_from_file_location("u2hq", os.path.join(HERE, "../../vllm/model_executor/layers/quantization/u2_headquant.py"))
hq = importlib.util.module_from_spec(spec); spec.loader.exec_module(hq)
ap = argparse.ArgumentParser(); ap.add_argument("--ref", default="/home/kevin/projects/lanes/u2-quant/ref.pt")
ap.add_argument("--out", default=os.path.join(HERE, "fidelity_embed.json")); ap.add_argument("--ckpt", default="/home/kevin/projects/lanes/u2-quant/ckpt_embed")
a = ap.parse_args()
T0 = time.time(); log = lambda *s: print(f"[{time.time()-T0:6.0f}s]", *s, flush=True)
class L: info = staticmethod(lambda f, *x: log("u2:", f % x)); warning = info
ref = torch.load(a.ref); ids, H0 = ref["ids"], ref["H"]
name = "model.language_model.embed_tokens.weight"
E = R.load_embed()
q = hq.get_quantized(R.MODEL_DIR, name, E, L)
R._embed = hq.dequantize_linear(q).bfloat16()  # what the kernel produces (scale dtype = bf16/fp16 output)
log("embed SQNR dB", 10 * torch.log10(E.float().pow(2).sum() / (E.float() - R._embed.float()).pow(2).sum()).item())
H1 = R.run_main(ids, ckpt_dir=a.ckpt, log=log)
B, T, D = H1.shape
cos = torch.nn.functional.cosine_similarity(H0.reshape(-1, D), H1.reshape(-1, D), dim=-1)
W = R.load_lm_head().float()
tgt = torch.full((B, T), -1, dtype=torch.long); tgt[:, :-1] = ids[:, 1:]
pos = torch.arange(T).repeat(B); late = (pos >= 16)
agree, kl, ce0, ce1, ok = [], [], [], [], []
hf0, hf1 = H0.reshape(-1, D), H1.reshape(-1, D); tf = tgt.reshape(-1)
for r0 in range(0, B * T, 192):
    z0, z1 = hf0[r0:r0 + 192] @ W.T, hf1[r0:r0 + 192] @ W.T
    l0, l1 = torch.log_softmax(z0, -1), torch.log_softmax(z1, -1)
    agree.append((z0.argmax(-1) == z1.argmax(-1)).float()); kl.append((l0.exp() * (l0 - l1)).sum(-1))
    t = tf[r0:r0 + 192]; v = t >= 0; tc = t.clamp(min=0)[:, None]
    ce0.append(torch.where(v, -l0.gather(1, tc)[:, 0], torch.zeros(()))); ce1.append(torch.where(v, -l1.gather(1, tc)[:, 0], torch.zeros(()))); ok.append(v.float())
agree, kl, ce0, ce1, ok = map(torch.cat, (agree, kl, ce0, ce1, ok))
m = late & (ok > 0)
res = {"top1_agree": agree[late].mean().item(), "KL_mean_nats": kl[late].mean().item(), "KL_p99_nats": kl[late].quantile(0.99).item(),
       "CE_base": ce0[m].mean().item(), "CE_embed_int4": ce1[m].mean().item(), "dCE_nats": (ce1[m] - ce0[m]).mean().item(),
       "H_cosine_mean": cos.mean().item(), "H_cosine_min": cos.min().item(), "rows": int(late.sum())}
print(json.dumps(res, indent=1)); json.dump(res, open(a.out, "w"), indent=1)
