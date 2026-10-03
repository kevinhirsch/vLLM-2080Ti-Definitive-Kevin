#!/usr/bin/env python
"""Lane K8 track H: layer-wise fidelity of 2:4 pruning on the deployed int4 weights vs equal-bytes dense int3.
For each sampled layer / linear: teacher W0 = the deployed int4 weights (dequantised); inputs = real activations from
capture.py (cal = first 4096 tokens, eval = remaining 2304).  Variants (all measured as ||(Wq-W0)X||/||W0 X|| on EVAL):
  wanda24      : 2:4 by |w|*||x||, kept values untouched (stay on the int4 grid), no update
  sgpt24grid   : SparseGPT 2:4 with OBS compensation, kept values re-snapped to the SAME int4 grid (deployable as-is)
  sgpt24fp     : SparseGPT 2:4 compensation, kept values free (upper bound for any 2:4 + higher-bit values)
  rtn3 / gptq3 : dense int3 g128 (equal bytes: 3.19 b/w) - the dominance baseline
  rtn4+gptq4 ctrl: GPTQ int4 re-fit from W0 (must be ~0)
Rows are sub-sampled (default 768 random rows/matrix; rows are independent given the Hessian) to bound CPU cost."""
import os, sys, json, time, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/home/kevin/projects/lanes/u2-quant"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import ref_dump as R
import prune_lib as P

CAP = "/home/kevin/projects/lanes/k8/cap"
NCAL = 4096
DAMP = 0.5  # 4096 cal tokens < n_in for most matrices: damp 0.01 overfits (cal 0.06 vs eval 0.21); 0.1-0.5 is the held-out optimum


def load_q(prefix):
    h = R._sf(R.MAIN_ST)
    packed = h.get_tensor(prefix + ".weight_packed"); scale = h.get_tensor(prefix + ".weight_scale").float()
    zpp = h.get_tensor(prefix + ".weight_zero_point"); shape = h.get_tensor(prefix + ".weight_shape")
    out_f, in_f = int(shape[0]), int(shape[1])
    q = ((packed.unsqueeze(-1) >> R._SHIFTS) & 15).reshape(out_f, in_f)
    zp = ((zpp.unsqueeze(1) >> R._SHIFTS.view(1, 8, 1)) & 15).reshape(out_f, -1)
    ng = in_f // 128
    w = (q.reshape(out_f, ng, 128) - zp.unsqueeze(-1)).float() * scale.unsqueeze(-1)
    return w.reshape(out_f, in_f), dict(scale=scale, zp=zp.float())


def mats(i, typ):
    p = f"{R.PFX}.layers.{i}"
    if typ == "full_attention":
        m = [("q_proj", f"{p}.self_attn.q_proj", "h_attn"), ("k_proj", f"{p}.self_attn.k_proj", "h_attn"),
             ("v_proj", f"{p}.self_attn.v_proj", "h_attn"), ("o_proj", f"{p}.self_attn.o_proj", "core")]
    else:
        m = [("in_proj_qkv", f"{p}.linear_attn.in_proj_qkv", "h_attn"), ("in_proj_z", f"{p}.linear_attn.in_proj_z", "h_attn"),
             ("out_proj", f"{p}.linear_attn.out_proj", "core")]
    return m + [("gate_proj", f"{p}.mlp.gate_proj", "h_mlp"), ("up_proj", f"{p}.mlp.up_proj", "h_mlp"), ("down_proj", f"{p}.mlp.down_proj", "act")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a"); ap.add_argument("--layers", default="6,29,15,62")
    ap.add_argument("--rows", type=int, default=768); ap.add_argument("--variants", default=""); ap.add_argument("--only", default="")
    ap.add_argument("--out", default="/home/kevin/projects/lanes/k8/prune_eval.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("K8_THREADS", "6")))
    g = torch.Generator(); g.manual_seed(0)
    for L in [int(x) for x in a.layers.split(",")]:
        d = torch.load(f"{CAP}/{a.tag}_lin_L{L:02d}.pt")
        typ = d["type"]
        Hc = {}
        for name, prefix, inp in mats(L, typ):
            if a.only and name not in a.only.split(","):
                continue
            t0 = time.time()
            X = d[inp].float()
            Xc, Xe = X[:NCAL], X[NCAL:]
            if inp not in Hc:
                Hc[inp] = P.hessian(Xc)
            H = Hc[inp]
            W0, grid = load_q(prefix)
            rows = torch.randperm(W0.shape[0], generator=g)[:a.rows].sort()[0]
            W0 = W0[rows]; grid = dict(scale=grid["scale"][rows], zp=grid["zp"][rows])
            res = {"layer": L, "type": typ, "mat": name, "shape": list(W0.shape), "rows": int(len(rows))}
            variants = {
                "wanda24": lambda: P.wanda(W0, Xc),
                "sgpt24grid": lambda: P.sparsegpt(W0, H, grid=grid, damp=DAMP),
                "sgpt34grid": lambda: P.sparsegpt(W0, H, n_keep=3, grid=grid, damp=DAMP),
                "sgpt24fp": lambda: P.sparsegpt(W0, H, nbits=16, damp=DAMP),
                "rtn3": lambda: P.rtn(W0, 3),
                "gptq3": lambda: P.sparsegpt(W0, H, nbits=3, prune=False, damp=DAMP),
                "gptq4ctl": lambda: P.sparsegpt(W0, H, nbits=4, prune=False, damp=DAMP),
            }
            for vn, fn in variants.items():
                if a.variants and vn not in a.variants.split(","):
                    continue
                tv = time.time(); Wq = fn()
                res[vn] = {"eval": P.rel_err(Wq, W0, Xe), "cal": P.rel_err(Wq, W0, Xc), "s": round(time.time() - tv, 1)}
            res["total_s"] = round(time.time() - t0, 1)
            print(json.dumps(res), flush=True)
            open(a.out, "a").write(json.dumps(res) + "\n")
            del H


if __name__ == "__main__":
    main()
