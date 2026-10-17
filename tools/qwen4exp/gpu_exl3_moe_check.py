#!/usr/bin/env python3
"""GPU gate for the EXL3 fused MoE on REAL pack tensors (run inside a window, production engine stopped).

Loads E real experts of one layer from the EXL3 pack (trellis/suh/svh/mul1), builds the plugin's fused
state exactly as Exl3MoEMethod does, routes small batches (distinct top-K per token, one non-local -1
route) and compares:
  fused   : apply_exl3_fused_moe -> exllamav3_ext.exl3_moe (the pinned runtime's ABI), deterministic
            (direct_output) and atomic modes
  ref     : apply_exl3_python_loop -> exllamav3 LinearEXL3.forward per expert (the EXL3 GEMM kernels,
            an independent code path)
Gates: the v2 ABI was used (no legacy fallback), outputs finite, rel max err vs ref within fp16 kernel
noise, deterministic mode bitwise-repeatable. Writes a JSON result; exit 0 = PASS.

usage: gpu_exl3_moe_check.py <pack_dir> <out.json> [--layer 0] [--experts 16] [--plugin-src DIR]
"""
import argparse
import json
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("out")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--plugin-src", default=None, help="import vllm_exl3 from this src dir (patched copy)")
    a = ap.parse_args()
    if a.plugin_src:
        sys.path.insert(0, a.plugin_src)
    import torch
    from safetensors import safe_open

    import vllm_exl3.exl3 as X

    dev = torch.device("cuda:0")
    idx = json.load(open(os.path.join(a.pack, "model.safetensors.index.json")))["weight_map"]
    pre = f"model.language_model.layers.{a.layer}.mlp.experts."
    handles = {}

    def get(name):
        fn = idx[name]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(a.pack, fn), framework="pt", device="cpu")
        return handles[fn].get_tensor(name).to(dev)

    def opt(name):
        return get(name) if name in idx else None

    inners = []
    for e in range(a.experts):
        pack = {}
        for which, proj in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
            p = f"{pre}{e}.{proj}."
            pack[which] = X.make_linear_exl3(get(p + "trellis"), get(p + "suh"), get(p + "svh"),
                                             opt(p + "mcg"), opt(p + "mul1"))
        inners.append(pack)

    class Layer:  # the attributes build_exl3_fused_state / apply_exl3_fused_moe read
        pass

    layer = Layer()
    layer.w13_trellis = inners[0]["gate"].trellis
    layer._exl3_hidden_size = int(inners[0]["gate"].in_features)
    layer._exl3_intermediate_local = int(inners[0]["gate"].out_features)
    layer._exl3_bits = int(inners[0]["gate"].trellis.shape[-1] // 16)  # words per 16x16 tile / 16 = K
    layer._exl3_codebook_flags = tuple(
        bool(getattr(inners[0][w], at, d)) for w in ("gate", "up", "down") for at, d in (("mcg", True), ("mul1", False))
    )
    X.build_exl3_fused_state(layer, inners)
    hidden = layer._exl3_hidden_size
    res = {"layer": a.layer, "experts": a.experts, "topk": a.topk, "K": layer._exl3_k,
           "codebook_flags": layer._exl3_codebook_flags, "temps": [list(t.shape) for t in layer._exl3_fused_temps],
           "cases": [], "pass": True}
    torch.manual_seed(0)
    for tokens in (1, 5, 40):
        x = torch.randn(tokens, hidden, dtype=torch.float16, device=dev)
        ids = torch.stack([torch.randperm(a.experts, device=dev)[: a.topk] for _ in range(tokens)])
        ids[0, 0] = -1  # one non-local / invalid route
        w = torch.softmax(torch.randn(tokens, a.topk, device=dev), -1)
        ref = X.apply_exl3_python_loop(x, ids, w, inners, None, None).float()
        for det in (True, False):
            X._MOE_DETERMINISTIC = det
            X._EXL3_MOE_ABI = None
            t0 = time.perf_counter()
            out = X.apply_exl3_fused_moe(x, ids, w, layer, inners, None, None).float()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3
            err = ((out - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
            repeat_equal = None
            if det:
                again = X.apply_exl3_fused_moe(x, ids, w, layer, inners, None, None).float()
                repeat_equal = bool(torch.equal(out, again))
            ok = (X._EXL3_MOE_ABI == "v2" and bool(torch.isfinite(out).all()) and err <= 2e-2
                  and repeat_equal is not False)
            res["pass"] &= ok
            row = {"tokens": tokens, "deterministic": det, "abi": X._EXL3_MOE_ABI, "rel_max_err": err,
                   "repeat_bitwise_equal": repeat_equal, "first_call_ms": round(ms, 3), "pass": ok}
            res["cases"].append(row)
            print(("PASS " if ok else "FAIL ") + json.dumps(row), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("RESULT", "PASS" if res["pass"] else "FAIL")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
