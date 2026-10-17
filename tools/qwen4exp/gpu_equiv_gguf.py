#!/usr/bin/env python3
"""GPU equivalence gate for the GGUF ops (gguf_ops / gguf.py) on the real device. PASS/FAIL JSON, exit 0 only on PASS.

For every ggml type Qwen3.8-Flash-Next uses (+ Q8_0):
  dequantize  : ggml_dequantize == gguf-py numpy dequant (Q2_0: our numpy reference), max abs err <= 1e-3 * max|w|
  mmvq        : ggml_mul_mat_vec_a8 vs x @ W_ref.T (fp32), rel err <= 3e-2 (Q8_1 activation quantization)
  gguf_mul_mat: the dispatcher at T = 1, 8, 64 (MMVQ / MMQ / dequant+GEMM paths)
  moe_vec     : gguf_moe with some topk_ids = -1 (expert-parallel non-local) == reference over local experts only
  PLE gather  : HostRowGather + IQ4_NL dequant == reference rows
The first call of each op is made inside a CUDA graph capture as well (K5 rule): the captured replay must match.
Usage: gpu_equiv_gguf.py OUT.json
"""
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

BLOCK = {"Q8_0": (8, 32, 34), "Q4_K": (12, 256, 144), "Q5_K": (13, 256, 176), "Q6_K": (14, 256, 210),
         "IQ2_XXS": (16, 256, 66), "IQ2_XS": (17, 256, 74), "IQ3_XXS": (18, 256, 98), "IQ4_NL": (20, 32, 18),
         "IQ3_S": (21, 256, 110), "IQ2_S": (22, 256, 82), "IQ4_XS": (23, 256, 136), "Q2_0": (42, 64, 18)}
SCALE_AT = {"Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (-2,)}


def random_blocks(name, n_elems, rng, d_range=(0.001, 0.02)):
    _, be, bb = BLOCK[name]
    nb = n_elems // be
    raw = rng.integers(0, 256, size=(nb, bb), dtype=np.uint8)
    for off in SCALE_AT.get(name, (0,)):
        d = rng.uniform(*d_range, size=nb).astype(np.float16).view(np.uint8).reshape(nb, 2)
        o = off % bb
        raw[:, o:o + 2] = d
    return raw.reshape(-1)


def ref_dequant(name, raw, n_elems):
    if name == "Q2_0":
        blk = raw.reshape(-1, 18)
        d = blk[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
        qs = blk[:, 2:]
        j = np.arange(64)
        q = (qs[:, j // 4] >> (2 * (j % 4))) & 3
        return (d * (q.astype(np.float32) - 1)).reshape(-1)[:n_elems]
    import gguf
    return gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(BLOCK[name][0])).reshape(-1)[:n_elems]


def main():
    out_path = sys.argv[1]
    from vllm.model_executor.layers.quantization import gguf_ops
    from vllm.model_executor.layers.quantization.gguf import _gguf_moe, gguf_mul_mat  # noqa: F401
    from vllm.models.qwen4_exp.nvidia.ple_host import HostRowGather

    dev = torch.device("cuda", 0)
    rng = np.random.default_rng(0)
    res = {"checks": [], "device": torch.cuda.get_device_name(0)}
    ok_all = True

    def check(name, ok, **kw):
        nonlocal ok_all
        ok_all &= bool(ok)
        res["checks"].append({"name": name, "pass": bool(ok), **kw})
        print(("PASS " if ok else "FAIL ") + name, kw, flush=True)

    rows, cols = 256, 2560
    for name, (tid, be, bb) in BLOCK.items():
        raw = random_blocks(name, rows * cols, rng)
        w_ref = torch.from_numpy(ref_dequant(name, raw, rows * cols).reshape(rows, cols).astype(np.float32))
        q = torch.from_numpy(raw.reshape(rows, cols // be * bb)).to(dev)
        w = gguf_ops.ggml_dequantize(q, tid, rows, cols, torch.float16).float().cpu()
        err = (w - w_ref).abs().max().item()
        check(f"dequant/{name}", err <= 1e-3 * w_ref.abs().max().item() + 1e-6, max_abs_err=err)
        for T in (1, 8, 64):
            x = torch.randn(T, cols, dtype=torch.float16, device=dev) * 0.5
            y_ref = x.float().cpu() @ w_ref.T
            y = gguf_mul_mat(x, q, tid, rows).float().cpu()
            rel = ((y - y_ref).norm() / y_ref.norm().clamp_min(1e-9)).item()
            check(f"mul_mat/{name}/T{T}", rel <= 3e-2, rel_err=rel)
        # captured first-call replay must match an eager call
        x = torch.randn(1, cols, dtype=torch.float16, device=dev)
        eager = gguf_ops.ggml_mul_mat_vec_a8(q, x, tid, rows)
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.cuda.graph(g):
            cap = gguf_ops.ggml_mul_mat_vec_a8(q, x, tid, rows)
        g.replay()
        torch.cuda.synchronize()
        check(f"graph_replay/{name}", torch.equal(cap, eager))

    # K = 640 rows (the expert down projections): 20 IQ4_NL / 10 Q2_0 blocks per row, not a multiple of 256
    for name in ("IQ4_NL", "Q2_0"):
        tid, be, bb = BLOCK[name]
        r640 = random_blocks(name, 2560 * 640, rng)
        w_ref = torch.from_numpy(ref_dequant(name, r640, 2560 * 640).reshape(2560, 640).astype(np.float32))
        q = torch.from_numpy(r640.reshape(2560, 640 // be * bb)).to(dev)
        for T in (1, 8, 64):
            x = torch.randn(T, 640, dtype=torch.float16, device=dev) * 0.5
            y = gguf_mul_mat(x, q, tid, 2560).float().cpu()
            y_ref = x.float().cpu() @ w_ref.T
            rel = ((y - y_ref).norm() / y_ref.norm().clamp_min(1e-9)).item()
            check(f"mul_mat_k640/{name}/T{T}", rel <= 3e-2, rel_err=rel)

    # MoE: E experts, top-k, a quarter of the ids marked non-local (-1).
    # Small scales + small activations keep the fp16 intermediate (silu(gate)*up) far from overflow:
    # random 3-bit grids at the default scale range overflowed fp16 (NaN) in W1 attempt 2.
    E, H, I, T, K = 16, 2560, 640, 5, 4
    for t13, t2 in (("IQ2_XXS", "Q2_0"), ("IQ3_S", "IQ4_NL"), ("IQ2_S", "Q2_0"), ("IQ3_XXS", "IQ4_NL"),
                    ("IQ2_S", "IQ4_NL")):
        id13, be13, bb13 = BLOCK[t13]
        id2, be2, bb2 = BLOCK[t2]
        r13 = random_blocks(t13, E * 2 * I * H, rng, (0.0005, 0.002))
        r2 = random_blocks(t2, E * H * I, rng, (0.0005, 0.002))
        w13_ref = torch.from_numpy(ref_dequant(t13, r13, E * 2 * I * H).reshape(E, 2 * I, H))
        w2_ref = torch.from_numpy(ref_dequant(t2, r2, E * H * I).reshape(E, H, I))
        w13 = torch.from_numpy(r13.reshape(E, 2 * I, H // be13 * bb13)).to(dev)
        w2 = torch.from_numpy(r2.reshape(E, H, I // be2 * bb2)).to(dev)
        x = torch.randn(T, H, dtype=torch.float16, device=dev) * 0.1
        ids = torch.randint(0, E, (T, K), device=dev, dtype=torch.int32)
        ids[0, 0] = -1
        ids[2, 1:] = -1
        wts = torch.softmax(torch.randn(T, K, device=dev), -1).half()
        y = _gguf_moe(x, w13, w2, wts, ids, id13, id2).float().cpu()
        xr = x.float().cpu()
        y_ref = torch.zeros(T, H)
        for t in range(T):
            for k in range(K):
                e = int(ids[t, k])
                if e < 0:
                    continue
                gu = w13_ref[e] @ xr[t]
                a = torch.nn.functional.silu(gu[:I]) * gu[I:]
                y_ref[t] += float(wts[t, k]) * (w2_ref[e] @ a)
        rel = ((y - y_ref).norm() / y_ref.norm()).item()
        check(f"moe_vec/{t13}+{t2}", rel <= 5e-2 and torch.isfinite(y).all().item(), rel_err=rel)

    # PLE host gather + IQ4_NL dequant
    rows_tab, dim = 4096, 160
    raw = random_blocks("IQ4_NL", rows_tab * dim, rng).reshape(rows_tab, dim // 32 * 18)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(b"\0" * 96)
        f.write(raw.tobytes())
        path = f.name
    try:
        gat = HostRowGather(path, 96, rows_tab, raw.shape[1])
        ids = rng.integers(0, rows_tab, size=3 * 16)
        buf = np.empty((ids.shape[0], raw.shape[1]), np.uint8)
        got = gat.gather(ids, buf)
        emb = gguf_ops.ggml_dequantize(torch.from_numpy(got.copy()).to(dev), 20, 3, 16 * dim, torch.float16)
        ref = torch.from_numpy(ref_dequant("IQ4_NL", raw[ids].reshape(-1), ids.shape[0] * dim)).reshape(3, 16 * dim)
        err = (emb.float().cpu() - ref).abs().max().item()
        check("ple_gather_iq4nl", err <= 1e-3 * ref.abs().max().item(), max_abs_err=err)
    finally:
        os.unlink(path)

    res["pass"] = ok_all
    res["time"] = time.strftime("%FT%T")
    json.dump(res, open(out_path, "w"), indent=1)
    print("RESULT", "PASS" if ok_all else "FAIL", json.dumps({"pass": ok_all, "n": len(res["checks"])}))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
