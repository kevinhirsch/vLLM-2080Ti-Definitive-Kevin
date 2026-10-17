#!/usr/bin/env python3
"""GPU gate for the UVA host-table PLE (VLLM_QWEN4_PLE_UVA=1) on the REAL table (run in a window, engine stopped).

Registers the pack's ngram_embedding.safetensors table in place (pinned, read-only, mapped). It needs MemAvailable
of at least the table size: 24.4 GiB for the EXL3 Flash-Next pack, so only with no engine running. Then:
  1. rows: UVA gather on the GPU equals the memmap rows read on the CPU, bit-exact (random ids, every head range);
  2. decode: the method's decode of the UVA rows equals its decode of the same rows uploaded from the CPU;
  3. graph: gather + decode captured in one CUDA graph, then replayed with new ids written into the static id
     buffer, equals eager (the lookup is capturable, so FULL decode graphs can include it);
  4. timing: median replay time per call at T = 1, 32, 256 tokens (16 rows/token).
Writes a JSON result; exit 0 = PASS.
usage: gpu_ple_uva_check.py <pack_dir> <out.json> [--tokens 1,32,256] [--iters 50]
"""
import argparse
import json
import os
import statistics
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("out")
    ap.add_argument("--tokens", default="1,32,256")
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    import torch
    from safetensors import safe_open

    from vllm.models.qwen4_exp.nvidia import model as _model  # noqa: F401  (import order: avoids the qsa cycle)
    from vllm.models.qwen4_exp.nvidia import ple_host as P

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    path = os.path.join(a.pack, "ngram_embedding.safetensors")
    m = P.Qwen4ExpPLEEXL3TableMethod(path)
    res = {"path": path, "rows": m.rows, "row_bytes": m.row_bytes, "bits": m.bits, "cases": [], "pass": True}
    t0 = time.time()
    table = P.HostRegisteredTable(m.path, m.offset, m.rows, m.row_bytes)
    res["register_s"] = round(time.time() - t0, 1)
    print(f"registered {m.rows} rows x {m.row_bytes} B in {res['register_s']} s", flush=True)

    # the decode reads head_bias / head_offsets from the layer: load them from the table file
    layer = type("L", (), {})()
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            for name in ("head_bias", "head_offsets"):
                if key.endswith("." + name):
                    setattr(layer, name, f.get_tensor(key).to(dev))
    heads = int(layer.head_offsets.numel())
    g = torch.Generator().manual_seed(0)

    def ids_for(tokens):
        return torch.randint(0, m.rows, (tokens * heads,), generator=g, dtype=torch.int64)

    for T in [int(x) for x in a.tokens.split(",")]:
        ids_cpu = ids_for(T)
        ids = ids_cpu.to(dev)
        rows = torch.empty(T * heads, m.row_bytes // 2, dtype=torch.int16, device=dev)
        P.uva_gather_rows(table.device_view, ids, rows)
        torch.cuda.synchronize()
        ref_rows = torch.from_numpy(table.array[ids_cpu.numpy()].copy()).view(torch.int16)
        rows_ok = torch.equal(rows.cpu(), ref_rows)
        out_dim = heads * m.row_dim
        dec_uva = m.decode(layer, rows.view(torch.uint8), ids, T, out_dim, torch.float16)
        dec_ref = m.decode(layer, ref_rows.view(torch.uint8).to(dev), ids, T, out_dim, torch.float16)
        dec_ok = torch.equal(dec_uva, dec_ref)
        # graph: static ids -> gather -> decode, captured once, replayed with fresh ids
        static_ids = ids.clone()
        static_rows = torch.empty_like(rows)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):  # warm-up on a side stream, as torch.cuda.graph expects
            P.uva_gather_rows(table.device_view, static_ids, static_rows)
            m.decode(layer, static_rows.view(torch.uint8), static_ids, T, out_dim, torch.float16)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            P.uva_gather_rows(table.device_view, static_ids, static_rows)
            static_out = m.decode(layer, static_rows.view(torch.uint8), static_ids, T, out_dim, torch.float16)
        new_ids = ids_for(T).to(dev)
        static_ids.copy_(new_ids)
        graph.replay()
        torch.cuda.synchronize()
        eager_rows = torch.empty_like(rows)
        P.uva_gather_rows(table.device_view, new_ids, eager_rows)
        eager_out = m.decode(layer, eager_rows.view(torch.uint8), new_ids, T, out_dim, torch.float16)
        graph_ok = torch.equal(static_out, eager_out)
        times = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(a.iters):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1e3 / a.iters)
        ok = rows_ok and dec_ok and graph_ok
        res["pass"] &= ok
        row = {"tokens": T, "rows_bit_exact": rows_ok, "decode_equal": dec_ok, "graph_replay_equal": graph_ok,
               "replay_us_median": round(statistics.median(times), 2), "pass": ok}
        res["cases"].append(row)
        print(("PASS " if ok else "FAIL ") + json.dumps(row), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("RESULT", "PASS" if res["pass"] else "FAIL")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
