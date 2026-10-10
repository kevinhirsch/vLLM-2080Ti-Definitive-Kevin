#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded E06 numerical fixture. No torch/CUDA import without --run-gpu.

Root must commission this in an exclusive GPU window. It loads only an
explicit SHA-bound existing TQ ELF; it never builds/JITs, starts a model or
changes services. It compares prefix output/LSE, not full MTP quality or speed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "vllm/v1/attention/ops/tq_gqa_cuda.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--expected-extension-sha256")
    parser.add_argument("--expected-source-head")
    parser.add_argument("--prefix-lengths", default="0,1,17,257")
    parser.add_argument("--ql", type=int, default=4)
    parser.add_argument("--hq", type=int, default=16)
    parser.add_argument("--hk", type=int, default=2)
    parser.add_argument("--num-splits", type=int, default=128)
    parser.add_argument("--graph", action="store_true",
                        help="also capture zero prefixes then replay actual metadata")
    args = parser.parse_args()
    lengths = [int(x) for x in args.prefix_lengths.split(",")]
    if not (2 <= len(lengths) <= 8 and all(0 <= x <= 32769 for x in lengths)
            and 1 <= args.ql <= 4 and 1 <= args.hk <= 8
            and 1 <= args.hq <= 64 and args.hq % args.hk == 0
            and args.hq // args.hk <= 16 and 1 <= args.num_splits <= 128):
        parser.error("fixture exceeds supported bounds")
    pages = [math.ceil(n / 16) for n in lengths]
    cache_bytes = (1 + sum(pages)) * 16 * args.hk * 230
    if cache_bytes > 128 * 1024 * 1024:
        parser.error("fixture KV cache exceeds 128 MiB bound")
    if not args.run_gpu:
        print(json.dumps({"status": "not-run", "cuda_imported": False,
                          "scope": "prefix output/LSE only; no timing or quality claim",
                          "cache_bytes": cache_bytes, "prefix_lengths": lengths,
                          "requires": ["exclusive root-owned GPU window", "SM75",
                                       "committed source HEAD", "SHA-bound existing TQ ELF"]}))
        return 0
    if not args.extension or not args.expected_extension_sha256 or not args.expected_source_head:
        parser.error("--run-gpu requires explicit extension SHA and source HEAD")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != args.expected_source_head:
        parser.error("source HEAD does not match requested frozen generation")
    source_hashes = {}
    for relative in ["vllm/v1/attention/ops/tq_gqa_cuda.py",
                     "vllm/v1/attention/ops/tq_prefix_batch_plan.py",
                     "vllm/v1/attention/backends/turboquant_attn.py",
                     "tools/compare_tq_prefix_batch_sm75.py"]:
        data = (ROOT / relative).read_bytes()
        if data != subprocess.check_output(["git", "show", head + ":" + relative], cwd=ROOT):
            parser.error("source differs from the committed generation: " + relative)
        source_hashes[relative] = hashlib.sha256(data).hexdigest()
    elf_sha = hashlib.sha256(args.extension.read_bytes()).hexdigest()
    if elf_sha != args.expected_extension_sha256:
        parser.error("extension SHA does not match")

    # Deliberately inside the authorized run branch. No vLLM package import.
    import torch

    if torch.cuda.get_device_capability(0) != (7, 5):
        parser.error("this fixture is restricted to SM75")
    spec = importlib.util.spec_from_file_location("tq_gqa_sm75", args.extension)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    spec = importlib.util.spec_from_file_location("e06_tq_wrapper", WRAPPER)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    wrapper._EXT = extension  # existing ELF only: _load cannot enter its JIT path
    device = "cuda:0"
    generator = torch.Generator(device=device).manual_seed(10610)
    cache = torch.randint(0, 256, (1 + sum(pages), 16, args.hk, 230),
                          dtype=torch.uint8, device=device, generator=generator)
    def metadata_half(value):
        return torch.full(cache.shape[:-1], value, dtype=torch.float16,
                          device=device).view(torch.uint8).reshape(*cache.shape[:-1], 2)
    cache[..., 96:98] = metadata_half(0.5)
    cache[..., 226:228] = metadata_half(0.02)
    cache[..., 228:230] = metadata_half(-0.15)
    # A nontrivial exactly representable orthogonal rotation, fixed on CPU.
    pi = torch.tensor([[(-1 if (i & j).bit_count() % 2 else 1) / 16
                        for j in range(256)] for i in range(256)],
                      dtype=torch.float32, device=device)
    pit = pi.T.contiguous()
    centroids = torch.linspace(-2, 2, 8, dtype=torch.float32, device=device)
    queries = [torch.randn(args.ql, args.hq, 256, dtype=torch.float16,
                           device=device, generator=generator) * 0.1 for _ in lengths]
    nbt = max(1, max(pages))
    tables, sequence_lengths = [], []
    page_offset = 1
    for length, count in zip(lengths, pages):
        table = list(range(page_offset, page_offset + count)) + [-1] * (nbt - count)
        page_offset += count
        # Legal page zero for an empty-prefix capture, matching the backend.
        tables.append(torch.tensor([table if length else [0] * nbt],
                                   dtype=torch.int32, device=device))
        sequence_lengths.append(torch.tensor([length], dtype=torch.int32, device=device))
    def masked(out, lse, index):
        has_prefix = sequence_lengths[index] > 0
        return (torch.where(has_prefix.reshape(1, 1, 1), out, torch.zeros_like(out)),
                torch.where(has_prefix.reshape(1, 1), lse,
                            torch.full_like(lse, float("-inf"))))
    baseline = []
    for i, query in enumerate(queries):
        lse = torch.empty(args.ql, args.hq, dtype=torch.float32, device=device)
        out = wrapper.tq_gqa_decode_attention(
            query, cache, tables[i], sequence_lengths[i], pi, centroids,
            1 / 16, True, q_per_seq=args.ql, num_splits=args.num_splits,
            PiT=pit, lse_buf=lse)
        baseline.append(masked(out, lse, i))
    def batched(table_inputs=tables, length_inputs=sequence_lengths):
        return wrapper.tq_gqa_decode_attention_batch(
            queries, cache, table_inputs, length_inputs, pi, centroids,
            1 / 16, True, num_splits=args.num_splits, PiT=pit)
    out, lse = batched()
    def compare(out, lse):
        results = []
        for i, (old_out, old_lse) in enumerate(baseline):
            rows = slice(i * args.ql, (i + 1) * args.ql)
            new_out, new_lse = masked(out[rows], lse[rows], i)
            results.append({"request": i, "prefix_length": lengths[i],
                            "output_equal": torch.equal(old_out, new_out),
                            "lse_equal": torch.equal(old_lse, new_lse),
                            "output_finite": bool(torch.isfinite(new_out).all().item()),
                            "lse_valid": bool((torch.isfinite(new_lse) if lengths[i]
                                               else torch.isneginf(new_lse)).all().item())})
        return results
    eager = compare(out, lse)
    graph_results = None
    if args.graph:
        graph_tables = [torch.zeros_like(t) for t in tables]
        graph_lengths = [torch.zeros_like(t) for t in sequence_lengths]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): batched(graph_tables, graph_lengths)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out, graph_lse = batched(graph_tables, graph_lengths)
        for target, actual in zip(graph_tables, tables): target.copy_(actual)
        for target, actual in zip(graph_lengths, sequence_lengths): target.copy_(actual)
        graph.replay()
        graph_results = compare(graph_out, graph_lse)
    torch.cuda.synchronize()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != head:
        parser.error("source generation changed during the comparison")
    for relative, expected in source_hashes.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            parser.error("source bytes changed during the comparison: " + relative)
    if hashlib.sha256(args.extension.read_bytes()).hexdigest() != elf_sha:
        parser.error("extension bytes changed during the comparison")
    results = eager + (graph_results or [])
    ok = all(r["output_equal"] and r["lse_equal"] and r["output_finite"]
             and r["lse_valid"] for r in results)
    print(json.dumps({"status": "pass" if ok else "fail", "source_head": head,
                      "source_hashes": source_hashes, "extension_sha256": elf_sha,
                      "torch_version": torch.__version__, "jit_or_build": False,
                      "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "ql": args.ql, "hq": args.hq, "hk": args.hk,
                      "num_splits": args.num_splits, "cache_bytes": cache_bytes,
                      "eager": eager, "zero_capture_then_metadata_replay": graph_results,
                      "limits": "Synthetic prefix output/LSE only; no real logits, MTP acceptance, recurrent state, APC, serving graph or timing proof."}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
