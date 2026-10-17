#!/usr/bin/env python3
"""Flash-Next latency map from torch-profiler traces of a running vLLM server.

The server must be booted with ``--profiler-config '{"profiler": "torch", "torch_profiler_dir": DIR,
"torch_profiler_with_stack": false, "torch_profiler_record_shapes": false}'``. For each concurrency point
(``--points``, e.g. C1, a midway point and the measured breaking point) this makes ``--reps`` captures of
``--tokens`` greedy decode tokens per stream (prompts prefilled before the capture), parses each rank's trace and
reports ms per decode step by component, plus GPU idle (host/scheduler/launch gaps) within the capture.
With ``--ttft-prompt-tokens N`` it also captures one N-token prefill (TTFT breakdown).

Components: moe_experts, dense_gemm, qsa_attention, qsa_indexer, gdn, hyperconn, ple_ngram, allreduce, norm_act,
sampling, copy_misc, other + gpu_idle.
Attribution, first match wins:
  1. communication kernels, and kernels whose name identifies the module (exl3_moe, GDN, QSA, HC, PLE), by name;
  2. the innermost ``nm9:<component>`` profiler scope around the kernel on its GPU stream. The model opens these only
     while a profiler runs, so a generic kernel (cuBLAS/cutlass GEMV, EXL3 GEMV, copies, top-k gating) is filed under
     the module that launched it;
  3. a kernel signature table (name, grid, block) learned from scoped kernels in the same capture or loaded with
     ``--signatures``. A CUDA-graph replay carries no scopes, so a graph map borrows the table of an eager map;
  4. generic kernel-name needles.
Each point also reports ``top_signatures``: ms and calls per step for every (kernel, grid, block). Usage:
  latency_map.py --port 8011 --trace-dir DIR --out map.json --points 1,8,24 [--reps 3] [--tokens 128]
"""
import argparse
import bisect
import collections
import concurrent.futures as cf
import glob
import gzip
import json
import os
import statistics
import time
import urllib.request

COMPONENTS = [
    ("moe_experts", ("exl3_moe", "moe_vec", "ggml_moe", "fused_moe", "moe_align", "expert", "fat_gemm")),
    ("qsa_indexer", ("qsa_indexer", "indexer", "persistent_topk", "cooperative_topk", "pre_indexer", "compress")),
    ("qsa_attention", ("qsa", "splitk", "sparse_paged", "merge_splitk", "reshape_and_cache")),
    ("gdn", ("gdn", "gated_delta", "flash_qla", "fused_recurrent", "chunk_", "causal_conv1d", "conv1d", "recurrent")),
    ("hyperconn", ("hc_", "hyper", "_hc", "gated_residual", "grouped_gemma")),
    ("ple_ngram", ("ple", "ngram", "dequantize_block_iq4", "lookup_ple")),
    ("allreduce", ("nccl", "allreduce", "all_reduce", "cross_device_reduce", "all_gather")),
    ("dense_gemm", ("exl3_gemm", "gemv", "hgemm", "mul_mat", "gemm", "cutlass", "cublas", "sm75_xmma", "turing_",
                    "volta_", "marlin", "wmma", "dot")),
    ("norm_act", ("rms_norm", "rmsnorm", "layernorm", "layer_norm", "silu", "act_and_mul", "sigmoid", "rotary", "rope")),
    ("sampling", ("sample", "topk", "top_k", "argmax", "softmax", "rejection", "penalt", "logits")),
    ("copy_misc", ("copy", "cat", "index", "scatter", "gather", "slice", "fill", "elementwise", "reduce")),
]


SCOPE_PREFIX = "nm9:"


def component(name):
    low = name.lower()
    for label, needles in COMPONENTS:
        if any(n in low for n in needles):
            return label
    return "other"


def signature(e):
    a = e.get("args", {})
    return f"{e['name'][:100]}|{tuple(a.get('grid', ()))}|{tuple(a.get('block', ()))}"


def _scope_index(ev):
    """GPU-side nm9 profiler scopes per (device, stream): (sorted (start, end, component), prefix max end)."""
    spans = collections.defaultdict(list)
    for e in ev:
        if (isinstance(e, dict) and e.get("cat") == "gpu_user_annotation" and e.get("ph") == "X"
                and str(e.get("name", "")).startswith(SCOPE_PREFIX)):
            spans[(e.get("pid"), e.get("tid"))].append((e["ts"], e["ts"] + e.get("dur", 0), e["name"][len(SCOPE_PREFIX):]))
    out = {}
    for key, v in spans.items():
        v.sort()
        reach, m = [], float("-inf")
        for _, end, _ in v:
            m = max(m, end)
            reach.append(m)
        out[key] = (v, reach)
    return out


def _innermost(index, ts):
    """Component of the innermost scope containing ts: the latest-starting container. reach[i] (the furthest end of
    spans[0..i]) stops the walk as soon as no earlier span can still contain ts."""
    spans, reach = index
    i = bisect.bisect_right(spans, (ts, float("inf"), "")) - 1
    while i >= 0 and reach[i] > ts:
        s0, e0, comp = spans[i]
        if s0 <= ts < e0:
            return comp
        i -= 1
    return None


# kernel names that identify their module; generic names (GEMV/GEMM, copies, elementwise, top-k) take the scope
SPECIFIC = {"moe_experts", "qsa_indexer", "qsa_attention", "gdn", "hyperconn", "ple_ngram"}


def attribute(kernels, ev, table):
    """Component per kernel: COMM > module-specific kernel name > nm9 scope > signature table > generic name.
    Returns the components and the signature table learned from scoped generic kernels."""
    scopes = _scope_index(ev)
    votes = collections.defaultdict(collections.Counter)
    out = []
    for e in kernels:
        by_name = component(e["name"])
        if by_name == "allreduce" or by_name in SPECIFIC:
            out.append(by_name)
            continue
        sp = scopes.get((e.get("pid"), e.get("tid")))
        comp = _innermost(sp, e["ts"]) if sp else None
        if comp:
            votes[signature(e)][comp] += 1
        out.append(comp)
    learned = {sig: c.most_common(1)[0][0] for sig, c in votes.items()}
    merged = {**(table or {}), **learned}
    return [c or merged.get(signature(e)) or component(e["name"]) for c, e in zip(out, kernels)], learned


def post(base, path, body=None):
    data = json.dumps(body).encode() if body is not None else b""
    r = urllib.request.Request(base + path, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=1800) as resp:
        return json.loads(resp.read() or b"{}") if path.startswith("/v1") else resp.status


def complete(base, model, prompt, n):
    out = post(base, "/v1/completions", {"model": model, "prompt": prompt, "max_tokens": n, "temperature": 0.0,
                                         "ignore_eos": True})
    return out["usage"]["completion_tokens"]


def capture(base, trace_dir, fn):
    before = set(glob.glob(os.path.join(trace_dir, "**", "*.json.gz"), recursive=True))
    post(base, "/start_profile")
    t0 = time.time()
    try:
        info = fn()
    finally:
        post(base, "/stop_profile")
    files = []
    for _ in range(300):  # traces are written asynchronously after stop
        files = sorted(set(glob.glob(os.path.join(trace_dir, "**", "*.json.gz"), recursive=True)) - before)
        if files and all(time.time() - os.path.getmtime(f) > 5 for f in files):
            break
        time.sleep(1)
    return {"wall_s": time.time() - t0, "info": info, "files": files}


def parse(path, steps, table=None):
    with gzip.open(path, "rt") as fh:
        ev = json.load(fh).get("traceEvents", [])
    k = [e for e in ev if isinstance(e, dict) and e.get("cat") == "kernel" and e.get("ph") == "X" and e.get("dur", 0) > 0]
    if not k:
        return None
    comps, learned = attribute(k, ev, table)
    comp = collections.Counter()
    top = collections.Counter()
    sig_us = collections.Counter()
    sig_n = collections.Counter()
    sig_comp = {}
    for e, c in zip(k, comps):
        comp[c] += e["dur"]
        top[e["name"][:100]] += e["dur"]
        sg = signature(e)
        sig_us[sg] += e["dur"]
        sig_n[sg] += 1
        sig_comp[sg] = c
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in k)
    busy, cs, ce = 0.0, iv[0][0], iv[0][1]
    for s, e2 in iv[1:]:
        if s > ce:
            busy += ce - cs
            cs, ce = s, e2
        else:
            ce = max(ce, e2)
    busy += ce - cs
    span = iv[-1][1] - iv[0][0] if iv else 0
    n = max(1, steps)
    ms = {c: us / 1e3 / n for c, us in comp.items()}
    ms["gpu_idle"] = (span - busy) / 1e3 / n
    return {"file": os.path.basename(path), "step_ms": span / 1e3 / n, "components_ms": ms,
            "top_kernels_ms": [(name, round(us / 1e3 / n, 4)) for name, us in top.most_common(20)],
            "top_signatures": [(sg, sig_comp[sg], round(us / 1e3 / n, 4), round(sig_n[sg] / n, 2))
                               for sg, us in sig_us.most_common(30)],
            "scoped_signatures": len(learned), "_learned": learned}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", required=True, help="concurrency points, e.g. 1,<mid>,<breaking>")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--tokens-per-step", type=float, default=1.0, help="accepted tokens per step (MTP > 1)")
    ap.add_argument("--ttft-prompt-tokens", type=int, default=0)
    ap.add_argument("--signatures", default="", help="JSON {signature: component} from an eager (scoped) map, for "
                    "CUDA-graph traces; this run's learned table is written next to --out as *.signatures.json")
    a = ap.parse_args()
    table = json.load(open(a.signatures)) if a.signatures else {}
    learned_all = {}
    base = f"http://127.0.0.1:{a.port}"
    model = json.loads(urllib.request.urlopen(base + "/v1/models", timeout=30).read())["data"][0]["id"]
    res = {"model": model, "args": vars(a), "points": []}
    for c in [int(x) for x in a.points.split(",")]:
        prompts = [f"Request {i}: explain the history of {['ships', 'bridges', 'clocks', 'maps'][i % 4]} in depth.\n"
                   for i in range(c)]
        for p in prompts:  # prefill outside the capture
            complete(base, model, p, 1)
        reps = []
        for _ in range(a.reps):
            cap = capture(base, a.trace_dir, lambda: list(cf.ThreadPoolExecutor(c).map(
                lambda p: complete(base, model, p, a.tokens), prompts)))
            steps = round(max(cap["info"]) / a.tokens_per_step)
            parsed = [x for x in (parse(f, steps, table) for f in cap["files"]) if x]
            for x in parsed:
                learned_all.update(x.pop("_learned"))
            reps.append({"wall_s": cap["wall_s"], "steps": steps, "ranks": parsed})
        rank0 = [r["ranks"][0] for r in reps if r["ranks"]]
        comps = sorted({k for r in rank0 for k in r["components_ms"]})
        summary = {k: statistics.median(r["components_ms"].get(k, 0.0) for r in rank0) for k in comps}
        step = statistics.median(r["step_ms"] for r in rank0) if rank0 else None
        row = {"conc": c, "step_ms": step, "step_ms_spread": (max(r["step_ms"] for r in rank0) -
               min(r["step_ms"] for r in rank0)) if rank0 else None,
               "components_ms": dict(sorted(summary.items(), key=lambda kv: -kv[1])), "reps": reps}
        res["points"].append(row)
        json.dump(res, open(a.out, "w"), indent=1)
        print(json.dumps({"conc": c, "step_ms": step, "components_ms": row["components_ms"]}), flush=True)
    if a.ttft_prompt_tokens:
        prompt = "The committee reviewed the quarterly report in detail. " * (a.ttft_prompt_tokens // 10)
        cap = capture(base, a.trace_dir, lambda: complete(base, model, prompt, 1))
        parsed = [x for x in (parse(f, 1, table) for f in cap["files"]) if x]
        for x in parsed:
            learned_all.update(x.pop("_learned"))
        res["ttft"] = {"prompt_tokens": a.ttft_prompt_tokens, "wall_s": cap["wall_s"],
                       "components_ms": parsed[0]["components_ms"] if parsed else None,
                       "top_kernels_ms": parsed[0]["top_kernels_ms"] if parsed else None}
        print(json.dumps({"ttft": res["ttft"]["wall_s"], "components_ms": res["ttft"]["components_ms"]}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    if learned_all:
        json.dump(learned_all, open(os.path.splitext(a.out)[0] + ".signatures.json", "w"), indent=1)


if __name__ == "__main__":
    main()
