"""CPU test for latency_map.py attribution: nm9 scopes, module-specific names, comm, learned signatures."""
import gzip, importlib.machinery, json, os, sys, tempfile
lm = importlib.machinery.SourceFileLoader("lm", sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "latency_map.py")).load_module()

def k(name, ts, dur, grid, tid=7):
    return {"cat": "kernel", "ph": "X", "name": name, "ts": ts, "dur": dur, "pid": 0, "tid": tid,
            "args": {"grid": grid, "block": [32, 4, 1]}}

def scope(name, ts, dur, tid=7):
    return {"cat": "gpu_user_annotation", "ph": "X", "name": name, "ts": ts, "dur": dur, "pid": 0, "tid": tid}

GEMV = "std::enable_if<true, void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float>"
kernels = [k(GEMV, 10, 30, [320, 1, 1]),                       # HC projection (generic name)
           k("void vllm::moe::topkGating<16, 512>", 105, 5, [1, 1, 1]),   # router top-k ("topk" -> sampling by name)
           k(GEMV, 112, 6, [128, 1, 1]),                          # router GEMV
           k("void exl3_moe_kernel<2, 128, 2>", 120, 20, [8, 1, 8]),
           k("void vllm::cross_device_reduce_1stage<__half, 2>", 150, 10, [2, 1, 1]),
           k("void at::native::vectorized_elementwise_kernel<4>", 300, 5, [4, 1, 1])]  # outside any scope
scopes = [scope("nm9:hyperconn", 0, 100), scope("nm9:moe_experts", 100, 100), scope("execute_context_0", 0, 400)]

def write(ev):
    fd, p = tempfile.mkstemp(suffix=".json.gz"); os.close(fd)
    with gzip.open(p, "wt") as fh: json.dump({"traceEvents": ev}, fh)
    return p

eager = lm.parse(write(kernels + scopes), 1)
comp = eager["components_ms"]
assert abs(comp["hyperconn"] - 0.030) < 1e-9, comp            # GEMV inside the HC scope
assert abs(comp["moe_experts"] - 0.031) < 1e-9, comp          # top-k + router GEMV + exl3_moe
assert abs(comp["allreduce"] - 0.010) < 1e-9, comp            # comm wins over the moe scope
assert abs(comp["copy_misc"] - 0.005) < 1e-9, comp            # unscoped generic kernel keeps its name class
learned = eager["_learned"]
assert len(learned) == 3 and set(learned.values()) == {"hyperconn", "moe_experts"}, learned

graph = lm.parse(write(kernels), 1, table=learned)            # CUDA-graph replay: no scopes, borrowed table
assert graph["components_ms"] == {**comp, "gpu_idle": graph["components_ms"]["gpu_idle"]}, graph["components_ms"]
bare = lm.parse(write(kernels), 1)                            # no scopes, no table: old name-only behaviour
assert "hyperconn" not in bare["components_ms"] and bare["components_ms"]["dense_gemm"] > 0, bare["components_ms"]
sig = {s[0]: (s[1], s[3]) for s in eager["top_signatures"]}
assert sig[lm.signature(kernels[0])] == ("hyperconn", 1.0), sig
print("latency_map attribution: ALL PASS")
