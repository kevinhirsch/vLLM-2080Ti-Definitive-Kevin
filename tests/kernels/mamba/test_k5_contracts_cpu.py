# SPDX-License-Identifier: Apache-2.0
"""K5 metadata/call-order and actual C++ bounds helper tests, with no CUDA import.

Run directly with Python. AST extraction executes the production preflight and
model method without importing vLLM/torch or collecting the historical GPU test.
The production device bounds helper is compiled as ordinary host C++ solely to
test its integer predicate; this does not test CUDA math, scheduling or graphs.
"""
from __future__ import annotations

import ast
import math
import shutil
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GDN = ROOT / "vllm/model_executor/layers/mamba/gdn"
DT = types.SimpleNamespace(float16="f16", float32="f32", int32="i32")


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def bind_k5(layer, namespace):
    """_k5_spec_decode (metadata lookup + E16 tree guard) delegates to _k5_spec_decode_md; bind both."""
    ns = extract(GDN / "qwen_gdn_linear_attn.py", {"_k5_spec_decode", "_k5_spec_decode_md"}, namespace)
    layer.tree_stash = None
    layer._k5_spec_decode_md = types.MethodType(ns["_k5_spec_decode_md"], layer)
    return ns["_k5_spec_decode"]


class Tensor:
    """Only host-visible tensor metadata. No tensor allocation or CUDA methods."""

    def __init__(self, shape, dtype="f16", strides=None, cuda=True, device="cuda:0", ptr=4096):
        self.shape = tuple(shape)
        self.dtype, self.is_cuda, self.device, self.ptr = dtype, cuda, device, ptr
        if strides is None:
            result, stride = [], 1
            for d in reversed(self.shape):
                result.insert(0, stride)
                stride *= d
            strides = result
        self.strides = tuple(strides)
        self.copies = 0

    def dim(self):
        return len(self.shape)

    def size(self, i=None):
        return self.shape if i is None else self.shape[i]

    def stride(self, i=None):
        return self.strides if i is None else self.strides[i]

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return self.strides == Tensor(self.shape).strides

    def data_ptr(self):
        return self.ptr

    def element_size(self):
        return 2 if self.dtype == "f16" else 4

    def __getitem__(self, keys):
        keys = keys if isinstance(keys, tuple) else (keys,)
        keys += (slice(None),) * (self.dim() - len(keys))
        shape, strides = [], []
        for d, stride, key in zip(self.shape, self.strides, keys):
            if isinstance(key, slice):
                start, stop, step = key.indices(d)
                shape.append(len(range(start, stop, step)))
                strides.append(stride * step)
        return Tensor(shape, self.dtype, strides, self.is_cuda, self.device, self.ptr)

    def reshape(self, *shape):
        shape = list(shape)
        if -1 in shape:
            shape[shape.index(-1)] = self.numel() // math.prod(d for d in shape if d != -1)
        result = Tensor(shape, self.dtype, cuda=self.is_cuda, device=self.device)
        if len(shape) == 3:
            result.strides = (self.strides[0], shape[2], 1)
        return result

    view = reshape

    def detach(self):
        self.copies += 1
        return self

    def float(self):
        return Tensor(self.shape, "f32", cuda=self.is_cuda, device=self.device)

    def contiguous(self):
        return Tensor(self.shape, self.dtype, cuda=self.is_cuda, device=self.device)

    def flatten(self, start=0):
        start %= self.dim()
        return Tensor(self.shape[:start] + (math.prod(self.shape[start:]),), self.dtype,
                      cuda=self.is_cuda, device=self.device)


NS = extract(GDN / "gdn_sm75_cuda.py", {"eligible", "inputs_supported"},
             {"torch": DT, "math": math, "_ENABLED": True, "_VARIANT": 2})


def fixture():
    # Current TP2 per-rank shapes, variable actual tokens and padded graph rows;
    # noncontiguous token rows occur in projected qkv/z and replicated ba views.
    return dict(mixed_qkv=Tensor((6, 5120), strides=(8192, 1)),
                a=Tensor((6, 24), strides=(96, 1)), b=Tensor((6, 24), strides=(96, 1)),
                A_log=Tensor((24,), "f32"), dt_bias=Tensor((24,)),
                state_indices=Tensor((4, 4), "i32"), cu_seqlens=Tensor((5,), "i32"),
                num_accepted=Tensor((4,), "i32"), state=Tensor((20, 24, 128, 128), "f32"),
                z=Tensor((6, 24, 128), strides=(8192, 128, 1)), norm_weight=Tensor((128,)),
                conv_state=Tensor((20, 5120, 6)), conv_weight=Tensor((5120, 4)),
                conv_bias=None, num_requests=2, num_tokens=6, scale=128**-0.5, eps=1e-6)


class PreflightTests(unittest.TestCase):
    def test_current_fp32_and_strided_projection_views(self):
        self.assertTrue(NS["inputs_supported"](**fixture()))

    def test_unsupported_contracts(self):
        cases = [
            ("mixed_qkv", Tensor((6, 5120), "f32")),
            ("mixed_qkv", Tensor((6, 5120), strides=(1, 6))),
            ("a", Tensor((5, 24))), ("b", Tensor((6, 23))),
            ("a", Tensor((6, 24), cuda=False, device="cpu")),
            ("A_log", Tensor((24,), "f32", device="cuda:1")),
            ("A_log", Tensor((23,), "f32")), ("dt_bias", Tensor((1, 24))),
            ("norm_weight", Tensor((127,))),
            ("state", Tensor((20, 24, 128, 128), "f32", ptr=4097)),
            ("state", Tensor((20, 24, 128, 128), "f32", strides=(393215, 16384, 128, 1))),
            ("state", Tensor((20, 24, 128, 128), "f32", strides=(393216, 1, 3072, 24))),
            ("state_indices", Tensor((2, 0), "i32")),
            ("state_indices", Tensor((2, 9), "i32")),
            ("state_indices", Tensor((1, 4), "i32")),
            ("state_indices", Tensor((2, 4), "f32")),
            ("state_indices", Tensor((2, 4), "i32", strides=(1, 2))),
            ("cu_seqlens", Tensor((2,), "i32")),
            ("cu_seqlens", Tensor((3,), "f32")),
            ("num_accepted", Tensor((1,), "i32")),
            ("conv_state", Tensor((19, 5120, 6))),
            ("conv_state", Tensor((20, 5120, 5))),
            ("conv_state", Tensor((20, 5120, 6), "f32")),
            ("conv_state", Tensor((20, 5120, 6), strides=(30720, 1, 1))),
            ("conv_weight", Tensor((5120, 6))),
            ("conv_bias", Tensor((5119,))),
            ("eps", float("nan")), ("scale", 0),
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value):
                d = fixture()
                d[name] = value
                self.assertFalse(NS["inputs_supported"](**d))

    def test_zero_head_count_falls_back_without_division(self):
        args = dict(head_k_dim=128, head_v_dim=128, num_k_heads=0, num_v_heads=24,
                    act_dtype="f16", state_dtype="f32", interleaved=False,
                    activation="silu", norm_before_gate=True, group_size=None)
        self.assertFalse(NS["eligible"](**args))
        args["num_k_heads"] = 8
        self.assertTrue(NS["eligible"](**args))

    def test_transposed_conv_layout_and_extra_actual_tokens(self):
        d = fixture()
        d["conv_state"] = Tensor((20, 5120, 6), strides=(30720, 1, 5120))
        self.assertTrue(NS["inputs_supported"](**d))
        d["num_requests"] = 1
        self.assertFalse(NS["inputs_supported"](**d))

    def test_model_rejection_precedes_parameter_copies_and_conv(self):
        d = fixture()
        d["state_indices"].dtype = "f32"
        md = types.SimpleNamespace(spec_state_indices_tensor=d["state_indices"],
              spec_sequence_masks=object(), num_prefills=0, num_decodes=0, num_spec_decodes=2,
              num_actual_tokens=6, num_accepted_tokens=d["num_accepted"],
              spec_query_start_loc=d["cu_seqlens"], tree_rows=0)
        layer = types.SimpleNamespace(prefix="layer", A_log=d["A_log"], dt_bias=d["dt_bias"],
              head_k_dim=128, head_v_dim=128, num_k_heads=16, num_v_heads=48, tp_size=2,
              key_dim=2048, value_dim=6144, hidden_size=5120, activation="silu",
              gqa_interleaved_layout=False, kv_cache=(d["conv_state"], d["state"]),
              norm=types.SimpleNamespace(activation="silu", norm_before_gate=True,
                   group_size=None, weight=d["norm_weight"], eps=1e-6),
              conv1d=types.SimpleNamespace(weight=Tensor((5120, 1, 4)), bias=None),
              split_ba=lambda _: (Tensor((8, 24), strides=(96, 1)), Tensor((8, 24), strides=(96, 1))))
        conv_calls = []
        def forbidden_conv(*args, **kwargs):
            conv_calls.append(kwargs)
            raise AssertionError("unsupported contract reached convolution")
        namespace = {"torch": DT, "_k5_gdn": types.SimpleNamespace(**{k: NS[k] for k in ("eligible", "inputs_supported")}),
            "get_forward_context": lambda: types.SimpleNamespace(attn_metadata={"layer": md}),
            "GDNAttentionMetadata": types.SimpleNamespace, "is_conv_state_dim_first": lambda: True,
            "causal_conv1d_update": forbidden_conv, "PAD_SLOT_ID": -1}
        method = bind_k5(layer, namespace)
        self.assertFalse(method(layer, Tensor((8, 8192)), Tensor((8, 96)), Tensor((8, 5120))))
        self.assertEqual(conv_calls, [])
        self.assertEqual(layer.dt_bias.copies + layer.norm.weight.copies, 0)


    def test_accepted_path_passes_native_parameters_without_derived_state(self):
        # Regression (2026-10-13 window k5-fused-gdn-aba): the first K5 call runs
        # inside FULL CUDA-graph capture. Lazily created fp32 copies of dt_bias
        # and the norm weight were allocated there but their cast was only
        # recorded, so replays read uninitialised graph-pool memory (evalkit
        # 0/48, MTP acceptance 2.26 -> 0.135). The accepted path must hand the
        # model's own parameters to the kernel and create no layer state.
        d = fixture()
        md = types.SimpleNamespace(spec_state_indices_tensor=d["state_indices"],
              spec_sequence_masks=object(), num_prefills=0, num_decodes=0, num_spec_decodes=2,
              num_actual_tokens=6, num_accepted_tokens=d["num_accepted"],
              spec_query_start_loc=d["cu_seqlens"], tree_rows=0)
        norm = types.SimpleNamespace(activation="silu", norm_before_gate=True,
                                     group_size=None, weight=d["norm_weight"], eps=1e-6)
        layer = types.SimpleNamespace(prefix="layer", A_log=d["A_log"], dt_bias=d["dt_bias"],
              head_k_dim=128, head_v_dim=128, num_k_heads=16, num_v_heads=48, tp_size=2,
              key_dim=2048, value_dim=6144, hidden_size=5120, activation="silu",
              gqa_interleaved_layout=False, kv_cache=(d["conv_state"], d["state"]), norm=norm,
              conv1d=types.SimpleNamespace(weight=Tensor((5120, 1, 4)), bias=None),
              split_ba=lambda _: (Tensor((8, 24), strides=(96, 1)), Tensor((8, 24), strides=(96, 1))),
              out_proj=lambda x: (x, None))
        calls = {}

        def conv(x, *args, **kwargs):
            calls["conv"] = kwargs
            return x

        def gdn_mtp(*args):
            calls["gdn"] = args

        class Output(Tensor):
            def copy_(self, other):
                calls["copied"] = True

        kernel = types.SimpleNamespace(**{k: NS[k] for k in ("eligible", "inputs_supported")},
                                       enabled=lambda: True, gdn_mtp=gdn_mtp)
        dt = types.SimpleNamespace(**vars(DT), zeros=lambda shape, **kw: Tensor(shape))
        namespace = {"torch": dt, "_k5_gdn": kernel,
            "get_forward_context": lambda: types.SimpleNamespace(attn_metadata={"layer": md}),
            "GDNAttentionMetadata": types.SimpleNamespace, "is_conv_state_dim_first": lambda: True,
            "causal_conv1d_update": conv, "PAD_SLOT_ID": -1}
        method = bind_k5(layer, namespace)
        before = set(vars(layer))
        out = Output((8, 5120))
        self.assertTrue(method(layer, Tensor((8, 8192)), Tensor((8, 96)), out))
        self.assertTrue(calls.get("copied"))
        args = calls["gdn"]
        self.assertIs(args[4], layer.dt_bias)
        self.assertIs(args[10], layer.norm.weight)
        self.assertIs(args[3], layer.A_log)
        self.assertEqual(set(vars(layer)), before)
        self.assertEqual(layer.dt_bias.copies + layer.norm.weight.copies, 0)
        self.assertTrue(calls["conv"]["guard_bounds"])

    def test_kernel_reads_parameters_in_native_dtype(self):
        source = (GDN / "gdn_sm75_ext/gdn_mtp_sm75.cu").read_text()
        self.assertNotIn("dt_bias.data_ptr<float>()", source)
        self.assertNotIn("norm_w.data_ptr<float>()", source)
        self.assertEqual(source.count("ld_param(dt_bias, hv, dt_f32)"), 2)
        self.assertEqual(source.count("ld_param(norm_w, v, nw_f32)"), 2)
        model = (GDN / "qwen_gdn_linear_attn.py").read_text()
        self.assertNotIn("_k5_dt_bias", model)
        self.assertNotIn("_k5_norm_w", model)

class DevicePredicateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (GDN / "gdn_sm75_ext/gdn_mtp_sm75.cu").read_text()
        start = source.index("__device__ __forceinline__ bool invalid_request(")
        opening = source.index("{", start)
        depth = 1
        end = opening + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        fn = source[start:end].replace("__device__ __forceinline__", "inline")
        start = source.index("__device__ __forceinline__ void zero_interval(")
        opening = source.index("{", start)
        depth, end = 1, opening + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        fn += "\n" + source[start:end].replace("__device__ __forceinline__", "inline")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.tmp.name) / "bounds"
        main = """
#include <iostream>
#include <vector>
constexpr int kMaxTok=8;
constexpr int kDimV=128;
constexpr int kThreads=512;
using __half=float;
inline float __float2half(float x) { return x; }
FN
int main() {
 int op,w,t,a,s,n; while(std::cin>>op) {
 if(op==1) {
  int bos,eos; std::cin>>bos>>eos;
  std::vector<float> out(5*kDimV,99);
  for(int tid=0;tid<kThreads;++tid) zero_interval(out.data(),bos,eos,5,0,tid,kDimV);
  for(int row=0;row<5;++row) {
   int zeros=0;for(int col=0;col<kDimV;++col) zeros+=out[row*kDimV+col]==0;
   std::cout<<zeros<<" ";
  }
  std::cout<<"\\n";continue;
 }
 std::cin>>w>>t>>a>>s>>n;
  std::vector<int> x(w); for(auto& v:x) std::cin>>v;
  int src=-1; std::cout<<invalid_request(x.data(),w,t,a,s,n,&src)<<" "<<src<<"\\n";
 }
}
""".replace("FN", fn)
        compiler = shutil.which("g++-15") or shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("host C++ compiler unavailable")
        subprocess.run([compiler, "-std=c++17", "-x", "c++", "-", "-o", str(cls.binary)],
                       input=main, text=True, check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_actual_bounds_predicate(self):
        cases = [
            ((4, 4, 1, 8, -1, [1, 2, 3, 4]), False),
            ((4, 2, 4, 8, -1, [1, 2, 3, 4]), False),
            ((4, 4, 0, 8, -1, [1, 2, 3, 4]), True),
            ((4, 4, 5, 8, -1, [1, 2, 3, 4]), True),
            ((4, 4, 1, 8, -1, [8, 2, 3, 4]), True),
            ((4, 4, 3, 8, -1, [1, 2, 8, 4]), True),
            ((4, 4, 1, 8, -1, [1, 2, 3, 8]), True),
            ((4, 4, 1, 8, -1, [1, -1, -2, 4]), False),
            ((4, 4, 2, 8, -1, [1, -1, 3, 4]), True),
            ((4, 4, 2, 8, -1, [-1, 2, 3, 4]), True),
            ((4, 4, 1, 8, 0, [0, 2, 3, 4]), True),
            ((4, 4, 2, 8, 0, [1, 0, 3, 4]), True),
            ((4, 5, 1, 8, -1, [1, 2, 3, 4]), True),
            ((4, 2, 1, 8, -1, [1, 2, 8, 8]), True),
        ]
        payload = "".join(" ".join(map(str, [0, *args[:5], *args[5]])) + "\n" for args, _ in cases)
        result = subprocess.run([str(self.binary)], input=payload, text=True,
                                capture_output=True, check=True).stdout.splitlines()
        for (args, expected), line in zip(cases, result):
            with self.subTest(metadata=args):
                self.assertEqual(bool(int(line.split()[0])), expected)
        self.assertEqual(len(result), len(cases))

    def test_actual_safe_zero_interval(self):
        intervals = [(-3, 2), (3, 9), (5, 2), (8, 10), (-9, -1), (1, 3)]
        payload = "".join(f"1 {bos} {eos}\n" for bos, eos in intervals)
        lines = subprocess.run([str(self.binary)], input=payload, text=True,
                               capture_output=True, check=True).stdout.splitlines()
        for (bos, eos), line in zip(intervals, lines):
            with self.subTest(interval=(bos, eos)):
                start, end = max(0, min(5, bos)), max(0, min(5, eos))
                self.assertEqual(list(map(int, line.split())),
                                 [128 if start <= row < end else 0 for row in range(5)])
        self.assertEqual(len(lines), len(intervals))


if __name__ == "__main__":
    unittest.main(verbosity=2)
