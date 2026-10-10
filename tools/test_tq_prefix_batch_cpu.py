#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stdlib-only E06 contract tests: no torch/vLLM import or CUDA initialization.

AST extraction executes the real new wrapper/backend methods with tensor
metadata spies. These tests establish grouping/ownership/fallback, not GPU
numerics, graph correctness, MTP acceptance or performance.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "vllm/v1/attention/ops"
BACKEND = ROOT / "vllm/v1/attention/backends/turboquant_attn.py"
spec = importlib.util.spec_from_file_location("plan", OPS / "tq_prefix_batch_plan.py")
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


def extracted(path, name, env):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0),
        node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env[name]


DEVICE = SimpleNamespace(type="cuda", index=0)


class Tensor:
    def __init__(self, shape, rows=None, dtype="f16", strides=None):
        self.shape = tuple(shape)
        self.rows = list(rows if rows is not None else [None] * shape[0])
        self.dtype, self.device = dtype, DEVICE
        self.strides = strides

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise AssertionError("the prototype must slice, not read device scalars")
        rows = self.rows[key]
        return Tensor((len(rows), *self.shape[1:]), rows, self.dtype, self.strides)

    def __setitem__(self, key, value):
        assert len(self.rows[key]) == len(value.rows)
        self.rows[key] = value.rows

    def reshape(self, *shape):
        return Tensor(shape, self.rows, self.dtype)

    def float(self):
        return Tensor(self.shape, self.rows, "f32")

    def __matmul__(self, matrix):
        Torch.rotations.append((self.shape, tuple(self.rows), matrix))
        return Tensor(self.shape, self.rows, "f32")

    def contiguous(self):
        return Tensor(self.shape, self.rows, self.dtype)

    def stride(self, dim):
        if self.strides is not None:
            return self.strides[dim]
        result = 1
        for size in self.shape[dim + 1:]:
            result *= size
        return result

    def to(self, dtype):
        return Tensor(self.shape, self.rows, dtype)

    def __sub__(self, value):
        return Tensor(self.shape, [x - value for x in self.rows], self.dtype)

    def clamp_min(self, value):
        return Tensor(self.shape, [max(x, value) for x in self.rows], self.dtype)

    def __gt__(self, value):
        return Tensor(self.shape, [x > value for x in self.rows], "bool")

    def tolist(self):
        raise AssertionError("device metadata must never be read for E06 planning")


class Torch:
    float16, float32, int32, uint8 = "f16", "f32", "i32", "u8"
    rotations = []

    @staticmethod
    def cat(tensors, dim=0):
        assert dim == 0
        assert all(t.shape[1:] == tensors[0].shape[1:] for t in tensors)
        return Tensor((sum(t.shape[0] for t in tensors), *tensors[0].shape[1:]),
                      [r for t in tensors for r in t.rows], tensors[0].dtype)

    @staticmethod
    def empty(*shape, dtype, device):
        return Tensor(shape, dtype=dtype)

    @staticmethod
    def empty_like(tensor):
        return Tensor(tensor.shape, dtype=tensor.dtype)

    @staticmethod
    def zeros_like(tensor):
        return Tensor(tensor.shape, [0] * tensor.shape[0], tensor.dtype)

    @staticmethod
    def full_like(tensor, value):
        return Tensor(tensor.shape, [value] * tensor.shape[0], tensor.dtype)

    @staticmethod
    def where(mask, yes, no):
        assert len(mask.rows) == 1
        return yes.contiguous() if mask.rows[0] else no.contiguous()

    @staticmethod
    def arange(length, **kwargs):
        return Tensor((length,), list(range(length)), kwargs["dtype"])


class Extension:
    def __init__(self):
        self.calls = []

    def decode(self, *args):
        self.calls.append(args)
        q_rot, kv, bt, sl, cent, mid, out, lse, QL = args[:9]
        assert q_rot.shape[0] == bt.shape[0] * QL
        assert q_rot.stride(0) == q_rot.shape[1] * q_rot.shape[2]
        assert bt.stride(0) == bt.shape[1]
        out.rows[:], lse.rows[:] = q_rot.rows, q_rot.rows


class Contracts(unittest.TestCase):
    def setUp(self):
        Torch.rotations = []
        self.ext = Extension()
        self.batch = extracted(OPS / "tq_gqa_cuda.py",
                               "tq_gqa_decode_attention_batch",
                               {"torch": Torch, "_load": lambda: self.ext})

    def groups(self, starts, rows, eligible=True):
        return plan.homogeneous_prefix_groups(
            starts, num_reqs=0 if starts is None else len(starts) - 1,
            total_rows=rows, eligible=eligible)

    def inputs(self, n=2, ql=4):
        return ([Tensor((ql, 8, 256), range(10 + i * ql, 10 + (i + 1) * ql),
                        strides=(4096, 512, 1)) for i in range(n)],
                Tensor((8, 16, 2, 230), dtype="u8", strides=(16000, 1000, 230, 1)),
                [Tensor((1, 3), [100 + i], "i32", strides=(7, 1)) for i in range(n)],
                [Tensor((1,), [20 + i], "i32") for i in range(n)])

    def call(self, inputs, **kwargs):
        return self.batch(*inputs, None, object(), 0.0625, True,
                          PiT="same-PiT", **kwargs)

    def test_homogeneous_c4(self):
        self.assertEqual(self.groups([0, 4, 8, 12, 16], 16), ((0, 1, 2, 3),))

    def test_singleton_disabled_and_missing_cpu(self):
        self.assertEqual(self.groups([0, 4], 4), ())
        self.assertEqual(self.groups([0, 4, 8], 8, False), ())
        self.assertEqual(self.groups(None, 8), ())

    def test_mixed_zero_and_padding(self):
        self.assertEqual(self.groups([0, 4, 6, 10, 10], 16), ((0, 2),))
        self.assertEqual(self.groups([0, 5, 10, 11, 12], 16), ((2, 3),))
        self.assertEqual(self.groups([0, 1, 2, 6, 10], 12), ((0, 1), (2, 3)))

    def test_bad_metadata_no_plan(self):
        for starts, rows in [([1, 5, 9], 9), ([0, 5, 4], 8),
                             ([0, -1, 4], 4), ([0, 4, 9], 8),
                             ([0, True, 2], 2), ([0, 1.0, 2], 2)]:
            with self.subTest(starts=starts):
                self.assertEqual(self.groups(starts, rows), ())
        self.assertEqual(plan.homogeneous_prefix_groups(
            [0], num_reqs=2, total_rows=8, eligible=True), ())

    def test_extra_cpu_buffer_capacity_is_not_a_request(self):
        self.assertEqual(plan.homogeneous_prefix_groups(
            [0, 4, 8, -99], num_reqs=2, total_rows=12, eligible=True), ((0, 1),))

    def test_per_request_rotation_then_one_launch_preserves_rows_and_strides(self):
        inputs = self.inputs()
        out, lse = self.call(inputs, num_splits=64)
        self.assertEqual(len(Torch.rotations), 2)
        self.assertEqual([x[0] for x in Torch.rotations], [(4, 8, 256)] * 2)
        self.assertEqual([x[2] for x in Torch.rotations], ["same-PiT"] * 2)
        self.assertEqual(out.rows, list(range(10, 18)))
        self.assertEqual(lse.rows, out.rows)
        self.assertEqual(len(self.ext.calls), 1)
        args = self.ext.calls[0]
        self.assertIs(args[1], inputs[1])  # cache object/strides unchanged
        self.assertEqual(args[2].rows, [100, 101])
        self.assertEqual(args[3].rows, [20, 21])
        self.assertEqual(args[2].stride(0), 3)  # padded source BT is packed
        self.assertEqual(args[8:13], (4, 4, 4, 16, 64))

    def test_invalid_wrapper_inputs_rejected_before_rotation_or_launch(self):
        cases = []
        for ns in (0, 129):
            cases.append((self.inputs(), {"num_splits": ns}))
        cases.extend([(self.inputs(n=1), {}), (self.inputs(ql=5), {})])
        inputs = self.inputs(); inputs[0][1].dtype = "f32"; cases.append((inputs, {}))
        inputs = self.inputs(); inputs[2][1].shape = (1, 4); cases.append((inputs, {}))
        inputs = self.inputs(); inputs[3][0].shape = (4,); cases.append((inputs, {}))
        inputs = self.inputs(); inputs[1].shape = (8, 8, 2, 230); cases.append((inputs, {}))
        inputs = self.inputs(); inputs[1].shape = (8, 16, 2, 234); cases.append((inputs, {}))
        inputs = self.inputs(); inputs[2][0].dtype = "i64"; cases.append((inputs, {}))
        for inputs, kwargs in cases:
            with self.subTest(kwargs=kwargs, shapes=[t.shape for t in inputs[0]]):
                with self.assertRaises(ValueError): self.call(inputs, **kwargs)
        self.assertEqual(Torch.rotations, [])
        self.assertEqual(self.ext.calls, [])

    def backend(self, enabled=True, bad_layout=False):
        api = SimpleNamespace(enabled=lambda: True, gqa_eligible=lambda **kw: True,
                              tq_gqa_decode_attention_batch=self.batch)
        fn = extracted(BACKEND, "_spec_decode_attention_raw_current", {
            "torch": Torch, "_TQ_BATCH_SPEC_PREFIX": enabled, "_gqa_cuda": api,
            "homogeneous_prefix_groups": plan.homogeneous_prefix_groups})
        query = Tensor((12, 8, 256), range(10, 22))
        key = Tensor((12, 2, 256), range(10, 22))
        value = Tensor((12, 2, 256), range(10, 22))
        cache = self.inputs()[1]
        if bad_layout: cache.shape = (8, 16, 2, 234)
        metadata = SimpleNamespace(
            query_start_loc_cpu=SimpleNamespace(tolist=lambda: [0, 4, 6, 10, 10]),
            query_start_loc=Tensor((5,)),
            seq_lens=Tensor((4,), [24, 22, 4, 0], "i32"),
            block_table=Tensor((4, 3), [100, 101, 102, 103], "i32"),
            max_seq_len=24, max_query_len=4)
        owner = SimpleNamespace(max_num_kv_splits=64, scale=0.0625,
            tq_config=SimpleNamespace(key_mse_bits=3, effective_value_quant_bits=4,
                key_fp8=False, key_packed_size=98, norm_correction=True))
        merges = []
        def merge(q, k, v, *args, prefix_state=None):
            self.assertEqual(q.rows, k.rows)
            self.assertEqual(q.rows, v.rows)
            merges.append((list(q.rows), prefix_state))
            return q.contiguous()
        owner._spec_continuation_decode_attention = merge
        out = fn(owner, query, key, value, cache, metadata, None, object(), "same-PiT")
        return out, merges

    def test_actual_backend_row_ownership_zero_prefix_singleton_and_padding(self):
        out, merges = self.backend()
        self.assertEqual(out.rows, list(range(10, 20)) + [None, None])
        self.assertEqual(len(self.ext.calls), 1)
        self.assertEqual(self.ext.calls[0][2].rows, [100, 0])  # safe zero page
        self.assertEqual(self.ext.calls[0][3].rows, [20, 0])
        self.assertEqual(merges[0][1][0].rows, [10, 11, 12, 13])
        self.assertIsNone(merges[1][1])  # incompatible QL=2 is singleton
        self.assertEqual(merges[2][1][0].rows, [0, 0, 0, 0])
        self.assertEqual(merges[2][1][1].rows, [float("-inf")] * 4)

    def test_default_off_and_ineligible_layout_preserve_singletons(self):
        for kw in ({"enabled": False}, {"bad_layout": True}):
            with self.subTest(kw=kw):
                out, merges = self.backend(**kw)
                self.assertTrue(all(prefix is None for _, prefix in merges))
                self.assertEqual(out.rows[-2:], [None, None])
        self.assertEqual(self.ext.calls, [])

    def test_original_wrapper_ast_unchanged(self):
        path = "vllm/v1/attention/ops/tq_gqa_cuda.py"
        before = subprocess.check_output(["git", "show", "7004df6ae8:" + path],
                                         cwd=ROOT, text=True)
        def node(text):
            return next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
                        and n.name == "tq_gqa_decode_attention")
        self.assertEqual(ast.dump(node(before)), ast.dump(node((ROOT / path).read_text())))

    def test_new_wrapper_has_no_device_scalar_reads(self):
        tree = ast.parse((OPS / "tq_gqa_cuda.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "tq_gqa_decode_attention_batch")
        calls = [n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)]
        self.assertFalse(set(calls) & {"item", "tolist", "cpu", "numpy", "nonzero"})

    def test_early_prefix_merge_preserves_request_inputs_and_output_alias(self):
        calls = []
        def merge(*args, **kwargs):
            calls.append((args, kwargs))
            return args[3]
        fn = extracted(BACKEND, "_spec_continuation_decode_attention", {
            "torch": Torch, "triton_turboquant_spec_continuation_merge": merge})
        owner = SimpleNamespace(scale=0.0625)
        query, key, value = Tensor((4, 8, 256)), Tensor((4, 2, 256)), Tensor((4, 2, 256))
        prefix = Tensor((4, 8, 256))
        lse = Tensor((4, 8), dtype="f32")
        result = fn(owner, query, key, value, None, None, Tensor((1,), [4], "i32"),
                    None, None, None, None, prefix_state=(prefix, lse))
        self.assertIs(result, prefix)
        self.assertEqual(calls[0][0][:5], (query, key, value, prefix, lse))
        self.assertIs(calls[0][1]["output"], prefix)
        with self.assertRaises(ValueError):
            fn(owner, query, key, value, None, None, Tensor((1,), [4], "i32"),
               None, None, None, None,
               prefix_state=(Tensor((8, 8, 256)), lse))
        self.assertEqual(len(calls), 1)

    def test_retained_compressed_cache_method_unchanged(self):
        path = "vllm/v1/attention/backends/turboquant_attn.py"
        before = subprocess.check_output(["git", "show", "7004df6ae8:" + path],
                                         cwd=ROOT, text=True)
        def node(text):
            return next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef)
                        and n.name == "_spec_decode_attention")
        self.assertEqual(ast.dump(node(before)), ast.dump(node(BACKEND.read_text())))

    def test_numerical_entrypoint_default_does_not_import_cuda(self):
        result = subprocess.check_output(
            [sys.executable, str(ROOT / "tools/compare_tq_prefix_batch_sm75.py")],
            text=True)
        receipt = json.loads(result)
        self.assertEqual(receipt["status"], "not-run")
        self.assertFalse(receipt["cuda_imported"])
        tree = ast.parse((ROOT / "tools/compare_tq_prefix_batch_sm75.py").read_text())
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(any(a.name.startswith(("torch", "vllm")) for a in n.names)
                             for n in imports))


if __name__ == "__main__":
    unittest.main()
