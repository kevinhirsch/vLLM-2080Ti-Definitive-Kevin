# SPDX-License-Identifier: Apache-2.0
"""Execute the optional K5 conv guard on CPU, without importing vLLM.

Run directly (avoids repository pytest/conftest platform initialization):
  CUDA_VISIBLE_DEVICES='' TRITON_INTERPRET=1 python \
    tests/kernels/mamba/test_causal_conv1d_k5_guard_cpu.py

AST extraction loads the actual production kernel and wrapper, rather than a
second implementation. CUDA initialization is forbidden throughout the run.
This proves interpreter behavior; it does not qualify compiled SM75 kernels.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if os.environ.get("TRITON_INTERPRET") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("this test requires TRITON_INTERPRET=1 and CUDA_VISIBLE_DEVICES=''")

import torch
import triton
import triton.language as tl

if torch.cuda.is_initialized():
    raise RuntimeError("CUDA was already initialized before the CPU test")

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "vllm/model_executor/layers/mamba/ops/causal_conv1d.py"
)
tree = ast.parse(SOURCE.read_text())
names = {"_causal_conv1d_update_kernel", "causal_conv1d_update"}
definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
if {node.name for node in definitions} != names:
    raise RuntimeError("production conv kernel/wrapper not found")
namespace = {
    "torch": torch,
    "triton": triton,
    "tl": tl,
    "NULL_BLOCK_ID": 0,
    "current_platform": SimpleNamespace(is_arch_support_pdl=lambda: False),
}
with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialization forbidden")):
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"), namespace)
update = namespace["causal_conv1d_update"]


class K5ConvBoundsCPU(unittest.TestCase):
    def setUp(self):
        self.cuda_patch = patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialization forbidden")
        )
        self.cuda_patch.start()

    def tearDown(self):
        self.cuda_patch.stop()
        self.assertFalse(torch.cuda.is_initialized())

    def run_row(self, *, interval=(1, 3), slot=1, accepted=2, recurrent=(1, 2, 3, 4),
                guard=True, alias=False, apc=None, include_recurrent=True):
        generator = torch.Generator(device="cpu").manual_seed(19)
        x = torch.randn((5, 8), generator=generator)
        weight = torch.randn((8, 4), generator=generator)
        state = torch.randn((3, 8, 6), generator=generator)
        state0, x0 = state.clone(), x.clone()
        out = x if alias else torch.full_like(x, 99)
        out0 = out.clone()
        kwargs = {}
        indices = torch.tensor([slot], dtype=torch.int32)
        if apc is not None:
            indices = torch.tensor([slot], dtype=torch.int32)
            kwargs.update(
                initial_state_idx=torch.tensor([apc[0]], dtype=torch.int32),
                block_idx_last_scheduled_token=torch.tensor([apc[1]], dtype=torch.int32),
            )
        if guard and include_recurrent:
            kwargs.update(
                state_indices_2d=torch.tensor([recurrent], dtype=torch.int32),
                recurrent_num_slots=5,
            )
        result = update(
            x, state, weight,
            conv_state_indices=indices,
            num_accepted_tokens=torch.tensor([accepted], dtype=torch.int32),
            query_start_loc=torch.tensor(interval, dtype=torch.int32),
            max_query_len=4, null_block_id=-1, out=out,
            guard_bounds=guard, **kwargs,
        )
        return result, state, state0, out0, x, x0

    def assert_rejected(self, **kwargs):
        result, state, before, out0, _, _ = self.run_row(**kwargs)
        torch.testing.assert_close(state, before, rtol=0, atol=0)
        bos, eos = kwargs.get("interval", (1, 3))
        start, end = max(0, min(5, bos)), max(0, min(5, eos))
        expected = out0.clone()
        if end > start:
            expected[start:end].zero_()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_bad_conv_slots_leave_state_untouched(self):
        for slot in (-2, -1, 3, 1000000):
            with self.subTest(slot=slot):
                self.assert_rejected(slot=slot)

    def test_query_bounds_zero_only_safe_allocated_rows(self):
        for interval in ((-1, 3), (1, 6), (3, 1), (6, 7), (-5, -1), (0, 5)):
            with self.subTest(interval=interval):
                self.assert_rejected(interval=interval)

    def test_accepted_bounds_and_sources_reject_before_conv_mutation(self):
        for accepted in (0, 5, 1000000):
            with self.subTest(accepted=accepted):
                self.assert_rejected(accepted=accepted)
        for source in (-2, -1, 5, 1000000):
            with self.subTest(source=source):
                self.assert_rejected(recurrent=(1, source, 3, 4))

    def test_positive_bad_destinations_reject_even_beyond_actual_query_length(self):
        for destination in (5, 1000000):
            with self.subTest(destination=destination):
                self.assert_rejected(recurrent=(1, 2, 3, destination))

    def test_negative_destinations_remain_skippable(self):
        guarded = self.run_row(recurrent=(-2, 2, -1, 4))
        original = self.run_row(guard=False)
        torch.testing.assert_close(guarded[0], original[0], rtol=0, atol=0)
        torch.testing.assert_close(guarded[1], original[1], rtol=0, atol=0)

    def test_apc_source_and_destination_slots_and_columns_reject(self):
        for slot, apc in (((1, 3), (0, 1)), ((3, 1), (0, 1)), ((1, 2), (-1, 1)),
                          ((1, 2), (0, 2))):
            with self.subTest(slot=slot, apc=apc):
                self.assert_rejected(slot=slot, apc=apc)

    def test_guard_matches_default_valid_padded_query_and_accepted_prefix(self):
        for accepted in (1, 2, 4):
            for interval in ((0, 0), (1, 2), (1, 3), (0, 4)):
                with self.subTest(accepted=accepted, interval=interval):
                    guarded = self.run_row(accepted=accepted, interval=interval)
                    original = self.run_row(accepted=accepted, interval=interval, guard=False)
                    torch.testing.assert_close(guarded[0], original[0], rtol=0, atol=0)
                    torch.testing.assert_close(guarded[1], original[1], rtol=0, atol=0)

    def test_guard_rejection_with_inplace_output_preserves_unowned_tokens(self):
        self.assert_rejected(interval=(-1, 3), alias=True)
        self.assert_rejected(slot=3, alias=True)

    def test_optional_conv_only_guard_retains_accepted_count_rejection(self):
        self.assert_rejected(accepted=0, include_recurrent=False)
        self.assert_rejected(accepted=5, include_recurrent=False)

    def test_guard_metadata_pair_is_required(self):
        with self.assertRaisesRegex(ValueError, "supplied together"):
            update(
                torch.ones(1, 8, 4), torch.ones(3, 8, 6), torch.ones(8, 4),
                conv_state_indices=torch.tensor([1], dtype=torch.int32),
                guard_bounds=True, recurrent_num_slots=5,
            )

    def test_real_k5_column_view_and_padded_metadata_preserve_valid_requests(self):
        generator = torch.Generator(device="cpu").manual_seed(41)
        x = torch.randn((5, 8), generator=generator)
        weight = torch.randn((8, 4), generator=generator)
        state = torch.randn((3, 8, 6), generator=generator)
        indices = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=torch.int32)
        conv_indices = indices[:, 0]
        self.assertFalse(conv_indices.is_contiguous())
        accepted = torch.tensor([1, 4, 999], dtype=torch.int32)
        cu = torch.tensor([0, 2, 5, 999], dtype=torch.int32)
        guarded_state, default_state = state.clone(), state.clone()
        args = dict(conv_state_indices=conv_indices, num_accepted_tokens=accepted,
                    query_start_loc=cu, max_query_len=4, null_block_id=-1)
        guarded = update(x.clone(), guarded_state, weight, guard_bounds=True,
                         state_indices_2d=indices, recurrent_num_slots=3, **args)
        original = update(x.clone(), default_state, weight, **args)
        torch.testing.assert_close(guarded, original, rtol=0, atol=0)
        torch.testing.assert_close(guarded_state, default_state, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
