# SPDX-License-Identifier: Apache-2.0
"""CPU regression for invalid accepted counts with either GDN null sentinel.

Run directly, without the repository pytest/conftest platform imports:
  CUDA_VISIBLE_DEVICES='' TRITON_INTERPRET=1 python \
    tests/kernels/test_gdn_pad_sentinel_cpu.py

The actual kernels and their launch wrappers are AST-extracted. No vLLM module
is imported, and CUDA initialization is forbidden. This is interpreter proof,
not compiled SM75 or normal-traffic fault evidence.
"""

import ast
import os
import unittest
from pathlib import Path
from unittest.mock import patch

if (
    os.environ.get("TRITON_INTERPRET") != "1"
    or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    or os.environ.get("FLA_USE_FAST_OPS", "0") != "0"
):
    raise unittest.SkipTest("requires CPU Triton interpreter and default FLA math ops")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

if torch.cuda.is_initialized():
    raise RuntimeError("CUDA was initialized before the CPU test")

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "vllm/third_party/flash_linear_attention/ops"


def interpreter_exp(value):
    # op.py aliases exp=tl.exp with default FLA math. Resolve the tl attribute
    # during interpretation so Triton's temporary language patch is honored.
    return tl.exp(value)


def extract(path, names, *, strip_decorators=False):
    definitions = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in definitions} != names:
        raise RuntimeError(f"production definitions missing in {path}")
    if strip_decorators:
        for node in definitions:
            node.decorator_list = []
    namespace = {
        "torch": torch,
        "triton": triton,
        "tl": tl,
        "NULL_BLOCK_ID": 0,
        "exp": interpreter_exp,
    }
    with patch.object(
        torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
    ):
        exec(
            compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
            namespace,
        )
    return namespace


sigmoid = extract(
    OPS / "fused_sigmoid_gating.py",
    {
        "fused_sigmoid_gating_delta_rule_update_kernel",
        "fused_sigmoid_gating_delta_rule_update",
    },
)["fused_sigmoid_gating_delta_rule_update"]
recurrent = extract(
    OPS / "fused_recurrent.py",
    {
        "fused_recurrent_gated_delta_rule_fwd_kernel",
        "fused_recurrent_gated_delta_rule_fwd",
    },
)["fused_recurrent_gated_delta_rule_fwd"]
update_counts = extract(
    ROOT / "vllm/v1/spec_decode/utils.py",
    {"update_num_computed_tokens_for_batch_change"},
    strip_decorators=True,
)["update_num_computed_tokens_for_batch_change"]


class GDNPadSentinelCPU(unittest.TestCase):
    def setUp(self):
        self.cuda_patch = patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
        )
        self.cuda_patch.start()

    def tearDown(self):
        self.cuda_patch.stop()
        self.assertFalse(torch.cuda.is_initialized())

    def run_kernel(self, kind, accepted, null, *, lengths=(2, 3), normalize=False):
        generator = torch.Generator(device="cpu").manual_seed(45)
        rand = lambda *shape: torch.rand(*shape, generator=generator)  # noqa: E731
        tokens, hk, hv, d = sum(lengths), 1, 2, 4
        q, k, v = rand(1, tokens, hk, d), rand(1, tokens, hk, d), rand(1, tokens, hv, d)
        a, b, A_log, dt_bias = rand(tokens, hv), rand(tokens, hv), rand(hv), rand(hv)
        state = rand(9, hv, d, d)
        before = state.clone()
        indices = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)
        cu = torch.tensor([0, lengths[0], tokens], dtype=torch.int32)
        accepted = torch.tensor(accepted, dtype=torch.int32)
        kwargs = dict(
            q=q, k=k, v=v, scale=d**-0.5, initial_state=state,
            inplace_final_state=True, cu_seqlens=cu, ssm_state_indices=indices,
            num_accepted_tokens=accepted, use_qk_l2norm_in_kernel=normalize,
            null_block_id=null,
        )
        if kind == "sigmoid":
            out, state = sigmoid(A_log=A_log, a=a, b=b, dt_bias=dt_bias, **kwargs)
            gates = -torch.exp(A_log) * torch.nn.functional.softplus(a + dt_bias)
            beta = b.sigmoid()
        else:
            gates, beta = -rand(1, tokens, hv), rand(1, tokens, hv)
            out, state = recurrent(g=gates, beta=beta, **kwargs)
            gates, beta = gates[0], beta[0]
        return out, state, before, (q, k, v, gates, beta, indices, cu, accepted)

    def test_invalid_counts_zero_output_and_leave_all_state_untouched(self):
        for kind in ("sigmoid", "recurrent"):
            for null in (0, -1):
                for bad in (0, 5, 1000000):
                    with self.subTest(kind=kind, null=null, accepted=bad):
                        out, state, before, _ = self.run_kernel(kind, (bad, bad), null)
                        torch.testing.assert_close(
                            out, torch.zeros_like(out), rtol=0, atol=0
                        )
                        torch.testing.assert_close(state, before, rtol=0, atol=0)

    def test_rejected_request_does_not_suppress_valid_neighbor(self):
        for kind in ("sigmoid", "recurrent"):
            for null in (0, -1):
                with self.subTest(kind=kind, null=null):
                    out, state, before, _ = self.run_kernel(kind, (0, 2), null)
                    torch.testing.assert_close(out[:, :2], torch.zeros_like(out[:, :2]))
                    self.assertGreater(torch.count_nonzero(out[:, 2:]).item(), 0)
                    torch.testing.assert_close(state[:5], before[:5], rtol=0, atol=0)
                    self.assertFalse(torch.equal(state[5:8], before[5:8]))

    def test_valid_accepted_prefix_matches_independent_torch_recurrence(self):
        for kind in ("sigmoid", "recurrent"):
            for null in (0, -1):
                for accepted in (1, 2, 4):
                    for normalize in (False, True):
                        with self.subTest(
                            kind=kind, null=null, accepted=accepted, normalize=normalize
                        ):
                            out, state, before, data = self.run_kernel(
                                kind, (accepted, accepted), null, normalize=normalize
                            )
                            q, k, v, gates, beta, indices, cu, counts = data
                            expected_state = before.clone()
                            expected_out = torch.zeros_like(out)
                            for row in range(2):
                                h = before[indices[row, counts[row] - 1]].clone()
                                for pos in range(cu[row], cu[row + 1]):
                                    qr, kr = q[0, pos, 0].clone(), k[0, pos, 0].clone()
                                    if normalize:
                                        qr *= torch.rsqrt((qr * qr).sum() + 1e-6)
                                        kr *= torch.rsqrt((kr * kr).sum() + 1e-6)
                                    qr *= q.size(-1)**-0.5
                                    h *= torch.exp(gates[pos])[:, None, None]
                                    delta = (v[0, pos] - (h * kr).sum(-1)) * beta[
                                        pos, :, None
                                    ]
                                    h += delta[:, :, None] * kr
                                    expected_out[0, pos] = (h * qr).sum(-1)
                                    expected_state[indices[row, pos - cu[row]]] = h
                            torch.testing.assert_close(
                                out, expected_out, rtol=2e-5, atol=2e-6
                            )
                            torch.testing.assert_close(
                                state, expected_state, rtol=2e-5, atol=2e-6
                            )

    def test_async_count_update_propagates_unclamped_prior_draft_count(self):
        # This proves malformed counts can reach the next-step buffer; it does
        # not claim that normal rejection sampling produces malformed counts.
        computed = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
        accepted = torch.ones(4, dtype=torch.int32)
        previous = torch.tensor([0, 1, -1, 3], dtype=torch.int32)
        valid = torch.tensor([0, 999, 2, 3], dtype=torch.int32)
        drafts = torch.tensor([3, 3, 0, 0], dtype=torch.int32)
        cpu_computed = torch.tensor([11, 21, 31, 41], dtype=torch.int32)
        update_counts(computed, accepted, previous, valid, drafts, cpu_computed)
        torch.testing.assert_close(
            accepted, torch.tensor([0, 999, 1, 1], dtype=torch.int32)
        )
        torch.testing.assert_close(
            computed, torch.tensor([10, 1019, 31, 41], dtype=torch.int32)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
