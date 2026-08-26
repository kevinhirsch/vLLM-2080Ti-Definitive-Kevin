#!/usr/bin/env python3
"""EXP-046: FlashQLA varlen-unbatching correctness + speed probe.

Compares, on identical random packed-varlen inputs (N sequences, GDN prefill
shapes), the Triton/FLA path (current fallback) vs the legacy-kernel loop
(VLLM_FLASHQLA_VARLEN_LOOP). Reports max-abs-diff of outputs + final states
and per-path wall time. Run on a GPU with the engine STOPPED.

Usage: flashqla_varlen_probe.py [n_seqs] [tokens_per_seq]
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_SEQS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
T_PER = int(sys.argv[2]) if len(sys.argv) > 2 else 2048

# Qwen3.8-27B GDN prefill shapes (per TP rank): H=16 v-heads, d=128, k-heads 8
H, HK, D = 16, 8, 128


def make_inputs(device):
    total = N_SEQS * T_PER
    g = torch.randn(1, total, H, device=device, dtype=torch.float32) * 0.1 - 1.0
    return dict(
        q=torch.randn(1, total, HK, D, device=device, dtype=torch.bfloat16),
        k=torch.randn(1, total, HK, D, device=device, dtype=torch.bfloat16),
        v=torch.randn(1, total, H, D, device=device, dtype=torch.bfloat16),
        g=g,
        beta=torch.rand(1, total, H, device=device, dtype=torch.bfloat16),
        initial_state=torch.zeros(
            N_SEQS, H, D, D, device=device, dtype=torch.float32
        ),
        cu_seqlens=torch.arange(
            0, total + 1, T_PER, device=device, dtype=torch.int32
        ),
    )


def main():
    device = "cuda:0"
    # Rebase rename: the varlen loop + forward_native/forward_cuda live on
    # ChunkGatedDeltaRule (gdn_linear_attn.py:226; gate read at :397-400).
    from vllm.model_executor.layers.mamba.gdn_linear_attn import (  # noqa: E402
        ChunkGatedDeltaRule,
    )

    be = ChunkGatedDeltaRule.__new__(ChunkGatedDeltaRule)
    inp = make_inputs(device)

    def run(path):
        args = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inp.items()}
        torch.cuda.synchronize()
        t0 = time.time()
        if path == "triton":
            out, st = be.forward_native(
                **args, output_final_state=True, use_qk_l2norm_in_kernel=True
            )
        else:
            os.environ["VLLM_FLASHQLA_VARLEN_LOOP"] = "1"
            out, st = be.forward_flashqla_legacy(
                **args, output_final_state=True, use_qk_l2norm_in_kernel=True
            )
        torch.cuda.synchronize()
        return out, st, time.time() - t0

    # warmup + timed
    for path in ("triton", "loop"):
        run(path)
    o1, s1, t1 = run("triton")
    o2, s2, t2 = run("loop")

    od = (o1.float() - o2.float()).abs().max().item()
    sd = (s1.float() - s2.float()).abs().max().item() if s1 is not None and s2 is not None else -1
    print(f"n_seqs={N_SEQS} t_per={T_PER} total={N_SEQS*T_PER}")
    print(f"triton: {t1*1000:.1f}ms   loop(legacy): {t2*1000:.1f}ms   speedup: {t1/t2:.2f}x")
    print(f"max|out diff|: {od:.4e}   max|state diff|: {sd:.4e}")
    ok = od < 5e-2 and (sd < 5e-2 or sd < 0)
    print("CORRECTNESS:", "PASS" if ok else "FAIL (investigate tolerance/layout)")


if __name__ == "__main__":
    main()
