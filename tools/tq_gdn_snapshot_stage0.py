# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-038 Stage-0 driver — GDN snapshot byte-exactness self-test.

Standalone, MANUAL script. It is NOT a pytest test (no ``test_`` prefix, not
under ``tests/``) and does nothing on import — everything is under
``if __name__ == "__main__"``. Run it ONLY in a scheduled maintenance window on
a THROWAWAY 2-GPU engine. NEVER point it at the production :8001 serve, and
NEVER run it while :8001 is serving on the same GPUs.

What it proves (Stage-0 atom): snapshotting one live sequence's GDN recurrent
state into a park slot is byte-exact (``torch.equal(parked, live)``) on BOTH TP
ranks. It drives the worker-side runner method
``GPUModelRunner.snapshot_mamba_self_test`` via ``collective_rpc`` so each rank
verifies its own shard.

Example (throwaway engine, same flags as the deploy):

    VLLM_TQ_GDN_SNAPSHOT=1 \
    VLLM_TQ_GDN_SNAPSHOT_CONFIRM_THROWAWAY=1 \
    python tools/tq_gdn_snapshot_stage0.py \
        --model /path/to/Qwen3.8-27B-hybrid \
        --tensor-parallel-size 2 \
        --kv-cache-dtype turboquant_k8v4 \
        --mamba-cache-mode align \
        --dtype half \
        --gpu-memory-utilization 0.82 \
        --block-aligned-tokens 2112
"""

from __future__ import annotations

import argparse
import sys


def _build_block_aligned_prompt(n_tokens: int) -> list[int]:
    """A deterministic, block-aligned dummy prompt (token ids). Block-aligned
    (N x block_size) prompts avoid the partial-block / accept_token_bias corner
    cases, isolating the snapshot atom (see EXP-038 Risk #2)."""
    # token id 13 is safe/among the first vocab entries for typical tokenizers;
    # content is irrelevant to a byte-exactness test of the recurrent state.
    return [13] * n_tokens


def main() -> int:
    import os

    import vllm.envs as envs
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--kv-cache-dtype", default="turboquant_k8v4")
    parser.add_argument("--mamba-cache-mode", default="align")
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument(
        "--block-aligned-tokens",
        type=int,
        default=2112,
        help="prompt length; keep a multiple of the mamba block_size",
    )
    args = parser.parse_args()

    if not envs.VLLM_TQ_GDN_SNAPSHOT:
        print(
            "REFUSING TO RUN: set VLLM_TQ_GDN_SNAPSHOT=1 to enable the "
            "snapshot feature (it is inert by default).",
            file=sys.stderr,
        )
        return 2

    # Hard guard: this must never share GPUs with the production serve.
    if os.getenv("VLLM_TQ_GDN_SNAPSHOT_CONFIRM_THROWAWAY") != "1":
        print(
            "REFUSING TO RUN: set VLLM_TQ_GDN_SNAPSHOT_CONFIRM_THROWAWAY=1 to "
            "confirm this is a THROWAWAY engine, not the :8001 serve.",
            file=sys.stderr,
        )
        return 2

    # A non-positive prompt length yields an empty prompt and never exercises the
    # snapshot path; reject it up front.
    if args.block_aligned_tokens <= 0:
        print(
            "REFUSING TO RUN: --block-aligned-tokens must be > 0 (got "
            f"{args.block_aligned_tokens}); a non-positive value yields an "
            "empty prompt.",
            file=sys.stderr,
        )
        return 2

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        kv_cache_dtype=args.kv_cache_dtype,
        mamba_cache_mode=args.mamba_cache_mode,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enforce_eager=True,
        # throwaway-test envelope: tiny window boots fast; pin the SM75 GDN
        # prefill backend the production serve validates (fork default may differ)
        max_model_len=8192,
        additional_config={"gdn_prefill_backend": "flashqla_legacy"},
    )

    # Drive a block-aligned prefill, then snapshot MID-GENERATION: the runner
    # drops ``self.requests[req_id]`` the moment a request finishes (while
    # ``mamba_state_idx`` lingers one step longer), so the self-test must catch
    # the sequence while it is genuinely in flight. Decode enough tokens to
    # give the polling loop a comfortable window under enforce_eager.
    import threading
    import time

    prompt = _build_block_aligned_prompt(args.block_aligned_tokens)
    gen_done = threading.Event()

    def _generate() -> None:
        try:
            llm.generate(
                {"prompt_token_ids": prompt},
                SamplingParams(max_tokens=256, temperature=0.0),
            )
        finally:
            gen_done.set()

    gen_thread = threading.Thread(target=_generate, daemon=True)
    gen_thread.start()

    # The self-test runs per rank on the worker's model_runner. The callable
    # form of collective_rpc receives the worker as ``self``. Poll until the
    # sequence is live on both ranks (utility RPCs interleave between engine
    # steps); keep the last attempt's results as the verdict.
    results: list[dict] = []
    while not gen_done.is_set():
        time.sleep(0.5)
        try:
            attempt = llm.collective_rpc(
                lambda worker: worker.model_runner.snapshot_mamba_self_test()
            )
        except Exception as exc:  # noqa: BLE001 - transient mid-gen RPC issues
            print(f"self-test RPC attempt failed transiently: {exc!r}")
            continue
        results = attempt
        if all(r.get("ok") for r in attempt):
            break
    gen_thread.join(timeout=120)
    if not results:
        print("never caught the sequence in flight", file=sys.stderr)
        return 1

    ok = all(r.get("ok") for r in results)
    for r in results:
        print(
            f"rank {r.get('rank')}: ok={r.get('ok')} "
            f"checked={r.get('num_state_tensors_checked')} "
            f"req={r.get('req_id')} :: {r.get('detail')}"
        )
    print(f"\nSTAGE-0 {'PASS' if ok else 'FAIL'} (both ranks byte-exact: {ok})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
