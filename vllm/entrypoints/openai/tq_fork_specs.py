# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-038 fork child-spec validation — pure, dependency-free.

Shared by the ``/tq/fork`` (v1) and ``/tq/fork2`` (v2) routes in
``tq_snapshot_router.py``. Deliberately imports NOTHING (no fastapi, torch, or
vllm) so the request-validation contract is unit-testable offline, without a
built engine — see ``tests/entrypoints/openai/test_tq_fork_specs.py``, which
loads this file directly by path (bypassing the heavy ``vllm`` package __init__).
"""

# Sampling kwargs a fork child may override. Everything else is rejected so a
# malformed body can't smuggle unexpected SamplingParams kwargs into the engine.
CHILD_SAMPLING_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "max_tokens",
        "min_tokens",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "stop_token_ids",
        "ignore_eos",
    }
)

# Default per-child decode budget when a spec omits max_tokens.
DEFAULT_CHILD_MAX_TOKENS = 64


def parse_child_specs(children: list) -> list[dict]:
    """Validate + normalize a fork ``children`` list into SamplingParams-kwarg
    dicts.

    Each child's keys are whitelisted against :data:`CHILD_SAMPLING_KEYS` so a
    malformed body can't smuggle arbitrary ``SamplingParams`` kwargs into the
    engine, and ``max_tokens`` is coerced to int (default
    :data:`DEFAULT_CHILD_MAX_TOKENS`). Raises ``ValueError`` (the HTTP caller
    maps this to 400) on any bad entry. Returns a fresh list of new dicts; the
    input is never mutated.
    """
    specs: list[dict] = []
    for i, child in enumerate(children):
        if not isinstance(child, dict):
            raise ValueError(f"children[{i}] must be an object")
        bad = set(child) - CHILD_SAMPLING_KEYS
        if bad:
            raise ValueError(
                f"children[{i}] has unsupported key(s): {sorted(bad)}; "
                f"allowed: {sorted(CHILD_SAMPLING_KEYS)}"
            )
        spec = dict(child)
        try:
            spec["max_tokens"] = int(spec.get("max_tokens", DEFAULT_CHILD_MAX_TOKENS))
        except (TypeError, ValueError) as e:
            raise ValueError(f"children[{i}].max_tokens must be an int: {e}")
        specs.append(spec)
    return specs
