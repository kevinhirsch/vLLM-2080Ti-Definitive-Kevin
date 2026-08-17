# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-039 (S4): pure-python unit tests for the scoped re-emission drafter.

These tests deliberately do NOT import the full vLLM/torch stack. The drafter
module only needs ``vllm.config.VllmConfig`` (an unused type import at runtime)
and ``vllm.logger.init_logger``; both are stubbed so the drafter's algorithm can
be exercised in isolation with numpy alone. This mirrors the way the drafter is
actually driven inside ``gpu_model_runner`` (see the docstring of
``_make_input_batch`` for the precise timing contract being reproduced).

Run: ``python3 tests/v1/spec_decode/test_scoped_reemission.py`` (numpy only).
"""

import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import numpy as np

# ---------------------------------------------------------------------------
# Stub the two vLLM symbols the drafter imports, then load the module directly
# from its file path so this stays a pure-python unit test (no torch, no CUDA).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_MODULE_PATH = os.path.join(
    _REPO, "vllm", "v1", "spec_decode", "scoped_reemission.py"
)


def _load_drafter_module():
    """(Re)load scoped_reemission.py with stubbed vllm deps."""
    vllm_pkg = types.ModuleType("vllm")
    cfg_mod = types.ModuleType("vllm.config")
    cfg_mod.VllmConfig = object  # only used as a type hint
    log_mod = types.ModuleType("vllm.logger")

    class _NullLogger:
        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

    log_mod.init_logger = lambda *_a, **_k: _NullLogger()
    sys.modules["vllm"] = vllm_pkg
    sys.modules["vllm.config"] = cfg_mod
    sys.modules["vllm.logger"] = log_mod

    spec = importlib.util.spec_from_file_location(
        "scoped_reemission_under_test", _MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SR = _load_drafter_module()


# ---------------------------------------------------------------------------
# Fakes for the two objects the drafter touches: vllm_config and input_batch.
# ---------------------------------------------------------------------------
def _make_config(num_spec_tokens=16, max_model_len=65536):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=num_spec_tokens),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )


def _make_input_batch(prompt, committed_seq):
    """Build a one-request fake input_batch.

    TIMING CONTRACT being reproduced (the crux of the bug): inside
    ``gpu_model_runner.execute_model`` the MTP/EAGLE path calls
    ``propose_draft_token_ids`` (which runs ``ScopedReemissionDrafter.merge``)
    BEFORE ``_bookkeeping_sync`` writes this step's freshly-sampled tokens into
    ``token_ids_cpu`` and advances ``num_tokens_no_spec``. So at merge time:

      * ``token_ids_cpu[:num_tokens_no_spec]`` == the sequence THROUGH THE END OF
        THE PREVIOUS STEP (``committed_seq`` here) -- it does NOT include the
        token(s) sampled this step.
      * the tokens sampled this step arrive ONLY via the ``sampled_token_ids``
        argument to ``merge`` (as MTP receives them via ``next_token_ids``).

    ``committed_seq`` is therefore the stale sequence; the caller supplies the
    just-sampled token(s) separately.
    """
    n = len(committed_seq)
    prompt = np.asarray(prompt, dtype=np.int64)
    width = max(n + 64, len(prompt) + 64)
    token_ids_cpu = np.zeros((1, width), dtype=np.int64)
    token_ids_cpu[0, :n] = np.asarray(committed_seq, dtype=np.int64)
    return SimpleNamespace(
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
        num_tokens_no_spec=np.asarray([n], dtype=np.int64),
        num_prompt_tokens=np.asarray([len(prompt)], dtype=np.int64),
        token_ids_cpu=token_ids_cpu,
    )


def _fresh_drafter(g=12, k_scoped=16, min_uniq=1, num_spec_tokens=16):
    os.environ["VLLM_S4_G"] = str(g)
    os.environ["VLLM_S4_K_SCOPED"] = str(k_scoped)
    os.environ["VLLM_S4_MIN_UNIQ"] = str(min_uniq)
    os.environ["VLLM_S4_LOG_EVERY"] = "0"
    return SR.ScopedReemissionDrafter(_make_config(num_spec_tokens=num_spec_tokens))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_verbatim_copy_single_token_step():
    """Gate-open, 1 token sampled this step: draft must be the TRUE continuation.

    Reproduces the measured failure: with the stale-stream read the drafter
    proposes ``prompt[e-1 : ...]`` (position 0 == the token JUST sampled this
    step, guaranteed to reject), instead of ``prompt[e : ...]``.
    """
    g, k = 12, 16
    P = 60
    prompt = list(range(100, 100 + P))  # distinct -> every g-gram unique

    m = 30  # generated has re-emitted prompt[0:m]; last generated = prompt[m-1]
    # True sequence (conceptually) = prompt + prompt[0:m].
    # Stale committed sequence (what token_ids_cpu holds at merge time) is
    # missing the last generated token prompt[m-1], sampled THIS step.
    committed = prompt + prompt[0 : m - 1]
    sampled_this_step = [prompt[m - 1]]  # arrives only via sampled_token_ids

    drafter = _fresh_drafter(g=g, k_scoped=k)
    ib = _make_input_batch(prompt, committed)
    mtp_draft = [[7, 8]]  # arbitrary MTP K2 draft that must be overridden

    merged = drafter.merge(mtp_draft, ib, [sampled_this_step])

    true_continuation = prompt[m : m + k]
    assert merged[0] == true_continuation, (
        f"gate-open draft misaligned:\n  got   {merged[0]}\n  "
        f"want  {true_continuation}\n  (got[0]={merged[0][0]} is "
        f"prompt[m-1]={prompt[m-1]}? {merged[0][0] == prompt[m-1]} "
        f"-> classic off-by-one / stale-stream)"
    )
    print("PASS test_verbatim_copy_single_token_step")


def test_verbatim_copy_multi_token_step():
    """Gate-open with s=3 tokens accepted this step (grew>1). Draft still aligns.

    Proves the fix advances the needle by ALL of this step's sampled tokens, not
    just one (an off-by-one fix that hard-coded +1 would fail this)."""
    g, k = 12, 16
    P = 70
    prompt = list(range(200, 200 + P))
    m = 40
    s = 3
    committed = prompt + prompt[0 : m - s]
    sampled_this_step = prompt[m - s : m]  # 3 tokens verified/sampled this step

    drafter = _fresh_drafter(g=g, k_scoped=k)
    ib = _make_input_batch(prompt, committed)
    merged = drafter.merge([[1, 2]], ib, [sampled_this_step])

    true_continuation = prompt[m : m + k]
    assert merged[0] == true_continuation, (
        f"multi-token step misaligned: got {merged[0]} want {true_continuation}"
    )
    print("PASS test_verbatim_copy_multi_token_step")


def test_needle_longer_than_g():
    """s >= g: the whole needle comes from this step's sampled tokens."""
    g, k = 6, 16
    P = 60
    prompt = list(range(300, 300 + P))
    m = 30
    s = 8  # >= g
    committed = prompt + prompt[0 : m - s]
    sampled_this_step = prompt[m - s : m]

    drafter = _fresh_drafter(g=g, k_scoped=k)
    ib = _make_input_batch(prompt, committed)
    merged = drafter.merge([[1, 2]], ib, [sampled_this_step])
    true_continuation = prompt[m : m + k]
    assert merged[0] == true_continuation, (
        f"s>=g needle misaligned: got {merged[0]} want {true_continuation}"
    )
    print("PASS test_needle_longer_than_g")


def test_gate_closed_passes_mtp_through():
    """Not in a copy span -> gate closed -> MTP draft returned unchanged."""
    g = 12
    prompt = list(range(100, 160))
    # Generated tail that does NOT occur in the prompt (out-of-range tokens).
    committed = prompt + [9000 + i for i in range(20)]
    sampled_this_step = [9999]

    drafter = _fresh_drafter(g=g, k_scoped=16)
    ib = _make_input_batch(prompt, committed)
    mtp_draft = [[42, 43]]
    merged = drafter.merge(mtp_draft, ib, [sampled_this_step])
    assert merged[0] == [42, 43], f"gate should be closed; got {merged[0]}"
    print("PASS test_gate_closed_passes_mtp_through")


def test_no_sampled_token_returns_mtp():
    """Partial prefill (no real sampled token) -> MTP passthrough, no draft."""
    g = 12
    prompt = list(range(100, 160))
    committed = prompt + prompt[0:20]
    drafter = _fresh_drafter(g=g, k_scoped=16)
    ib = _make_input_batch(prompt, committed)
    merged = drafter.merge([[5, 6]], ib, [[]])  # empty sampled row
    assert merged[0] == [5, 6], f"expected MTP passthrough; got {merged[0]}"
    print("PASS test_no_sampled_token_returns_mtp")


def _run_all():
    tests = [
        test_verbatim_copy_single_token_step,
        test_verbatim_copy_multi_token_step,
        test_needle_longer_than_g,
        test_gate_closed_passes_mtp_through,
        test_no_sampled_token_returns_mtp,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
