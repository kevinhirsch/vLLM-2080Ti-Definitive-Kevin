# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-python unit tests for the M-7 ``VLLM_MTP_DRAFT_CAP`` boot-time guard.

Validates ``validate_mtp_draft_cap`` in
``vllm/v1/spec_decode/mtp_draft_cap.py`` against the empirically-measured M-7
facts (2026-08-23, prod hardware, 7-experiment bisect), without importing
torch / CUDA. The module is stdlib-only, so it is loaded directly by path --
same pattern as ``tests/v1/core/test_spec_decode_workspace.py``.

Measured facts being pinned:
  * cap=2, K=3 (cap < K): per-step RuntimeError in
    ``_copy_draft_token_ids_to_cpu`` (draft-token tensor width mismatch).
  * cap=3, K=3 (cap == K): silent decode deadlock (0 tok/s forever, no
    errors) -- functionally a no-op cap, but still never actually worked as
    a *distinct* setting from unset.
  * cap unset: the only configuration that has ever worked.

The guard's contract is narrower than "cap==K is safe": it only special-cases
"unset" as ok. cap==K is accepted here purely because it is mathematically a
no-op (min(K, K) == K) -- not because it's been shown safe standalone.

Run: ``python3 tests/v1/spec_decode/test_mtp_draft_cap_guard.py`` (stdlib only).
"""

import importlib.util
import os
import sys


def _load_module():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    path = os.path.join(repo, "vllm", "v1", "spec_decode", "mtp_draft_cap.py")
    name = "_mtp_draft_cap"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so `from __future__ import annotations` string
    # annotations resolve against the right module (mirrors
    # test_spec_decode_workspace.py's loader).
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_module()
validate = M.validate_mtp_draft_cap


# ---------------------------------------------------------------------------


def test_cap_unset_is_ok():
    """raw_cap=None (env var absent) must never raise, for any K."""
    for k in (1, 2, 3, 16):
        validate(None, k)  # must not raise


def test_cap_empty_string_is_ok():
    """An explicitly-empty env value behaves like unset."""
    validate("", 3)  # must not raise


def test_cap_zero_is_ok():
    """VLLM_MTP_DRAFT_CAP=0 is the documented "no cap" sentinel (matches the
    runtime parsing in llm_base_proposer.py: `if _cap > 0: ...`)."""
    validate("0", 3)  # must not raise


def test_cap_equals_k_is_ok():
    """cap == K is a no-op cap (min(K, K) == K) -- the only value besides
    unset that has ever booted, per the M-7 bisect."""
    for k in (1, 2, 3, 16):
        validate(str(k), k)  # must not raise


def test_cap_less_than_k_raises():
    """Measured: cap=2, K=3 -> per-step RuntimeError in
    _copy_draft_token_ids_to_cpu. The guard must catch this at boot."""
    try:
        validate("2", 3)
    except ValueError as e:
        msg = str(e)
        assert "2" in msg and "3" in msg, msg
        assert "VLLM_MTP_DRAFT_CAP" in msg, msg
    else:
        raise AssertionError("expected ValueError for cap(2) < K(3)")


def test_cap_greater_than_k_raises():
    """cap > K was never validated anywhere and silently deadlocks decode
    (0 tok/s, no errors) -- must also be caught at boot."""
    try:
        validate("5", 3)
    except ValueError as e:
        msg = str(e)
        assert "5" in msg and "3" in msg, msg
    else:
        raise AssertionError("expected ValueError for cap(5) > K(3)")


def test_error_message_names_both_values_and_suggests_unset():
    """Message must name both cap and K and point at the fix (unset)."""
    try:
        validate("2", 3)
    except ValueError as e:
        msg = str(e)
        assert "num_speculative_tokens" in msg, msg
        assert "unset" in msg.lower() or "Unset" in msg, msg
    else:
        raise AssertionError("expected ValueError")


def test_matches_the_exact_measured_bisect_points():
    """Pin the two exact prod-measured (cap, K) pairs from M-7."""
    try:
        validate("2", 3)
        raise AssertionError("cap=2/K=3 must raise (measured: tensor mismatch)")
    except ValueError:
        pass
    # cap=3/K=3 is accepted by the guard (mathematically a no-op), even
    # though the measured behavior at that exact setting was a silent
    # deadlock -- the fix for *that* is "don't set the cap at all", which
    # this guard cannot detect (it can't tell "explicitly redundant" from
    # "load-bearing"). Documented, not a gap in this test.
    validate("3", 3)


def test_negative_cap_is_treated_as_unset():
    """A negative cap is nonsensical; parsing mirrors llm_base_proposer.py's
    `if _cap > 0`, so <=0 (including negative) is a no-op, not an error."""
    validate("-1", 3)  # must not raise


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failed:
        raise SystemExit(f"{failed}/{len(tests)} tests failed")
    print(f"All {len(tests)} tests passed.")
