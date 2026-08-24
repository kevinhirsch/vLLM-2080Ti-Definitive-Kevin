# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-038 fork v2 — OFFLINE unit tests for the shared child-spec validation.

These run WITHOUT a built engine / torch / fastapi: they load
``vllm/entrypoints/openai/tq_fork_specs.py`` directly by file path, bypassing the
heavy ``vllm`` package __init__ (which imports torch). This is the pure-python
half of the fork2 test plan; the route-level + hardware e2e coverage that needs a
throwaway engine is described in docs/exp038-fork-v2.md.
"""
import contextlib
import importlib.util
import pathlib
import re

try:
    import pytest

    raises = pytest.raises
except ModuleNotFoundError:  # bare checkout: no pytest — provide a shim so the
    # __main__ self-runner below still exercises every assertion.
    @contextlib.contextmanager
    def raises(exc, match=None):
        try:
            yield
        except exc as e:  # noqa: BLE001
            if match is not None and not re.search(match, str(e)):
                raise AssertionError(
                    f"pattern {match!r} not found in {str(e)!r}"
                ) from e
        else:
            raise AssertionError(f"{exc.__name__} not raised")


# Load the dependency-free module by path so importing it does NOT trigger
# `vllm/__init__.py` (torch) — keeps this test runnable in a bare checkout.
_SPECS_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "vllm"
    / "entrypoints"
    / "openai"
    / "tq_fork_specs.py"
)
_spec = importlib.util.spec_from_file_location("tq_fork_specs", _SPECS_PATH)
tq_fork_specs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tq_fork_specs)

parse_child_specs = tq_fork_specs.parse_child_specs
CHILD_SAMPLING_KEYS = tq_fork_specs.CHILD_SAMPLING_KEYS
DEFAULT = tq_fork_specs.DEFAULT_CHILD_MAX_TOKENS


def test_default_max_tokens_applied():
    specs = parse_child_specs([{"temperature": 0.0}])
    assert specs == [{"temperature": 0.0, "max_tokens": DEFAULT}]


def test_max_tokens_coerced_to_int():
    specs = parse_child_specs([{"max_tokens": "128"}])
    assert specs[0]["max_tokens"] == 128
    assert isinstance(specs[0]["max_tokens"], int)


def test_all_whitelisted_keys_pass_through():
    child = {k: (0 if k != "stop" else ["</s>"]) for k in CHILD_SAMPLING_KEYS}
    specs = parse_child_specs([child])
    # Every whitelisted key survives; nothing extra is added beyond normalization.
    assert set(specs[0]) == set(child)


def test_unsupported_key_rejected():
    with raises(ValueError, match="unsupported key"):
        parse_child_specs([{"temperature": 0.0, "logits_processors": []}])


def test_non_dict_child_rejected():
    with raises(ValueError, match=r"children\[1\] must be an object"):
        parse_child_specs([{"temperature": 0.0}, "not-a-dict"])


def test_bad_max_tokens_rejected():
    with raises(ValueError, match="max_tokens must be an int"):
        parse_child_specs([{"max_tokens": "abc"}])


def test_input_not_mutated():
    child = {"temperature": 0.5}
    parse_child_specs([child])
    assert child == {"temperature": 0.5}  # no max_tokens injected into the input


def test_error_index_points_at_offending_child():
    with raises(ValueError, match=r"children\[2\]"):
        parse_child_specs(
            [{"temperature": 0.0}, {"top_p": 0.9}, {"bogus": 1}]
        )


if __name__ == "__main__":
    # Allow ``python test_tq_fork_specs.py`` in a bare env (no pytest runner).
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
