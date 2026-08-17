# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the thinking-budget stale-rescan / runaway-think bug.

Ported alongside the fix stolen from upstream PR #100 to
``vllm/v1/sample/thinking_budget_state.py``.

Bug (pre-fix): ``_update_think_state`` caches ``start_thinking`` /
``end_thinking`` and only rescans while they are ``-1``. After a *natural*
``</think>`` (the model closes the block on its own, within budget), those
offsets stay frozen at the first block's positions. A later ``<think>`` block
is therefore never re-detected: the state machine keeps seeing
``start < end`` (block 1's frozen positions), stays in the "exiting think
mode" branch forever, and never counts the second block against the budget ->
the second reasoning block runs away past ``thinking_token_budget`` with no
forced end token.

Fix: on a natural end (start >= 0, end >= 0, end > start, not in_end, not
continue_thinking) reset ``start_thinking`` / ``end_thinking`` to ``-1``,
clear the block counters, and advance ``scan_offset`` to ``len(output)`` so
the next block is scanned fresh from after the closed block.

This test is deliberately engine-free ("pure python"): it drives the pure
``_update_think_state`` state machine directly and needs neither a GPU nor a
running model. When torch + vllm are importable (the normal engine test env)
it imports the real class; otherwise it stubs the two import-time
dependencies and loads the module straight from the repo file.
"""

import importlib.util
import os
import sys
import types

THINK_START = 100  # stand-in <think> token id
THINK_END = 200  # stand-in </think> token id
FILLER = 1  # any non-marker token
BUDGET = 5


def _load_holder_class():
    """Return ThinkingBudgetStateHolder, importing the repo file directly.

    Falls back to stubbing ``torch`` and the logits-processor interface so the
    state-machine logic (which touches neither) can be exercised without an
    engine install.
    """
    try:  # real engine environment
        from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

        return ThinkingBudgetStateHolder
    except Exception:
        pass

    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.device = lambda *a, **k: "cpu"
        torch.bool = "bool"
        torch.long = "long"
        torch.zeros = lambda *a, **k: None
        torch.full = lambda *a, **k: None
        torch.Tensor = object
        sys.modules["torch"] = torch
    for name in (
        "vllm",
        "vllm.v1",
        "vllm.v1.sample",
        "vllm.v1.sample.logits_processor",
        "vllm.v1.sample.logits_processor.interface",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    iface = sys.modules["vllm.v1.sample.logits_processor.interface"]
    iface.BatchUpdate = object

    class _MoveDirectionality:
        SWAP = "swap"

    iface.MoveDirectionality = _MoveDirectionality

    repo_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "vllm",
            "v1",
            "sample",
            "thinking_budget_state.py",
        )
    )
    spec = importlib.util.spec_from_file_location(
        "_thinking_budget_state_under_test", repo_file
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ThinkingBudgetStateHolder


Holder = _load_holder_class()


def _make_holder():
    """Bypass __init__ (which builds torch tensors) and set only what the
    pure state machine reads."""
    holder = Holder.__new__(Holder)
    holder.think_start_token_ids = [THINK_START]
    holder.think_end_token_ids = [THINK_END]
    return holder


def _step(holder, state, output):
    """Mimic the no-spec path of ``update_state`` for one appended token."""
    state["output_tok_ids"] = list(output)
    state["spec_token_ids"] = []
    state["in_spec_mode"] = False
    state["force_index"] = []
    holder._update_think_state(state)


def _drive():
    """Two reasoning blocks: block 1 closes naturally within budget; block 2
    runs long past budget. Returns per-step observations."""
    holder = _make_holder()
    state = holder._init_state_entry(None, BUDGET)  # pure generation, no prompt

    # block1: <think> filler </think>  (natural end, 1 think token << budget)
    # gap:    filler filler            (ordinary answer text)
    # block2: <think> filler*10        (runaway: 10 think tokens >> budget)
    tokens = [THINK_START, FILLER, THINK_END] + [FILLER, FILLER] + [THINK_START] + [FILLER] * 10
    block2_start_idx = tokens.index(THINK_START, 3)  # index of block2's <think>

    out = []
    observations = []
    for i, tok in enumerate(tokens):
        out.append(tok)
        _step(holder, state, out)
        observations.append(
            {
                "i": i,
                "tok": tok,
                "output_len": len(out),
                "start_thinking": state["start_thinking"],
                "end_thinking": state["end_thinking"],
                "scan_offset": state.get("scan_offset"),
                "in_think": state["in_think"],
                "in_end": state["in_end"],
                "force_index": list(state.get("force_index", [])),
            }
        )
    return observations, block2_start_idx


def test_scan_window_resets_after_natural_end():
    """After block 1's natural </think>, the scan window advances and the
    cached block offsets are cleared (the core of the fix)."""
    observations, _ = _drive()
    # block1's </think> is emitted at output index 2 (third token).
    natural_end = observations[2]
    assert natural_end["tok"] == THINK_END
    assert natural_end["scan_offset"] == natural_end["output_len"], (
        "scan_offset must jump to len(output) after a natural </think>; "
        f"got {natural_end['scan_offset']} vs {natural_end['output_len']}. "
        "Pre-fix this key is absent/0 and old offsets are reused."
    )
    assert natural_end["start_thinking"] == -1
    assert natural_end["end_thinking"] == -1


def test_second_block_is_budget_bounded():
    """The second reasoning block must be counted against the budget and get a
    forced end -- pre-fix it is invisible to the counter and runs away."""
    observations, block2_start_idx = _drive()

    enforced = [
        o
        for o in observations
        if o["i"] > block2_start_idx and o["in_end"] and o["force_index"]
    ]
    assert enforced, (
        "second reasoning block never triggered budget forcing (in_end + "
        "force_index) -> runaway think. This is the pre-fix bug."
    )
    # Forcing must arrive promptly: no more than BUDGET (+small slack) think
    # tokens past the block start.
    first_forced_step = enforced[0]["i"]
    think_tokens_before_force = first_forced_step - block2_start_idx
    assert think_tokens_before_force <= BUDGET + 2, (
        f"forcing arrived {think_tokens_before_force} tokens into block 2; "
        f"budget is {BUDGET} -- looks like a runaway."
    )


def test_first_block_natural_end_not_force_ended():
    """Sanity: a block that closes within budget is not force-ended (no false
    positive from the reset)."""
    observations, block2_start_idx = _drive()
    for o in observations[: block2_start_idx]:
        assert not (o["in_end"] and o["force_index"]), (
            f"unexpected forced end at step {o['i']} before block 2"
        )


if __name__ == "__main__":
    test_scan_window_resets_after_natural_end()
    test_second_block_is_budget_bounded()
    test_first_block_natural_end_not_force_ended()
    print("all thinking-budget scan-reset tests passed")
