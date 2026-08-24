# SPDX-License-Identifier: Apache-2.0

from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder


def test_reset_retains_split_start_marker_after_end_marker():
    holder = object.__new__(ThinkingBudgetStateHolder)
    holder.think_start_token_ids = [10, 11, 12]
    holder.think_end_token_ids = [20, 21]
    state = {
        "output_tok_ids": [1, 2, 20, 21, 10, 11],
        "end_thinking": 2,
        "thinking_token_budget": 32,
        "in_think": True,
        "think_count": 7,
        "continue_thinking": False,
        "start_thinking": 0,
    }

    holder._reset_after_thinking_block(state)

    # Keep the two-token prefix of the next <think> marker, but start after
    # the completed </think> marker so it cannot be rediscovered.
    assert state["scan_offset"] == 4
    assert state["prev_output_length"] == len(state["output_tok_ids"])
    assert state["start_thinking"] == -1
    assert state["end_thinking"] == -1

