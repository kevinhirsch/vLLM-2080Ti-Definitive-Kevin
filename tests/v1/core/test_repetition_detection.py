# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.core.sched.utils import check_sequence_repetition


def test_repetition_detection_is_still_available_when_explicitly_configured():
    params = RepetitionDetectionParams(
        max_pattern_size=2,
        min_pattern_size=1,
        min_count=3,
    )
    assert check_sequence_repetition([1, 2, 1, 2, 1, 2], params)


def test_repetition_detection_does_not_match_non_repeating_tool_arguments():
    params = RepetitionDetectionParams(
        max_pattern_size=32,
        min_pattern_size=1,
        min_count=8,
    )
    # Markdown table separators can repeat in a JSON string without the
    # generated argument being a repeated model response.
    tokens = [10, 11, 12, 13, 14, 15, 16, 17] * 2 + [18, 19]
    assert not check_sequence_repetition(tokens, params)
