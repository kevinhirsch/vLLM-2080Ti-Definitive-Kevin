# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""F-5 (backlog RANK 2, f5-forced-k lane): unit tests for the doubled-token
anomaly checkers in ``vllm.v1.worker.f5_probe_checks``.

Pure string/int parsing -- no torch, no GPU, no vLLM engine. These pin the
checkers' behavior on hand-built examples, including the exact "1 2 2 3 4
4 5" doubling signature quoted in the backlog for this investigation.
"""

from vllm.v1.worker.f5_probe_checks import (
    AnomalyKind,
    find_doubled_alphabet_cycle,
    find_doubled_integers,
)

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class TestFindDoubledIntegers:
    def test_clean_sequence_has_no_anomalies(self) -> None:
        text = "\n".join(str(i) for i in range(1, 11))
        assert find_doubled_integers(text, 1, 10) == []

    def test_backlog_doubling_signature(self) -> None:
        # The exact example from the backlog context: "1 2 2 3 4 4 5".
        text = "1 2 2 3 4 4 5"
        anomalies = find_doubled_integers(text, 1, 5)
        repeats = [a for a in anomalies if a.kind == AnomalyKind.REPEAT]
        assert len(repeats) == 2
        assert (repeats[0].expected, repeats[0].actual) == (3, 2)
        assert (repeats[1].expected, repeats[1].actual) == (5, 4)

    def test_single_repeat_detected_with_position(self) -> None:
        text = "1 2 3 3 4 5"
        anomalies = find_doubled_integers(text, 1, 5)
        assert len(anomalies) == 1
        assert anomalies[0].kind == AnomalyKind.REPEAT
        assert anomalies[0].position == 3  # 0-based index of the extra "3"

    def test_skip_detected(self) -> None:
        text = "1 2 4 5"  # 3 is missing
        anomalies = find_doubled_integers(text, 1, 5)
        assert len(anomalies) == 1
        assert anomalies[0].kind == AnomalyKind.SKIP
        assert anomalies[0].expected == 3

    def test_trailing_skip_when_generation_stops_early(self) -> None:
        text = "1 2 3"  # golden range is 1..5; 4 and 5 never appear
        anomalies = find_doubled_integers(text, 1, 5)
        assert [a.expected for a in anomalies] == [4, 5]
        assert all(a.kind == AnomalyKind.SKIP for a in anomalies)

    def test_reorder_detected(self) -> None:
        text = "1 2 4 3 5"  # 4 and 3 swapped
        anomalies = find_doubled_integers(text, 1, 5)
        kinds = [a.kind for a in anomalies]
        assert AnomalyKind.REORDER in kinds or AnomalyKind.SKIP in kinds
        # Whichever classification, the divergence must be reported at all.
        assert len(anomalies) >= 1

    def test_extra_prose_around_numbers_is_ignored(self) -> None:
        # Prompt says "one integer per line, nothing else", but the checker
        # should still work if the model adds incidental text -- as long as
        # the numbers themselves are extractable and in order.
        text = "Sure, here you go:\n1\n2\n3\n4\n5\nHope that helps!"
        assert find_doubled_integers(text, 1, 5) == []

    def test_empty_text_reports_all_expected_as_skipped(self) -> None:
        anomalies = find_doubled_integers("", 1, 3)
        assert len(anomalies) == 3
        assert all(a.kind == AnomalyKind.SKIP for a in anomalies)


class TestFindDoubledAlphabetCycle:
    def test_clean_repetition_has_no_anomalies(self) -> None:
        text = ALPHABET * 3
        assert find_doubled_alphabet_cycle(text, ALPHABET, 3) == []

    def test_ignores_incidental_whitespace(self) -> None:
        text = "\n".join([ALPHABET] * 3)  # newline-joined instead of bare concat
        assert find_doubled_alphabet_cycle(text, ALPHABET, 3) == []

    def test_doubled_cycle_detected(self) -> None:
        # One extra full cycle inserted after the first: a duplicated run.
        text = ALPHABET + ALPHABET + ALPHABET + ALPHABET  # 4 cycles instead of 3
        anomalies = find_doubled_alphabet_cycle(text, ALPHABET, 3)
        assert len(anomalies) == 1
        assert anomalies[0].kind == AnomalyKind.REPEAT

    def test_dropped_cycle_detected(self) -> None:
        text = ALPHABET * 2  # 2 cycles instead of 3
        anomalies = find_doubled_alphabet_cycle(text, ALPHABET, 3)
        assert len(anomalies) == 1
        assert anomalies[0].kind == AnomalyKind.SKIP

    def test_corruption_mid_cycle_detected(self) -> None:
        corrupted = ALPHABET[:10] + "X" + ALPHABET[11:]  # one letter wrong
        text = ALPHABET + corrupted + ALPHABET
        anomalies = find_doubled_alphabet_cycle(text, ALPHABET, 3)
        assert len(anomalies) == 1
        assert anomalies[0].position == len(ALPHABET) + 10
