# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""F-5 (backlog RANK 2, f5-forced-k lane): pure-Python "doubled token"
anomaly checkers for the forced-K probe's deterministic prompts (see
PROBE.md "Prompts" / "Doubled-token signature (grep / detection)").

These have nothing to do with the engine itself -- they parse a completed
generation's TEXT against a mechanically-known golden sequence and report
where it diverges. Kept as a standalone module (not a test helper) so the
same checkers can be run standalone against a saved completion file from
the actual forced-K probe window (``python -m
vllm.v1.worker.f5_probe_checks <file>``), not only from pytest.

No torch, no vLLM engine imports -- deliberately importable and runnable
on CPU-only hardware, independent of everything else in this lane.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from enum import Enum


class AnomalyKind(str, Enum):
    REPEAT = "REPEAT"  # the same expected value appears twice
    SKIP = "SKIP"  # an expected value never appears
    REORDER = "REORDER"  # a value appears out of the expected order


@dataclass(frozen=True)
class Anomaly:
    kind: AnomalyKind
    position: int  # index into the observed sequence (0-based)
    expected: object
    actual: object

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.kind.value} at position {self.position}: expected {self.expected!r}, got {self.actual!r}"


def _extract_integers(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"-?\d+", text)]


def find_doubled_integers(
    text: str, expected_start: int, expected_end: int
) -> list[Anomaly]:
    """Check a `count-1-300`-style completion (PROBE.md prompt 1) against
    the golden sequence ``expected_start..expected_end`` inclusive.

    Walks the *expected* sequence and the *observed* integers in lockstep:

      - if the next observed integer equals the value the walk is currently
        stalled on (i.e. the previous expected value, not the next one),
        that is a REPEAT -- the "1 2 2 3 4 4 5" signature from the backlog.
      - if the next observed integer skips ahead of what is expected, every
        value strictly between them is logged as SKIP and the walk jumps
        forward to resync (so one dropped number doesn't cascade into
        spurious anomalies for the rest of the sequence).
      - if the next observed integer is less than the previous expected
        value (and not an exact repeat of it), that is a REORDER.

    This is intentionally simple/greedy rather than an edit-distance
    alignment: for this probe's purpose (name the *first* divergence and
    get a rough count of how many, not compute a minimal diff) greedy
    resync is sufficient and much easier to reason about/test.
    """
    observed = _extract_integers(text)
    expected_seq = list(range(expected_start, expected_end + 1))
    anomalies: list[Anomaly] = []

    exp_idx = 0
    prev_expected: int | None = None
    for obs_idx, value in enumerate(observed):
        if exp_idx >= len(expected_seq):
            break
        expected_value = expected_seq[exp_idx]
        if value == expected_value:
            prev_expected = value
            exp_idx += 1
            continue
        if prev_expected is not None and value == prev_expected:
            anomalies.append(
                Anomaly(AnomalyKind.REPEAT, obs_idx, expected_value, value)
            )
            continue  # do not advance exp_idx: still waiting for expected_value
        if value > expected_value:
            # One or more expected values were skipped; log each, then
            # resync exp_idx to just past the observed value (if it's in
            # range) so subsequent comparisons aren't all spurious SKIPs.
            skip_to = min(value, expected_seq[-1] + 1)
            while exp_idx < len(expected_seq) and expected_seq[exp_idx] < skip_to:
                anomalies.append(
                    Anomaly(AnomalyKind.SKIP, obs_idx, expected_seq[exp_idx], None)
                )
                exp_idx += 1
            if exp_idx < len(expected_seq) and expected_seq[exp_idx] == value:
                prev_expected = value
                exp_idx += 1
            continue
        # value < expected_value and not a repeat of prev_expected:
        anomalies.append(Anomaly(AnomalyKind.REORDER, obs_idx, expected_value, value))

    # Anything left in expected_seq never showed up at all.
    for missing in expected_seq[exp_idx:]:
        anomalies.append(Anomaly(AnomalyKind.SKIP, len(observed), missing, None))

    return anomalies


def find_doubled_alphabet_cycle(text: str, unit: str, cycles: int) -> list[Anomaly]:
    """Check an `alphabet-x8`-style completion (PROBE.md prompt 2): `unit`
    (e.g. the 26-letter uppercase alphabet) repeated `cycles` times with no
    separators, expected output ``unit * cycles`` byte-for-byte.

    Reports at most one anomaly (this check is a straight byte compare,
    not a resync walk like the integer checker -- a single dropped/doubled
    cycle shifts every subsequent character, so enumerating every
    downstream mismatch individually would just be noise; the first
    divergence is what matters for triage).
    """
    # Keep only characters that belong to the unit's alphabet, so
    # incidental whitespace/newlines the model adds don't register as
    # spurious mismatches.
    allowed = set(unit)
    observed = "".join(ch for ch in text if ch in allowed)
    expected = unit * cycles

    if observed == expected:
        return []

    min_len = min(len(observed), len(expected))
    for i in range(min_len):
        if observed[i] != expected[i]:
            return [Anomaly(AnomalyKind.REORDER, i, expected[i], observed[i])]

    # One is a strict prefix of the other.
    if len(observed) < len(expected):
        return [
            Anomaly(
                AnomalyKind.SKIP, min_len, expected[min_len : min_len + 1], None
            )
        ]
    return [
        Anomaly(AnomalyKind.REPEAT, min_len, None, observed[min_len : min_len + 1])
    ]


def _main(argv: list[str]) -> int:  # pragma: no cover - CLI glue
    if len(argv) != 2:
        print(f"usage: {argv[0]} <completion-text-file>", file=sys.stderr)
        return 2
    with open(argv[1]) as f:
        text = f.read()
    anomalies = find_doubled_integers(text, 1, 300)
    if not anomalies:
        print("no anomalies found (count-1-300 golden range)")
        return 0
    for a in anomalies:
        print(a)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv))
