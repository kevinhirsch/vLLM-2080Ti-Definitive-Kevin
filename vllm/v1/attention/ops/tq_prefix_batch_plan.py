# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-metadata-only planning for the default-off E06 prefix prototype.

This module imports no torch/vLLM runtime and never reads device metadata.
Unclaimed trailing graph-padding rows and zero-length requests are left alone.
"""

from collections import defaultdict
from collections.abc import Sequence


def homogeneous_prefix_groups(
    query_start_loc: Sequence[int] | None,
    *,
    num_reqs: int,
    total_rows: int,
    eligible: bool,
    max_query_len: int = 4,
) -> tuple[tuple[int, ...], ...]:
    """Return only multi-request groups, or no plan for unsafe metadata.

    The caller establishes common device, NS, cache layout and backend. This
    prototype supports production MTP3's at-most-four verifier rows. Metadata
    must already be on the CPU; absence must not trigger a device copy.
    """
    if not eligible or query_start_loc is None:
        return ()
    if (
        type(num_reqs) is not int
        or type(total_rows) is not int
        or num_reqs < 0
        or total_rows < 0
        or len(query_start_loc) < num_reqs + 1
    ):
        return ()
    starts = query_start_loc[:num_reqs + 1]
    if (
        any(type(x) is not int for x in starts)
        or starts[0] != 0
        or starts[-1] > total_rows
        or any(a > b for a, b in zip(starts, starts[1:]))
    ):
        return ()
    groups: dict[int, list[int]] = defaultdict(list)
    for request_idx, (start, end) in enumerate(zip(starts, starts[1:])):
        q_len = end - start
        if 0 < q_len <= max_query_len:
            groups[q_len].append(request_idx)
    return tuple(
        sorted(
            (tuple(group) for group in groups.values() if len(group) > 1),
            key=lambda group: group[0],
        )
    )
