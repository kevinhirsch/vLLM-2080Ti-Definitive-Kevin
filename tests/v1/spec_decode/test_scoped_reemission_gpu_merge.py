# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-039 v3: real-tensor tests for the GPU-side scoped-drafter merge.

Unlike ``test_scoped_reemission.py`` (numpy only), these exercise the tensorized
``merge_gpu`` path with **real torch tensors** -- the scatter into the pinned
staging buffer, the H2D copy, and the on-device ``torch.where`` select. They run
on CUDA when available (the 2x2080Ti box) and fall back to CPU tensors otherwise,
so the merge logic is verified even without a GPU.

The drafter module itself is loaded with stubbed ``vllm.config`` / ``vllm.logger``
(same trick as the v2 test) so we do NOT import the full vLLM stack; only torch is
real. ``merge_gpu`` imports torch lazily, so the module stays importable without it.

Run: ``python3 tests/v1/spec_decode/test_scoped_reemission_gpu_merge.py``
     (needs torch + numpy; CUDA optional).
"""

import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is present in the target venv
    print("SKIP: torch unavailable")
    sys.exit(0)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_MODULE_PATH = os.path.join(_REPO, "vllm", "v1", "spec_decode", "scoped_reemission.py")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_drafter_module():
    """Load scoped_reemission.py with stubbed vllm deps (see v2 test for why)."""
    vllm_pkg = types.ModuleType("vllm")
    cfg_mod = types.ModuleType("vllm.config")
    cfg_mod.VllmConfig = object
    log_mod = types.ModuleType("vllm.logger")

    class _NullLogger:
        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

    log_mod.init_logger = lambda *_a, **_k: _NullLogger()

    _keys = ("vllm", "vllm.config", "vllm.logger")
    _saved = {k: sys.modules.get(k) for k in _keys}
    try:
        sys.modules["vllm"] = vllm_pkg
        sys.modules["vllm.config"] = cfg_mod
        sys.modules["vllm.logger"] = log_mod
        spec = importlib.util.spec_from_file_location(
            "scoped_reemission_gpu_under_test", _MODULE_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k in _keys:
            if _saved[k] is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = _saved[k]
    return mod


SR = _load_drafter_module()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
def _make_config(num_spec=16, max_model_len=65536, max_num_seqs=8):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=num_spec),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def _make_batch(rows):
    """``rows`` = list of (prompt_list, committed_list) -> multi-request batch.

    Reproduces the merge-time timing contract: ``token_ids_cpu`` holds the
    sequence through the PREVIOUS step (``committed``); this step's sampled
    tokens are passed separately. The prompt index is built from the first
    ``len(prompt)`` committed tokens (in the verbatim setup ``committed`` starts
    with the prompt), matching the v2 unit test.
    """
    n = len(rows)
    width = max(max(len(p), len(c)) for p, c in rows) + 64
    token_ids_cpu = np.zeros((n, width), dtype=np.int64)
    num_prompt = np.zeros(n, dtype=np.int64)
    num_tokens = np.zeros(n, dtype=np.int64)
    req_ids, req_id_to_index = [], {}
    for i, (p, c) in enumerate(rows):
        token_ids_cpu[i, : len(c)] = np.asarray(c, dtype=np.int64)
        num_prompt[i] = len(p)
        num_tokens[i] = len(c)
        req_ids.append(f"r{i}")
        req_id_to_index[f"r{i}"] = i
    return SimpleNamespace(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        num_tokens_no_spec=num_tokens,
        num_prompt_tokens=num_prompt,
        token_ids_cpu=token_ids_cpu,
    )


def _sampled_tensor(sampled_rows, num_spec):
    """Pack per-request sampled-token lists into a padded [n, num_spec+1] tensor."""
    width = num_spec + 1
    arr = np.full((len(sampled_rows), width), -1, dtype=np.int64)
    for i, row in enumerate(sampled_rows):
        arr[i, : len(row)] = np.asarray(row, dtype=np.int64)
    return torch.as_tensor(arr, device=DEVICE)


def _mtp_tensor(rows_k, num_spec):
    """MTP draft tensor [n, K_mtp] (K_mtp<num_spec simulates VLLM_MTP_DRAFT_CAP)."""
    return torch.as_tensor(np.asarray(rows_k, dtype=np.int64), device=DEVICE)


def _fresh(gpu_merge, g=12, k_scoped=16, min_uniq=1, num_spec=16, max_num_seqs=8):
    os.environ["VLLM_S4_G"] = str(g)
    os.environ["VLLM_S4_K_SCOPED"] = str(k_scoped)
    os.environ["VLLM_S4_MIN_UNIQ"] = str(min_uniq)
    os.environ["VLLM_S4_LOG_EVERY"] = "0"
    os.environ["VLLM_S4_GPU_MERGE"] = "1" if gpu_merge else "0"
    return SR.ScopedReemissionDrafter(
        _make_config(num_spec=num_spec, max_num_seqs=max_num_seqs)
    )


def _strip(row):
    return [t for t in row if t != -1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_gpu_merge_returns_padded_tensor():
    """merge_gpu returns a [n, num_spec] int32 device tensor (NOT a list)."""
    num_spec = 16
    prompt = list(range(100, 160))
    m = 30
    committed = prompt + prompt[0 : m - 1]
    batch = _make_batch([(prompt, committed)])
    mtp = _mtp_tensor([[7, 8]], num_spec)  # K_mtp=2 (capped)
    sampled = _sampled_tensor([[prompt[m - 1]]], num_spec)

    drafter = _fresh(gpu_merge=True)
    merged = drafter.merge_gpu(mtp, batch, sampled)

    assert isinstance(merged, torch.Tensor), "merge_gpu must return a tensor"
    assert merged.shape == (1, num_spec), f"bad shape {tuple(merged.shape)}"
    assert merged.dtype == torch.int32, f"bad dtype {merged.dtype}"
    assert merged.device.type == DEVICE.type, "wrong device"
    print("PASS test_gpu_merge_returns_padded_tensor")


def test_gpu_merge_gate_open_row_is_scoped_continuation():
    """A gate-open request's row == scoped continuation, left-aligned, -1 padded."""
    num_spec, k = 16, 16
    prompt = list(range(100, 160))
    m = 30
    committed = prompt + prompt[0 : m - 1]
    batch = _make_batch([(prompt, committed)])
    mtp = _mtp_tensor([[7, 8]], num_spec)
    sampled = _sampled_tensor([[prompt[m - 1]]], num_spec)

    drafter = _fresh(gpu_merge=True)
    row = drafter.merge_gpu(mtp, batch, sampled).cpu().tolist()[0]

    want = prompt[m : m + k]
    assert _strip(row) == want, f"gate-open row: got {_strip(row)} want {want}"
    # Positions beyond the scoped length must be -1 (not stale MTP tokens).
    assert row[len(want) :] == [-1] * (num_spec - len(want)), "missing -1 pad"
    assert 7 not in row and 8 not in row, "MTP tokens leaked into a gate-open row"
    print("PASS test_gpu_merge_gate_open_row_is_scoped_continuation")


def test_gpu_merge_gate_closed_row_is_width_normalized_mtp():
    """Gate-closed row == MTP draft placed into width-num_spec, -1 padded.

    This is the width-normalization that makes VLLM_MTP_DRAFT_CAP async-safe:
    the K_mtp=2 tensor becomes [m0, m1, -1, ... -1] (width 16)."""
    num_spec = 16
    prompt = list(range(100, 160))
    # Tail that does not occur in the prompt -> gate closed.
    committed = prompt + [9000 + i for i in range(20)]
    batch = _make_batch([(prompt, committed)])
    mtp = _mtp_tensor([[42, 43]], num_spec)  # K_mtp=2
    sampled = _sampled_tensor([[9999]], num_spec)

    drafter = _fresh(gpu_merge=True)
    row = drafter.merge_gpu(mtp, batch, sampled).cpu().tolist()[0]

    assert row[:2] == [42, 43], f"MTP tokens not preserved: {row[:2]}"
    assert row[2:] == [-1] * (num_spec - 2), f"bad width-normalization: {row}"
    print("PASS test_gpu_merge_gate_closed_row_is_width_normalized_mtp")


def test_gpu_merge_mixed_batch():
    """Batch with one gate-open and one gate-closed request, merged correctly."""
    num_spec, k = 16, 16
    p0 = list(range(100, 160))  # gate-open request
    m = 30
    c0 = p0 + p0[0 : m - 1]
    p1 = list(range(500, 560))  # gate-closed request (novel tail)
    c1 = p1 + [8000 + i for i in range(20)]

    batch = _make_batch([(p0, c0), (p1, c1)])
    mtp = _mtp_tensor([[7, 8], [42, 43]], num_spec)
    sampled = _sampled_tensor([[p0[m - 1]], [9999]], num_spec)

    drafter = _fresh(gpu_merge=True)
    merged = drafter.merge_gpu(mtp, batch, sampled).cpu().tolist()

    assert _strip(merged[0]) == p0[m : m + k], f"row0 scoped wrong: {_strip(merged[0])}"
    assert _strip(merged[1]) == [42, 43], f"row1 MTP wrong: {_strip(merged[1])}"
    print("PASS test_gpu_merge_mixed_batch")


def test_gpu_merge_multi_token_accept_step():
    """s=3 tokens accepted this step: scoped row still aligns (needle advances by
    ALL sampled tokens, not a hard-coded +1)."""
    num_spec, k = 16, 16
    prompt = list(range(200, 270))
    m, s = 40, 3
    committed = prompt + prompt[0 : m - s]
    sampled_row = prompt[m - s : m]
    batch = _make_batch([(prompt, committed)])
    mtp = _mtp_tensor([[1, 2]], num_spec)
    sampled = _sampled_tensor([sampled_row], num_spec)

    drafter = _fresh(gpu_merge=True)
    row = drafter.merge_gpu(mtp, batch, sampled).cpu().tolist()[0]
    assert _strip(row) == prompt[m : m + k], f"multi-accept misaligned: {_strip(row)}"
    print("PASS test_gpu_merge_multi_token_accept_step")


def test_gpu_merge_no_sampled_row_passes_mtp_through():
    """Partial-prefill row (no sampled token) -> width-normalized MTP, no draft."""
    num_spec = 16
    prompt = list(range(100, 160))
    committed = prompt + prompt[0:20]
    batch = _make_batch([(prompt, committed)])
    mtp = _mtp_tensor([[5, 6]], num_spec)
    sampled = _sampled_tensor([[]], num_spec)  # empty sampled row

    drafter = _fresh(gpu_merge=True)
    row = drafter.merge_gpu(mtp, batch, sampled).cpu().tolist()[0]
    assert row[:2] == [5, 6] and row[2:] == [-1] * (num_spec - 2), (
        f"expected MTP passthrough; got {row}"
    )
    print("PASS test_gpu_merge_no_sampled_row_passes_mtp_through")


def test_gpu_merge_matches_v2_list_merge():
    """The GPU tensor path and the v2 CPU-list path adjudicate identically.

    Strip the -1 padding from each tensor row and it must equal the v2 list row,
    request-for-request -- proving merge_gpu is an async-safe re-expression of
    merge, not a different policy (a clean A/B requires this)."""
    num_spec, k = 16, 16
    p0 = list(range(100, 160))
    m = 30
    c0 = p0 + p0[0 : m - 1]
    p1 = list(range(500, 560))
    c1 = p1 + [8000 + i for i in range(20)]
    p2 = list(range(700, 770))  # a second gate-open request
    m2 = 25
    c2 = p2 + p2[0 : m2 - 1]

    rows = [(p0, c0), (p1, c1), (p2, c2)]
    mtp_rows = [[7, 8], [42, 43], [9, 10]]
    sampled_rows = [[p0[m - 1]], [9999], [p2[m2 - 1]]]

    # v2 list path (fresh drafter, identical env/config).
    d_cpu = _fresh(gpu_merge=False)
    v2 = d_cpu.merge(
        _mtp_tensor(mtp_rows, num_spec),
        _make_batch(rows),
        _sampled_tensor(sampled_rows, num_spec),
    )

    # v3 GPU path (fresh drafter).
    d_gpu = _fresh(gpu_merge=True)
    v3 = d_gpu.merge_gpu(
        _mtp_tensor(mtp_rows, num_spec),
        _make_batch(rows),
        _sampled_tensor(sampled_rows, num_spec),
    )
    v3_stripped = [_strip(r) for r in v3.cpu().tolist()]

    assert v3_stripped == v2, f"path divergence:\n  v2 {v2}\n  v3 {v3_stripped}"
    # Sanity: rows 0 and 2 are scoped copies, row 1 is MTP.
    assert v2[0] == p0[m : m + k] and v2[2] == p2[m2 : m2 + k]
    assert v2[1] == [42, 43]
    print("PASS test_gpu_merge_matches_v2_list_merge")


def test_gpu_merge_buffers_grow_across_batches():
    """Buffers allocated for a small batch grow when a larger batch arrives."""
    num_spec = 16
    drafter = _fresh(gpu_merge=True, max_num_seqs=2)

    def one_batch(n):
        rows, mtp_rows, sampled_rows = [], [], []
        for i in range(n):
            prompt = list(range(1000 + 100 * i, 1000 + 100 * i + 60))
            m = 30
            rows.append((prompt, prompt + prompt[0 : m - 1]))
            mtp_rows.append([7, 8])
            sampled_rows.append([prompt[m - 1]])
        return (
            _mtp_tensor(mtp_rows, num_spec),
            _make_batch(rows),
            _sampled_tensor(sampled_rows, num_spec),
        )

    merged_small = drafter.merge_gpu(*one_batch(2))
    assert merged_small.shape == (2, num_spec)
    # Bigger than max_num_seqs=2 -> buffers must grow, no crash.
    merged_big = drafter.merge_gpu(*one_batch(5))
    assert merged_big.shape == (5, num_spec)
    print("PASS test_gpu_merge_buffers_grow_across_batches")


def _run_all():
    tests = [
        test_gpu_merge_returns_padded_tensor,
        test_gpu_merge_gate_open_row_is_scoped_continuation,
        test_gpu_merge_gate_closed_row_is_width_normalized_mtp,
        test_gpu_merge_mixed_batch,
        test_gpu_merge_multi_token_accept_step,
        test_gpu_merge_no_sampled_row_passes_mtp_through,
        test_gpu_merge_matches_v2_list_merge,
        test_gpu_merge_buffers_grow_across_batches,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n[device={DEVICE.type}] {len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
