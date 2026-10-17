"""CPU test: QSA's selected-block rows are put in a fixed order after the top-k (run-to-run reproducibility).

persistent_topk writes its picks through atomicAdd, so a row's order varies between runs and the sparse attention
sums in that order. sort_selected_blocks_ makes the order a function of the set alone, invalid (-1) entries last.
"""
import torch

from vllm.models.qwen4_exp.nvidia import model as _model  # noqa: F401  (import order: avoids the qsa cycle)
from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as Q


def test_rows_sorted_with_invalid_entries_last():
    rows = torch.tensor([[7, 3, -1, 9, -1], [0, -1, -1, -1, -1], [4, 2, 8, 6, 5]], dtype=torch.int32)
    Q.sort_selected_blocks_(rows)
    assert rows.tolist() == [[3, 7, 9, -1, -1], [0, -1, -1, -1, -1], [2, 4, 5, 6, 8]]


def test_any_permutation_of_a_row_gives_the_same_output():
    g = torch.Generator().manual_seed(0)
    base = torch.randperm(1024, generator=g)[:512].to(torch.int32)
    base[400:] = -1  # fewer visible blocks than the budget
    ref = Q.sort_selected_blocks_(base.clone().unsqueeze(0))
    for _ in range(20):  # what atomic compaction does: same set, any order
        perm = base[torch.randperm(512, generator=g)].unsqueeze(0)
        assert torch.equal(Q.sort_selected_blocks_(perm), ref)
    assert int((ref[0] >= 0).sum()) == 400 and bool((ref[0, :400].diff() > 0).all())


def test_in_place_on_a_strided_view():
    buf = torch.full((4, 8), -7, dtype=torch.int32)
    view = buf[:, 2:6]  # a static buffer's slice, as CUDA-graph decode passes
    view.copy_(torch.tensor([[3, 1, -1, 2]] * 4, dtype=torch.int32))
    Q.sort_selected_blocks_(view)
    assert buf[:, 2:6].tolist() == [[1, 2, 3, -1]] * 4 and bool((buf[:, :2] == -7).all())


def test_kill_switch_is_read():
    assert isinstance(Q._SORT_SELECTED, bool)
