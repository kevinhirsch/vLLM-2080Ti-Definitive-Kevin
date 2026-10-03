# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the K6 lane plumbing (VLLM_K6_MUX). No GPU needed."""
import threading

import pytest
import torch

import vllm.k6_mux as k6
from vllm import forward_context as fc
from vllm.distributed import parallel_state as ps


class _G:
    def __init__(self, name):
        self.unique_name = name


@pytest.fixture
def mux_on(monkeypatch):
    monkeypatch.setattr(k6, "ENABLED", True)
    yield


def test_default_off_is_global_semantics():
    assert k6.ENABLED is False  # env unset in CI
    sentinel = object()
    with fc.override_forward_context(sentinel):
        assert fc.get_forward_context() is sentinel
        with k6.lane(k6.PREFILL):  # gate off: lane state is ignored
            assert fc.get_forward_context() is sentinel
    assert not fc.is_forward_context_available()


def test_prefill_lane_has_private_forward_context(mux_on):
    main_ctx, lane_ctx = object(), object()
    seen = {}
    gate = threading.Event(); done = threading.Event()

    def prefill_thread():
        with k6.lane(k6.PREFILL):
            assert not fc.is_forward_context_available()
            with fc.override_forward_context(lane_ctx):
                seen["lane_inside"] = fc.get_forward_context()
                gate.set(); done.wait(5)
            seen["lane_after"] = fc.is_forward_context_available()

    with fc.override_forward_context(main_ctx):
        t = threading.Thread(target=prefill_thread); t.start()
        assert gate.wait(5)
        assert fc.get_forward_context() is main_ctx  # main lane unaffected while the lane is inside
        done.set(); t.join()
        assert fc.get_forward_context() is main_ctx
    assert seen == {"lane_inside": lane_ctx, "lane_after": False}
    assert not fc.is_forward_context_available()


def test_group_resolution_per_lane(mux_on, monkeypatch):
    a, b = _G("tp:0"), _G("tp:0-k6prefill")
    monkeypatch.setattr(ps, "_TP", a)
    monkeypatch.setitem(ps._groups, "tp:0", lambda: a)
    assert ps.get_tp_group() is a and ps._resolve_group("tp:0") is a
    with k6.lane(k6.PREFILL, groups={"tp:0": b}):
        assert ps.get_tp_group() is b
        assert ps._resolve_group("tp:0") is b
    assert ps.get_tp_group() is a and ps._resolve_group("tp:0") is a
    # another thread never sees this thread's override
    out = {}
    with k6.lane(k6.PREFILL, groups={"tp:0": b}):
        th = threading.Thread(target=lambda: out.setdefault("g", ps.get_tp_group())); th.start(); th.join()
    assert out["g"] is a


def test_marlin_workspace_private_per_lane(mux_on):
    ws = torch.zeros(68, dtype=torch.int)
    assert k6.marlin_workspace(ws) is ws  # decode lane (main thread) keeps the captured address
    with k6.lane(k6.PREFILL):
        w1 = k6.marlin_workspace(ws); w2 = k6.marlin_workspace(ws)
    assert w1 is w2 and w1 is not ws and w1.shape == ws.shape and w1.dtype == ws.dtype
    assert int(w1.abs().sum()) == 0
    ws2 = torch.zeros(68, dtype=torch.int)
    with k6.lane(k6.PREFILL):
        assert k6.marlin_workspace(ws2) is not w1  # one per layer workspace


@pytest.mark.parametrize("rng,dec,pre,k3,exp", [
    ((0, -5), None, None, -1, {"decode": -5, "k3_allreduce": -1, "prefill": 0}),
    ((0, -5), 0, None, -1, {"decode": -1, "k3_allreduce": -1, "prefill": 0}),      # decode never below K3
    ((0, -5), -9, -3, -1, {"decode": -5, "k3_allreduce": -1, "prefill": -1}),      # clamp + prefill not above K3
    ((0, -1), None, None, -1, {"decode": -1, "k3_allreduce": -1, "prefill": 0}),
])
def test_priorities(monkeypatch, rng, dec, pre, k3, exp):
    monkeypatch.delenv("VLLM_K6_MUX_DECODE_PRIO", raising=False)
    monkeypatch.delenv("VLLM_K6_MUX_PREFILL_PRIO", raising=False)
    assert k6.choose_priorities(rng, decode=dec, prefill=pre, k3=k3) == exp


def test_lane_restores_state():
    assert k6.current_lane() == k6.DECODE
    with k6.lane(k6.PREFILL, groups={"x": 1}):
        with k6.lane(k6.DECODE):
            assert k6.current_lane() == k6.DECODE and k6.group_override("x") is None
        assert k6.current_lane() == k6.PREFILL and k6.group_override("x") == 1
    assert k6.current_lane() == k6.DECODE and k6.group_override("x") is None
