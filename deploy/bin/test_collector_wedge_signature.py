#!/usr/bin/env python3
"""EF2: a watchdog wedge kill keeps the fault signature already in the journal (2026-10-02 12:00:30 TQ Xid filed as wedge)."""
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "efc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine-fault-collector.py"))
efc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(efc)

FENCE_JOURNAL = (
    "(Worker_TP1 pid=3780125) ERROR 10-02 12:00:30 [ef_fence.py:42] EF-FENCE first observed CUDA fault at tq:prefill-portion: "
    "CUDA error: an illegal memory access was encountered\n"
    "(Worker_TP1 pid=3780125) torch.AcceleratorError: CUDA error: an illegal memory access was encountered\n")


def test_fault_before_wedge_kill_keeps_fault_signature():
    sig, detail = efc.classify(FENCE_JOURNAL, "")
    assert sig == "cuda-illegal-address" and detail == "fence=tq"
    sig2, detail2 = efc.wedge_signature(sig, detail)
    assert sig2 == "cuda-illegal-address"
    assert detail2.startswith("fence=tq; killed by watchdog")


def test_genuine_wedge_without_fault_evidence():
    sig, detail = efc.classify("INFO: 200 OK\n", "")
    assert sig == "unknown-exit"
    assert efc.wedge_signature(sig, detail) == ("generation-wedge", efc.WEDGE_DETAIL)


def test_dead_core_with_live_api_is_not_a_wedge():
    sig, detail = efc.classify("vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.\n", "")
    assert efc.wedge_signature(sig, detail)[0] == "engine-dead-other"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
