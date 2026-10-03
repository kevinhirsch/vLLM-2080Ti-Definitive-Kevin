"""RL (L106): the fault collector names the systemd unit that owned a FOREIGN Xid pid (S4 only dropped it)."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("efc_rl", os.path.join(HERE, "engine-fault-collector.py"))
efc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(efc)
sys.path.insert(0, HERE)
import unitrun  # noqa: E402

JOURNAL = ("Oct 03 06:34:06 HNET00 serve-active.sh[4192564]: (EngineCore pid=4192564) INFO trigger received signal=SIGTERM\n"
           "Oct 03 06:34:06 HNET00 serve-active.sh[4192747]: (Worker_TP0 pid=4192747) INFO Parent process exited\n")
FOREIGN = "2026-10-03T06:34:02-07:00 HNET00 kernel: NVRM: Xid (PCI:0000:04:00): 31, pid=41733, name=python, channel 0x22. MMU Fault\n"
OWN = "2026-10-03T06:34:02-07:00 HNET00 kernel: NVRM: Xid (PCI:0000:04:00): 31, pid=4192747, name=python, channel 0x22\n"


class FakeResolver:
    def __init__(self):
        self.calls = []

    def owner_of_pid(self, pid, at=None):
        self.calls.append((pid, at))
        return {"unit": "k2-bench.service", "lane": "k2", "job": "bench", "how": "registry", "comm": "python"}


def test_foreign_xid_names_the_owning_unit():
    r = FakeResolver()
    out = efc.foreign_xid_owners(JOURNAL, FOREIGN + OWN, resolver=r)
    assert len(out) == 1
    assert out[0]["pid"] == 41733 and out[0]["xid"] == 31 and out[0]["unit"] == "k2-bench.service" and out[0]["lane"] == "k2"
    assert r.calls[0][0] == 41733 and r.calls[0][1] is not None      # attributed at the Xid's own time


def test_own_xid_and_no_engine_pids_give_no_foreign_rows():
    assert efc.foreign_xid_owners(JOURNAL, OWN, resolver=FakeResolver()) == []
    assert efc.foreign_xid_owners("no pids here", FOREIGN, resolver=FakeResolver()) == []


def test_unattributed_pid_still_listed_with_its_kernel_name():
    class Nobody:
        def owner_of_pid(self, pid, at=None):
            return None
    out = efc.foreign_xid_owners(JOURNAL, FOREIGN, resolver=Nobody())
    assert out[0]["unit"] is None and out[0]["how"] == "unattributed" and out[0]["comm"] == "python"


def test_real_unitrun_registry_attributes_a_dead_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(unitrun, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(unitrun, "PROC", str(tmp_path / "noproc"))
    t = efc._xid_ts(FOREIGN)
    with open(tmp_path / "history.jsonl", "w") as fh:
        fh.write(json.dumps({"unit": "lp-g8smoke.service", "lane": "lp", "job": "g8smoke", "started": t - 30, "finished": t + 2,
                             "pids": {"41733": {"comm": "python", "first": t - 29, "last": t}}}) + "\n")
    out = efc.foreign_xid_owners(JOURNAL, FOREIGN, resolver=unitrun)
    assert out[0]["unit"] == "lp-g8smoke.service" and out[0]["lane"] == "lp" and out[0]["how"] == "registry"


def test_classification_unchanged_by_attribution():
    sig, _ = efc.classify(JOURNAL, FOREIGN)
    assert sig == "unknown-exit"
    sig, _ = efc.classify(JOURNAL, OWN)
    assert sig == "cuda-illegal-address"
