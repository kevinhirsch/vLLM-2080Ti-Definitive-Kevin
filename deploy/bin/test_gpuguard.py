"""RL: the boot GPU gate (foreign compute apps shrink the KV pool), the cooperative busy signal, KV pool parsing."""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpuguard  # noqa: E402
import unitrun  # noqa: E402


def _fake_smi(tmp_path, apps):
    p = tmp_path / "nvidia-smi"
    p.write_text("#!/bin/bash\n"
                 "case \"$*\" in\n"
                 "  *query-gpu*) echo '0, GPU-aaa'; echo '1, GPU-bbb';;\n"
                 f"  *query-compute-apps*) printf '%s' '{apps}';;\n"
                 "esac\n")
    p.chmod(0o755)
    return str(p)


def test_compute_apps_parse(tmp_path, monkeypatch):
    monkeypatch.setattr(gpuguard, "NVIDIA_SMI", _fake_smi(tmp_path, "10, VLLM::Worker_TP0, 20456, GPU-aaa\n33, python, 202, GPU-bbb\n"))
    apps = gpuguard.compute_apps()
    assert apps == [{"pid": 10, "name": "VLLM::Worker_TP0", "used_mib": 20456, "gpu": 0},
                    {"pid": 33, "name": "python", "used_mib": 202, "gpu": 1}]


def test_foreign_apps_excludes_the_engine_cgroup_and_names_the_owner(tmp_path, monkeypatch):
    apps = [{"pid": 10, "name": "w0", "used_mib": 1, "gpu": 0}, {"pid": 33, "name": "python", "used_mib": 202, "gpu": 0}]
    monkeypatch.setattr(unitrun, "owner_of_pid", lambda pid, at=None: {"unit": "lp-smoke.service", "lane": "lp", "how": "live-cgroup", "comm": "python"})
    monkeypatch.setattr(unitrun, "proc_cgroup", lambda pid: None)
    monkeypatch.setattr(gpuguard, "proc_cmdline", lambda pid: "python tools/lp/g8_smoke.py")
    f = gpuguard.foreign_apps(apps, eng={10})
    assert [x["pid"] for x in f] == [33]
    assert f[0]["unit"] == "lp-smoke.service" and f[0]["lane"] == "lp"
    d = gpuguard.describe(f)
    assert "pid 33" in d and "unit=lp-smoke.service" in d and "g8_smoke.py" in d


def test_wait_no_foreign_waits_then_fails_with_the_list(monkeypatch):
    seq = [[{"pid": 1}], [{"pid": 1}], []]
    monkeypatch.setattr(gpuguard, "foreign_apps", lambda: seq.pop(0) if len(seq) > 1 else seq[0])
    t = [0.0]
    ok, apps = gpuguard.wait_no_foreign(100, poll_s=5, clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s))
    assert ok and apps == []
    monkeypatch.setattr(gpuguard, "foreign_apps", lambda: [{"pid": 9}])
    t[0] = 0.0
    ok, apps = gpuguard.wait_no_foreign(10, poll_s=5, clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s))
    assert not ok and apps == [{"pid": 9}]


def test_busy_signal_lifecycle(tmp_path):
    f = str(tmp_path / "busy.json")
    assert gpuguard.busy_state(f) == {"busy": False}
    gpuguard.set_busy("K5", "k5 window", 60, window="k5w", phase="window", path=f)
    st = gpuguard.busy_state(f)
    assert st["busy"] and st["window"] == "k5w" and st["pid"] == os.getpid()
    assert not gpuguard.clear_busy("other", path=f)          # never clears someone else's window
    assert gpuguard.busy_state(f)["busy"]
    assert gpuguard.busy_state(f, now=time.time() + 120)["stale"] == "expired"
    assert gpuguard.clear_busy("k5w", path=f) and gpuguard.busy_state(f) == {"busy": False}


def test_busy_signal_goes_stale_when_the_owner_dies(tmp_path):
    f = str(tmp_path / "busy.json")
    p = subprocess.Popen(["sleep", "30"])
    gpuguard.set_busy("X", "r", 600, window="w", pid=p.pid, path=f)
    assert gpuguard.busy_state(f)["busy"]
    p.kill()
    p.wait()
    assert gpuguard.busy_state(f).get("stale") == "owner-dead"


def test_check_lets_the_owning_window_through(tmp_path, monkeypatch):
    f = str(tmp_path / "busy.json")
    monkeypatch.setattr(gpuguard, "BUSY_FILE", f)
    gpuguard.set_busy("K5", "k5 window", 60, window="k5w", path=f)
    assert gpuguard.check()[0] is False
    assert gpuguard.check("k5w")[0] is True
    assert gpuguard.check("other")[0] is False


def test_kv_pool_parse():
    txt = ("INFO [kv_cache_utils.py:2815] GPU KV cache size: 965,886 tokens, Maximum concurrency ...\n"
           "INFO [kv_cache_utils.py:2815] GPU KV cache size: 899,704 tokens, Maximum concurrency ...\n")
    assert gpuguard.kv_pools(txt) == [965886, 899704]
    assert gpuguard.kv_pools("nothing") == []


def test_shared_gpuok_sh_reads_the_busy_signal(tmp_path):
    """~/projects/lanes/windows/gpuok.sh (the lane-wide gate) refuses while a live busy signal is held by another window."""
    sh = os.path.expanduser("~/projects/lanes/windows/gpuok.sh")
    if not os.path.exists(sh):
        return
    f = str(tmp_path / "busy.json")
    gpuguard.set_busy("K5", "k5 window", 60, window="k5w", path=f)
    env = {**os.environ, "GPU_BUSY_FILE": f, "GATEWAY": "http://127.0.0.1:9", "ENGINE": "http://127.0.0.1:9"}
    r = subprocess.run(["bash", sh, "0", "1"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 1 and "GPU busy" in r.stdout
    # the owning window passes the busy check (then fails on the unreachable gateway only because it is not in-window)
    r = subprocess.run(["bash", sh, "0", "1"], capture_output=True, text=True, env={**env, "WINDOW_ID": "k5w"}, timeout=60)
    assert "GPU busy" not in r.stdout


# ---- L138: GPU leases ---------------------------------------------------------------------------------------------------
def _app(pid, mib, gpu=1, unit=None):
    return {"pid": pid, "name": "python", "used_mib": mib, "gpu": gpu, "unit": unit, "lane": None, "comm": "python", "cmd": "x"}


def test_lease_lifecycle_and_expiry(tmp_path):
    f = str(tmp_path / "leases.json")
    a = gpuguard.lease_acquire("lp/bench", 1, 600, unit="lp-bench.service", ttl_s=60, path=f)
    b = gpuguard.lease_acquire("k7/x", "all", 300, unit="k7-x.service", ttl_s=60, path=f)
    assert {r["owner"] for r in gpuguard.leases(path=f)} == {"lp/bench", "k7/x"}
    assert gpuguard.lease_release(a["id"], path=f) and not gpuguard.lease_release(a["id"], path=f)
    assert [r["id"] for r in gpuguard.leases(path=f)] == [b["id"]]
    assert gpuguard.leases(path=f, now=time.time() + 120) == []                    # expired = void


def test_a_lease_whose_owner_process_died_is_void(tmp_path):
    f = str(tmp_path / "leases.json")
    p = subprocess.Popen(["sleep", "30"])
    gpuguard.lease_acquire("lp/bench", 1, 600, pid=p.pid, ttl_s=600, path=f)
    assert len(gpuguard.leases(path=f)) == 1
    p.kill()
    p.wait()
    assert gpuguard.leases(path=f) == []


def test_a_unit_lease_survives_until_the_unit_should_have_started_then_needs_the_unit(tmp_path, monkeypatch):
    f = str(tmp_path / "leases.json")
    row = gpuguard.lease_acquire("lp/bench", 1, 600, unit="lp-bench.service", ttl_s=600, path=f)
    monkeypatch.setattr(unitrun, "is_active", lambda u: False)
    assert len(gpuguard.leases(path=f)) == 1                                       # launch not yet visible: inside the grace
    assert gpuguard.leases(path=f, now=row["since"] + gpuguard.LEASE_UNIT_GRACE_S + 5) == []


def test_audit_classifies_leased_over_and_unleased(monkeypatch):
    monkeypatch.setattr(unitrun, "owner_of_pid", lambda pid, at=None: {"unit": {10: "lp-bench.service", 11: "lp-bench.service", 12: "mystery.service"}.get(pid)})
    monkeypatch.setattr(unitrun, "proc_cgroup", lambda pid: None)
    monkeypatch.setattr(gpuguard, "proc_cmdline", lambda pid: "x")
    rows = [{"id": "L1", "owner": "lp/bench", "gpu": 1, "mib": 1000, "unit": "lp-bench.service", "pid": None}]
    apps = [_app(10, 600), _app(11, 500), _app(12, 50)]                            # 1100 MiB under a 1000 + 400 slack lease: ok; pid 12 unleased
    res = gpuguard.audit(apps, rows, eng=set())
    assert [a["state"] for a in res["apps"]] == ["leased-ok", "leased-ok", "unleased"] and not res["ok"]
    assert [a["pid"] for a in res["unleased"]] == [12]
    apps = [_app(10, 1100), _app(11, 500)]                                         # the lease's budget is shared by its apps: 1600 > 1400
    res = gpuguard.audit(apps, rows, eng=set())
    assert [a["state"] for a in res["apps"]] == ["leased-over", "leased-over"] and [a["pid"] for a in res["over"]] == [10, 11]
    assert gpuguard.audit([_app(10, 100)], rows, eng=set())["ok"]


def test_a_lease_on_another_gpu_does_not_cover_the_app(monkeypatch):
    monkeypatch.setattr(unitrun, "owner_of_pid", lambda pid, at=None: {"unit": "lp-bench.service"})
    monkeypatch.setattr(unitrun, "proc_cgroup", lambda pid: None)
    monkeypatch.setattr(gpuguard, "proc_cmdline", lambda pid: "x")
    rows = [{"id": "L1", "owner": "lp/bench", "gpu": 0, "mib": 1000, "unit": "lp-bench.service", "pid": None}]
    assert gpuguard.audit([_app(10, 100, gpu=1)], rows, eng=set())["apps"][0]["state"] == "unleased"
    rows[0]["gpu"] = "all"
    assert gpuguard.audit([_app(10, 100, gpu=1)], rows, eng=set())["apps"][0]["state"] == "leased-ok"


def test_gpucap_run_holds_a_lease_for_the_job_and_releases_it(tmp_path, monkeypatch):
    import gpucap
    monkeypatch.setattr(gpuguard, "LEASES_FILE", str(tmp_path / "leases.json"))
    seen = {}

    def fake_run(lane, job, cmd, **kw):
        seen["during"] = [r["owner"] for r in gpuguard.leases()]
        seen["mib"] = [r["mib"] for r in gpuguard.leases()]
        return {"rc": 0, "result": "success"}
    monkeypatch.setattr(unitrun, "run", fake_run)
    monkeypatch.setattr(gpucap.Watcher, "loop", lambda self: None)
    res = gpucap.run("lp", "smoke", ["true"], gpu=1, cap_mib=600, margin_mib=300, gate=False)
    assert seen["during"] == ["lp/smoke"] and seen["mib"] == [900]
    assert gpuguard.leases() == [] and res["rc"] == 0
