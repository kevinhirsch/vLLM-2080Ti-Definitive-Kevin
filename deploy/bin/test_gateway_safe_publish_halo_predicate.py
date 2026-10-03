"""LV (2026-10-03): the publisher asks "is a Halo run live?" exactly the way the incident supervisor counts occupancy.

GW2's 08:01 publish aborted after its drain because two incidents finished their runs mid-drain and became repair-requested with run
status "stopping" -- terminal to the supervisor (slot free; the next round waits on the start pause), active to the publisher."""
import ast
import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("gateway_safe_publish_pred", os.path.join(HERE, "gateway_safe_publish.py"))
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)

SUPERVISOR_CANDIDATES = [
    os.environ.get("HALO_SUPERVISOR_SOURCE", ""),
    "/home/kevin/projects/lanes/estate-independence-spine/tools/halo_incident_supervisor.py",
    "/home/kevin/.local/share/estate-overseer/tools/halo_incident_supervisor.py",
]


def lease(s=600):
    return {"expires_at_epoch": time.time() + s, "tool_calls_used": 1, "tool_calls_max": 40}


class Predicate(unittest.TestCase):
    def test_0801_finished_runs_mid_drain_are_not_active(self):
        for run in ("stopping", "completed", "stopped", "interrupted"):
            self.assertFalse(pub.halo_run_live({"status": "repair-requested", "lease": lease(), "halo_run": {"status": run}}), run)

    def test_live_runs_are_active_whatever_the_label(self):
        self.assertTrue(pub.halo_run_live({"status": "running", "lease": lease(), "halo_run": {"status": "running"}}))
        self.assertTrue(pub.halo_run_live({"status": "repair-requested", "lease": lease(), "halo_run": {"status": "queued"}}))
        # an investigation run is a live run too (the old label check missed it, then saw it "appear" when it finished)
        self.assertTrue(pub.halo_run_live({"status": "investigate", "lease": lease(), "halo_run": {"status": "running"}}))

    def test_dead_leases_and_closed_incidents_do_not_block(self):
        self.assertFalse(pub.halo_run_live({"status": "running", "lease": lease(-1), "halo_run": {"status": "running"}}))
        self.assertFalse(pub.halo_run_live({"status": "running", "lease": {"expires_at_epoch": time.time() + 60,
                                                                           "tool_calls_used": 40, "tool_calls_max": 40},
                                            "halo_run": {"status": "running"}}))
        self.assertFalse(pub.halo_run_live({"status": "recovered", "lease": lease(), "halo_run": {"status": "running"}}))
        self.assertFalse(pub.halo_run_live({"status": "running", "halo_run": {"status": "running"}}))

    def test_directory_scan(self):
        with tempfile.TemporaryDirectory() as d, patch.object(pub, "HALO_INCIDENTS", Path(d)):
            Path(d, "a.json").write_text(json.dumps({"id": "a", "status": "repair-requested", "lease": lease(), "halo_run": {"status": "stopping"}}))
            Path(d, "b.json").write_text(json.dumps({"id": "b", "status": "running", "lease": lease(), "halo_run": {"status": "running"}}))
            self.assertEqual(pub._halo_active_runs(), ["b"])


class LivenessQuiesceHold(unittest.TestCase):
    """The publish takes the liveness authority's quiesce hold (owned by its pid) and always gives it back."""

    def test_acquire_and_release_through_the_real_actuator(self):
        with tempfile.TemporaryDirectory() as d:
            act_src = os.path.join(HERE, "engine-actuator.py")
            # run the real actuator against a private BASE by rewriting its HOME
            env = dict(os.environ, HOME=d)
            os.makedirs(os.path.join(d, ".local/share/vllm-qwen27b"))
            calls = []
            real_run = pub.subprocess.run

            def run(cmd, **kw):
                calls.append(cmd)
                return real_run(cmd, env=env, **kw)
            with patch.object(pub, "ACTUATOR", Path(act_src)), patch.object(pub.subprocess, "run", run), \
                    patch.object(pub, "AUDIT_LOG", Path(d) / "audit.log"):
                lease = pub._liveness_hold(900)
                self.assertTrue(lease)
                holds = json.load(open(os.path.join(d, ".local/share/vllm-qwen27b/liveness-holds.json")))
                self.assertEqual(holds[0]["kind"], "quiesce")
                self.assertEqual(holds[0]["pid"], os.getpid())
                self.assertEqual(holds[0]["mode"], "owned")
                pub._liveness_release(lease)
                self.assertEqual(json.load(open(os.path.join(d, ".local/share/vllm-qwen27b/liveness-holds.json"))), [])

    def test_missing_actuator_is_audited_not_fatal(self):
        with tempfile.TemporaryDirectory() as d, patch.object(pub, "ACTUATOR", Path(d) / "nope.py"), \
                patch.object(pub, "AUDIT_LOG", Path(d) / "audit.log"):
            self.assertIsNone(pub._liveness_hold(900))
            self.assertIn("liveness-hold-skipped", (Path(d) / "audit.log").read_text())

    def test_phase_marker_only_on_our_lease(self):
        with tempfile.TemporaryDirectory() as d, patch.object(pub, "HALO_START_PAUSE", Path(d) / "pause.json"):
            tok = pub._begin_halo_quiesce(60, 60)
            pub._mark_halo_quiesce_phase(tok, "draining")
            row = json.loads((Path(d) / "pause.json").read_text())
            self.assertEqual((row["phase"], row["token"]), ("draining", tok))
            pub._mark_halo_quiesce_phase("other", "swapping")
            self.assertEqual(json.loads((Path(d) / "pause.json").read_text())["phase"], "draining")


class NoDriftFromTheSupervisor(unittest.TestCase):
    """The sets are copies of the supervisor's; fail the suite the moment the supervisor changes them."""

    def setUp(self):
        src = next((p for p in SUPERVISOR_CANDIDATES if p and os.path.exists(p)), None)
        if not src:
            self.skipTest("halo_incident_supervisor.py not present on this host")
        self.consts = {}
        for node in ast.parse(open(src).read()).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in ("TERMINAL_HALO_RUN_STATUSES", "OPEN_STATES", "ACTIVE_EXECUTION_STATES"):
                    v = node.value
                    if isinstance(v, ast.Call):          # frozenset({...})
                        v = v.args[0]
                    self.consts[name] = set(ast.literal_eval(v))

    def test_same_sets(self):
        self.assertEqual(self.consts.get("TERMINAL_HALO_RUN_STATUSES"), set(pub.HALO_TERMINAL_RUN_STATUSES))
        self.assertEqual(self.consts.get("OPEN_STATES"), set(pub.HALO_OPEN_STATES))
        self.assertEqual(self.consts.get("ACTIVE_EXECUTION_STATES"), set(pub.HALO_ACTIVE_EXECUTION_STATES))


if __name__ == "__main__":
    unittest.main()
