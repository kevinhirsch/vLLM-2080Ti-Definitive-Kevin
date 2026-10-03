#!/usr/bin/env python3
"""Lane CFG: engine_config_check.py -- resolved chain, drift and whereis-knob, on a synthetic unit (no systemd, no engine).

Pins the failure classes that produced wrong answers in production:
  * an override sourced by the serve script, then silently overwritten by a later export (the PYTHONPATH class);
  * a knob read from a file that is NOT in the chain (serve-profile-v02.sh vs vllm-qwen27b.env);
  * a worker /proc environ clobbered by setproctitle being read as "knob absent";
  * a knob set in the env file that no engine code reads (dead knob);
  * a ledger claim checked against the running engine (--expect).
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_config_check as E  # noqa: E402


class Fixture(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        d = self.d = Path(self.td.name)
        (d / "tree" / "vllm").mkdir(parents=True)
        (d / "tree" / "vllm" / "envs.py").write_text(
            'X = {"VLLM_REAL_KNOB": lambda: int(os.getenv("VLLM_REAL_KNOB", "7"))}\n')
        (d / "tree" / "vllm" / "worker.py").write_text('import os\nos.environ.get("VLLM_DIRECT", "0")\n')
        (d / "unit.env").write_text("# engine env\nVLLM_REAL_KNOB=3\nVLLM_DEAD_KNOB=1\nVLLM_UNSET_ME=1\n"
                                    "PYTHONPATH=/from/envfile\nVLLM_MNBT=3632\n")
        (d / "override.env").write_text("PYTHONPATH=/from/override\nVLLM_NOT_EXPORTED=1\nexport VLLM_DIRECT=1\n")
        (d / "serve-real.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "[ -f %s ] && . %s\n"
            "export PYTHONPATH=/from/script\n"
            "unset VLLM_UNSET_ME || true\n"
            "ARGS=(python -m vllm.entrypoints.openai.api_server --port 8001 --max-num-batched-tokens ${VLLM_MNBT:-3584})\n"
            'exec "${ARGS[@]}"\n' % (d / "override.env", d / "override.env"))
        (d / "serve-other.sh").write_text("#!/usr/bin/env bash\nunset VLLM_REAL_KNOB\nexec python -m x\n")
        (d / "active-serve").write_text("serve-real.sh\n")
        (d / "serve-active.sh").write_text(
            '#!/usr/bin/env bash\nD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
            'target="$(cat "$D/active-serve" 2>/dev/null || true)"\ncase "$target" in\n'
            '  serve-real.sh) exec bash "$D/serve-real.sh" ;;\n  *) exec bash "$D/serve-other.sh" ;;\nesac\n')
        self.props = {"FragmentPath": str(d / "x.service"), "DropInPaths": "", "EnvironmentFiles": [(str(d / "unit.env"), False)],
                      "Environment": [], "ExecStart": "", "WorkingDirectory": str(d), "MainPID": "4242",
                      "ExecMainStartTimestamp": "now", "ExecArgv": [str(d / "serve-active.sh")]}
        self.patches = [patch.object(E, "unit_props", lambda unit=None: dict(self.props)),
                        patch.object(E, "RUNTIME_DIR", d), patch.object(E, "REPO_COPIES", [d / "repo"])]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        E._TREE_CACHE.clear()

    def chain(self):
        return E.build_chain("x.service", self.d / "tree")


class ChainTests(Fixture):
    def test_dispatch_layers_and_attribution(self):
        c = self.chain()
        self.assertEqual(c["serve_script"], str(self.d / "serve-real.sh"))
        kinds = [l["kind"] for l in c["layers"]]
        self.assertEqual(kinds[0], "EnvironmentFile")
        self.assertTrue(kinds[1].startswith("sourced by serve-real.sh:3"))
        self.assertEqual(kinds[2], "serve script body")
        f = c["final_env"]
        self.assertEqual(f["PYTHONPATH"], "/from/script")
        self.assertNotIn("VLLM_UNSET_ME", f)
        self.assertEqual(f["VLLM_DIRECT"], "1")
        self.assertIn("unit.env:2", c["attribution"]["VLLM_REAL_KNOB"])
        self.assertIn("after line 3", c["attribution"]["VLLM_DIRECT"])
        self.assertIn("serve script body", c["attribution"]["PYTHONPATH"])
        self.assertEqual(c["argv"][-2:], ["--max-num-batched-tokens", "3632"])

    def test_default_arm_when_pointer_unknown(self):
        (self.d / "active-serve").write_text("nonsense\n")
        c = self.chain()
        self.assertEqual(c["serve_script"], str(self.d / "serve-other.sh"))
        self.assertIn("default arm", c["dispatch"]["note"])

    def test_refuses_process_control_scripts(self):
        (self.d / "serve-real.sh").write_text("#!/bin/bash\nsudo systemctl restart foo\nexec python\n")
        c = self.chain()
        self.assertTrue(any("refusing" in e for e in c["errors"]))

    def test_lookalikes_lists_files_outside_the_chain(self):
        (self.d / "serve-real.sh.bak-1").write_text("x")
        looks = E.lookalikes(self.chain())
        self.assertIn(str(self.d / "serve-other.sh"), looks["not_in_chain"])
        self.assertNotIn(str(self.d / "serve-real.sh"), looks["not_in_chain"])
        self.assertEqual(looks["backup_variants_not_in_chain"], 1)


class CheckTests(Fixture):
    def _actual(self, env, argv=None, clob=None):
        c = self.chain()
        argv = argv if argv is not None else c["argv"]
        with patch.object(E, "proc_env", lambda pid: (dict(env), clob)), patch.object(E, "proc_argv", lambda pid: argv), \
                patch.object(E, "descendants", lambda pid: []), patch.object(E, "proc_start_epoch", lambda pid: None):
            return c, E.check(c, ["VLLM_REAL_KNOB=3", "VLLM_ALLREDUCE=32"])

    def codes(self, res):
        return {(f["code"], f["key"]) for f in res["findings"]}

    def test_in_sync_engine_reports_layer_conflicts_and_dead_knobs_only(self):
        c = self.chain()
        c2, res = self._actual(c["final_env"])
        codes = self.codes(res)
        self.assertIn(("override-dropped", "PYTHONPATH"), codes)      # override.env value never reached the engine
        self.assertIn(("envfile-unset", "VLLM_UNSET_ME"), codes)
        self.assertIn(("dead-knob", "VLLM_DEAD_KNOB"), codes)
        self.assertNotIn(("dead-knob", "VLLM_REAL_KNOB"), codes)
        self.assertNotIn(("dead-knob", "VLLM_MNBT"), codes)           # consumed by the script (argv), not dead
        self.assertIn(("claim-ok", "VLLM_REAL_KNOB"), codes)
        self.assertIn(("override-not-exported", "VLLM_NOT_EXPORTED"), codes)   # plain KEY=V never reaches the engine
        self.assertIn(("claim-false", "VLLM_ALLREDUCE"), codes)       # ledger claim vs the running engine
        self.assertFalse([f for f in res["findings"] if f["code"] in ("value-drift", "missing-in-engine", "argv-drift")])
        self.assertNotIn("VLLM_NOT_EXPORTED", c2["final_env"])

    def test_drift_in_env_and_argv(self):
        c = self.chain()
        env = dict(c["final_env"])
        env["VLLM_REAL_KNOB"] = "9"
        env.pop("VLLM_DIRECT")
        env["VLLM_STALE"] = "1"
        argv = list(c["argv"])
        argv[-1] = "3584"
        _, res = self._actual(env, argv)
        codes = self.codes(res)
        self.assertIn(("value-drift", "VLLM_REAL_KNOB"), codes)
        self.assertIn(("missing-in-engine", "VLLM_DIRECT"), codes)
        self.assertIn(("extra-in-engine", "VLLM_STALE"), codes)
        self.assertIn(("argv-drift", "--max-num-batched-tokens"), codes)
        self.assertTrue(res["drift"])

    def test_clobbered_environ_detection(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"\0" * 900 + b"VLLM_A=1\0VLLM_B=2\0")
        try:
            with patch.object(E, "Path", lambda p: Path(f.name) if p.startswith("/proc/") else Path(p)):
                env, note = E.proc_env(1)
        finally:
            os.unlink(f.name)
        self.assertEqual(env, {"VLLM_A": "1", "VLLM_B": "2"})
        self.assertIn("clobbered", note)


class WhereisTests(Fixture):
    def test_whereis_reports_production_chain_and_lookalikes(self):
        c = self.chain()
        with patch.object(E, "proc_env", lambda pid: ({"VLLM_REAL_KNOB": "3"}, None)), \
                patch.object(E, "proc_argv", lambda pid: c["argv"]):
            r = {x["knob"]: x for x in E.whereis(c, ["VLLM_REAL_KNOB", "VLLM_MNBT", "VLLM_DEAD_KNOB"])}
        k = r["VLLM_REAL_KNOB"]
        self.assertEqual(k["production_value"], "3")
        self.assertIn("unit.env:2", k["set_by"][0]["where"])
        self.assertTrue(any("serve-other.sh" in m for m in k["mentioned_outside_chain"]))   # the K1 trap, flagged
        self.assertEqual(k["code_default"]["value"], "7")
        self.assertEqual(r["VLLM_MNBT"]["argv_flags"][0]["flag"], "--max-num-batched-tokens")
        self.assertEqual(r["VLLM_MNBT"]["production_value"], "<absent>")
        self.assertIn("DEAD", r["VLLM_DEAD_KNOB"]["verdict"])

    def test_secrets_masked(self):
        self.assertTrue(E.mask("HF_TOKEN", "hf_abcdefgh").startswith("<set"))
        self.assertEqual(E.mask("VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS", "524288"), "524288")


if __name__ == "__main__":
    unittest.main()
