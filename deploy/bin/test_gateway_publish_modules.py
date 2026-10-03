#!/usr/bin/env python3
"""Lane SH: the publisher ships a MANIFEST of startup-loaded modules beside the shim, and verifies what the restarted
gateway actually loaded.

- A module-only change (shim bytes unchanged) used to take the no-restart "current" path: the new module sat on disk
  unloaded until some later restart paired it with whatever shim was live then. Now any module difference is a full,
  drained publish.
- Every manifest module: committed-HEAD check, atomic install, rollback that restores (or removes a first install).
- Readback: GET /gateway/modules reports the sha256 of the exact bytes each module executed.

Run:  python -m pytest -q test_gateway_publish_modules.py
"""
import asyncio
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("gateway_safe_publish_modules_test", HERE / "gateway_safe_publish.py")
pub = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pub)


class _Tree(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="gw-pubmod-")
        self.addCleanup(self._td.cleanup)
        d = Path(self._td.name)
        (d / "src").mkdir()
        (d / "live").mkdir()
        self.src, self.live = d / "src", d / "live"
        (self.src / "keepalive-shim.py").write_text("shim v1\n")
        (self.live / "keepalive-shim.py").write_text("shim v1\n")
        (self.src / "dash.html").write_text("<html>p</html>")
        for p in (patch.object(pub, "SOURCE", self.src / "keepalive-shim.py"),
                  patch.object(pub, "RUNTIME", self.live / "keepalive-shim.py"),
                  patch.object(pub, "DASH_SOURCE", self.src / "dash.html"),
                  patch.object(pub, "DASH_RUNTIME", self.live / "dash.html"),
                  patch.object(pub, "GATEWAY_MODULES", (("mod_a.py", "mod_a"), ("mod_b.py", "mod_b")))):
            p.start()
            self.addCleanup(p.stop)

    def git_show(self, args, **kw):
        if args[:2] == ["git", "show"]:
            return (self.src / args[2].rsplit("/", 1)[1].replace("gateway_dashboard.html", "dash.html")).read_bytes()
        raise AssertionError(args)


class ManifestInstallRestore(_Tree):
    def test_install_restore_and_first_install_removed(self):
        (self.src / "mod_a.py").write_text("A2\n")
        (self.live / "mod_a.py").write_text("A1\n")
        (self.src / "mod_b.py").write_text("B1\n")                    # first install
        st = pub._install_gateway_files(b"<html>p</html>")
        self.assertEqual((st["mod_a"], st["mod_b"]), ("installed", "installed"))
        self.assertEqual((self.live / "mod_a.py").read_text(), "A2\n")
        self.assertEqual(pub._install_gateway_files(b"<html>p</html>")["mod_a"], "current")
        pub._restore_gateway_files(st)
        self.assertEqual((self.live / "mod_a.py").read_text(), "A1\n")
        self.assertFalse((self.live / "mod_b.py").exists())

    def test_absent_module_is_not_an_error(self):
        self.assertEqual(pub._install_gateway_files(b"<html>p</html>")["mod_a"], "absent")


class ModuleOnlyChangeRestarts(_Tree):
    def _publish_until_first_gateway_call(self):
        def stop(*a, **k):
            raise RuntimeError("reached the drained publish path")
        with patch.object(pub.subprocess, "check_output", self.git_show), patch.object(pub, "_http", stop), \
                patch.dict(os.environ, {"SHIM_ADMIN_TOKEN": "t"}):
            return pub._publish(60, 60)

    def test_module_change_with_same_shim_takes_the_drained_path(self):
        (self.src / "mod_a.py").write_text("A2\n")
        (self.live / "mod_a.py").write_text("A1\n")
        with self.assertRaisesRegex(RuntimeError, "drained publish path"):
            self._publish_until_first_gateway_call()
        self.assertEqual((self.live / "mod_a.py").read_text(), "A1\n")     # nothing installed before the drain

    def test_everything_identical_is_current_without_restart(self):
        (self.src / "mod_a.py").write_text("A1\n")
        (self.live / "mod_a.py").write_text("A1\n")
        self.assertEqual(self._publish_until_first_gateway_call()["status"], "current")

    def test_uncommitted_module_is_refused(self):
        (self.src / "mod_a.py").write_text("A2\n")

        def show(args, **kw):
            return b"committed A\n" if args[2].endswith("mod_a.py") else self.git_show(args)
        with patch.object(pub.subprocess, "check_output", show):
            with self.assertRaisesRegex(RuntimeError, "mod_a.py source differs from committed HEAD"):
                pub._publish(60, 60)


class LoadedModulesReadback(unittest.TestCase):
    def test_truth_table(self):
        sha = hashlib.sha256(b"A\n").hexdigest()
        srcs = {"mod_a.py": b"A\n"}
        cases = [({"modules": {"mod_a.py": {"loaded": True, "sha256": sha}}}, True),
                 ({"modules": {"mod_a.py": {"loaded": True, "sha256": "0" * 64}}}, False),
                 ({"modules": {"mod_a.py": {"loaded": False, "sha256": sha}}}, False),
                 ({"modules": {}}, False)]
        for reply, want in cases:
            with self.subTest(reply=reply), patch.object(pub, "_http", lambda *a, **k: reply):
                self.assertIs(pub._modules_loaded_match("t", srcs), want)

    def test_old_gateway_without_endpoint_only_excuses_the_optional_schema(self):
        def gone(*a, **k):
            raise OSError("404")
        with patch.object(pub, "_http", gone):
            self.assertTrue(pub._modules_loaded_match("t", {pub.CONFIG_SCHEMA_NAME: b"x"}))
            self.assertFalse(pub._modules_loaded_match("t", {"mod_a.py": b"x"}))


class ShimReportsWhatItLoaded(unittest.TestCase):
    def test_gateway_modules_reports_the_schema_bytes_it_executed(self):
        h = importlib.util.spec_from_file_location("golden_harness_pubmod", HERE / "test_gateway_golden_routing.py")
        G = importlib.util.module_from_spec(h)
        h.loader.exec_module(G)
        with tempfile.TemporaryDirectory(prefix="gw-pubmod-shim-") as td:
            m = G._fresh_shim(td, 0)
            body = json.loads(asyncio.run(m.gateway_modules(None)).body)
        want = hashlib.sha256((HERE / "gateway_config_schema.py").read_bytes()).hexdigest()
        rec = body["modules"]["gateway_config_schema.py"]
        self.assertEqual((rec["loaded"], rec["sha256"], rec["required"]), (True, want, False))
        self.assertEqual(body["gateway_sha256"], hashlib.sha256((HERE / "keepalive-shim.py").read_bytes()).hexdigest())

    def test_manifest_lists_the_schema(self):
        self.assertIn(("gateway_config_schema.py", "config_schema"), pub.GATEWAY_MODULES)


if __name__ == "__main__":
    unittest.main()
