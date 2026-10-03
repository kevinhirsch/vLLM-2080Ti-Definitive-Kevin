#!/usr/bin/env python3
"""Lane SH: the shim's include parts (gateway_parts.py) stay a pure, complete, guarded file split.

- GATEWAY_PARTS (the literal tuple), the _include_gateway_part(...) calls and the gateway_part_*.py files on disk are
  the same list in the same order: an unlisted part could hide SHIM_* reads from the config-schema guard (CFG) or be
  shipped without being loaded.
- every part carries the header, compiles, and is executed only through the shim (refused if unlisted);
- the expanded source compiles, and scan_code() sees the parts' env reads;
- the publisher ships every part and verifies it.

Run:  python -m pytest -q test_gateway_parts.py
"""
import ast
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SHIM = HERE / "keepalive-shim.py"


def _load(name, fname):
    s = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


P = _load("gateway_parts_t", "gateway_parts.py")
S = _load("gateway_config_schema_parts_t", "gateway_config_schema.py")
PUB = _load("gateway_safe_publish_parts_t", "gateway_safe_publish.py")


class PartsAreOneList(unittest.TestCase):
    def test_tuple_includes_and_files_agree(self):
        text = SHIM.read_text()
        listed = S.gateway_parts(str(SHIM))
        included = P.part_names(text)
        on_disk = sorted(p.name for p in HERE.glob("gateway_part_*.py"))
        self.assertEqual(listed, included, "GATEWAY_PARTS and the _include_gateway_part() calls differ")
        self.assertEqual(sorted(listed), on_disk, "a gateway_part_*.py file is not listed in GATEWAY_PARTS (or vice versa)")
        self.assertEqual(len(set(listed)), len(listed), "a part is included twice")

    def test_every_part_has_the_header_and_compiles(self):
        for name in S.gateway_parts(str(SHIM)):
            with self.subTest(part=name):
                text = (HERE / name).read_text()
                self.assertTrue(text.startswith(P.HEADER_PREFIX), "part must start with the '# gateway-part:' header")
                compile(text, name, "exec")

    def test_expanded_source_compiles_and_has_no_includes_left(self):
        src = P.expanded_source(str(SHIM))
        ast.parse(src)
        self.assertEqual(P.part_names(src), [])

    def test_scan_code_sees_reads_inside_parts(self):
        with tempfile.TemporaryDirectory(prefix="gw-parts-") as td:
            shim = Path(td) / "keepalive-shim.py"
            shim.write_text('import os\nGATEWAY_PARTS = ("gateway_part_x.py",)\nA = os.environ.get("SHIM_A", "1")\n'
                            '_include_gateway_part("gateway_part_x.py")\n')
            (Path(td) / "gateway_part_x.py").write_text('# gateway-part: t\nB = os.environ.get("SHIM_B", "2")\n')
            hits = S.scan_code(str(shim))
        self.assertEqual(sorted(hits), ["SHIM_A", "SHIM_B"])
        self.assertEqual(hits["SHIM_B"][0][1], "2")

    def test_split_then_expand_is_byte_identical(self):
        with tempfile.TemporaryDirectory(prefix="gw-parts-") as td:
            shim = Path(td) / "keepalive-shim.py"
            original = "a = 1\n\n# ---- banner\ndef f():\n    return 2\n\n\nb = f()\n"
            shim.write_text(original)
            P.split(str(shim), 3, 7, "demo", "demo part")
            self.assertIn('_include_gateway_part("gateway_part_demo.py")', shim.read_text())
            self.assertEqual(P.expanded_source(str(shim)), original)

    def test_registry_normalisation(self):
        a = 'x\nGATEWAY_PARTS = (\n    "gateway_part_a.py",\n    "gateway_part_b.py",\n)\ny\n'
        self.assertEqual(P.without_registry(a), 'x\nGATEWAY_PARTS = (\n)\ny\n')


class ShimExecutesPartsInItsNamespace(unittest.TestCase):
    def setUp(self):
        G = _load("golden_harness_parts_t", "test_gateway_golden_routing.py")
        self._td = tempfile.TemporaryDirectory(prefix="gw-parts-shim-")
        self.addCleanup(self._td.cleanup)
        self.m = G._fresh_shim(self._td.name, 0)

    def test_moved_names_live_in_the_shim_namespace(self):
        self.assertTrue(self.m.DASHBOARD_HTML.startswith("<!doctype html>"))
        self.assertIn("gateway_dashboard", self.m.gateway_dashboard.__name__)

    def test_loaded_parts_are_recorded_with_their_sha(self):
        import hashlib
        for name in S.gateway_parts(str(SHIM)):
            rec = self.m._GATEWAY_MODULES_LOADED[name]
            self.assertEqual((rec["loaded"], rec["required"], rec["kind"]), (True, True, "part"))
            self.assertEqual(rec["sha256"], hashlib.sha256((HERE / name).read_bytes()).hexdigest())

    def test_an_unlisted_part_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "not listed in GATEWAY_PARTS"):
            self.m._include_gateway_part("gateway_part_rogue.py")


class PublisherShipsParts(unittest.TestCase):
    def test_parts_are_in_the_manifest_from_the_published_source(self):
        with patch.object(PUB, "SOURCE", SHIM):
            names = PUB._module_names()
        for name in S.gateway_parts(str(SHIM)):
            self.assertIn(name, names)


if __name__ == "__main__":
    unittest.main()
