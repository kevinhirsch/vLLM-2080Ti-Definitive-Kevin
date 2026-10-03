"""Unit checks for tools/ir/offload_kv_config.py (CPU only, stdlib).  Run: python3 -m unittest tools.ir.test_offload_kv_config"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offload_kv_config as m  # noqa: E402


class T(unittest.TestCase):
    def test_config_json_shape(self):
        cfg = m.build_config("/ssd/kv", 12 * m.GIB, "abc123")
        self.assertEqual(cfg["kv_connector"], "OffloadingConnector")
        self.assertEqual(cfg["kv_load_failure_policy"], "recompute")
        self.assertEqual(cfg["engine_id"], m.ENGINE_ID)
        ex = cfg["kv_connector_extra_config"]
        self.assertEqual(ex["spec_name"], "TieringOffloadingSpec")
        self.assertEqual(ex["cpu_bytes_to_use"], 12 * m.GIB)
        self.assertEqual(ex["secondary_tiers"][0]["root_dir"], "/ssd/kv/checkpoint-abc123")
        self.assertNotIn("store_threshold", ex)
        json.dumps(cfg)  # serialisable

    def test_store_threshold_only_when_meaningful(self):
        self.assertNotIn("store_threshold", m.build_config("/x", 1, "f", store_threshold=1)["kv_connector_extra_config"])
        self.assertEqual(m.build_config("/x", 1, "f", store_threshold=2)["kv_connector_extra_config"]["store_threshold"], 2)

    def test_bad_policy(self):
        with self.assertRaises(ValueError):
            m.build_config("/x", 1, "f", failure_policy="abort")

    def test_size_guard(self):
        ok4, _ = m.check_cpu_bytes(4 * m.GIB)
        ok12, _ = m.check_cpu_bytes(12 * m.GIB)
        self.assertFalse(ok4)   # lane UP's 4 GiB result
        self.assertTrue(ok12)
        self.assertGreater(m.recommended_cpu_bytes(), 8 * m.GIB)  # the old 8 GiB default is below the working set
        self.assertLessEqual(m.recommended_cpu_bytes(), 14 * m.GIB)

    def test_cli_refuses_small_then_allows(self):
        base = ["config", "--root", "/r", "--fingerprint", "f", "--cpu-bytes", str(4 * m.GIB)]
        self.assertEqual(m.main(base), 2)
        self.assertEqual(m.main(base + ["--allow-small"]), 0)

    def test_stale_cleanup_respects_held(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "vllm_offload_aaa.mmap")
            b = os.path.join(d, "vllm_offload_bbb.mmap")
            other = os.path.join(d, "unrelated.bin")
            for p in (a, b, other):
                open(p, "w").close()
            held = {os.path.realpath(b)}
            self.assertEqual(m.stale_offload_files(d, held), [a])
            self.assertEqual(m.clean_stale(d, dry_run=True, held=held), [a])
            self.assertTrue(os.path.exists(a))
            self.assertEqual(m.clean_stale(d, held=held), [a])
            self.assertFalse(os.path.exists(a))
            self.assertTrue(os.path.exists(b) and os.path.exists(other))

    def test_stale_cleanup_with_real_open_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "vllm_offload_live.mmap")
            with open(p, "w") as fh:  # held open by THIS process
                self.assertEqual(m.stale_offload_files(d), [])
                fh.write("x")
            self.assertEqual(m.stale_offload_files(d), [p])


if __name__ == "__main__":
    unittest.main()
