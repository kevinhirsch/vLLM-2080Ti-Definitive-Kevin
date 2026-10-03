#!/usr/bin/env python3
"""Lane CFG: config_drift_report.py on fixture notes + fixture facts (no gateway, no engine, no vault)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_drift_report as C  # noqa: E402

FACTS = {"gw.local_budget": 14, "gw.local_max_out": 16384, "gw.pool_tokens": 922358, "gw.token_budget": "auto",
         "gw.interactive_never_overflow": 0, "gw.tiny_tokens": 1500, "cap.kv_pool_live": 899704,
         "env.__pid__": 4242, "env.VLLM_TQ_GQA_CUDA": "1", "argv.--max-num-seqs": "16", "flag.--language-model-only": True}


class Notes(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        d = Path(self.td.name)
        (d / "Qwen3.8 Live Config.md").write_text(
            "---\ndescription: live config, KV pool 922,358, MTP-3\n---\n# old\nKV pool: 637,560 tokens (history)\n"
            "## CURRENT as of 2026-10-01\nKV pool: 111 tokens\n"
            "## CURRENT as of 2026-10-02 23:40\n- **KV pool: 922,358 tokens**. `--max-num-seqs 16`. The gateway's "
            "`SHIM_POOL_TOKENS` (637,560) and `SHIM_TOKEN_BUDGET` (500,000) are STALE. VLLM_TQ_GQA_CUDA=1 and "
            "VLLM_CUSTOM_ALLREDUCE_MAX_SIZE_MB=32 and SHIM_LOCAL_BUDGET=14\n## Next\nKV pool: 5 tokens\n")
        self.p = patch.object(C, "VAULT", d)
        self.p.start()
        self.addCleanup(self.p.stop)

    def test_scoped_claims_and_kv_harvest(self):
        rows = [r for r in C.check_notes(FACTS) if r["source"] == "Qwen3.8 Live Config"]
        by = {(r["fact"], r["doc"]): r["status"] for r in rows}
        self.assertEqual(by[("cap.kv_pool_live", "922,358")], "MISMATCH")       # description + newest CURRENT block
        self.assertNotIn(("cap.kv_pool_live", "637,560"), by)                  # history above the block ignored
        self.assertNotIn(("cap.kv_pool_live", "111"), by)                      # older CURRENT block ignored
        self.assertNotIn(("cap.kv_pool_live", "5"), by)                        # next section ignored
        self.assertEqual(by[("argv.--max-num-seqs", "16")], "ok")
        self.assertEqual(by[("gw.pool_tokens", "637,560")], "MISMATCH")
        self.assertEqual(by[("gw.token_budget", "500,000")], "MISMATCH")        # documented number vs live 'auto'
        self.assertEqual(by[("env.VLLM_TQ_GQA_CUDA", "1")], "ok")
        self.assertEqual(by[("env.VLLM_CUSTOM_ALLREDUCE_MAX_SIZE_MB", "32")], "MISMATCH")   # absent in the engine
        self.assertEqual(by[("gw.local_budget", "14")], "ok")

    def test_engine_down_is_unresolved_not_mismatch(self):
        facts = {k: v for k, v in FACTS.items() if not k.startswith(("env.", "argv.", "flag."))}
        rows = [r for r in C.check_notes(facts) if r["fact"].startswith(("env.", "argv."))]
        self.assertTrue(rows and all(r["status"] in ("unresolved", "note-missing") for r in rows), rows)
        self.assertIn("unresolved", {r["status"] for r in rows})


class Dashboard(unittest.TestCase):
    def test_dead_fields_reads_inverted_default_hint_and_configured_vs_live(self):
        html = ('<select id=f_preset></select><input id=f_remote_key type=password>'
                '<span class=hint>0 = never, it queues (default) &middot; 1 = old</span>\n <input id=f_interactive_never_overflow>'
                '<span class=hint>tokens</span><input id=f_local_budget><input id=f_ghost>'
                '<script>cfg.local_budget; cfg.vanished_field;</script>')
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
            f.write(html)
        try:
            rows = C.check_dashboard(FACTS, f.name)
        finally:
            os.unlink(f.name)
        claims = {r["claim"].split(":")[0] for r in rows}
        self.assertIn("form field f_ghost", claims)
        self.assertIn("page reads cfg.vanished_field", claims)
        self.assertIn("hint for f_interactive_never_overflow", claims)        # says 0 is the default; code default is 1
        self.assertIn("KV pool row (shows configured SHIM_POOL_TOKENS)", claims)
        self.assertFalse(any("f_preset" in c or "f_remote_key" in c or "f_local_budget" in c for c in claims))


if __name__ == "__main__":
    unittest.main()
