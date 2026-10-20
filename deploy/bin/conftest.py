# Every deploy/bin test runs under live_guard: no write or exec under the live engine/gateway dirs (RL 2026-10-03).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tempfile
# conftest_rep_retry_compat: the repetition-retry suite (test_rep_retry.py, from lane gw-rep-retry via gw-combined,
# 2026-10-20) imports its fakes as `from conftest import ...`; the plugin supplies them alongside live_guard.
pytest_plugins = ["live_guard", "conftest_rep_retry_compat"]
# The strict-tool-history suite drives the real remote path, whose flight recorder defaults to the LIVE engine tree;
# SHIM_FLIGHTREC_DIR (the shim's own override) keeps those files in a temp dir so live_guard stays clean.
if "SHIM_FLIGHTREC_DIR" not in os.environ:
    import atexit
    import shutil
    os.environ["SHIM_FLIGHTREC_DIR"] = tempfile.mkdtemp(prefix="deploy-bin-tests-flightrec-")
    # test_no_production_ledger_touch requires every temp dir a run creates to be cleaned up
    atexit.register(shutil.rmtree, os.environ["SHIM_FLIGHTREC_DIR"], True)
