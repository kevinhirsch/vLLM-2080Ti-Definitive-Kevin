# Every deploy/bin test runs under live_guard: no write or exec under the live engine/gateway dirs (RL 2026-10-03).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
pytest_plugins = ["live_guard"]


# BS3 (found 2026-10-04): test_gateway_spend_authority / test_gateway_spend_v6 drive the router with a 100K-token request, and the gateway's
# flight recorder (_flightrec_dir(), default = the LIVE ~/.local/share/vllm-qwen27b/flightrec) wrote *_100000tok.json test files into the live
# 40-file ring -- every test run evicted real Xid/empty-answer repro payloads. The directory is an env knob; point it at a per-session tmp dir
# unless the test (or a human) already chose one.
import tempfile  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _flightrec_in_tmp():
    if os.environ.get("SHIM_FLIGHTREC_DIR"):
        yield
        return
    with tempfile.TemporaryDirectory(prefix="flightrec-test-") as d:
        os.environ["SHIM_FLIGHTREC_DIR"] = d
        try:
            yield
        finally:
            os.environ.pop("SHIM_FLIGHTREC_DIR", None)
