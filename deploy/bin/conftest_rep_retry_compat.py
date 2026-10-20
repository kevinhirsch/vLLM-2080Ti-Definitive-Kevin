"""Runs test_rep_retry.py (fix 2) under the same tree conftest as the live suite.

The live suite's conftest (pytest_plugins = ["live_guard", ...]) does not provide fix 2's
`shim` / `patch_stream` fixtures, so this plugin supplies them from fix 2's own harness,
conftest_rep_retry.py. test_rep_retry.py imports its fakes (FakeRequest, FakeUpstream) from
conftest_rep_retry directly.

2026-10-20 (coordinator): it used to do `from conftest import ...`, and this plugin published
the fakes into the live conftest at pytest_configure. Plain unittest (the production-ledger
ratchet, test_no_production_ledger_touch.py) loads no pytest plugin, so the import failed
there and the ratchet failed. The fakes now come from the module that defines them, and the
publishing step is gone.
"""
import pytest

import conftest_rep_retry
from conftest_rep_retry import FakeRequest, FakeStreamResponse, FakeUpstream  # noqa: F401 (re-exported)


@pytest.fixture
def shim():
    return conftest_rep_retry._load_shim()


@pytest.fixture
def patch_stream(monkeypatch):
    """Same effect as fix 2's patch_stream: StreamResponse.prepare/write/write_eof are faked
    and the written bytes are recorded for the test to inspect."""
    import aiohttp.web
    rec = {"prepared": False, "written": [], "eof": False}

    async def fake_prepare(self, request):
        rec["prepared"] = True

    async def fake_write(self, data):
        rec["written"].append(bytes(data) if isinstance(data, (bytes, bytearray))
                              else str(data).encode())

    async def fake_write_eof(self, data=None):
        rec["eof"] = True

    monkeypatch.setattr(aiohttp.web.StreamResponse, "prepare", fake_prepare)
    monkeypatch.setattr(aiohttp.web.StreamResponse, "write", fake_write)
    monkeypatch.setattr(aiohttp.web.StreamResponse, "write_eof", fake_write_eof)
    return rec
