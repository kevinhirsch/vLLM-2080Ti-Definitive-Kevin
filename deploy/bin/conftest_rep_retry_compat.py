"""Loads test_rep_retry.py (fix 2) under the same tree conftest as the live suite.

test_rep_retry.py does `from conftest import FakeRequest, FakeUpstream`, which under the
live suite's conftest (pytest_plugins = ["live_guard"]) finds no such names. This module
supplies the fakes and the `shim` / `patch_stream` fixtures from fix 2's own harness,
unchanged; it just also installs the live's Fakes alongside them. Unchanged sources:
tests/test_rep_retry.py and tests/conftest_rep_retry.py (one line repointed to this repo's
keepalive-shim.py).
"""
import importlib.util
import sys

import pytest

conv = importlib.util.spec_from_file_location(
    "conftest_rep_retry", __file__.replace("conftest_rep_retry_compat.py", "conftest_rep_retry.py"))
conftest_rep_retry = importlib.util.module_from_spec(conv)
conv.loader.exec_module(conftest_rep_retry)

FakeRequest = conftest_rep_retry.FakeRequest
FakeUpstream = conftest_rep_retry.FakeUpstream
FakeStreamResponse = conftest_rep_retry.FakeStreamResponse

_spec = importlib.util.spec_from_file_location(
    "test_rep_retry", __file__.replace("conftest_rep_retry_compat.py", "test_rep_retry.py"))
_mod = importlib.util.module_from_spec(_spec)


def pytest_configure(config):
    """test_rep_retry.py imports its fakes as `from conftest import ...`, which under this
    tree's conftest (the live one) has no such names: publish them under that name, then
    load the (unchanged) suite, which binds them at import."""
    sys.modules["conftest"].FakeRequest = FakeRequest
    sys.modules["conftest"].FakeUpstream = FakeUpstream
    if not hasattr(_mod, "TOOL"):
        _spec.loader.exec_module(_mod)


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
