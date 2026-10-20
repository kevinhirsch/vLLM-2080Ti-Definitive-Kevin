"""Test harness for the GW-REP-RETRY change in keepalive-shim.py.

Loads the (mirror) gateway module once per session and provides small
fake doubles (request, transport, upstream, content) so the tests can
drive ``_relay`` with a mocked upstream -- no real engine, no network.
"""
import importlib.util
import sys
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = HERE   # the candidate/control copy under tests/ is the file under test
SHIM = os.path.join(SRC, "keepalive-shim.py")


_MOD = None


def _load_shim():
    # sys.dont_write_bytecode: the module name matches the live suite's, so a run that loads
    # both suites shares one module; no bytecode is cached next to the file under test.
    # Cached in _MOD: module_from_spec does not register the module, so a second load would
    # re-execute the gateway and discard any patch a test made on the first one.
    global _MOD
    if _MOD is not None:
        return _MOD
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("keepalive_shim", SHIM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MOD = mod
    return mod


@pytest.fixture(scope="session")
def shim():
    """The loaded gateway module (session-scoped: load once).

    The import is lazy (unchanged behaviour: same module, same path, loaded once). A load
    failure is reported per test instead of aborting collection, so on a gateway WITHOUT
    fix 2 each test fails on its own (the helpers are missing) rather than the run erroring."""
    global LOAD_ERROR
    try:
        return _load_shim()
    except Exception as e:          # noqa: BLE001 -- surfaced per test, never swallowed
        LOAD_ERROR = e
        pytest.fail(f"gateway module failed to load: {type(e).__name__}: {e}")


LOAD_ERROR = None


@pytest.fixture
def patch_stream(monkeypatch):
    """Patch aiohttp StreamResponse prepare/write/write_eof so the
    streaming tests can run without a real transport. Records what
    would be streamed to the client (the spliced retry, etc.)."""
    import aiohttp.web
    rec = {"prepared": False, "written": [], "eof": False}

    async def fake_prepare(self, request):
        rec["prepared"] = True
        return None

    async def fake_write(self, data):
        rec["written"].append(bytes(data) if isinstance(data, (bytes, bytearray))
                               else str(data).encode())

    async def fake_write_eof(self, data=None):
        rec["eof"] = True

    monkeypatch.setattr(aiohttp.web.StreamResponse, "prepare", fake_prepare)
    monkeypatch.setattr(aiohttp.web.StreamResponse, "write", fake_write)
    monkeypatch.setattr(aiohttp.web.StreamResponse, "write_eof", fake_write_eof)
    return rec


# --------------------------------------------------------------------------- #
# Fake doubles
# --------------------------------------------------------------------------- #
class FakeTransport:
    """A transport that is not closing (so _client_gone() is False)."""
    def is_closing(self):
        return False


class FakeRequest:
    """A minimal stand-in for the aiohttp client request.

    Supports attribute access (path, headers, remote, transport) and
    dict-style access (request["k"], request.get("k")) -- the two shapes
    _relay()/_prepare_stream_response() use.
    """
    def __init__(self, path="/v1/chat/completions", remote="127.0.0.1"):
        self.path = path
        self.headers = {}
        self.remote = remote
        self.transport = FakeTransport()
        self.method = "POST"
        self._data = {}

    def __getitem__(self, k):
        return self._data[k]

    def __setitem__(self, k, v):
        self._data[k] = v

    def get(self, k, d=None):
        return self._data.get(k, d)


class FakeContent:
    """A fake response body stream: iter_any() yields chunks, readany()
    pops one chunk (b"" when exhausted)."""
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def iter_any(self):
        for c in self._chunks:
            yield c

    async def readany(self):
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class FakeUpstream:
    """A fake upstream response.

    Non-streaming: ``read()`` returns ``body``.
    Streaming: ``content.iter_any()`` / ``content.readany()`` yield ``chunks``.
    """
    def __init__(self, status=200, body=b"", chunks=None, content_type="application/json"):
        self.status = status
        self.headers = {"Content-Type": content_type}
        self._body = body
        self._chunks = chunks if chunks is not None else ([body] if body else [])
        self._content = FakeContent(self._chunks)
        self.closed = False
        self.released = False

    async def read(self):
        return self._body

    def close(self):
        self.closed = True

    def release(self):
        self.released = True

    @property
    def content(self):
        return self._content


class FakeStreamResponse:
    """A fake aiohttp StreamResponse that records writes.

    Records prepare(), write() and write_eof() so the tests can verify
    what was streamed to the client (e.g. the spliced retry).
    """
    def __init__(self, status=200, headers=None):
        self.status = status
        self.headers = dict(headers or {})
        self.written = []
        self.prepared = False
        self.eof = False

    async def prepare(self, request):
        self.prepared = True
        return None

    async def write(self, data):
        self.written.append(bytes(data))

    async def write_eof(self, data=None):
        self.eof = True

    @property
    def body(self):
        return b"".join(self.written)
