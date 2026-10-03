# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L52: a SIGTERM'd server must exit well inside the service manager's stop timeout (30 s).

Measured on the 2x2080Ti box: ~10 of 45 stops ended in SIGKILL because uvicorn sat in "Waiting for
connections to close" for streams whose engine had already been shut down (workers were TERMed by
systemd at the same instant), and because workers could outlive a swallowed SystemExit / a hung
NCCL teardown. These tests need no GPU and no model.
"""

import inspect
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time

import pytest

from vllm.utils import shutdown_deadline
from vllm.utils.network_utils import get_open_port

SERVER = textwrap.dedent(
    """
    import asyncio, sys
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from vllm.entrypoints.launchers.launcher import serve_http

    class _VC:
        shutdown_timeout = 0

    class _Engine:
        vllm_config = _VC()
        errored = False
        is_running = True
        def shutdown(self, timeout=None):
            pass

    app = FastAPI()
    app.state.engine_client = _Engine()

    @app.get("/stream")
    async def stream():
        async def gen():
            yield b"first-chunk\\n"
            await asyncio.sleep(3600)   # the engine is gone: this stream can never finish
        return StreamingResponse(gen())

    async def main():
        t = await serve_http(app, None, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
        await t

    asyncio.run(main())
    """
)


def _start_server(port, env_extra):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **env_extra}
    proc = subprocess.Popen(
        [sys.executable, "-c", SERVER, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            return proc
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError("server died during startup")
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("server did not start")


def _open_stream(port):
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(b"GET /stream HTTP/1.1\r\nHost: x\r\n\r\n")
    got = b""
    deadline = time.time() + 5
    while b"first-chunk" not in got and time.time() < deadline:
        got += s.recv(4096)
    assert b"first-chunk" in got
    return s  # kept open: an in-flight request


def _run_stop(env_extra, wait_s):
    port = get_open_port()
    proc = _start_server(port, env_extra)
    conn = _open_stream(port)
    try:
        t0 = time.time()
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=wait_s)
            return proc.returncode, time.time() - t0
        except subprocess.TimeoutExpired:
            return None, time.time() - t0
    finally:
        conn.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_open_stream_no_longer_holds_the_server_past_the_grace():
    rc, took = _run_stop({"VLLM_HTTP_GRACEFUL_SHUTDOWN_S": "2"}, wait_s=15)
    assert rc is not None, "server still alive 15 s after SIGTERM with an open stream"
    assert took < 10, took


def test_control_without_the_grace_the_old_hang_reproduces():
    """VLLM_HTTP_GRACEFUL_SHUTDOWN_S=0 is the old behaviour: proves the test above detects the bug."""
    rc, took = _run_stop(
        {"VLLM_HTTP_GRACEFUL_SHUTDOWN_S": "0", "VLLM_API_SERVER_EXIT_DEADLINE_S": "0"},
        wait_s=8,
    )
    assert rc is None, f"expected the old unbounded wait, exited rc={rc} after {took:.1f}s"


def test_api_exit_deadline_is_the_last_resort_even_without_the_http_grace():
    rc, took = _run_stop(
        {"VLLM_HTTP_GRACEFUL_SHUTDOWN_S": "0", "VLLM_API_SERVER_EXIT_DEADLINE_S": "3"},
        wait_s=15,
    )
    assert rc == shutdown_deadline.EXIT_CODE_DEADLINE, (rc, took)
    assert 2.5 < took < 12, took


def test_arm_exit_deadline_calls_exit_with_143():
    shutdown_deadline._reset_for_tests()
    calls = []
    assert shutdown_deadline.arm_exit_deadline("t-fire", 0.05, exit_fn=calls.append)
    deadline = time.time() + 5
    while not calls and time.time() < deadline:
        time.sleep(0.02)
    assert calls == [143]


def test_arm_exit_deadline_is_idempotent_and_disableable():
    shutdown_deadline._reset_for_tests()
    calls = []
    assert shutdown_deadline.arm_exit_deadline("t-once", 0.05, exit_fn=calls.append)
    assert not shutdown_deadline.arm_exit_deadline("t-once", 0.05, exit_fn=calls.append)
    assert not shutdown_deadline.arm_exit_deadline("t-off", 0, exit_fn=calls.append)
    assert not shutdown_deadline.arm_exit_deadline("t-off2", -1, exit_fn=calls.append)
    assert not shutdown_deadline.arm_exit_deadline("t-off3", None, exit_fn=calls.append)
    time.sleep(0.4)
    assert calls == [143]  # armed once, fired once


def test_a_process_that_exits_by_itself_is_not_delayed_by_the_timer():
    code = (
        "from vllm.utils.shutdown_deadline import arm_exit_deadline;"
        "arm_exit_deadline('x', 600); print('done')"
    )
    t0 = time.time()
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0 and "done" in out.stdout
    assert time.time() - t0 < 100  # a non-daemon 600 s timer would hang here


def test_a_hung_process_is_exited_by_its_deadline():
    code = (
        "import time;from vllm.utils.shutdown_deadline import arm_exit_deadline;"
        "arm_exit_deadline('hung', 1); time.sleep(300)"
    )
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", code], timeout=120, capture_output=True)
    assert p.returncode == 143
    assert time.time() - t0 < 100


def test_workers_and_api_server_are_wired_to_the_deadline():
    from vllm.entrypoints.launchers import launcher
    from vllm.v1.executor.multiproc_executor import WorkerProc

    assert "arm_exit_deadline(\"Worker\"" in inspect.getsource(WorkerProc.worker_main)
    assert "arm_exit_deadline(" in inspect.getsource(launcher.serve_http)
    assert "timeout_graceful_shutdown" in inspect.getsource(launcher.serve_http)


@pytest.mark.parametrize(
    "name,default",
    [
        ("VLLM_HTTP_GRACEFUL_SHUTDOWN_S", 5),
        ("VLLM_API_SERVER_EXIT_DEADLINE_S", 20),
        ("VLLM_WORKER_EXIT_DEADLINE_S", 10),
    ],
)
def test_defaults_sit_well_inside_a_30s_stop_timeout(name, default, monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) == default
    assert default < 30
