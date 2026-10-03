# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A hard backstop for process shutdown (lane FX, L52).

Why: ``systemctl stop`` sends SIGTERM to every process of the unit at once and SIGKILLs the whole
cgroup after ``TimeoutStopSec``. A process that handles SIGTERM "gracefully" can still hang for
longer than that -- uvicorn waiting forever for open streams whose engine is already gone, a worker
whose ``SystemExit`` was swallowed inside a weakref/``__del__`` callback, NCCL/CUDA teardown waiting
on a peer that has already exited. A SIGKILL at the timeout is not a clean exit (no atexit, no
``ExecStopPost`` ordering guarantees, an "unplanned" classification, a 30 s outage), so each process
arms its own deadline the moment it is told to stop: if it is still alive ``seconds`` later it
exits immediately with ``os._exit`` (the kernel then releases its CUDA context exactly as it would
after a SIGKILL, just earlier than systemd would do it).

The timer runs in a daemon thread, so a clean exit is never delayed by it.
"""

import os
import threading
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

_armed: dict[str, float] = {}
_lock = threading.Lock()

EXIT_CODE_DEADLINE = 143  # 128 + SIGTERM: what a TERM'd process reports


def arm_exit_deadline(
    name: str,
    seconds: float,
    *,
    exit_code: int = EXIT_CODE_DEADLINE,
    exit_fn=os._exit,
) -> bool:
    """Exit this process ``seconds`` from now unless it has exited by itself.

    Idempotent per ``name`` (a second SIGTERM must not push the deadline out). ``seconds <= 0``
    disables the backstop. Returns True only when this call armed the timer.
    """
    if seconds is None or seconds <= 0:
        return False
    with _lock:
        if name in _armed:
            return False
        _armed[name] = time.monotonic() + seconds

    def _fire() -> None:
        logger.error(
            "[shutdown] %s: still running %.0fs after the stop request; exiting now "
            "(a hung teardown would otherwise be SIGKILLed by the service manager)",
            name,
            seconds,
        )
        exit_fn(exit_code)

    t = threading.Timer(seconds, _fire)
    t.daemon = True
    t.name = f"ExitDeadline-{name}"
    t.start()
    return True


def _reset_for_tests() -> None:
    with _lock:
        _armed.clear()
