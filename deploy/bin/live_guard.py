"""live_guard -- a pytest plugin that makes a test run unable to touch the LIVE engine/gateway (lane RL, 2026-10-03).

Why: at 10:23:12 a K6 stubbed test reached the real ~/.local/share/vllm-qwen27b/engine-actuator.py ("restart --by K6"):
no engine restart, but a false restart-job.json "done" row, two fake estate events and a touched restart.lock. Earlier
(09:15) a suite run overwrote the live gateway_config_schema.py. CFG's ratchet covers six gateway files only.

Active for the whole pytest process (a sys.addaudithook cannot be removed), it RAISES PermissionError -- the operation
does not happen -- on:
  * any write under a live directory: open() for write/append/create/truncate, os.rename/replace, os.remove/unlink,
    os.mkdir/rmdir, os.chmod, os.utime, os.truncate, os.link/symlink, shutil copy/move/rmtree;
  * any subprocess / exec / spawn whose argv names a path under a live directory (running the deployed
    engine-actuator.py, gateway-offline.py, serve scripts, ...), any `sudo`, and `systemctl` (system manager, i.e.
    without --user) acting on the engine, gateway or watchdog units;
  * os.system / os.popen strings containing any of the above.
Reads are allowed. Live directories: ~/.local/share/vllm-qwen27b, ~/.local/share/vllm-releases (current / history),
~/.local/share/estate-overseer/events, ~/.local/share/lane-units, and anything in LIVE_GUARD_EXTRA (os.pathsep list).
Every child process also inherits PYTEST_CURRENT_TEST, which RL's tools (and the deployed actuator, once it carries
the same check) refuse mutating actions on.

Adopt:
  * deploy/bin: conftest.py loads it for every test.
  * a lane's tools: put this in the lane's conftest.py
        import sys; sys.path.insert(0, "/home/kevin/projects/lanes/windows"); pytest_plugins = ["live_guard"]
    or run `pytest -p live_guard` with that directory on PYTHONPATH.
  * a test that must legitimately exercise a live path redirects it to tmp first; there is no allow-list by design,
    except LIVE_GUARD_OFF=1 for a human debugging session (never in CI or the gate).
"""
from __future__ import annotations

import os
import re
import sys

HOME = os.path.expanduser("~")
LIVE_DIRS = [os.path.join(HOME, ".local/share/vllm-qwen27b"), os.path.join(HOME, ".local/share/vllm-releases"),
             os.path.join(HOME, ".local/share/estate-overseer/events"), os.path.join(HOME, ".local/share/lane-units")]
LIVE_DIRS += [p for p in os.environ.get("LIVE_GUARD_EXTRA", "").split(os.pathsep) if p]
LIVE_UNITS = re.compile(r"\b(vllm-qwen27b[\w.@-]*|vllm-keepalive-shim[\w.@-]*)\b")
WRITE_EVENTS = {"os.rename", "os.remove", "os.mkdir", "os.rmdir", "os.chmod", "os.chown", "os.utime", "os.truncate",
                "os.link", "os.symlink", "shutil.copyfile", "shutil.copymode", "shutil.copystat", "shutil.copytree",
                "shutil.move", "shutil.rmtree", "os.removexattr", "os.setxattr"}
EXEC_EVENTS = {"subprocess.Popen", "os.exec", "os.posix_spawn", "os.spawn", "os.system", "os.popen", "pty.spawn"}
_state = {"installed": False, "dirs": [], "violations": []}


class LiveTouch(PermissionError):
    pass


def _norm(p):
    try:
        return os.path.realpath(os.fsdecode(p)) if not isinstance(p, int) else None
    except (TypeError, ValueError):
        return None


def _under(path, dirs):
    if not path:
        return None
    for d in dirs:
        if path == d or path.startswith(d + os.sep):
            return d
    return None


def _open_writes(args):
    mode = args[1] if len(args) > 1 else "r"
    flags = args[2] if len(args) > 2 else 0
    if isinstance(mode, str) and any(c in mode for c in "wax+"):
        return True
    return isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))


def _argv_strings(args):
    out = []
    for a in args:
        if isinstance(a, (str, bytes, os.PathLike)):
            out.append(os.fsdecode(a))
        elif isinstance(a, (list, tuple)):
            out.extend(os.fsdecode(x) for x in a if isinstance(x, (str, bytes, os.PathLike)))
    return out


def exec_violation(argv, dirs):
    """Why this command line must not run from a test (None = fine)."""
    toks = []
    for s in argv:
        toks.extend(s.split())
    raw = [os.path.expanduser(t.strip("'\"")) for t in toks]
    for t in raw:
        if t.startswith(("/", "~", ".")) and _under(_norm(t), dirs):
            return f"runs a live path {t}"
    base = [os.path.basename(t) for t in raw]
    if "sudo" in base:
        return "runs sudo"
    if "systemctl" in base and "--user" not in raw and LIVE_UNITS.search(" ".join(raw)):
        return "runs systemctl on a live unit"
    return None


def _hook(event, args):
    if os.environ.get("LIVE_GUARD_OFF") == "1":
        return
    dirs = _state["dirs"]
    why = None
    if event == "open":
        if args and _open_writes(args):
            d = _under(_norm(args[0]), dirs)
            if d:
                why = f"writes {os.fsdecode(args[0])}"
    elif event in WRITE_EVENTS:
        for a in args:
            if isinstance(a, (str, bytes, os.PathLike)):
                d = _under(_norm(a), dirs)
                if d:
                    why = f"{event} on {os.fsdecode(a)}"
                    break
    elif event in EXEC_EVENTS:
        why = exec_violation(_argv_strings(args), dirs)
        if why:
            why = f"{event}: {why}"
    if why:
        msg = f"live_guard: a test {why} -- tests must never touch the live engine/gateway (redirect it to tmp)"
        _state["violations"].append(msg)
        raise LiveTouch(msg)


def install(dirs=None):
    _state["dirs"] = [os.path.realpath(d) for d in (dirs or LIVE_DIRS)]
    if not _state["installed"]:
        sys.addaudithook(_hook)
        _state["installed"] = True


def pytest_configure(config):
    install()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if _state["violations"]:
        terminalreporter.section("live_guard")
        for v in sorted(set(_state["violations"])):
            terminalreporter.write_line(v)


def pytest_sessionfinish(session, exitstatus):
    # a violation caught (and swallowed) inside product code still fails the run
    if _state["violations"] and session.exitstatus == 0:
        session.exitstatus = 1
