#!/usr/bin/env python3
"""Idempotently harden the existing private wg0 config without printing secrets.

wg-quick inherits `set -e`: a cleanup command that reports an already absent
route/rule can prevent the interface from starting. This script changes only
the known cleanup hooks, keeps an owner-only backup, and refuses an unfamiliar
PreUp stanza for human or Halo diagnosis.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import time

CONFIG = Path("/etc/wireguard/wg0.conf")
OLD = "ip route flush table 200 2>/dev/null; while iptables"
NEW = "ip route flush table 200 2>/dev/null || true; while iptables"


def hardened(data: bytes) -> bytes:
    text = data.decode("utf-8")
    if text.count("PreUp = ") != 1 or text.count("PostDown = ") < 1:
        raise ValueError("unexpected wg0 cleanup hook layout")
    if OLD in text:
        if text.count(OLD) != 1:
            raise ValueError("ambiguous route cleanup hook")
        text = text.replace(OLD, NEW)
    elif text.count(NEW) != 1:
        raise ValueError("unknown route cleanup hook")
    lines = []
    for line in text.splitlines(keepends=True):
        if line.startswith("PostDown = ") and not line.rstrip().endswith("|| true"):
            end = "\n" if line.endswith("\n") else ""
            line = line.rstrip("\n") + " || true" + end
        lines.append(line)
    return "".join(lines).encode("utf-8")


def apply(path: Path = CONFIG) -> bool:
    prior = path.read_bytes()
    result = hardened(prior)
    if result == prior:
        return False
    stat = path.stat()
    backup = path.with_name(path.name + f".before-hook-guard-{int(time.time())}")
    fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(prior)
        out.flush()
        os.fsync(out.fileno())
    os.chown(backup, stat.st_uid, stat.st_gid)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".wg0-hook-", delete=False) as out:
        temp = Path(out.name)
        out.write(result)
        out.flush()
        os.fsync(out.fileno())
    try:
        os.chmod(temp, stat.st_mode & 0o777)
        os.chown(temp, stat.st_uid, stat.st_gid)
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        if os.geteuid() != 0:
            raise PermissionError("root is required to edit wg0.conf")
        print("updated" if apply() else "already-current")
        return 0
    print("already-current" if hardened(CONFIG.read_bytes()) == CONFIG.read_bytes() else "needs-update")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
