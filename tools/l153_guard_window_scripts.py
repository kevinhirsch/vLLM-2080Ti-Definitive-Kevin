#!/usr/bin/env python3
"""L153 (BS3): move legacy engine-window shell scripts from "stop vllm-qwen27b-watchdog.timer ... start it again" to the liveness hold.

A script that stops the timer blinds the whole liveness authority (the timer is its clock) and leaves it stopped when the script dies
between the stop and the start (DFT 02:49-07:14, 4.5 h unwatched). deploy/bin/window_guard.sh is the one way a window owns the engine now.
This tool patches a script in place (idempotent):
  1. archives the original under <archive>/<relative path>.orig (never deleted);
  2. inserts, right after the shebang, `. GUARD; window_guard_enter LANE "reason" TTL -- "$@"`;
  3. replaces every `[sudo [-n]] systemctl stop|start vllm-qwen27b-watchdog.timer` with a no-op `: "..."` (the rest of the line, `;`, `&&`,
     redirects, is untouched, so the script stays syntactically valid and its own restore logic still runs);
  4. checks the result with `bash -n` and refuses to write a script that does not parse.

  l153_guard_window_scripts.py [--apply] [--archive DIR] [--guard FILE] [--ttl S] FILE...
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

TIMER_CMD = re.compile(r"(?:sudo\s+(?:-n\s+)?)?systemctl\s+(?:stop|start)\s+vllm-qwen27b-watchdog\.timer")
NOOP = ': "L153: watchdog timer untouched - window_guard holds the engine"'
GUARD_MARK = "window_guard_enter"
ENGINE_CTL = re.compile(r"systemctl\s+(?:stop|restart)\s+vllm-qwen27b\b(?!-)")


def patch_text(text: str, lane: str, reason: str, guard: str, ttl: int = 21600):
    """Returns (new_text, changed). Pure."""
    if GUARD_MARK in text:
        return text, False
    lines = text.split("\n")
    body = "\n".join(lines)
    # only the real command lines: a mention in a comment is left alone
    out = []
    for ln in lines:
        if ln.lstrip().startswith("#"):
            out.append(ln)
        else:
            out.append(TIMER_CMD.sub(NOOP, ln))
    insert = [f"# L153 (BS3): the engine is owned through a liveness hold (window_guard.sh); this script no longer stops the watchdog timer.",
              f'. "{guard}"',
              f'window_guard_enter {lane} "{reason}" {ttl} -- "$@"']
    at = 1 if out and out[0].startswith("#!") else 0
    out[at:at] = insert
    new = "\n".join(out)
    return new, new != body


def lane_of(path: str) -> str:
    parts = os.path.abspath(path).split(os.sep)
    for marker in ("lanes", "tools"):
        if marker in parts:
            i = parts.index(marker)
            if i + 1 < len(parts) - 1:
                return re.sub(r"[^A-Za-z0-9_-]+", "-", parts[i + 1])[:24]
    return "legacy"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--archive", default=os.path.expanduser("~/projects/lanes/_l153-archive"))
    ap.add_argument("--guard", default=os.path.expanduser("~/projects/lanes/windows/window_guard.sh"))
    ap.add_argument("--ttl", type=int, default=21600)
    a = ap.parse_args(argv)
    rc = 0
    for f in a.files:
        f = os.path.abspath(f)
        text = open(f).read()
        new, changed = patch_text(text, lane_of(f), f"legacy window {os.path.basename(f)} (L153 guard)", a.guard, a.ttl)
        if not changed:
            print(f"skip   {f} (already guarded or nothing to do)")
            continue
        p = subprocess.run(["bash", "-n", "/dev/stdin"], input=new, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"REFUSE {f}: patched script does not parse: {p.stderr.strip()[:200]}")
            rc = 1
            continue
        left = [n for n, ln in enumerate(new.split("\n"), 1) if not ln.lstrip().startswith("#") and TIMER_CMD.search(ln)]
        if left:
            print(f"REFUSE {f}: timer commands remain at lines {left}")
            rc = 1
            continue
        if not a.apply:
            print(f"would patch {f}")
            continue
        dst = os.path.join(a.archive, f.lstrip("/") + ".orig")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy2(f, dst)
        mode = os.stat(f).st_mode
        tmp = f + ".l153tmp"
        with open(tmp, "w") as fh:
            fh.write(new)
        os.chmod(tmp, mode)
        os.replace(tmp, f)
        print(f"patched {f}  (original: {dst})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
