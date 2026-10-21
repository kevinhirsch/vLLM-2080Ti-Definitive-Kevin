#!/usr/bin/env python3
"""live_drift_check -- does the LIVE ops layer still match git? (EB-103, 2026-10-21)

The gateway, the engine launchers and their env files run from ~/.local/share/vllm-qwen27b on the serving host,
copied there from this repo's deploy/ tree by install.sh. Edits made there in place leave git behind without anyone
noticing: in October 2026 the live gateway matched no branch for weeks, and production's launcher called a
gpu-boot-gate.sh that only an unmerged lane branch held. This check hashes every top-level live file and compares
it with the file install.sh would copy there from a git ref (default: the main line). Read-only: `git ls-tree`,
`git show`, and `git hash-object` without -w. It never copies, deploys or restarts anything.

Mapping (install.sh's):
  deploy/bin/<f>                -> <f>     every file except test_* and *.pyc (install.sh copies the rest)
  deploy/templates/<f>.jinja    -> <f>
  deploy/env/vllm-qwen27b.env   -> vllm-qwen27b.env
Host config is private ("code public, config private", Kevin 2026-10-21): only redacted examples are in git
(deploy/env/host-local/), and the allowlist marks the real files `local`.
Each live file is then:
  ok          identical to its source at the ref
  DRIFT       differs from its source                                  -> fails, unless the allowlist says `report`
  UNTRACKED   no source at the ref and not covered by the allowlist     -> fails, unless the allowlist says `report`
  host-local  the allowlist says `local` (private host config and secret carriers; never compared, never in git)
  ignored     the allowlist says `ignore` (runtime state, banked copies, logs, retired launchers)
Sources at the ref with no live copy are listed as `not deployed` (information only).

The allowlist is deploy/env/live-layer-allow.txt read FROM THE REF, so it is reviewed like code: lines
`ignore <glob>`, `local <glob>` or `report <glob>` (fnmatch on the file name), `#` comments.
Exit: 0 clean, 1 drift or untracked files, 2 usage or git error.  --json prints the full result.
Usage: live_drift_check.py [--repo DIR] [--ref REF] [--live DIR] [--allow-file FILE] [--json]
"""
import argparse
import fnmatch
import json
import os
import subprocess
import sys

DEFAULT_REPO = os.path.expanduser("~/Desktop/vLLM-2080Ti-Definitive")
DEFAULT_REF = "frontier-pastnative-20260816"
DEFAULT_LIVE = os.path.expanduser("~/.local/share/vllm-qwen27b")
ALLOW_PATH = "deploy/env/live-layer-allow.txt"


class GitError(RuntimeError):
    pass


def git(repo, *args):
    p = subprocess.run(["git", "--no-optional-locks", "-C", repo, *args], capture_output=True, text=True,
                       errors="replace")
    if p.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout


def sources_at(repo, ref):
    """{live file name: (repo path, blob)} by install.sh's mapping."""
    out = {}
    for line in git(repo, "ls-tree", "-r", "--full-tree", ref, "--", "deploy").splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, blob = meta.split()
        if kind != "blob":
            continue
        parts = path.split("/")
        name = parts[-1]
        if len(parts) == 3 and parts[1] == "bin":
            ok = not name.startswith("test_") and not name.endswith(".pyc")
        elif len(parts) == 3 and parts[1] == "templates":
            ok = name.endswith(".jinja")
        elif path == "deploy/env/vllm-qwen27b.env":
            ok = True
        else:
            ok = False
        if not ok:
            continue
        if name in out:
            raise GitError(f"two sources install to the same live name {name}: {out[name][0]} and {path}")
        out[name] = (path, blob)
    return out


def parse_allow(text):
    rules = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        verb, _, pattern = line.partition(" ")
        if verb not in ("ignore", "local", "report") or not pattern.strip():
            raise ValueError(f"bad allowlist line: {raw!r}")
        rules.append((verb, pattern.strip()))
    return rules


def verdict(rules, name):
    for verb, pattern in rules:
        if fnmatch.fnmatchcase(name, pattern):
            return verb
    return None


def check(repo, ref, live, allow_text=None):
    src = sources_at(repo, ref)
    if allow_text is None:
        try:
            allow_text = git(repo, "show", f"{ref}:{ALLOW_PATH}")
        except GitError:
            allow_text = ""
    rules = parse_allow(allow_text)
    res = {"repo": repo, "ref": ref, "commit": git(repo, "rev-parse", "--short=10", ref).strip(), "live": live,
           "ok": [], "drift": [], "drift_reported": [], "untracked": [], "untracked_reported": [], "host_local": [],
           "ignored": [], "not_deployed": []}
    seen = set()
    for name in sorted(os.listdir(live)):
        path = os.path.join(live, name)
        if not os.path.isfile(path) or os.path.islink(path):
            continue
        v = verdict(rules, name)
        if v in ("ignore", "local"):
            res["ignored" if v == "ignore" else "host_local"].append(name)
            continue
        if name in src:
            seen.add(name)
            h = git(repo, "hash-object", "--no-filters", path).strip()
            if h == src[name][1]:
                res["ok"].append(name)
            else:
                res["drift_reported" if v == "report" else "drift"].append({"name": name, "source": src[name][0]})
        else:
            res["untracked_reported" if v == "report" else "untracked"].append(name)
    res["not_deployed"] = sorted(n for n in src if n not in seen and not os.path.exists(os.path.join(live, n)))
    res["clean"] = not res["drift"] and not res["untracked"]
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--live", default=DEFAULT_LIVE)
    ap.add_argument("--allow-file", help="use this allowlist instead of the one at the ref (testing a change)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        allow = open(a.allow_file).read() if a.allow_file else None
        r = check(a.repo, a.ref, a.live, allow)
    except (GitError, ValueError, OSError) as e:
        print(f"live_drift_check: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(r, indent=1))
    else:
        print(f"live {r['live']} vs {r['ref']} ({r['commit']}): {len(r['ok'])} identical, {len(r['drift'])} drift, "
              f"{len(r['untracked'])} untracked, {len(r['drift_reported']) + len(r['untracked_reported'])} reported, "
              f"{len(r['host_local'])} host-local, {len(r['ignored'])} ignored, {len(r['not_deployed'])} not deployed")
        for d in r["drift"]:
            print(f"  DRIFT      {d['name']}  (source {d['source']})")
        for n in r["untracked"]:
            print(f"  UNTRACKED  {n}")
        for d in r["drift_reported"]:
            print(f"  reported   {d['name']} differs from {d['source']} (allowlist: report)")
        for n in r["untracked_reported"]:
            print(f"  reported   {n} has no source in git (allowlist: report)")
    return 0 if r["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
