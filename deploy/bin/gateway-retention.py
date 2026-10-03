#!/usr/bin/env python3
"""Bounded retention for the gateway/engine runtime state on HNET00 (Lane RT, 2026-10-03).

Why this exists: the root disk ran at 86-88% and the estate disk guard tripped. The gateway
runtime dir (~/.local/share/vllm-qwen27b) and the engine's compile cache have several
append-only or one-file-per-event stores. The shim prunes only its own request logs, by age:
30 days, plus a 200 MB/day cap. Nothing else had a bound:

  telemetry/requests-*.jsonl, hw-*.jsonl  bounded by age only. The worst case is 30 x 200 MB = 6 GB.
  incidents/fault-*/, xid-*/              one bundle per engine death or Xid burst, mostly copied
                                          flightrec prompt bodies (uncompressed JSON). No bound.
  <file>.bak-<epoch>-<pid>                one backup per governed publish (gateway_safe_publish.py).
  <file>.bak-bundle-<ts>                  one per bundle install. No bound.
  watchdog.log and other *.log / *.out    appended forever; vllm-watchdog.sh says "never truncated".
  ~/.cache/vllm/torch_compile_cache       one dir per compile-config hash. No bound.

What it does. Every policy has an explicit byte or count budget, so disk use is bounded by
construction and not by someone noticing:

  * ARCHIVE BEFORE DELETE. Data with value (request logs, logs, publish backups) is gzipped into
    <root>/archive/ before it leaves its live path. The archive has its own budget, and the oldest
    archive goes first. Fault evidence is never deleted: bundle dirs, META files, the ledger and
    the engine journals stay. Over budget, only the packed flightrec prompt copies of the oldest
    bundles are dropped.
  * Live paths that readers glob never change shape. requests-*.jsonl stays plain JSONL while it
    is live, because spend_reconcile.py, engine-fault-collector.py and the estate vitals read it
    with plain open(). An incident bundle keeps its dir and META.txt, which incident_timeline.py
    globs.
  * DRY RUN BY DEFAULT. Nothing changes without --apply. Every run writes a report to
    <root>/retention-last.json, so the estate can read the bound as a fact.
  * Single instance (flock). Never follows symlinks. Never touches a path outside its configured
    roots. Kill switch: a file named RETENTION_DISABLED in <root>, or GATEWAY_RETENTION_DISABLED=1.

Usage: gateway-retention.py [--apply] [--json] [--root DIR] [--cache-root DIR]
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import os
import re
import shutil
import sys
import tarfile
import time

MB = 1024 * 1024
DAY = 86400

DEFAULT_ROOT = os.path.expanduser("~/.local/share/vllm-qwen27b")
DEFAULT_CACHE_ROOT = os.path.expanduser("~/.cache/vllm/torch_compile_cache")

# Budgets. Every one can be overridden by env, so a lane can test without editing the file.
def _env_num(name, default):
    try:
        return type(default)(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

TELEMETRY_LIVE_BUDGET_MB   = _env_num("RETENTION_TELEMETRY_LIVE_MB", 1536)
TELEMETRY_PROTECT_DAYS     = _env_num("RETENTION_TELEMETRY_PROTECT_DAYS", 3)
TELEMETRY_ARCHIVE_MB       = _env_num("RETENTION_TELEMETRY_ARCHIVE_MB", 512)
INCIDENT_PACK_AFTER_DAYS   = _env_num("RETENTION_INCIDENT_PACK_DAYS", 7)
INCIDENT_GZIP_MIN_BYTES    = _env_num("RETENTION_INCIDENT_GZIP_MIN_BYTES", 256 * 1024)
INCIDENT_BUDGET_MB         = _env_num("RETENTION_INCIDENT_BUDGET_MB", 1024)
BACKUP_KEEP_PER_FILE       = _env_num("RETENTION_BACKUP_KEEP", 10)
BACKUP_ARCHIVE_MB          = _env_num("RETENTION_BACKUP_ARCHIVE_MB", 256)
LOG_ROTATE_MB              = _env_num("RETENTION_LOG_ROTATE_MB", 16)
LOG_KEEP_ROTATIONS         = _env_num("RETENTION_LOG_KEEP", 4)
LOG_ARCHIVE_MB             = _env_num("RETENTION_LOG_ARCHIVE_MB", 256)
CACHE_IDLE_DAYS            = _env_num("RETENTION_CACHE_IDLE_DAYS", 21)
CACHE_KEEP_NEWEST          = _env_num("RETENTION_CACHE_KEEP", 16)
CACHE_BUDGET_MB            = _env_num("RETENTION_CACHE_BUDGET_MB", 12288)

TELEMETRY_RE = re.compile(r"^(requests|hw)-(\d{8})\.jsonl$")
INCIDENT_RE  = re.compile(r"^(fault|xid)-(\d{8})-(\d{6})$")
# Machine-generated backups only. Hand-named ones (bak-pre-*, BANKED-*, .before-*) are Kevin's
# or a lane's deliberate snapshots, so they are never counted or moved.
BACKUP_RES = (re.compile(r"^(?P<base>.+)\.bak-(?P<ts>\d{10})-\d+$"),
              re.compile(r"^(?P<base>.+)\.bak-bundle-(?P<ts>\d{8}-\d{6})$"))
LOG_GLOBS = ("", "watchdog")            # dirs (relative to root) whose *.log / *.out are rotated
LOG_SUFFIXES = (".log", ".out")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _is_plain_file(path):
    try:
        st = os.lstat(path)
    except OSError:
        return False
    import stat as _s
    return _s.S_ISREG(st.st_mode)


def _is_plain_dir(path):
    try:
        st = os.lstat(path)
    except OSError:
        return False
    import stat as _s
    return _s.S_ISDIR(st.st_mode)


def _tree_bytes(path):
    if _is_plain_file(path):
        return os.lstat(path).st_size
    total = 0
    for dp, dns, fns in os.walk(path, followlinks=False):
        for f in fns:
            try:
                total += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return total


def _inside(path, root):
    rp, rr = os.path.realpath(path), os.path.realpath(root)
    return rp == rr or rp.startswith(rr + os.sep)


def _gzip_to(src, dst):
    """Copy src to dst.gz-style file atomically (tmp + rename). Returns compressed size."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    with open(src, "rb") as fi, gzip.open(tmp, "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    # verify it decompresses fully before anything is removed
    with gzip.open(tmp, "rb") as fv:
        while fv.read(1 << 20):
            pass
    os.replace(tmp, dst)
    try:
        os.chmod(dst, os.lstat(src).st_mode & 0o777)
    except OSError:
        pass
    return os.lstat(dst).st_size


class Run:
    def __init__(self, apply):
        self.apply = apply
        self.policies = []

    def policy(self, name, root):
        p = {"name": name, "root": root, "bytes_before": 0, "bytes_after": 0,
             "freed_bytes": 0, "actions": [], "errors": []}
        self.policies.append(p)
        return p

    def act(self, p, kind, path, freed=0, **extra):
        row = {"kind": kind, "path": path, "freed_bytes": int(freed)}
        row.update(extra)
        p["actions"].append(row)
        p["freed_bytes"] += int(freed)


def _enforce_archive_budget(run, p, adir, budget_mb, pattern=None):
    """Drop the oldest archive files until adir is within budget_mb."""
    if not _is_plain_dir(adir):
        return
    files = []
    for dp, _d, fns in os.walk(adir, followlinks=False):
        for f in fns:
            fp = os.path.join(dp, f)
            if f.endswith(".tmp") or not _is_plain_file(fp):
                continue
            if pattern and not pattern.search(f):
                continue
            st = os.lstat(fp)
            files.append((st.st_mtime, st.st_size, fp))
    files.sort()
    total = sum(s for _m, s, _f in files)
    for _m, size, fp in files:
        if total <= budget_mb * MB:
            break
        if run.apply:
            os.unlink(fp)
        run.act(p, "archive-expire", fp, size)
        total -= size


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------
def telemetry_policy(run, root, now):
    tdir = os.path.join(root, "telemetry")
    adir = os.path.join(root, "archive", "telemetry")
    p = run.policy("telemetry-request-logs", tdir)
    if not _is_plain_dir(tdir):
        return p
    rows = []
    for f in os.listdir(tdir):
        m = TELEMETRY_RE.match(f)
        fp = os.path.join(tdir, f)
        if m and _is_plain_file(fp):
            rows.append((m.group(2), f, fp, os.lstat(fp).st_size))
    rows.sort()
    total = sum(r[3] for r in rows)
    p["bytes_before"] = total
    protect_from = time.strftime("%Y%m%d", time.gmtime(now - max(0, TELEMETRY_PROTECT_DAYS - 1) * DAY))   # today + N-1 previous UTC days
    for day, f, fp, size in rows:
        if total <= TELEMETRY_LIVE_BUDGET_MB * MB:
            break
        if day >= protect_from:
            p["errors"].append(f"over budget but only protected days remain ({total // MB} MB)")
            break
        dst = os.path.join(adir, f + ".gz")
        try:
            gz = _gzip_to(fp, dst) if run.apply else 0
            if run.apply:
                os.unlink(fp)
            run.act(p, "archive-gzip", fp, size - gz, archive=dst)
            total -= size
        except OSError as e:
            p["errors"].append(f"{f}: {e}")
    _enforce_archive_budget(run, p, adir, TELEMETRY_ARCHIVE_MB)
    p["bytes_after"] = total
    return p


def incidents_policy(run, root, now):
    idir = os.path.join(root, "incidents")
    p = run.policy("incident-bundles", idir)
    if not _is_plain_dir(idir):
        return p
    p["bytes_before"] = _tree_bytes(idir)
    bundles = []
    for f in sorted(os.listdir(idir)):
        m = INCIDENT_RE.match(f)
        bp = os.path.join(idir, f)
        if not (m and _is_plain_dir(bp)):
            continue
        try:
            t = time.mktime(time.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S"))
        except ValueError:
            continue
        bundles.append((t, bp))
    bundles.sort()
    for t, bp in bundles:
        if now - t < INCIDENT_PACK_AFTER_DAYS * DAY:
            continue
        # (1) flightrec/ -> flightrec.tar.gz (prompt copies, the bulk of every bundle)
        fr = os.path.join(bp, "flightrec")
        if _is_plain_dir(fr):
            before = _tree_bytes(fr)
            dst = os.path.join(bp, "flightrec.tar.gz")
            try:
                if run.apply:
                    tmp = dst + ".tmp"
                    with tarfile.open(tmp, "w:gz") as tf:
                        tf.add(fr, arcname="flightrec")
                    with tarfile.open(tmp, "r:gz") as tv:   # verify every member reads back
                        for mem in tv.getmembers():
                            if mem.isfile():
                                tv.extractfile(mem).read()
                    os.replace(tmp, dst)
                    os.chmod(dst, 0o600)
                    shutil.rmtree(fr)
                    after = os.lstat(dst).st_size
                else:
                    after = before // 6
                run.act(p, "pack-flightrec", fr, before - after, archive=dst)
            except (OSError, tarfile.TarError) as e:
                p["errors"].append(f"{fr}: {e}")
        # (2) gzip big plain files (engine journals); META*.txt and small json stay readable
        for f in sorted(os.listdir(bp)):
            fp = os.path.join(bp, f)
            if (not _is_plain_file(fp) or f.endswith(".gz") or f.upper().startswith("META")
                    or os.lstat(fp).st_size < INCIDENT_GZIP_MIN_BYTES):
                continue
            size = os.lstat(fp).st_size
            try:
                gz = _gzip_to(fp, fp + ".gz") if run.apply else size // 6
                if run.apply:
                    os.unlink(fp)
                run.act(p, "gzip", fp, size - gz, archive=fp + ".gz")
            except OSError as e:
                p["errors"].append(f"{fp}: {e}")
    # (3) budget: drop packed prompt copies of the OLDEST bundles only, never META/journal/ledger
    total = _tree_bytes(idir) if run.apply else p["bytes_before"] - p["freed_bytes"]
    for _t, bp in bundles:
        if total <= INCIDENT_BUDGET_MB * MB:
            break
        dst = os.path.join(bp, "flightrec.tar.gz")
        if _is_plain_file(dst):
            size = os.lstat(dst).st_size
            if run.apply:
                os.unlink(dst)
            run.act(p, "expire-flightrec-copy", dst, size)
            total -= size
    if total > INCIDENT_BUDGET_MB * MB:
        p["errors"].append(f"incidents still {total // MB} MB > {INCIDENT_BUDGET_MB} MB after expiring "
                           "prompt copies; the remaining evidence is never auto-deleted")
    p["bytes_after"] = total
    return p


def backups_policy(run, root, now):
    p = run.policy("runtime-backups", root)
    adir = os.path.join(root, "archive", "backups")
    groups = {}
    for sub in ("", "watchdog"):
        d = os.path.join(root, sub)
        if not _is_plain_dir(d):
            continue
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            if not _is_plain_file(fp):
                continue
            for rx in BACKUP_RES:
                m = rx.match(f)
                if m:
                    groups.setdefault((d, m.group("base")), []).append((os.lstat(fp).st_mtime, fp))
                    break
    stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(now))
    victims = []
    for (_d, base), rows in sorted(groups.items()):
        rows.sort(reverse=True)
        p["bytes_before"] += sum(os.lstat(fp).st_size for _m, fp in rows)
        victims.extend(fp for _m, fp in rows[BACKUP_KEEP_PER_FILE:])
    if victims:
        size = sum(os.lstat(fp).st_size for fp in victims)
        dst = os.path.join(adir, f"backups-{stamp}.tar.gz")
        try:
            if run.apply:
                os.makedirs(adir, exist_ok=True)
                tmp = dst + ".tmp"
                with tarfile.open(tmp, "w:gz") as tf:
                    for fp in victims:
                        tf.add(fp, arcname=os.path.relpath(fp, root))
                with tarfile.open(tmp, "r:gz") as tv:
                    names = set(tv.getnames())
                missing = [fp for fp in victims if os.path.relpath(fp, root) not in names]
                if missing:
                    raise OSError(f"archive missing {len(missing)} members")
                os.replace(tmp, dst)
                os.chmod(dst, 0o600)
                for fp in victims:
                    os.unlink(fp)
                gz = os.lstat(dst).st_size
            else:
                gz = size // 4
            run.act(p, "archive-backups", dst, size - gz, files=len(victims))
        except (OSError, tarfile.TarError) as e:
            p["errors"].append(f"backups: {e}")
    _enforce_archive_budget(run, p, adir, BACKUP_ARCHIVE_MB)
    p["bytes_after"] = p["bytes_before"] - (p["freed_bytes"] if victims else 0)
    return p


def logs_policy(run, root, now):
    p = run.policy("runtime-logs", root)
    adir = os.path.join(root, "archive", "logs")
    stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(now))
    for sub in LOG_GLOBS:
        d = os.path.join(root, sub)
        if not _is_plain_dir(d):
            continue
        for f in sorted(os.listdir(d)):
            fp = os.path.join(d, f)
            if not (f.endswith(LOG_SUFFIXES) and _is_plain_file(fp)):
                continue
            size = os.lstat(fp).st_size
            p["bytes_before"] += size
            if size <= LOG_ROTATE_MB * MB:
                continue
            tag = (sub.replace(os.sep, "_") + "_" if sub else "") + f
            dst = os.path.join(adir, f"{tag}.{stamp}.gz")
            try:
                if run.apply:
                    # copytruncate: writers keep their O_APPEND fd. Lines appended between the copy
                    # and the truncate (microseconds) are the only loss, and that is documented.
                    gz = _gzip_to(fp, dst)
                    with open(fp, "r+b") as fh:
                        fh.truncate(0)
                else:
                    gz = size // 8
                run.act(p, "rotate-log", fp, size - gz, archive=dst)
            except OSError as e:
                p["errors"].append(f"{fp}: {e}")
            # keep only the newest LOG_KEEP_ROTATIONS archives of this log
            if _is_plain_dir(adir):
                olds = sorted((g for g in os.listdir(adir) if g.startswith(tag + ".") and g.endswith(".gz")),
                              reverse=True)
                for g in olds[LOG_KEEP_ROTATIONS:]:
                    gp = os.path.join(adir, g)
                    gsz = os.lstat(gp).st_size
                    if run.apply:
                        os.unlink(gp)
                    run.act(p, "archive-expire", gp, gsz)
    _enforce_archive_budget(run, p, adir, LOG_ARCHIVE_MB)
    p["bytes_after"] = p["bytes_before"] - sum(a["freed_bytes"] for a in p["actions"] if a["kind"] == "rotate-log")
    return p


def cache_policy(run, cache_root, now):
    """torch compile cache: one dir per config hash, regenerable (costs one recompile at the next
    boot of that config). Recency = newest atime/mtime of anything inside, so a config that is
    booted daily is never idle even if its files were written weeks ago."""
    p = run.policy("torch-compile-cache", cache_root)
    if not _is_plain_dir(cache_root):
        return p
    entries = []
    for parent in (cache_root, os.path.join(cache_root, "torch_aot_compile")):
        if not _is_plain_dir(parent):
            continue
        for f in os.listdir(parent):
            ep = os.path.join(parent, f)
            if f == "torch_aot_compile" or not _is_plain_dir(ep):
                continue
            newest, size = 0.0, 0
            for dp, _d, fns in os.walk(ep, followlinks=False):
                for g in fns:
                    try:
                        st = os.lstat(os.path.join(dp, g))
                    except OSError:
                        continue
                    newest = max(newest, st.st_atime, st.st_mtime)
                    size += st.st_size
            entries.append((newest or os.lstat(ep).st_mtime, size, ep))
    entries.sort(reverse=True)
    total = sum(s for _n, s, _e in entries)
    p["bytes_before"] = total
    for i, (newest, size, ep) in enumerate(entries):
        if i < CACHE_KEEP_NEWEST:
            continue
        idle = now - newest > CACHE_IDLE_DAYS * DAY
        if idle or total > CACHE_BUDGET_MB * MB:
            if run.apply:
                shutil.rmtree(ep)
            run.act(p, "drop-cache-entry", ep, size, idle_days=round((now - newest) / DAY, 1))
            total -= size
    if total > CACHE_BUDGET_MB * MB:
        p["errors"].append(f"cache {total // MB} MB > budget but only the {CACHE_KEEP_NEWEST} newest entries remain")
    p["bytes_after"] = total
    return p


# ---------------------------------------------------------------------------
def run_all(root, cache_root, apply, now=None):
    now = time.time() if now is None else now
    run = Run(apply)
    for fn, r in ((telemetry_policy, root), (incidents_policy, root), (backups_policy, root),
                  (logs_policy, root), (cache_policy, cache_root)):
        try:
            fn(run, r, now)
        except Exception as e:      # one broken policy never stops the others
            run.policy(fn.__name__, r)["errors"].append(f"policy crashed: {e!r}")
    return {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)), "applied": apply,
            "root": root, "cache_root": cache_root,
            "freed_bytes": sum(p["freed_bytes"] for p in run.policies),
            "bytes_after": sum(p["bytes_after"] for p in run.policies),
            "errors": [e for p in run.policies for e in p["errors"]],
            "policies": run.policies}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="make changes (default: dry run)")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    ap.add_argument("--root", default=os.environ.get("GATEWAY_RETENTION_ROOT", DEFAULT_ROOT))
    ap.add_argument("--cache-root", default=os.environ.get("GATEWAY_RETENTION_CACHE_ROOT", DEFAULT_CACHE_ROOT))
    a = ap.parse_args(argv)
    if os.environ.get("GATEWAY_RETENTION_DISABLED") == "1" or os.path.exists(os.path.join(a.root, "RETENTION_DISABLED")):
        print("gateway-retention: disabled by kill switch")
        return 2
    os.makedirs(a.root, exist_ok=True)
    with open(os.path.join(a.root, ".retention.lock"), "a+") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("gateway-retention: another run holds the lock")
            return 3
        rep = run_all(a.root, a.cache_root, a.apply)
        out = os.path.join(a.root, "retention-last.json" if a.apply else "retention-dryrun.json")
        tmp = out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(rep, fh, indent=1)
        os.replace(tmp, out)
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        for p in rep["policies"]:
            print(f"{p['name']:24s} before={p['bytes_before'] / MB:8.1f}MB after={p['bytes_after'] / MB:8.1f}MB "
                  f"freed={p['freed_bytes'] / MB:7.1f}MB actions={len(p['actions'])} errors={len(p['errors'])}")
        print(f"{'TOTAL':24s} freed={rep['freed_bytes'] / MB:.1f}MB applied={rep['applied']}")
        for e in rep["errors"]:
            print("  !", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
