"""Lane RT 2026-10-03: every gateway runtime store has a byte or count bound, and data with value is archived before it leaves."""
import gzip
import importlib.util
import os
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("gateway_retention_test", Path(__file__).with_name("gateway-retention.py"))
ret = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ret)

MB = 1024 * 1024
NOW = time.mktime(time.strptime("20261003120000", "%Y%m%d%H%M%S"))


def _write(path, data, mtime=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _day(n):
    return time.strftime("%Y%m%d", time.gmtime(NOW - n * 86400))


class Telemetry(unittest.TestCase):
    def test_over_budget_archives_oldest_and_protects_recent_days(self):
        with tempfile.TemporaryDirectory() as d, patch.object(ret, "TELEMETRY_LIVE_BUDGET_MB", 3), \
                patch.object(ret, "TELEMETRY_PROTECT_DAYS", 3):
            t = Path(d) / "telemetry"
            for n in range(6):
                _write(t / f"requests-{_day(n)}.jsonl", (b'{"x": %d}\n' % n) * (MB // 9))
            _write(t / "prefixcache.jsonl", b"keep\n")
            original = (t / f"requests-{_day(5)}.jsonl").read_bytes()
            rep = ret.run_all(d, os.path.join(d, "nocache"), apply=True, now=NOW)
            live = sorted(f.name for f in t.glob("requests-*.jsonl"))
            self.assertLessEqual(sum((t / f).stat().st_size for f in live), 3 * MB)
            for n in range(3):
                self.assertIn(f"requests-{_day(n)}.jsonl", live)          # protected days never move
            arch = Path(d) / "archive" / "telemetry" / f"requests-{_day(5)}.jsonl.gz"
            self.assertEqual(gzip.decompress(arch.read_bytes()), original)   # archived byte-identical
            self.assertTrue((t / "prefixcache.jsonl").exists())
            self.assertEqual(rep["errors"], [])

    def test_dry_run_changes_nothing(self):
        with tempfile.TemporaryDirectory() as d, patch.object(ret, "TELEMETRY_LIVE_BUDGET_MB", 0):
            p = _write(Path(d) / "telemetry" / f"requests-{_day(10)}.jsonl", b"a" * 4096)
            rep = ret.run_all(d, os.path.join(d, "nocache"), apply=False, now=NOW)
            self.assertTrue(p.exists())
            self.assertFalse((Path(d) / "archive").exists())
            self.assertEqual(rep["policies"][0]["actions"][0]["kind"], "archive-gzip")

    def test_symlink_is_never_followed_or_moved(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as outside, \
                patch.object(ret, "TELEMETRY_LIVE_BUDGET_MB", 0):
            target = _write(Path(outside) / "precious.jsonl", b"z" * 4096)
            (Path(d) / "telemetry").mkdir()
            os.symlink(target, Path(d) / "telemetry" / f"requests-{_day(10)}.jsonl")
            ret.run_all(d, os.path.join(d, "nocache"), apply=True, now=NOW)
            self.assertEqual(target.read_bytes(), b"z" * 4096)
            self.assertTrue(os.path.islink(Path(d) / "telemetry" / f"requests-{_day(10)}.jsonl"))


class Incidents(unittest.TestCase):
    def _bundle(self, root, name, mtime):
        b = Path(root) / "incidents" / name
        _write(b / "META.txt", b"hits_10m=3\n")
        _write(b / "engine-journal.txt", b"journal line\n" * 40000)
        for i in range(3):
            _write(b / "flightrec" / f"{i}_100tok.json", b'{"prompt": "%d"}' % i * 5000)
        os.utime(b, (mtime, mtime))
        return b

    def test_old_bundles_pack_prompt_copies_and_keep_meta(self):
        with tempfile.TemporaryDirectory() as d:
            old = self._bundle(d, "xid-20260901-101500", NOW - 30 * 86400)
            new = self._bundle(d, "fault-20261002-101500", NOW - 86400)
            _write(Path(d) / "incidents" / "ledger.jsonl", b'{"kind": "FAULT"}\n')
            ret.run_all(d, os.path.join(d, "nocache"), apply=True, now=NOW)
            self.assertEqual((old / "META.txt").read_bytes(), b"hits_10m=3\n")
            self.assertFalse((old / "flightrec").exists())
            with tarfile.open(old / "flightrec.tar.gz") as tf:
                self.assertEqual(sorted(tf.getnames())[-3:], [f"flightrec/{i}_100tok.json" for i in range(3)])
            self.assertEqual(gzip.decompress((old / "engine-journal.txt.gz").read_bytes()), b"journal line\n" * 40000)
            self.assertTrue((new / "flightrec").is_dir())                  # recent bundle untouched
            self.assertTrue((new / "engine-journal.txt").exists())
            self.assertTrue((Path(d) / "incidents" / "ledger.jsonl").exists())

    def test_budget_expires_only_oldest_prompt_copies_never_evidence(self):
        with tempfile.TemporaryDirectory() as d, patch.object(ret, "INCIDENT_BUDGET_MB", 0):
            a = self._bundle(d, "xid-20260901-101500", NOW - 30 * 86400)
            b = self._bundle(d, "xid-20260902-101500", NOW - 29 * 86400)
            rep = ret.run_all(d, os.path.join(d, "nocache"), apply=True, now=NOW)
            for x in (a, b):
                self.assertFalse((x / "flightrec.tar.gz").exists())
                self.assertTrue((x / "META.txt").exists())
                self.assertTrue((x / "engine-journal.txt.gz").exists())
            self.assertTrue(any("never auto-deleted" in e for e in rep["errors"]))   # loud, not silent


class Backups(unittest.TestCase):
    def test_keeps_newest_per_file_archives_rest_and_ignores_hand_named(self):
        with tempfile.TemporaryDirectory() as d, patch.object(ret, "BACKUP_KEEP_PER_FILE", 2):
            for i in range(5):
                _write(Path(d) / f"keepalive-shim.py.bak-179100000{i}-12{i}", b"v%d" % i, mtime=NOW - 1000 + i)
            _write(Path(d) / "shim.env.bak-bundle-20261003-002821", b"e")
            hand = _write(Path(d) / "keepalive-shim.py.bak-pre-spend-v6-20260928-042806", b"banked")
            banked = _write(Path(d) / "serve-tqk8v4-fg.sh.BANKED-retention-shipped-20260817-085236", b"b")
            ret.run_all(d, os.path.join(d, "nocache"), apply=True, now=NOW)
            left = sorted(p.name for p in Path(d).glob("keepalive-shim.py.bak-1*"))
            self.assertEqual(left, ["keepalive-shim.py.bak-1791000003-123", "keepalive-shim.py.bak-1791000004-124"])
            self.assertTrue(hand.exists() and banked.exists())
            self.assertTrue((Path(d) / "shim.env.bak-bundle-20261003-002821").exists())
            arch = list((Path(d) / "archive" / "backups").glob("backups-*.tar.gz"))
            with tarfile.open(arch[0]) as tf:
                self.assertEqual(len(tf.getnames()), 3)


class Logs(unittest.TestCase):
    def test_rotates_in_place_keeps_inode_and_bounded_rotations(self):
        with tempfile.TemporaryDirectory() as d, patch.object(ret, "LOG_ROTATE_MB", 0), \
                patch.object(ret, "LOG_KEEP_ROTATIONS", 2):
            log = _write(Path(d) / "watchdog.log", b"line\n" * 1000)
            ino = log.stat().st_ino
            with open(log, "ab") as writer:           # a live O_APPEND writer survives the rotation
                for k in range(4):
                    writer.write(b"round %d\n" % k * 100)
                    writer.flush()
                    ret.run_all(d, os.path.join(d, "nocache"), apply=True, now=NOW + k)
                    self.assertEqual(log.stat().st_size, 0)
                writer.write(b"after\n")
            self.assertEqual(log.stat().st_ino, ino)
            self.assertEqual(log.read_bytes(), b"after\n")              # no sparse hole after truncate
            gz = sorted((Path(d) / "archive" / "logs").glob("watchdog.log.*.gz"))
            self.assertEqual(len(gz), 2)


class Cache(unittest.TestCase):
    def test_idle_entries_dropped_newest_kept(self):
        with tempfile.TemporaryDirectory() as d, patch.object(ret, "CACHE_KEEP_NEWEST", 1), \
                patch.object(ret, "CACHE_IDLE_DAYS", 21):
            c = Path(d) / "cache"
            _write(c / "torch_aot_compile" / "aaaa" / "m.bin", b"x", mtime=NOW - 60 * 86400)
            _write(c / "torch_aot_compile" / "bbbb" / "m.bin", b"x", mtime=NOW - 40 * 86400)
            _write(c / "c2af9568fa" / "rank0" / "m.bin", b"x", mtime=NOW - 2 * 86400)
            ret.run_all(os.path.join(d, "rt"), str(c), apply=True, now=NOW)
            self.assertTrue((c / "c2af9568fa").exists())
            self.assertFalse((c / "torch_aot_compile" / "aaaa").exists())
            self.assertFalse((c / "torch_aot_compile" / "bbbb").exists())

    def test_old_but_recently_read_entry_is_not_idle(self):
        with tempfile.TemporaryDirectory() as d, patch.object(ret, "CACHE_KEEP_NEWEST", 0):
            c = Path(d) / "cache"
            f = _write(c / "torch_aot_compile" / "live" / "m.bin", b"x")
            os.utime(f, (NOW - 3600, NOW - 90 * 86400))     # written 90 days ago, read an hour ago
            ret.run_all(os.path.join(d, "rt"), str(c), apply=True, now=NOW)
            self.assertTrue(f.exists())


class Cli(unittest.TestCase):
    def test_kill_switch(self):
        with tempfile.TemporaryDirectory() as d:
            _write(Path(d) / "RETENTION_DISABLED", b"")
            self.assertEqual(ret.main(["--root", d, "--apply"]), 2)

    def test_report_written(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ret.main(["--root", d, "--cache-root", os.path.join(d, "c")]), 0)
            self.assertTrue((Path(d) / "retention-dryrun.json").exists())


if __name__ == "__main__":
    unittest.main()
