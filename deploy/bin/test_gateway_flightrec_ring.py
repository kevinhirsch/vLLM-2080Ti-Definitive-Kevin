#!/usr/bin/env python3
"""Lane A6: the flight-recorder ring cleanup raced with itself.

LIVE 2026-10-03: 14 x 'WARNING flightrec: [Errno 2] No such file or directory: .../flightrec/<old>.json' (the path of an OLD file, not
the one just written). _write_flightrec runs in the default thread pool, so two big requests finishing together both list the ring,
both pick the same stale files, and the second unlink raised FileNotFoundError: the request was fine, but the failure was logged as a
flight-recorder fault (and an EXCEPTION out of the executor job). Removing a file that is already gone is the goal, not an error.

Run:  python -m pytest -q test_gateway_flightrec_ring.py
"""
import os
import tempfile
import threading
import unittest

import test_gateway_cache_model as C

shim = C.shim


class RingCleanup(unittest.TestCase):
    def test_concurrent_writers_never_raise_and_the_ring_stays_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(60):                                          # an over-full ring: every writer wants the same victims
                open(os.path.join(d, "%010d_1000tok.json" % (1700000000 + i)), "wb").write(b"{}")
            errors, start, clock, lock = [], threading.Barrier(8), [1800000000], threading.Lock()

            def writer(k):
                start.wait()
                for j in range(25):
                    with lock:
                        clock[0] += 1               # names sort by time, like the real int(time.time())_<tok> names
                        fn = os.path.join(d, "%010d_%dtok.json" % (clock[0], 2000 + k))
                    try:
                        shim._write_flightrec(d, fn, b"{}")
                    except Exception as e:                               # noqa: BLE001
                        errors.append(repr(e))

            ts = [threading.Thread(target=writer, args=(k,)) for k in range(8)]
            [t.start() for t in ts]
            [t.join() for t in ts]
            self.assertEqual(errors, [])
            self.assertLessEqual(len(os.listdir(d)), 40 + 8)             # bounded (the last writers may still be one ahead)

    def test_missing_victim_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(45):
                open(os.path.join(d, "%010d_1000tok.json" % (1700000000 + i)), "wb").write(b"{}")
            real_listdir = os.listdir

            def listdir_then_steal(p):
                names = real_listdir(p)
                os.unlink(os.path.join(p, sorted(names)[0]))             # another writer removed it between list and unlink
                return names

            orig = shim.os.listdir
            shim.os.listdir = listdir_then_steal
            try:
                shim._write_flightrec(d, os.path.join(d, "1800000000_5tok.json"), b"{}")
            finally:
                shim.os.listdir = orig
            self.assertLessEqual(len(os.listdir(d)), 40)


class Private(unittest.TestCase):
    def test_bodies_are_born_0600_whatever_the_umask(self):
        old = os.umask(0o000)
        try:
            with tempfile.TemporaryDirectory() as d:
                fn = os.path.join(d, "1_1tok.json")
                shim._write_flightrec(d, fn, b"{}")
                self.assertEqual(os.stat(fn).st_mode & 0o777, 0o600)
        finally:
            os.umask(old)


if __name__ == "__main__":
    unittest.main()
