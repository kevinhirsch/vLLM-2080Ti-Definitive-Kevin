"""L133 / L152 (BS3): a boot configuration that crash-loops identically is reverted to the last-known-good one by the liveness authority.

Live incidents: 10-02 22:11 (6 boots, KeyError 'weight'), 10-03 01:19 (10 boots, no KV memory), 03:01 (4 boots); each ended only when the window
owner intervened, because Restart=always + StartLimitIntervalSec=0 has no boot-success gate."""
import json
import os
import unittest
from datetime import datetime
from unittest.mock import patch

from test_engine_liveness_authority import Base, ea


def iso(t):
    return datetime.fromtimestamp(t).isoformat(timespec="seconds")


class BootRevert(Base):
    def setUp(self):
        super().setUp()
        d = self.tmp.name
        for name, val in (("RELEASES_CURRENT", f"{d}/releases/current"),):
            p = patch.object(ea, name, val)
            p.start()
            self.addCleanup(p.stop)
        self.base = d
        self.put("active-serve", "serve-hauhaucs-v02.sh\n")
        self.put("v02.override.env", "export V02_STACK=1\n")
        self.put("diag.env", "")
        self.put("vllm-qwen27b.env", "FOO=1\n")

    def put(self, name, text, mtime=None):
        path = f"{self.base}/{name}"
        open(path, "w").write(text)
        if mtime is not None:
            os.utime(path, (mtime, mtime))

    def read(self, name):
        return open(f"{self.base}/{name}").read()

    def fault(self, t, sig="boot-failed"):
        self.ledger.append({"ts": iso(t), "kind": "FAULT", "signature": sig})

    def good_boot(self):
        """tick once while healthy and stable: records last_healthy and (when allowed) the last-known-good config."""
        ea.tick(act=False)

    def crash_loop(self, n=4, sig="boot-failed", t0=None):
        """The engine goes unhealthy; n identical FAULT rows land after the last healthy moment."""
        t0 = self.now if t0 is None else t0
        self.now = t0 + 30
        self.healthy = False
        self.unit.update(ActiveState="activating", SubState="auto-restart", ExecMainStartTimestamp=t0 + 20)
        for i in range(n):
            self.fault(t0 + 5 + i * 3, sig)
        self.now = t0 + 60

    # ---- recording --------------------------------------------------------
    def test_records_last_known_good_after_a_stable_boot(self):
        self.good_boot()
        m = ea.lkg_meta()
        self.assertEqual(m["boot_started"], self.unit["ExecMainStartTimestamp"])
        self.assertEqual(open(f"{self.base}/lkg/v02.override.env").read(), "export V02_STACK=1\n")
        self.assertIsNotNone(m["files"]["active-serve"])
        self.assertEqual(self.state()["lkg"]["boot_started"], m["boot_started"])

    def test_a_young_boot_is_not_known_good_yet(self):
        self.unit["ExecMainStartTimestamp"] = self.now - 100
        self.good_boot()
        self.assertIsNone(ea.lkg_meta())

    def test_a_file_edited_after_the_boot_is_not_recorded(self):
        self.put("v02.override.env", "export V02_STACK=1\nexport NEXT_ARM=1\n", mtime=self.now - 100)   # staged for the next boot
        self.good_boot()
        self.assertIsNone(ea.lkg_meta())

    def test_nothing_is_recorded_while_a_window_holds_the_engine(self):
        json.dump([{"kind": "engine", "by": "K5", "reason": "window", "lease": "L", "acquired": self.now - 60, "until": self.now + 600}],
                  open(ea.HOLDS, "w"))
        self.good_boot()
        self.assertIsNone(ea.lkg_meta())

    # ---- reverting --------------------------------------------------------
    def window_breaks_the_boot(self):
        self.good_boot()
        self.put("v02.override.env", "export V02_STACK=1\nexport VLLM_K5_GDN_FUSED=1\n")      # the window's arm
        self.put("diag.env", "CUDA_LAUNCH_BLOCKING=1\n")

    def test_identical_boot_failures_revert_to_last_known_good_and_restart(self):
        self.window_breaks_the_boot()
        self.crash_loop(4)
        out = ea.tick(by="t")
        self.assertEqual(self.read("v02.override.env"), "export V02_STACK=1\n")
        self.assertEqual(self.read("diag.env"), "")
        acts = [a for a in self.actions() if a.get("action") == "revert"]
        self.assertEqual(len(acts), 1)
        self.assertEqual(acts[0]["cause"], "boot-failed-boot-failed")
        self.assertIn("v02.override.env", acts[0]["evidence"])
        self.assertIn(["restart", "--no-block", ea.UNIT], self.sudo)
        self.assertEqual(out["state"], "RECOVERING")
        kept = acts[0]["id"]
        self.assertIn("VLLM_K5_GDN_FUSED", open(f"{self.base}/lkg/reverted-{kept}/v02.override.env").read())   # the failing config is kept
        ev = [k for (a, k) in self.events if k.get("action") == "engine-boot-revert"]
        self.assertEqual(len(ev), 1)
        self.assertTrue(ev[0].get("handoff"))

    def test_it_reverts_even_though_the_window_holds_the_engine(self):
        self.window_breaks_the_boot()
        json.dump([{"kind": "engine", "by": "K5", "reason": "window", "lease": "L", "acquired": self.now - 60, "until": self.now + 6000}],
                  open(ea.HOLDS, "w"))
        self.crash_loop(4)
        # the hold must outlive the crash-loop clock jump
        json.dump([{"kind": "engine", "by": "K5", "reason": "window", "lease": "L", "acquired": self.now - 60, "until": self.now + 6000}],
                  open(ea.HOLDS, "w"))
        ea.tick(by="t")
        self.assertEqual(self.read("v02.override.env"), "export V02_STACK=1\n")
        self.assertEqual(len([a for a in self.actions() if a.get("action") == "revert"]), 1)

    def test_three_failures_are_not_enough(self):
        self.window_breaks_the_boot()
        self.crash_loop(3)
        ea.tick(by="t")
        self.assertIn("VLLM_K5_GDN_FUSED", self.read("v02.override.env"))
        self.assertEqual([a for a in self.actions() if a.get("action") == "revert"], [])

    def test_different_signatures_are_weather_not_configuration(self):
        self.window_breaks_the_boot()
        self.crash_loop(4)
        self.ledger[-1]["signature"] = "xid-31"
        ea.tick(by="t")
        self.assertIn("VLLM_K5_GDN_FUSED", self.read("v02.override.env"))
        self.assertEqual([a for a in self.actions() if a.get("action") == "revert"], [])

    def test_a_streak_with_the_config_already_good_is_not_reverted_and_says_why(self):
        self.good_boot()                      # nothing changed since the last-known-good record
        self.crash_loop(5)
        out = ea.tick(by="t")
        self.assertEqual([a for a in self.actions() if a.get("action") == "revert"], [])
        self.assertIn("already equals", out["boot_revert"]["blocked"])
        self.assertEqual(out["state"], "CRASH_LOOP")                                  # Halo / Kevin, as before

    def test_no_record_yet_is_reported_not_guessed(self):
        self.crash_loop(4)
        out = ea.tick(by="t")
        self.assertIn("no last-known-good", out["boot_revert"]["blocked"])
        self.assertEqual([a for a in self.actions() if a.get("action") == "revert"], [])

    def test_a_revert_happens_once_then_the_gate_and_the_equal_config_stop_a_loop(self):
        self.window_breaks_the_boot()
        self.crash_loop(4)
        ea.tick(by="t")
        self.now += 120
        ea.tick(by="t")
        self.assertEqual(len([a for a in self.actions() if a.get("action") == "revert"]), 1)

    def test_switch_off(self):
        self.window_breaks_the_boot()
        self.crash_loop(4)
        with patch.object(ea, "BOOT_REVERT_FAULTS", 0):
            ea.tick(by="t")
        self.assertIn("VLLM_K5_GDN_FUSED", self.read("v02.override.env"))

    def test_a_file_the_last_known_good_did_not_have_is_removed(self):
        os.unlink(f"{self.base}/diag.env")
        self.good_boot()
        self.put("diag.env", "CUDA_LAUNCH_BLOCKING=1\n")
        self.crash_loop(4)
        ea.tick(by="t")
        self.assertFalse(os.path.exists(f"{self.base}/diag.env"))

    def test_the_release_symlink_is_reverted_when_the_old_release_still_exists(self):
        rel = os.path.dirname(ea.RELEASES_CURRENT)
        os.makedirs(f"{rel}/rel-a")
        os.makedirs(f"{rel}/rel-b")
        os.symlink("rel-a", ea.RELEASES_CURRENT)
        self.good_boot()
        os.unlink(ea.RELEASES_CURRENT)
        os.symlink("rel-b", ea.RELEASES_CURRENT)
        self.crash_loop(4)
        ea.tick(by="t")
        self.assertEqual(os.readlink(ea.RELEASES_CURRENT), "rel-a")


if __name__ == "__main__":
    unittest.main()
