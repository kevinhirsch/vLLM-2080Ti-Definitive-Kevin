"""S4 (2026-10-03): the fault collector must not blame the engine for a kernel Xid raised by a pid outside the engine's process tree."""
import importlib.util, os, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("efc", os.path.join(HERE, "engine-fault-collector.py"))
efc = importlib.util.module_from_spec(spec); spec.loader.exec_module(efc)

JOURNAL = (
    "Oct 03 06:34:03 HNET00 serve-active.sh[4191191]: (APIServer pid=4191191) INFO: 127.0.0.1 - GET /v1/models 200 OK\n"
    "Oct 03 06:34:06 HNET00 serve-active.sh[4192564]: (EngineCore pid=4192564) INFO [shutdown] EngineCore: trigger received signal=SIGTERM\n"
    "Oct 03 06:34:06 HNET00 serve-active.sh[4192747]: (Worker_TP0 pid=4192747) INFO Parent process exited, terminating worker queues\n"
    "Oct 03 06:34:28 HNET00 systemd[1]: Stopping vLLM 2080Ti Definitive...\n"
)
FOREIGN = "2026-10-03T06:34:02-07:00 HNET00 kernel: NVRM: Xid (PCI:0000:04:00): 31, pid=41733, name=python, channel 0x00000022, intr 00000000. MMU Fault: ENGINE GRAPHICS\n"
OWN = "2026-10-03T06:34:02-07:00 HNET00 kernel: NVRM: Xid (PCI:0000:04:00): 31, pid=4192747, name=python, channel 0x00000022, intr 00000000. MMU Fault: ENGINE GRAPHICS\n"


class ForeignXid(unittest.TestCase):
    def test_foreign_xid31_is_not_blamed_on_engine(self):
        sig, _ = efc.classify(JOURNAL, FOREIGN)
        self.assertEqual(sig, "unknown-exit")

    def test_own_xid31_is_still_illegal_address(self):
        sig, _ = efc.classify(JOURNAL, OWN)
        self.assertEqual(sig, "cuda-illegal-address")

    def test_mixed_lines_keep_only_own(self):
        kept = efc.own_kernel_lines(JOURNAL, FOREIGN + OWN)
        self.assertIn("pid=4192747", kept)
        self.assertNotIn("pid=41733", kept)

    def test_no_engine_pids_in_journal_keeps_old_behaviour(self):
        sig, _ = efc.classify("Traceback ...\n", FOREIGN)
        self.assertEqual(sig, "cuda-illegal-address")


if __name__ == "__main__":
    unittest.main()
