"""EF2 (ported by LV, 2026-10-03): estate-watchdog's Xid check counts only NVIDIA `NVRM: Xid` lines. The r8169 NIC prints
'XID 641' in its boot banner, and the old case-insensitive grep turned every boot into a false xid-* incident."""
import os, re, stat, subprocess, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "estate-watchdog.sh")).read()
FN = re.search(r"(?ms)^check_kernel_xid\(\) \{.*?^\}", SRC).group(0)

NIC = "2026-10-02T20:53:38-07:00 HNET00 kernel: r8169 0000:06:00.0 eth0: RTL8125B, 50:eb:f6:ce:c1:33, XID 641, IRQ 158"
GPU = "2026-10-03T06:32:01-07:00 HNET00 kernel: NVRM: Xid (PCI:0000:01:00): 31, pid=4242, name=python3, Ch 00000008"


class XidCheck(unittest.TestCase):
    def run_check(self, lines):
        with tempfile.TemporaryDirectory() as d:
            b = os.path.join(d, "bin"); os.makedirs(b)
            jp = os.path.join(b, "journalctl")
            open(jp, "w").write("#!/bin/sh\ncat <<'EOF'\n" + "\n".join(lines) + "\nEOF\n")
            os.chmod(jp, os.stat(jp).st_mode | stat.S_IEXEC)
            script = f"STATE_DIR={d}\nWATCHDOG_INCIDENT_ARCHIVE=0\n{FN}\ncheck_kernel_xid\necho \"$CHECK_STATUS|$CHECK_DETAIL\"\n"
            out = subprocess.run(["bash", "-c", script], env=dict(os.environ, PATH=f"{b}:{os.environ['PATH']}", HOME=d),
                                 capture_output=True, text=True, timeout=30)
            return out.stdout.strip()

    def test_nic_banner_is_not_an_xid(self):
        self.assertTrue(self.run_check([NIC]).startswith("OK|"))

    def test_nvrm_xid_is(self):
        out = self.run_check([NIC, GPU])
        self.assertTrue(out.startswith("WARN|1 Xid"), out)


if __name__ == "__main__":
    unittest.main()
