"""LV (2026-10-03): only the engine-liveness authority may start a stopped engine (it honours holds, windows, the rate limit).

A unit dependency that pulls vllm-qwen27b.service in (Wants=/Requires=/BindsTo=/Upholds=/Requisite=) starts it behind the
authority's back: the watchdog unit's Wants= started a held engine every minute before vllm-watchdog.sh could defer."""
import os, re, unittest

DEPLOY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSD = os.path.join(DEPLOY, "systemd")
PULL = re.compile(r"^\s*(Wants|Requires|BindsTo|Upholds|Requisite|PartOf)\s*=.*\bvllm-qwen27b\.service\b", re.M)


class NoUnitPullsTheEngineIn(unittest.TestCase):
    def test_repo_units(self):
        bad = []
        for root, _d, files in os.walk(SYSD):
            for f in files:
                if f.endswith((".service", ".timer", ".conf", ".target", ".path", ".socket")):
                    p = os.path.join(root, f)
                    for m in PULL.finditer(open(p).read()):
                        bad.append(f"{os.path.relpath(p, SYSD)}: {m.group(0).strip()}")
        self.assertEqual(bad, [])

    def test_watchdog_unit_still_orders_after_the_engine(self):
        unit = open(os.path.join(SYSD, "vllm-qwen27b-watchdog.service")).read()
        self.assertRegex(unit, r"(?m)^After=vllm-qwen27b\.service$")
        self.assertNotRegex(unit, r"(?m)^Wants=")

    def test_engine_boot_start_is_its_own_install_section(self):
        unit = open(os.path.join(SYSD, "vllm-qwen27b.service")).read()
        self.assertRegex(unit, r"(?m)^WantedBy=multi-user\.target$")


if __name__ == "__main__":
    unittest.main()
