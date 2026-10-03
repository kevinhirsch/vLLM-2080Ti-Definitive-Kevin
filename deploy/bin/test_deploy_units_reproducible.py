"""AU 2026-10-03: the host must be rebuildable from the repo. Measured drift: warmup.conf, stop-timeout.conf (engine) and the
shim's hardening.conf existed only on the host; the repo unit still booted serve-tqk8v4-fg.sh while the host boots
serve-active.sh; install.sh installed only the engine's drop-in dir and copied 32 test files into the runtime dir."""
import os, re, unittest

DEPLOY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSD = os.path.join(DEPLOY, "systemd")
RUNTIME = "/home/kevin/.local/share/vllm-qwen27b/"


class Reproducible(unittest.TestCase):
    def test_install_ships_every_drop_in_dir(self):
        install = open(os.path.join(DEPLOY, "install.sh")).read()
        self.assertIn('"$HERE"/systemd/*.service.d', install, "install.sh must loop over every <unit>.service.d dir")

    def test_install_does_not_ship_tests(self):
        install = open(os.path.join(DEPLOY, "install.sh")).read()
        self.assertNotRegex(install, r'cp -v "\$HERE"/bin/\*\s')

    def test_every_runtime_path_a_unit_or_drop_in_runs_is_in_the_repo(self):
        missing = []
        for root, _d, files in os.walk(SYSD):
            for f in files:
                if not (f.endswith(".service") or f.endswith(".conf")):
                    continue
                for line in open(os.path.join(root, f)):
                    m = re.match(r"\s*Exec\w+=[-+!@]*(?:/usr/bin/python3\s+)?(\S+)", line)
                    if m and m.group(1).startswith(RUNTIME):
                        name = m.group(1)[len(RUNTIME):]
                        if not os.path.exists(os.path.join(DEPLOY, "bin", name)):
                            missing.append(f"{f}: {name}")
        self.assertEqual(missing, [], "unit/drop-in runs a runtime file the repo does not ship")

    def test_engine_unit_boots_the_profile_pointer(self):
        unit = open(os.path.join(SYSD, "vllm-qwen27b.service")).read()
        self.assertIn("ExecStart=" + RUNTIME + "serve-active.sh", unit)


if __name__ == "__main__":
    unittest.main()
