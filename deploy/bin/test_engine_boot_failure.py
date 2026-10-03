"""AU 2026-10-03: failed boots are their own fault class, and the warm-up hook stops waiting when the engine process dies.

Measured: 10-02 22:11-22:16 six boots died with Worker failed with error ''weight'' and 10-03 01:19-01:28 ten boots died
with 'No available memory for the cache blocks'; both were filed as engine-dead-other / unknown-exit. 10-03 03:01-03:30
four boots died ~20 s in but the unit stayed 'activating' 7 min 16 s each (warm-up hook waiting its 420 s cap), so
ExecStopPost read an empty 240 s journal window."""
import importlib.util, os, subprocess, tempfile, textwrap, time, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("efc_boot", os.path.join(HERE, "engine-fault-collector.py"))
efc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(efc)

KV = textwrap.dedent("""\
    Oct 03 01:21:44 HNET00 serve-active.sh[1905255]: (EngineCore pid=1905255) ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization`
    Oct 03 01:21:45 HNET00 serve-active.sh[1903500]: (APIServer pid=1903500) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
    Oct 03 01:21:47 HNET00 systemd[1]: vllm-qwen27b.service: Main process exited, code=exited, status=1/FAILURE
""")
WEIGHT = textwrap.dedent("""\
    Oct 02 22:13:46 HNET00 serve-active.sh[457499]: (EngineCore pid=457499) RuntimeError: Worker failed with error ''weight'', please check the stack trace above for the root cause
    Oct 02 22:13:47 HNET00 serve-active.sh[456510]: (APIServer pid=456510) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
""")


class BootFailureClass(unittest.TestCase):
    def test_no_kv_memory_boot(self):
        self.assertEqual(efc.classify(KV, "")[0], "boot-failed")
        self.assertIn("no-kv-memory", efc.classify(KV, "")[1])

    def test_weight_load_boot(self):
        sig, detail = efc.classify(WEIGHT, "")
        self.assertEqual(sig, "boot-failed")
        self.assertEqual(detail, "'weight'")

    def test_runtime_deaths_unchanged(self):
        self.assertEqual(efc.classify("EngineDeadError: x", "")[0], "engine-dead-other")
        self.assertEqual(efc.classify("torch.OutOfMemoryError: CUDA out of memory", "")[0], "oom")
        self.assertEqual(efc.classify("", "")[0], "unknown-exit")

    def test_illegal_address_at_boot_stays_cuda(self):
        j = "cudaErrorIllegalAddress\n" + WEIGHT
        self.assertEqual(efc.classify(j, "")[0], "cuda-illegal-address")


class WarmupStopsWhenEngineDies(unittest.TestCase):
    def test_exits_promptly_when_main_pid_is_gone(self):
        src = open(os.path.join(HERE, "warmup-after-start.sh")).read()
        with tempfile.TemporaryDirectory() as d:
            # point the hook at a dead engine port and a scratch log/queue; MAINPID of a process that already exited
            src = src.replace("ENGINE=http://127.0.0.1:8001", "ENGINE=http://127.0.0.1:9")
            src = src.replace("/home/kevin/.local/share/vllm-qwen27b/frontier-queue", f"{d}/q")
            src = src.replace("/home/kevin/.local/share/vllm-qwen27b/warmup.log", f"{d}/w.log")
            path = f"{d}/w.sh"
            open(path, "w").write(src)
            dead = subprocess.Popen(["true"]); dead.wait()
            t = time.time()
            r = subprocess.run(["bash", path], env=dict(os.environ, MAINPID=str(dead.pid), WARMUP_CAP_SECS="60"),
                               capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, 0)
            self.assertLess(time.time() - t, 10)
            self.assertIn("main process exited during boot", open(f"{d}/w.log").read())


if __name__ == "__main__":
    unittest.main()
