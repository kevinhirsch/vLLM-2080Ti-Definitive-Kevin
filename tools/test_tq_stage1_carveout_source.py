#!/usr/bin/env python3
"""CPU source-contract checks, not a compiled-C++/CUDA execution test."""
import pathlib
import re
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
REL = "vllm/v1/attention/ops/tq_gqa_ext/tq_gqa.cu"
SOURCE = (ROOT / REL).read_text()
BASE = subprocess.run(
    ["git", "show", "7004df6ae82c020c8a8e01c51f2f18f8889ab796:" + REL],
    cwd=ROOT, check=True, capture_output=True, text=True,
).stdout
PARSER = SOURCE.split("static int tq_stage1_requested_shared_kib() {", 1)[1].split(
    "static void tq_stage1_check_attribute_api", 1)[0]
CONFIG = SOURCE.split("static void tq_stage1_configure_shared_preference", 1)[1].split(
    "void tq_gqa_decode", 1)[0]


def source_policy(value):
    """Interpret only the literal source table; no native execution claimed."""
    literals = dict((key, int(result)) for key, result in re.findall(
        r'std::strcmp\(value, "(default|32|64)"\) == 0\) return (\d+);', PARSER))
    if value is None:
        if "value == nullptr" not in PARSER:
            raise AssertionError("missing absent-env branch")
        return 0
    if value in literals:
        return literals[value]
    if 'TORCH_CHECK(false,' not in PARSER:
        raise AssertionError("missing refusal")
    raise ValueError("invalid literal")


class CarveoutSourceContract(unittest.TestCase):
    def test_literal_policy_and_absence(self):
        self.assertEqual([source_policy(x) for x in (None, "default", "32", "64")],
                         [0, 0, 32, 64])
        self.assertEqual(PARSER.count("std::getenv("), 1)
        self.assertIn("static const int requested = []()", PARSER)

    def test_invalid_values_refused(self):
        for value in ("", "0", "50", "100", "032", "64 ", " 32", "32\n",
                      "DEFAULT", "-1", "32,64", "32=64"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                source_policy(value)

    def test_default_has_no_api_or_lock(self):
        before_return = CONFIG.split("if (requested_kib == 0) return;", 1)[0]
        self.assertNotIn("cuda", before_return.split("{", 1)[1])
        self.assertNotIn("mutex", before_return)
        self.assertNotIn("fprintf", before_return)

    def test_percentages_and_exact_stage1_symbol(self):
        self.assertIn("requested_kib == 32 ? 50 : 100", CONFIG)
        self.assertIn("cudaFuncSetAttribute(tq_gqa_stage1, cudaFuncAttributePreferredSharedMemoryCarveout, preferred_percent)", CONFIG)
        self.assertIn("cudaFuncGetAttributes(&attributes, tq_gqa_stage1)", CONFIG)
        self.assertNotIn("tq_gqa_stage2", CONFIG)

    def test_once_per_device_after_success(self):
        self.assertLess(CONFIG.index("configured_devices.count(device)"),
                        CONFIG.index("cudaFuncSetAttribute("))
        self.assertLess(CONFIG.index("cudaFuncGetAttributes("),
                        CONFIG.index("configured_devices.insert(device)"))
        self.assertLess(CONFIG.index("std::fprintf("),
                        CONFIG.index("configured_devices.insert(device)"))
        self.assertEqual(CONFIG.count("configured_devices.insert(device)"), 1)
        self.assertIn("std::lock_guard<std::mutex>", CONFIG)

    def test_error_capture_and_architecture_admission(self):
        self.assertIn("TORCH_CHECK(status == cudaSuccess", SOURCE)
        self.assertIn("TORCH_CHECK(major == 7 && minor == 5", CONFIG)
        self.assertIn("TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone", CONFIG)
        self.assertLess(CONFIG.index("capture_status == cudaStreamCaptureStatusNone"),
                        CONFIG.index("cudaFuncSetAttribute("))
        self.assertIn("attributes.preferredShmemCarveout == preferred_percent", CONFIG)

    def test_all_device_source_unchanged(self):
        # Host helpers were inserted after both kernels. Exact original device
        # function bodies, constants and inline PTX remain byte-identical.
        base_device = BASE.split("#define HD", 1)[1].split("void tq_gqa_decode", 1)[0]
        candidate_device = SOURCE.split("#define HD", 1)[1].split(
            "// Experimental host-only preference.", 1)[0]
        self.assertEqual(base_device, candidate_device)

    def test_original_launch_body_only_one_host_call_added(self):
        old = BASE.split("void tq_gqa_decode", 1)[1]
        new = SOURCE.split("void tq_gqa_decode", 1)[1]
        call = "  tq_stage1_configure_shared_preference(q_rot.get_device(), stream);\n"
        self.assertEqual(new.count(call), 1)
        self.assertEqual(new.replace(call, ""), old)
        self.assertLess(new.index(call.strip()), new.index("tq_gqa_stage1<<<"))


if __name__ == "__main__":
    unittest.main()
