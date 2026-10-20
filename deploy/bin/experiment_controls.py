"""Read-only engine identity used by window and gateway measurement controls."""
import hashlib
import json
import os
import re
import subprocess
import urllib.request


FLAGS = {
    "--model", "--served-model-name", "--max-model-len", "--max-num-seqs",
    "--max-num-batched-tokens", "--gpu-memory-utilization", "--tensor-parallel-size",
    "--long-prefill-token-threshold", "--max-num-partial-prefills", "--max-long-partial-prefills",
    "--pipeline-parallel-size", "--quantization", "--dtype", "--kv-cache-dtype",
    "--speculative-config", "--compilation-config", "--hf-overrides",
    "--enable-prefix-caching", "--disable-custom-all-reduce", "--enforce-eager",
}


def selected_flags(argv):
    flags = {}
    for i, arg in enumerate(argv):
        key, sep, value = arg.partition("=")
        if key in FLAGS:
            if not sep:
                value = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else True
            flags[key] = value
    return flags


def main_pid(unit="vllm-qwen27b"):
    try:
        r = subprocess.run(["systemctl", "show", "-p", "MainPID", "--value", unit],
                           capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip()) if r.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def runtime_identity(unit="vllm-qwen27b", *, proc="/proc", pid=None):
    """Fail closed when the SYSTEM unit or its process cannot be identified.

    No environment or arbitrary command line values are returned. The fingerprint
    binds the complete command line without exposing credentials it might contain.
    """
    try:
        if pid is None:
            pid = main_pid(unit)
        if pid is None:
            return None
        if int(pid) <= 0:
            return None
        base = os.path.join(proc, str(pid))
        with open(os.path.join(base, "stat")) as f:
            start = f.read().rsplit(")", 1)[1].split()[19]
        with open(os.path.join(base, "cmdline"), "rb") as f:
            raw = f.read()
        argv = [v.decode("utf-8", "replace") for v in raw.split(b"\0") if v]
        root = os.path.realpath(os.readlink(os.path.join(base, "cwd")))
        model = None
        if "serve" in argv:
            i = argv.index("serve") + 1
            if i < len(argv) and not argv[i].startswith("-"):
                model = argv[i]
        flags = selected_flags(argv)
        model = flags.get("--model", model)
        if not model:
            return None
        with open(os.path.join(base, "environ"), "rb") as f:
            environment = f.read().split(b"\0")
        kernel_prefixes = ("VLLM_TURBOQUANT_", "VLLM_GDN_", "VLLM_U2_",
                           "VLLM_QWOPUS_MTP_", "VLLM_CUSTOM_ALLREDUCE_", "VLLM_PREFIX_CACHE_", "VLLM_SCHED_")
        runtime_flags = {}
        for item in environment:
            if b"=" not in item:
                continue
            key, value = item.decode("utf-8", "replace").split("=", 1)
            if key.startswith(kernel_prefixes) or re.match(r"VLLM_K[0-9]+_", key):
                runtime_flags[key] = value
        row = {"engine_pid": int(pid), "pid_start": start, "root": root,
               "model_path": model, "flags": flags, "runtime_flags": runtime_flags,
               "argv_sha256": hashlib.sha256(raw).hexdigest()}
        row["fingerprint"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        return row
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def same_configuration(a, b):
    """Process generation changes on restart; the exact serving configuration must not."""
    return bool(a and b and all(a.get(k) == b.get(k)
                               for k in ("root", "model_path", "flags", "runtime_flags", "argv_sha256")))


def drain_complete(gateway, engine):
    """Unknown metrics are never evidence of a completed drain."""
    return (isinstance(gateway, dict) and gateway.get("local_active") == 0
            and isinstance(engine, dict) and engine.get("running") == 0 and engine.get("waiting") == 0)


def engine_inflight(engine, timeout=5):
    try:
        with urllib.request.urlopen(engine.rstrip("/") + "/metrics", timeout=timeout) as response:
            txt = response.read().decode("utf-8", "replace")
        out = {}
        for key in ("running", "waiting"):
            matches = re.findall(r"^vllm:num_requests_" + key + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", txt, re.M)
            if not matches:
                return None
            out[key] = sum(int(float(value)) for value in matches)
        return out
    except (OSError, ValueError):
        return None
