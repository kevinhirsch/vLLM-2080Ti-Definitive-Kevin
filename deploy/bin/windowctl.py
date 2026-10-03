#!/usr/bin/env python3
"""windowctl.py -- ONE framework for engine windows (lane RL, 2026-10-03, lead L147).

Lanes used to hand-write a window script each (k5_win.sh, k6_win.sh, k3_win.sh, lp_win1.sh, ...), and every one of them
re-implemented override save/restore, boots, leases and health waits. The leftovers were real: the watchdog timer was
left stopped for 4.5 h; gateway-offline leases were stranded when wrappers were killed; `pkill -f` matched its own
command line; trial guards raced the window's boots; and foreign CUDA processes shrank the KV pool of the boots
(966K -> 922K -> 900K tokens on 2026-10-03).

A lane now writes only a declarative SPEC (YAML or JSON). This framework owns everything else:
  * gateway local-offline window (gateway-offline.py; lease recorded with our pid, renewed, always closed);
  * snapshot + EXACT restore: v02.override.env bytes, systemd timer states, the production root's .so hashes (JIT build
    dirs are backed up and put back if a boot rebuilt them), engine health, the KV pool size;
  * Xid baseline + new-Xid detection after every step (each new Xid attributed to a unit via unitrun.owner_of_pid);
  * a GPU gate before EVERY boot: no non-engine compute app on either GPU (waits, then fails naming the owner unit);
  * the KV pool of every boot is recorded; a production restore below the expected pool is a FAILED restore;
  * the cooperative "GPU busy" signal (gpuguard.py; lane benches check ~/projects/lanes/windows/gpuok.sh);
  * spend check before every step: at >= spend_pause_usd (default $20) the window pauses = stops, restores, reports
    (the gateway's own $25/day cap is never touched);
  * every command runs in its own systemd user unit <lane>-<window>-<step> (unitrun.py), killed by unit, never by pattern;
  * a dead-man timer unit: if this process dies (even SIGKILL) the restore still happens within ~2 minutes;
  * results dir + summary.json (+ per-step logs).

Usage:
  windowctl.py validate SPEC              # schema + variable check, prints the resolved plan; touches nothing
  windowctl.py run SPEC [--dry-run]       # run the window in the foreground (WQ normally uses `submit`)
  windowctl.py submit SPEC [--wait]       # run it inside its own unit win-<lane>-<window> (stop it = systemctl --user stop)
  windowctl.py status                     # the active window, if any
  windowctl.py deadman --state FILE       # internal: the dead-man timer's entry point

Spec (YAML):
  lane: k5                      # unit prefix + results default ~/projects/lanes/<lane>/windows/<window>-<ts>
  window: k5-gdn-fused-ab
  reason: "K5 GDN fused decode A/B"
  by: K5
  max_s: 3600                   # whole-window budget; past it remaining steps are skipped and restore runs
  ttl_s: 1800                   # gateway lease TTL (renewed every 60 s while the window lives)
  wait_s: 90                    # wait for accepted local work after opening the gateway window
  spend_pause_usd: 20
  expected_kv_pool: 960000      # optional floor for the production restore (default: the pool at window start)
  kv_pool_tolerance: 0.005
  gpu_gate_wait_s: 300
  conflicts: [trial_guard.sh]   # refuse to start while such a process runs (exact /proc scan, never matches itself)
  watch_timers: [vllm-qwen27b-watchdog.timer]   # snapshotted + restored exactly (default)
  pause_timers: []              # stopped for the window, restarted by restore AND by the dead-man. The engine watchdog
                                # timer is added automatically when any step stops the engine (keep_watchdog_timer: true opts out)
  on_new_xid: abort             # abort | continue
  remote_quiet_s: 900           # refuse to open unless the remote valve is healthy AND had no breaker/402/429 for this
                                # long; during the window it is polled every 15 s and 2 bad polls abort + restore local
  snapshot_files: []            # extra files restored verbatim (e.g. a deployed serve script)
  promote: {release: ID, files: [...]}   # ONLY when every step passed: becomes the new restore target (prod default)
  vars: {L: /home/kevin/projects/lanes/k5}
  steps:
    - name: kernel-test
      engine: stop                        # stop | start
    - name: test
      run: "cd {{K5}} && python tools/k5/test_gdn_mtp.py"   # str = bash -c, list = argv
      # or  script: [/path/lane_script.sh, args...]   -> runs from a read-only snapshot taken at step start (snaprun)
      timeout_s: 600
      env: {CUDA_VISIBLE_DEVICES: "1"}
      outputs: ["{{L}}/test_gdn_mtp.json"]
      capture: {KPASS: {json: "{{L}}/test_gdn_mtp.json", key: pass, default: "False"}}
      on_fail: continue                   # abort (default) | continue
    - name: k5-arm
      when: "{{KPASS}} == True"
      boot:
        release: {sha: 968a2b10e3, label: k5}   # or an id, "current", or a path; built before the window if missing
        env: {VLLM_K5_GDN_FUSED: "1"}
        extra_args_append: "--profiler-config {...}"
        timeout_s: 1200
Template variables are {{NAME}} (vars, captures, and RESULTS WINDOW LANE STEP PROD_ROOT). Unknown names are an error.
Every run step gets WINDOW_ID=<window>, WINDOW_RESULTS=<dir> in its env (gpuok.sh lets the owning window through).
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpuguard  # noqa: E402
import unitrun  # noqa: E402

HOME = os.path.expanduser("~")
BASE = os.environ.get("WINDOWCTL_BASE", f"{HOME}/.local/share/vllm-qwen27b")
OVERRIDE = os.environ.get("V02_OVERRIDE_ENV", f"{BASE}/v02.override.env")
ACTUATOR = os.environ.get("ENGINE_ACTUATOR", f"{BASE}/engine-actuator.py")
SPEND_FILE = os.environ.get("GATEWAY_SPEND_FILE", f"{BASE}/gateway-spend.json")
PLANNED = f"{BASE}/planned-restart.json"
LOCK = os.environ.get("WINDOWCTL_LOCK", f"{BASE}/window.lock")
MARKER = os.environ.get("WINDOWCTL_MARKER", f"{BASE}/window-active.json")
GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8000")
ENGINE = os.environ.get("ENGINE_URL", "http://127.0.0.1:8001")
ENGINE_UNIT = os.environ.get("ENGINE_UNIT", "vllm-qwen27b")
LEGACY_ROOT = f"{HOME}/Desktop/wt-integrate"
RELEASES = os.environ.get("VLLM_RELEASES_ROOT", f"{HOME}/.local/share/vllm-releases")
JIT_DIRS = [".deps/tq_gqa_build", ".deps/FlashQLA-SM70-SM75/.torch_extensions_vllm_flashqla_legacy"]
DEFAULT_WATCH_TIMERS = ["vllm-qwen27b-watchdog.timer"]
VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
STEP_KINDS = ("run", "script", "boot", "engine", "sleep")
WATCHDOG_TIMER = "vllm-qwen27b-watchdog.timer"   # its service Wants= the engine: a tick STARTS a stopped engine
CODE_FILES = ("windowctl.py", "unitrun.py", "gpuguard.py", "release.py", "release_ab_probe.py", "gateway-offline.py", "mini_yaml.py")
BUILTINS = ("RESULTS", "WINDOW", "LANE", "STEP", "PROD_ROOT")


class SpecError(Exception):
    pass


class WindowAbort(Exception):
    def __init__(self, status, why):
        super().__init__(why)
        self.status, self.why = status, why


class Terminated(BaseException):
    def __init__(self, signum):
        super().__init__(f"signal {signum}")
        self.signum = signum


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gateway_offline_mod():
    return _load_module("gateway_offline", os.path.join(HERE, "gateway-offline.py"))


def release_mod():
    import release
    return release


def actuator_hold(*args, timeout=60):
    """Lane LV's engine-liveness hold (`engine-actuator.py hold ...`). While a window holds it the liveness authority
    and the watchdog take no automatic action, and stops are attributed to the holder. Returns the parsed JSON, or
    None when the deployed actuator has no `hold` subcommand yet (then the window runs as before)."""
    try:
        r = subprocess.run(["/usr/bin/python3", ACTUATOR, "hold", *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode == 2 and "invalid choice" in (r.stderr or ""):
        return None
    try:
        return json.loads((r.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": (r.stderr or r.stdout or "")[-300:], "rc": r.returncode}


def restart_argv(by, reason, hold=None):
    argv = ["/usr/bin/python3", ACTUATOR, "restart", "--by", str(by), "--reason", reason, "--no-drain", "--foreground"]
    if hold:
        argv += ["--hold", hold]          # LV: a planned restart is refused while a hold is held unless it names it
    return argv


REMOTE_TROUBLE = ("remote provider refused", "breaker open", "(402)", "(429)", "remote dead")


def remote_health(quiet_s: float = 0.0, now=None):
    """Is the remote valve able to carry the estate while local is offline? (2026-10-03 09:41: DeepSeek 402 = balance
    exhausted, breaker open, while K6's window had local stopped -> gateway mode "none", nothing could serve.)
    Healthy = capacity readable, remote configured + usable + inside budget, not dead, mode != none, and (quiet_s > 0)
    no breaker-open / 402 / 429 mode change within the last quiet_s seconds. Returns (ok, why)."""
    cap = http_json(f"{GATEWAY}/gateway/capacity")
    if not isinstance(cap, dict) or cap.get("error"):
        return False, f"gateway capacity unreadable: {str((cap or {}).get('error'))[:120]}"
    why = []
    if not cap.get("remote_configured"):
        why.append("remote not configured")
    if not cap.get("remote_usable"):
        why.append("remote not usable" + (f" ({'; '.join(cap.get('why') or [])[:160]})" if cap.get("why") else ""))
    if cap.get("remote_balance_exhausted"):
        why.append("remote provider balance exhausted (402)")
    if cap.get("remote_budget_ok") is False:
        why.append("remote budget exhausted")
    if cap.get("remote_dead_for_s"):
        why.append(f"remote dead for {cap['remote_dead_for_s']} s")
    if cap.get("mode") == "none":
        why.append("gateway mode is none (nothing can serve)")
    if quiet_s:
        now = time.time() if now is None else now
        for ch in cap.get("mode_changes") or []:
            if now - float(ch.get("t") or 0) <= quiet_s and any(k in r for r in ch.get("why") or [] for k in REMOTE_TROUBLE):
                why.append(f"remote trouble {int(now - float(ch['t']))} s ago: {'; '.join(ch.get('why'))[:120]}")
                break
    return not why, "; ".join(why)


def admin_token():
    try:
        return open(f"{BASE}/admin.token").read().strip()
    except OSError:
        return ""


def http_json(url, method="GET", payload=None, timeout=8):
    req = urllib.request.Request(url, method=method, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "X-Admin-Token": admin_token()})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode("utf-8", "replace")[:300], "http": e.code}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:200]}


def engine_health(timeout=3):
    try:
        with urllib.request.urlopen(f"{ENGINE}/health", timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:  # noqa: BLE001
        return None


def wait_health(budget_s, poll_s=4.0, clock=time.time, sleep=time.sleep):
    t0 = clock()
    while clock() - t0 < budget_s:
        if engine_health() == 200:
            return True, round(clock() - t0, 1)
        sleep(poll_s)
    return engine_health() == 200, round(clock() - t0, 1)


def spend_usd():
    try:
        return float(json.load(open(SPEND_FILE)).get("spent") or 0.0)
    except (OSError, ValueError, TypeError):
        return None


def xid_lines():
    out = subprocess.run(["journalctl", "-k", "--no-pager", "-o", "short-unix"], capture_output=True, text=True, timeout=60).stdout
    return [line for line in out.splitlines() if "NVRM: Xid" in line]


def attribute_xid(line):
    m = re.search(r"pid=(\d+)", line)
    ts = None
    try:
        ts = float(line.split()[0])
    except (ValueError, IndexError):
        pass
    if not m:
        return {"line": line[:240], "pid": None, "owner": None}
    pid = int(m.group(1))
    own = unitrun.owner_of_pid(pid, ts)
    return {"line": line[:240], "pid": pid, "owner": own, "engine": bool(own and own.get("unit") == f"{ENGINE_UNIT}.service")}


def timer_state(timer):
    a = subprocess.run(["systemctl", "is-active", timer], capture_output=True, text=True).stdout.strip()
    return a or "unknown"


def set_timer(timer, active: bool):
    verb = "start" if active else "stop"
    r = subprocess.run(["sudo", "-n", "systemctl", verb, timer], capture_output=True, text=True, timeout=60)
    return r.returncode == 0


def running_root():
    r = subprocess.run(["systemctl", "show", "-p", "MainPID", "--value", ENGINE_UNIT], capture_output=True, text=True)
    try:
        pid = int(r.stdout.strip() or 0)
        return os.path.realpath(os.readlink(f"/proc/{pid}/cwd")) if pid else None
    except (OSError, ValueError):
        return None


def so_hashes(root):
    out = {}
    if not root or not os.path.isdir(root):
        return out
    for sub in ("vllm", ".deps"):
        base = os.path.join(root, sub)
        for d, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs if x not in ("__pycache__",) and not os.path.islink(os.path.join(d, x))]
            for f in files:
                if f.endswith(".so"):
                    p = os.path.join(d, f)
                    if os.path.islink(p):
                        continue
                    h = hashlib.sha256()
                    with open(p, "rb") as fh:
                        for c in iter(lambda: fh.read(1 << 20), b""):
                            h.update(c)
                    out[os.path.relpath(p, root)] = h.hexdigest()
    return out


def override_value(override_bytes: bytes, var: str) -> str:
    """The value a variable ends up with after sourcing the override (evaluated by bash, like the serve script does)."""
    r = subprocess.run(["bash", "-c", f'set -a; eval "$1" >/dev/null 2>&1; printf %s "${{{var}:-}}"', "_",
                        override_bytes.decode("utf-8", "replace")], capture_output=True, text=True, timeout=10,
                       env={"PATH": "/usr/bin:/bin"})
    return r.stdout


def find_conflicts(patterns, proc="/proc"):
    """Processes whose argv contains a pattern -- never this process, its ancestors or other windowctl processes
    (the `pkill -f` self-match class)."""
    if not patterns:
        return []
    mine = set()
    pid = os.getpid()
    while pid and pid not in mine:
        mine.add(pid)
        try:
            with open(f"{proc}/{pid}/stat") as fh:
                pid = int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    hits = []
    for name in os.listdir(proc):
        if not name.isdigit() or int(name) in mine:
            continue
        try:
            with open(f"{proc}/{name}/cmdline", "rb") as fh:
                argv = [a.decode("utf-8", "replace") for a in fh.read().split(b"\0") if a]
        except OSError:
            continue
        if not argv or any("windowctl.py" in a for a in argv[:3]):
            continue
        for pat in patterns:
            if any(pat in a for a in argv):
                hits.append({"pid": int(name), "pattern": pat, "cmd": " ".join(argv)[:200],
                             "unit": unitrun.unit_of_cgroup(unitrun.proc_cgroup(int(name)) or "")})
                break
    return hits


# ---------------------------------------------------------------- spec

def load_spec(path):
    text = open(path).read()
    if path.endswith((".yaml", ".yml")):
        # always the dependency-free subset parser (the test gate has no PyYAML; specs must parse the same everywhere)
        import mini_yaml
        try:
            spec = mini_yaml.load(text)
        except mini_yaml.MiniYAMLError as e:
            raise SpecError(f"{path}: {e}")
    else:
        spec = json.loads(text)
    if not isinstance(spec, dict):
        raise SpecError("spec must be a mapping")
    return spec


def pause_timers_of(spec):
    """pause_timers + (by default) the engine watchdog timer whenever a step stops the engine: WQ 2026-10-03 08:11:58,
    a watchdog tick (vllm-qwen27b-watchdog.service Wants=vllm-qwen27b.service) re-started the engine 52 s after K3's
    window stopped it, and the microbench ran beside a booting engine. Opt out: keep_watchdog_timer: true."""
    out = list(spec.get("pause_timers") or [])
    stops = any(isinstance(st, dict) and st.get("engine") == "stop" for st in spec.get("steps") or [])
    if stops and not spec.get("keep_watchdog_timer") and WATCHDOG_TIMER not in out:
        out.append(WATCHDOG_TIMER)
    return out


def _names_in(obj):
    if isinstance(obj, str):
        return set(VAR_RE.findall(obj))
    if isinstance(obj, dict):
        return set().union(*[_names_in(v) for v in obj.values()]) if obj else set()
    if isinstance(obj, list):
        return set().union(*[_names_in(v) for v in obj]) if obj else set()
    return set()


def validate(spec):
    errs = []
    for k in ("lane", "window", "reason", "steps"):
        if not spec.get(k):
            errs.append(f"missing '{k}'")
    if spec.get("reason") and len(str(spec["reason"]).strip()) < 8:
        errs.append("reason must say what the window is for (>= 8 chars)")
    for k in ("max_s", "ttl_s", "wait_s", "gpu_gate_wait_s"):
        if k in spec and not (isinstance(spec[k], (int, float)) and spec[k] >= 0):
            errs.append(f"{k} must be a non-negative number")
    if spec.get("ttl_s") and not 30 <= spec["ttl_s"] <= 3600:
        errs.append("ttl_s must be 30..3600 (the gateway's limit; the framework renews it)")
    if spec.get("on_new_xid", "abort") not in ("abort", "continue"):
        errs.append("on_new_xid must be abort|continue")
    known = set(BUILTINS) | set((spec.get("vars") or {}).keys())
    known |= {release_var(st["name"]) for st in spec.get("steps") or [] if isinstance(st, dict) and st.get("name")
              and isinstance(st.get("boot"), dict) and "release" in st["boot"]}
    seen = set()
    for i, st in enumerate(spec.get("steps") or []):
        if not isinstance(st, dict):
            errs.append(f"step {i} is not a mapping")
            continue
        name = st.get("name")
        if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", str(name)):
            errs.append(f"step {i}: name must be [A-Za-z0-9_.-]+")
        if name in seen:
            errs.append(f"step {i}: duplicate name {name}")
        seen.add(name)
        kinds = [k for k in STEP_KINDS if k in st]
        if len(kinds) != 1:
            errs.append(f"step {name}: exactly one of {STEP_KINDS} (got {kinds})")
            continue
        if "engine" in st and st["engine"] not in ("stop", "start"):
            errs.append(f"step {name}: engine must be stop|start")
        if "boot" in st:
            b = st["boot"]
            if not isinstance(b, dict) or len([k for k in ("release", "tree", "default") if k in b]) != 1:
                errs.append(f"step {name}: boot needs exactly one of release / tree / default")
        if st.get("on_fail", "abort") not in ("abort", "continue"):
            errs.append(f"step {name}: on_fail must be abort|continue")
        used = _names_in({k: v for k, v in st.items() if k != "capture"})
        for u in sorted(used - known):
            errs.append(f"step {name}: unknown variable {{{{{u}}}}} (not in vars, builtins or an EARLIER capture)")
        for var in (st.get("capture") or {}):
            known.add(var)
    if errs:
        raise SpecError("; ".join(errs))
    return True


def release_var(step_name):
    """{{RELEASE_<step>}}: the resolved release directory of a boot step (resolved/built before the window opens)."""
    return "RELEASE_" + re.sub(r"[^A-Za-z0-9_]", "_", str(step_name))


def render(obj, env):
    if isinstance(obj, str):
        def sub(m):
            if m.group(1) not in env:
                raise SpecError(f"unknown variable {{{{{m.group(1)}}}}}")
            return str(env[m.group(1)])
        return VAR_RE.sub(sub, obj)
    if isinstance(obj, dict):
        return {k: render(v, env) for k, v in obj.items()}
    if isinstance(obj, list):
        return [render(v, env) for v in obj]
    return obj


def truthy(expr: str) -> bool:
    """`A == B`, `A != B`, or a bare value (true/1/yes/on = true). No eval."""
    expr = str(expr).strip()
    for op in ("==", "!="):
        if op in expr:
            a, b = (x.strip().strip("'\"") for x in expr.split(op, 1))
            return (a == b) if op == "==" else (a != b)
    return expr.lower() in ("true", "1", "yes", "on")


def do_capture(spec_cap, stdout_path):
    out = {}
    for var, how in (spec_cap or {}).items():
        val = None
        try:
            if "json" in how:
                d = json.load(open(how["json"]))
                for k in str(how.get("key", "")).split(".") if how.get("key") else []:
                    d = d[int(k)] if isinstance(d, list) else d[k]
                val = d
            elif "regex" in how:
                text = open(stdout_path, errors="replace").read() if stdout_path and os.path.exists(stdout_path) else ""
                ms = re.findall(how["regex"], text, re.M)
                if ms:
                    val = ms[-1] if isinstance(ms[-1], str) else ms[-1][0]
            elif "cmd" in how:
                r = subprocess.run(["bash", "-c", how["cmd"]], capture_output=True, text=True, timeout=int(how.get("timeout_s", 60)))
                if r.returncode == 0 and r.stdout.strip():
                    val = r.stdout.strip().splitlines()[-1]
        except Exception:  # noqa: BLE001
            val = None
        out[var] = str(how.get("default", "")) if val is None else str(val)
    return out


# ---------------------------------------------------------------- the window

class Window:
    def __init__(self, spec, spec_path=None, dry_run=False, results=None, preset=None):
        validate(spec)
        self.spec, self.spec_path, self.dry = spec, spec_path, dry_run
        self.lane, self.wid = str(spec["lane"]), str(spec["window"])
        self.by = str(spec.get("by") or self.lane.upper())
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.results = results or spec.get("results") or f"{HOME}/projects/lanes/{self.lane}/windows/{self.wid}-{ts}"
        self.results = os.path.abspath(os.path.expanduser(self.results))
        self.max_s = float(spec.get("max_s", 3600))
        self.t0 = time.time()
        self.state = {"window": self.wid, "lane": self.lane, "by": self.by, "results": self.results, "pid": os.getpid(),
                      "pid_start": unitrun.proc_start(os.getpid()), "lease": None, "boots": 0, "engine_stopped": False,
                      "deadman_unit": None, "restored": False, "started": now_iso(), "deadline": self.t0 + self.max_s}
        self.steps = []
        self.vars = dict(preset or {})       # --set VAR=VAL (dry runs: exercise a branch without running the step)
        self.preset = set((preset or {}).keys())
        self.snapshot = {}
        self.summary = {"window": self.wid, "lane": self.lane, "by": self.by, "reason": spec["reason"],
                        "results": self.results, "spec": spec_path, "started": now_iso(), "status": "running",
                        "steps": self.steps, "boots": [], "xids": [], "notes": []}
        self._lease_stop = threading.Event()
        self.renew_s = 60.0
        self._remote_stop = threading.Event()
        self.remote_poll_s = 15.0
        self._remote_strikes = 0
        self.remote_bad = None       # gateway TTL is capped at 3600 s: windows longer than that live on these renewals
        self._xid_count = None

    # -- bookkeeping
    def path(self, *p):
        return os.path.join(self.results, *p)

    def log(self, msg):
        line = f"{now_iso()} {msg}"
        print(line, flush=True)
        try:
            with open(self.path("window.log"), "a") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    def save_state(self):
        if self.dry:
            return
        tmp = self.path("state.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(self.state, fh, indent=1)
        os.replace(tmp, self.path("state.json"))

    def save_summary(self):
        with open(self.path("summary.json"), "w") as fh:
            json.dump(self.summary, fh, indent=1, default=str)

    # -- phases
    def resolve_release(self, ref):
        """A boot's `release:` -> a concrete release directory (built here, BEFORE the window, when missing)."""
        rel = release_mod()
        if isinstance(ref, dict):
            if ref.get("tree_head"):          # "whatever that tree's HEAD is right now" (e.g. the legacy prod tree)
                ref = {**ref, "sha": subprocess.run(["git", "-C", ref["tree_head"], "rev-parse", "HEAD"], capture_output=True,
                                                    text=True).stdout.strip(), "from": ref.get("from", ref["tree_head"])}
            sha = subprocess.run(["git", "-C", rel.REPO, "rev-parse", f"{ref.get('sha')}^{{commit}}"], capture_output=True,
                                 text=True).stdout.strip() if ref.get("sha") else ""
            if not sha:
                raise SpecError(f"release sha {ref.get('sha')} not found in {rel.REPO}")
            for row in rel.list_releases():
                m = rel.manifest_of(row["id"])
                if m["sha"] == sha and (ref.get("label") in (None, m.get("label"))):
                    return os.path.join(rel.ROOT, row["id"])
            if self.dry:
                return f"<to be built: {ref.get('label') or ''} {sha[:10]}>"
            self.log(f"building release {ref.get('label')} {sha[:10]} (before the window opens)")
            m = rel.build(sha, label=ref.get("label"), from_tree=ref.get("from", rel.DEFAULT_FROM), by=self.by,
                          log=self.log)
            return os.path.join(rel.ROOT, m["id"])
        if str(ref).startswith("<to be built"):
            return str(ref)
        if ref in ("current",):
            return rel.release_dir("current")
        if os.path.isabs(str(ref)):
            return str(ref)
        return rel.release_dir(str(ref))

    def preflight(self):
        spec = self.spec
        problems = []
        ok, why = remote_health(float(spec.get("remote_quiet_s", 900)))
        if not ok and not spec.get("allow_no_remote"):
            problems.append(f"remote valve unhealthy, refusing to take local offline: {why}")
        hits = find_conflicts(spec.get("conflicts") or [])
        if hits:
            problems.append("conflicting processes running: " + "; ".join(f"pid {h['pid']} ({h['pattern']}) unit={h['unit']}" for h in hits))
        if engine_health() != 200 and not spec.get("allow_unhealthy_start"):
            problems.append("engine is not healthy at window start (restore target unknown)")
        sp = spend_usd()
        if sp is not None and sp >= float(spec.get("spend_pause_usd", 20)):
            problems.append(f"spend ${sp:.2f} >= pause threshold ${spec.get('spend_pause_usd', 20)}")
        cap = http_json(f"{GATEWAY}/gateway/capacity")
        if cap.get("remote_balance_exhausted") and not spec.get("allow_no_remote"):
            # GW2/L172: with the provider balance empty the remote valve is gone; an engine window would leave the
            # estate with no serving path at all. Kevin tops up; the gateway's balance probe re-enables remote.
            problems.append("remote provider balance exhausted (402): an engine window would leave no serving path")
        off = http_json(f"{GATEWAY}/gateway/offline")
        if off.get("offline"):
            problems.append(f"another gateway offline window is open (by {off.get('by')}: {off.get('reason')})")
        busy = gpuguard.busy_state()
        if busy.get("busy") and busy.get("window") != self.wid:
            problems.append(f"GPU busy signal held by {busy.get('by')} ({busy.get('reason')})")
        return problems

    def take_snapshot(self):
        try:
            ov = open(OVERRIDE, "rb").read()
        except OSError:
            ov = None
        root = running_root() or LEGACY_ROOT
        snap = {
            "taken": now_iso(), "override_exists": ov is not None, "override_b64": (ov or b"").hex(),
            "override_sha256": hashlib.sha256(ov or b"").hexdigest(), "override_bytes": len(ov or b""),
            "timers": {t: timer_state(t) for t in (self.spec.get("watch_timers") or DEFAULT_WATCH_TIMERS) + pause_timers_of(self.spec)},
            "prod_root": root, "so": so_hashes(root), "health": engine_health(),
            "kv_pool": gpuguard.kv_pool_since(), "spend": spend_usd(),
            "xid_count": len(xid_lines()), "jit_backup": {},
            "files": {},
        }
        for f in self.spec.get("snapshot_files") or []:
            f = os.path.expanduser(f)
            try:
                snap["files"][f] = open(f, "rb").read().hex()
            except OSError:
                snap["files"][f] = None
        for rel in JIT_DIRS:
            src = os.path.join(root, rel)
            if os.path.isdir(src) and not os.path.islink(src):
                dst = self.path("snapshot", "jit", rel.replace("/", "_"))
                if not self.dry:
                    shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)
                snap["jit_backup"][rel] = dst
        for t, st in snap["timers"].items():
            if st != "active":
                self.summary["notes"].append(f"timer {t} was already {st} at window start (restored to that state)")
        self.snapshot = snap
        self._xid_count = snap["xid_count"]
        if not self.dry:
            with open(self.path("snapshot.json"), "w") as fh:
                json.dump(snap, fh, indent=1)

    def open_gateway(self):
        go = gateway_offline_mod()
        st, lease = go.open_window(f"{self.wid}: {self.spec['reason']}"[:120], self.by, int(self.spec.get("ttl_s", 1800)),
                                   int(self.spec.get("wait_s", 90)), mode="run")
        if not lease:
            raise WindowAbort("refused", f"gateway offline window not opened: {json.dumps(st)[:300]}")
        self.state["lease"] = lease
        self.save_state()
        self.log(f"gateway offline window open (local_active={st.get('local_active')})")
        threading.Thread(target=self._renew_loop, daemon=True).start()

    def _renew_loop(self):
        ttl = int(self.spec.get("ttl_s", 1800))
        while not self._lease_stop.wait(self.renew_s):
            lease = self.state.get("lease")
            if not lease:
                return
            self.renew_once(lease, ttl)

    def renew_once(self, lease, ttl):
        """Extend our window. The offline window lives only in the shim's memory: a gateway restart mid-window (seen
        2026-10-03 09:13:22, during K5's window) silently forgets it, and then the POST OPENS A NEW window with a new
        lease. Adopt that lease (record it for close/restore/orphan reaping) so the window is never left unfenced."""
        r = http_json(f"{GATEWAY}/gateway/offline", "POST", {"lease": lease, "ttl_s": ttl, "by": self.by,
                                                             "reason": f"{self.wid}: {self.spec['reason']}"[:120]})
        if r.get("error"):
            self.log(f"lease renew failed: {r.get('error')[:120]}")
            return r
        new = r.get("lease")
        if new and new != lease:
            self.state["lease"] = new
            self.save_state()
            try:
                gateway_offline_mod().save_lease(new, self.by, self.spec["reason"][:120], ttl, mode="run")
            except Exception:  # noqa: BLE001
                pass
            self.summary["notes"].append(f"{now_iso()} gateway forgot the offline window (restart?); re-opened it")
            self.log("gateway had forgotten the offline window (gateway restart?): re-opened it with a new lease")
        return r

    def take_hold(self):
        ttl = int(min(28800, max(60, self.max_s + 1800)))
        h = actuator_hold("acquire", "--kind", "engine", "--by", "windowctl", "--reason", f"{self.wid}: {self.spec['reason']}"[:120],
                          "--ttl", str(ttl), "--owner-pid", str(os.getpid()))
        if h is None:
            self.summary["notes"].append("engine-actuator has no `hold` yet (LV not deployed): window runs without a liveness hold")
            return
        if not h.get("lease"):
            raise WindowAbort("refused", f"engine liveness hold not granted: {json.dumps(h)[:300]}")
        self.state["hold"] = h["lease"]
        self.save_state()
        self.log(f"engine liveness hold {h['lease'][:8]}.. until {h.get('until')}")

    def arm_deadman(self):
        unit = unitrun.unit_name(self.lane, f"{self.wid}-deadman")
        r = subprocess.run(["systemd-run", "--user", f"--unit={unit}", "--collect", "--quiet",
                            "--on-active=120", "--on-unit-active=120", "--timer-property=AccuracySec=10s",
                            sys.executable, os.path.abspath(__file__), "deadman", "--state", self.path("state.json")],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            self.state["deadman_unit"] = unit
            self.save_state()
        else:
            self.summary["notes"].append(f"dead-man timer NOT armed: {r.stderr.strip()[:200]}")

    def remote_tick(self):
        """One poll of the remote watch: two consecutive unhealthy polls (~30 s) = abort. The running step's unit is
        stopped at once so the window reaches its step boundary and restores local now, not after a 20-minute bench."""
        if self.remote_bad or self.spec.get("allow_no_remote"):
            return
        ok, why = remote_health(0)
        if ok:
            self._remote_strikes = 0
            return
        self._remote_strikes += 1
        self.log(f"remote valve unhealthy ({self._remote_strikes}/2): {why}")
        if self._remote_strikes >= 2:
            self.remote_bad = why
            self.log("remote valve down while local is offline: stopping the running step, restoring local")
            unitrun.stop_lane(self.lane, prefix=self.wid,
                              exclude=(unitrun.unit_name(self.lane, f"{self.wid}-deadman") + ".service",))

    def _remote_watch_loop(self):
        while not self._remote_stop.wait(self.remote_poll_s):
            try:
                self.remote_tick()
            except Exception as e:  # noqa: BLE001
                self.log(f"remote watch error: {e!r}")

    def check_between_steps(self):
        if self.remote_bad:
            raise WindowAbort("aborted-remote", f"remote valve unhealthy while local was offline: {self.remote_bad}")
        ok, why = (True, "") if self.spec.get("allow_no_remote") else remote_health(0)
        if not ok:
            raise WindowAbort("aborted-remote", f"remote valve unhealthy while local was offline: {why}")
        if time.time() - self.t0 > self.max_s:
            raise WindowAbort("budget", f"window budget max_s={self.max_s:.0f} spent")
        sp = spend_usd()
        if sp is not None and sp >= float(self.spec.get("spend_pause_usd", 20)):
            raise WindowAbort("paused-spend", f"gateway spend ${sp:.2f} reached the pause threshold ${self.spec.get('spend_pause_usd', 20)}")
        lines = xid_lines()
        if self._xid_count is not None and len(lines) > self._xid_count:
            new = [attribute_xid(x) for x in lines[self._xid_count:]]
            self._xid_count = len(lines)
            self.summary["xids"].extend(new)
            self.log(f"NEW Xid(s): {json.dumps(new)[:400]}")
            if self.spec.get("on_new_xid", "abort") == "abort":
                raise WindowAbort("aborted-xid", f"{len(new)} new Xid(s); first: {new[0]['line'][:160]}")

    def env_for(self, step_name):
        return {**{k: str(v) for k, v in (self.spec.get("vars") or {}).items()}, **self.vars, "RESULTS": self.results,
                "WINDOW": self.wid, "LANE": self.lane, "STEP": step_name, "PROD_ROOT": self.snapshot.get("prod_root", "")}

    def gpu_gate(self, what):
        wait = float(self.spec.get("gpu_gate_wait_s", 300))
        ok, foreign = gpuguard.wait_no_foreign(wait)
        if not ok:
            raise WindowAbort("gpu-foreign", f"{what}: foreign compute apps on the GPUs after {wait:.0f} s "
                                             f"(would shrink the KV pool): {gpuguard.describe(foreign)}")

    def restart_engine(self, label, timeout_s):
        """Drained restart through the existing engine actuator, in its own unit. The window already holds the
        gateway offline window, so the actuator does not open another one (--no-drain)."""
        self.gpu_gate(f"boot {label}")
        gpuguard.set_busy(self.by, f"{self.wid} boot {label}", ttl_s=timeout_s + 300, window=self.wid, phase="boot")
        t_boot = time.time()
        res = unitrun.run(self.lane, f"{self.wid}-boot-{label}",
                          restart_argv(self.by, f"window {self.wid} boot {label}", self.state.get("hold")),
                          timeout_s=timeout_s, out=self.path("steps", f"boot-{label}.log"))
        ok, waited = wait_health(max(30.0, timeout_s - (time.time() - t_boot)))
        gpuguard.set_busy(self.by, f"{self.wid} window", ttl_s=self.max_s + 1800, window=self.wid, phase="window")
        boot = {"label": label, "unit": res.get("unit"), "actuator_rc": res.get("rc"), "healthy": ok,
                "boot_s": round(time.time() - t_boot, 1), "kv_pool": gpuguard.kv_pool_since(t_boot - 5),
                "running_root": running_root()}
        self.summary["boots"].append(boot)
        self.log(f"boot {label}: healthy={ok} in {boot['boot_s']} s, KV pool {boot['kv_pool']}, root {boot['running_root']}")
        return boot

    def write_override(self, extra_lines):
        base = bytes.fromhex(self.snapshot["override_b64"])
        body = base + (b"" if not base or base.endswith(b"\n") else b"\n") + "".join(extra_lines).encode()
        tmp = OVERRIDE + f".tmp{os.getpid()}"
        with open(tmp, "wb") as fh:
            fh.write(body)
        os.replace(tmp, OVERRIDE)
        return body

    def step_boot(self, st, rendered):
        b = rendered["boot"]
        label = b.get("label") or st["name"]
        lines = [f"# windowctl {self.wid} arm {label} {now_iso()}\n"]
        if "release" in b:
            root = self.resolve_release(b["release"])
            lines.append(f"export V02_ROOT={shlex.quote(root)}\n")
            try:   # lane JIT build dirs prebuilt inside the release (release.py --jit-ext): point the boot at them
                for k, v in (json.load(open(os.path.join(root, "RELEASE.json"))).get("boot_env") or {}).items():
                    if k not in (b.get("env") or {}):
                        lines.append(f"export {k}={shlex.quote(v)}\n")
            except (OSError, ValueError):
                pass
        elif "tree" in b:
            root = os.path.realpath(b["tree"])
            for rel in JIT_DIRS[:1] + [".deps/FlashQLA-SM70-SM75"]:
                p = os.path.join(b["tree"], rel)
                if os.path.islink(p) and not b.get("allow_shared_jit_dirs"):
                    raise WindowAbort("refused", f"tree {b['tree']}: {rel} is a symlink into {os.path.realpath(p)}; a boot "
                                                 f"would JIT-rebuild .so files in that tree. Build a release "
                                                 f"(release.py build <sha> --label {self.lane}) and boot it instead.")
            lines.append(f"export V02_ROOT={shlex.quote(root)}\n")
        for k, v in (b.get("env") or {}).items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
                raise SpecError(f"bad env name {k}")
            lines.append(f"export {k}={shlex.quote(str(v))}\n")
        if b.get("extra_args_append"):
            cur = override_value(bytes.fromhex(self.snapshot["override_b64"]), "VLLM_SERVE_EXTRA_ARGS")
            if "VLLM_SERVE_EXTRA_ARGS" in (b.get("env") or {}):
                cur = str(b["env"]["VLLM_SERVE_EXTRA_ARGS"])
            lines.append(f"export VLLM_SERVE_EXTRA_ARGS={shlex.quote((cur + ' ' + b['extra_args_append']).strip())}\n")
        if self.dry:
            return {"rc": 0, "dry": True, "override_lines": lines}
        self.write_override(lines)
        self.state["boots"] += 1
        self.state["engine_stopped"] = False
        self.save_state()
        boot = self.restart_engine(label, int(b.get("timeout_s", 1200)))
        boot["override_lines"] = lines
        if not boot["healthy"]:
            return {"rc": 1, "error": "boot not healthy", "boot": boot}
        return {"rc": 0, "boot": boot}

    def step_engine(self, st):
        if self.dry:
            return {"rc": 0, "dry": True}
        if st["engine"] == "stop":
            json.dump({"ts": now_iso(), "by": self.by, "reason": f"window {self.wid} step {st['name']}: engine stop"},
                      open(PLANNED, "w"))
            self.state["engine_stopped"] = True
            self.save_state()
            res = unitrun.run(self.lane, f"{self.wid}-{st['name']}", ["sudo", "-n", "systemctl", "stop", ENGINE_UNIT],
                              timeout_s=240, out=self.path("steps", f"{st['name']}.log"))
            return {"rc": res.get("rc"), "unit": res.get("unit")}
        self.gpu_gate(f"engine start ({st['name']})")
        res = unitrun.run(self.lane, f"{self.wid}-{st['name']}", ["sudo", "-n", "systemctl", "start", ENGINE_UNIT],
                          timeout_s=900, out=self.path("steps", f"{st['name']}.log"))
        ok, _ = wait_health(float(st.get("timeout_s", 900)))
        self.state["engine_stopped"] = not ok
        self.save_state()
        return {"rc": 0 if ok else 1, "unit": res.get("unit"), "kv_pool": gpuguard.kv_pool_since(time.time() - 900)}

    def snapshot_scripts(self, st, cmd):
        """snaprun for `run:` too: every existing *.sh file the command names is copied to a READ-ONLY snapshot in the
        results dir and the command is rewritten to run the snapshot (a lane editing its script mid-window can no longer
        make bash resume mid-line, which skipped K3's restore at 08:36). Returns (cmd, {original: snapshot})."""
        if isinstance(cmd, str):
            # only scripts in EXECUTION position: at a command start, optionally after bash/sh/source/./exec/nohup/timeout N
            # (never a redirect target or a cp/install destination)
            toks = [m.group("path") for m in re.finditer(
                r"(?:^|[;&|\n(]\s*)(?:(?:bash|sh|source|\.|exec|nohup|timeout\s+\S+)\s+)*(?P<path>/[^\s'\";|&<>()]+\.sh)\b",
                cmd, re.M)]
        else:
            argv0 = [str(x) for x in cmd]
            toks = [argv0[1]] if len(argv0) > 1 and os.path.basename(argv0[0]) in ("bash", "sh") else argv0[:1]
            toks = [t for t in toks if t.endswith(".sh")]
        found = {}
        for tok in toks:
            if os.path.isfile(tok) and tok not in found and not tok.startswith(self.results + os.sep):
                snap = self.path("steps", f"{st['name']}.{len(found)}.{os.path.basename(tok)[:-3]}.snap.sh")
                if not self.dry:
                    shutil.copy2(tok, snap)
                    os.chmod(snap, 0o444)
                found[tok] = snap
        if not found:
            return cmd, {}
        def swap(x):
            for k, v in found.items():
                x = re.sub(r"(?<![\w./-])" + re.escape(k) + r"\b", v, x)
            return x
        return (swap(cmd) if isinstance(cmd, str) else [swap(str(x)) for x in cmd]), found

    def step_run(self, st, rendered):
        cmd, snaps = self.snapshot_scripts(st, rendered["run"])
        argv = ["bash", "-c", cmd] if isinstance(cmd, str) else [str(x) for x in cmd]
        env = {k: str(v) for k, v in (rendered.get("env") or {}).items()}
        env.update(WINDOW_ID=self.wid, WINDOW_RESULTS=self.results, WINDOW_LANE=self.lane)
        out = self.path("steps", f"{st['name']}.log")
        if self.dry:
            return {"rc": 0, "dry": True, "argv": argv, "env": env}
        res = unitrun.run(self.lane, f"{self.wid}-{st['name']}", argv, timeout_s=int(st.get("timeout_s", 1800)), env=env,
                          cwd=rendered.get("cwd"), out=out)
        missing = [p for p in (rendered.get("outputs") or []) if not os.path.exists(p)]
        res["missing_outputs"] = missing
        if snaps:
            res["script_snapshots"] = snaps
        if missing and res.get("rc") == 0:
            res["rc"] = 3
            res["error"] = f"declared outputs missing: {missing}"
        return res

    def step_script(self, st, rendered):
        """`script: [path, args...]`: the lane script runs from a READ-ONLY snapshot taken now (WQ's snaprun rule: bash
        reads a running script incrementally, so editing it mid-run made K3's cleanup resume mid-line and skip restore)."""
        argv = rendered["script"] if isinstance(rendered["script"], list) else shlex.split(rendered["script"])
        src = os.path.realpath(argv[0])
        snap = self.path("steps", f"{st['name']}.snap{os.path.splitext(src)[1] or '.sh'}")
        if not self.dry:
            shutil.copy2(src, snap)
            os.chmod(snap, 0o444)
        st2 = {**st, "run": ["bash", snap, *argv[1:]] if not src.endswith(".py") else ["python3", snap, *argv[1:]]}
        res = self.step_run(st2, {**rendered, "run": st2["run"]})
        res["script_sha256"] = hashlib.sha256(open(src, "rb").read()).hexdigest()[:16] if os.path.exists(src) else None
        return res

    def run_steps(self):
        for st in self.spec["steps"]:
            name = st["name"]
            self.check_between_steps()
            env = self.env_for(name)
            row = {"name": name, "kind": next(k for k in STEP_KINDS if k in st), "started": now_iso()}
            self.steps.append(row)
            if "when" in st:
                cond = render(st["when"], env)
                if not truthy(cond):
                    row.update(skipped=True, when=cond)
                    self.log(f"step {name}: skipped (when: {cond})")
                    continue
            rendered = render({k: v for k, v in st.items() if k not in ("capture",)}, env)
            if row["kind"] != "boot" and not self.dry and not self.state["engine_stopped"] and engine_health() != 200:
                ok, _ = wait_health(300)
                if not ok:
                    raise WindowAbort("engine-down", f"engine unhealthy before step {name}")
            self.log(f"step {name} ({row['kind']}) start")
            t = time.time()
            if row["kind"] == "run":
                res = self.step_run(st, rendered)
            elif row["kind"] == "script":
                res = self.step_script(st, rendered)
            elif row["kind"] == "boot":
                res = self.step_boot(st, rendered)
            elif row["kind"] == "engine":
                res = self.step_engine(st)
            else:
                if not self.dry:
                    time.sleep(float(rendered["sleep"]))
                res = {"rc": 0}
            row.update({k: v for k, v in res.items() if k not in ("pids",)}, duration_s=round(time.time() - t, 1))
            if st.get("capture"):
                caps = do_capture(render(st["capture"], env), self.path("steps", f"{name}.log")) if not self.dry else \
                    {k: str(v.get("default", "")) for k, v in st["capture"].items()}
                caps = {k: v for k, v in caps.items() if k not in self.preset}
                self.vars.update(caps)
                row["captured"] = caps
            self.log(f"step {name}: rc={res.get('rc')} {res.get('result') or ''} {res.get('error') or ''} "
                     f"({row['duration_s']} s){' captured ' + json.dumps(row.get('captured')) if row.get('captured') else ''}")
            self.save_summary()
            if self.remote_bad:
                raise WindowAbort("aborted-remote", f"remote valve unhealthy during step {name}: {self.remote_bad}")
            if res.get("rc") not in (0, None) and st.get("on_fail", "abort") == "abort":
                raise WindowAbort("failed-step", f"step {name} failed rc={res.get('rc')} {res.get('error') or res.get('result') or ''}")

    # -- promotion: the ONLY way a window changes the production default, and only after every step passed
    def promote(self):
        """spec.promote = {release: <id|path|{sha,label}>, files: [paths]}. Run only when every step succeeded: the
        restore target becomes the promoted state (pointer flipped through release.py; the listed files kept), so the
        restore boot lands on the NEW default and is checked against it (root, .so hashes, KV pool, health)."""
        p = self.spec["promote"]
        out = {}
        if p.get("release"):
            rel = release_mod()
            root = self.resolve_release(p["release"])
            rid = os.path.basename(root)
            out["release"] = rel.activate(rid, reason=f"window {self.wid} promoted: {self.spec['reason']}"[:200], by=self.by,
                                          restart=False)
            self.snapshot["prod_root"] = os.path.realpath(root)
            self.snapshot["so"] = so_hashes(self.snapshot["prod_root"])
            self.snapshot["jit_backup"] = {}
            self.state["boots"] = max(1, self.state.get("boots", 0))     # the restore boot puts the new default live
        for f in p.get("files") or []:
            f = os.path.expanduser(f)
            try:
                self.snapshot["files"][f] = open(f, "rb").read().hex()
            except OSError:
                self.snapshot["files"][f] = None
            out.setdefault("files", []).append(f)
        with open(self.path("snapshot.json"), "w") as fh:
            json.dump(self.snapshot, fh, indent=1)
        self.save_state()
        self.log(f"PROMOTED: {json.dumps(out, default=str)[:400]}")
        return out

    # -- the restore
    def restore(self):
        return restore_from(self.snapshot, self.state, self.spec, log=self.log, results=self.results)

    def run(self):
        os.makedirs(self.path("steps"), exist_ok=True)
        lockfh = open(LOCK, "a+")
        try:
            fcntl.flock(lockfh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.summary.update(status="refused", why="another window holds the window lock")
            self.save_summary()
            return self.summary
        prev = {}
        try:
            for sg in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                prev[sg] = signal.signal(sg, _raise_terminated)
            problems = self.preflight()
            if problems and not self.dry:
                self.summary.update(status="refused", why=problems)
                self.log(f"REFUSED: {problems}")
                return self.summary
            # releases needed by boots are resolved (and built) BEFORE anything is touched
            for st in self.spec["steps"]:
                if "boot" in st and "release" in st["boot"]:
                    ref = st["boot"]["release"]
                    if isinstance(ref, dict):
                        st["boot"]["release"] = self.resolve_release(ref)
                    self.vars[release_var(st["name"])] = st["boot"]["release"]
            self.take_snapshot()
            self.log(f"snapshot: override {self.snapshot['override_bytes']} B, timers {self.snapshot['timers']}, prod root "
                     f"{self.snapshot['prod_root']}, {len(self.snapshot['so'])} .so, KV pool {self.snapshot['kv_pool']}, "
                     f"Xid count {self.snapshot['xid_count']}, spend ${self.snapshot['spend']}")
            if self.dry:
                self.run_steps()
                self.summary.update(status="dry-run-ok")
                return self.summary
            self.save_state()
            with open(MARKER, "w") as fh:
                json.dump({"window": self.wid, "lane": self.lane, "by": self.by, "pid": os.getpid(),
                           "pid_start": self.state["pid_start"], "results": self.results, "state": self.path("state.json"),
                           "started": now_iso(), "deadline": self.state["deadline"]}, fh, indent=1)
            gpuguard.set_busy(self.by, f"{self.wid} window", ttl_s=self.max_s + 1800, window=self.wid, phase="window")
            self.arm_deadman()
            self.take_hold()
            for t in pause_timers_of(self.spec):
                if self.snapshot["timers"].get(t) == "active":
                    set_timer(t, False)
                    self.log(f"paused timer {t} (restore + dead-man restart it)")
            self.open_gateway()
            threading.Thread(target=self._remote_watch_loop, daemon=True).start()
            self.run_steps()
            self.summary["status"] = "ok"
            if self.spec.get("promote"):
                self.summary["promoted"] = self.promote()
        except WindowAbort as e:
            self.summary.update(status=e.status, why=e.why)
            self.log(f"window stopped: {e.status}: {e.why}")
        except Terminated as e:
            self.summary.update(status="interrupted", why=f"signal {e.signum}")
            self.log(f"window interrupted by signal {e.signum}")
        except SpecError as e:
            self.summary.update(status="spec-error", why=str(e))
            self.log(f"spec error: {e}")
        except Exception as e:  # noqa: BLE001
            self.summary.update(status="error", why=repr(e)[:400])
            self.log(f"window error: {e!r}")
        finally:
            for sg in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(sg, signal.SIG_IGN)       # the restore is not interrupted halfway
            try:
                self._lease_stop.set()
                self._remote_stop.set()          # the watch must never stop the restore's own units
                if not self.dry and self.snapshot:
                    self.summary["restore"] = self.restore()
            finally:
                for sg, h in prev.items():
                    signal.signal(sg, h)
                self.summary["finished"] = now_iso()
                self.summary["duration_s"] = round(time.time() - self.t0, 1)
                self.summary["vars"] = self.vars
                self.save_summary()
                fcntl.flock(lockfh, fcntl.LOCK_UN)
                lockfh.close()
        return self.summary


def _raise_terminated(signum, _frame):
    raise Terminated(signum)


def restore_from(snap, state, spec, log=print, results=None):
    """Put production back EXACTLY as the snapshot found it. Idempotent: the window's finally and the dead-man share it."""
    rep = {"ok": True, "problems": []}

    def bad(msg):
        rep["ok"] = False
        rep["problems"].append(msg)
        log(f"RESTORE PROBLEM: {msg}")
    lane, wid = state["lane"], state["window"]
    # 1. nothing of this window keeps running
    rep["stopped_units"] = unitrun.stop_lane(lane, prefix=wid, exclude=(unitrun.unit_name(lane, f"{wid}-deadman") + ".service",))
    # 2. override bytes, verbatim
    want = bytes.fromhex(snap["override_b64"])
    try:
        cur = open(OVERRIDE, "rb").read()
    except OSError:
        cur = None
    if snap["override_exists"]:
        if cur != want:
            tmp = OVERRIDE + f".tmp{os.getpid()}"
            with open(tmp, "wb") as fh:
                fh.write(want)
            os.replace(tmp, OVERRIDE)
    elif cur is not None:
        os.unlink(OVERRIDE)
    try:
        rep["override_verbatim"] = (open(OVERRIDE, "rb").read() == want) if snap["override_exists"] else not os.path.exists(OVERRIDE)
    except OSError:
        rep["override_verbatim"] = False
    if not rep["override_verbatim"]:
        bad("override env not restored verbatim")
    # 2b. files the spec asked to snapshot (e.g. deployed serve scripts), verbatim
    rep["files"] = {}
    for f, hx in (snap.get("files") or {}).items():
        try:
            cur_f = open(f, "rb").read()
        except OSError:
            cur_f = None
        want_f = None if hx is None else bytes.fromhex(hx)
        if cur_f != want_f:
            if want_f is None:
                os.unlink(f)
            else:
                mode = os.stat(f).st_mode if os.path.exists(f) else 0o755
                tmp = f + f".tmp{os.getpid()}"
                with open(tmp, "wb") as fh:
                    fh.write(want_f)
                os.chmod(tmp, mode & 0o7777)
                os.replace(tmp, f)
        try:
            ok_f = (open(f, "rb").read() == want_f) if want_f is not None else not os.path.exists(f)
        except OSError:
            ok_f = False
        rep["files"][f] = "verbatim" if ok_f else "DIFFERS"
        if not ok_f:
            bad(f"file {f} not restored verbatim")
    # 3. engine: back on the snapshot's root and config, with the expected KV pool
    need_boot = state.get("boots", 0) > 0 or state.get("engine_stopped") or engine_health() != 200 \
        or (running_root() and running_root() != snap.get("prod_root"))
    expected = max(int(spec.get("expected_kv_pool") or 0), int(snap.get("kv_pool") or 0))
    tol = float(spec.get("kv_pool_tolerance", 0.005))
    rep["kv_pool_expected"] = expected
    if need_boot:
        for attempt in (1, 2):
            ok_gpu, foreign = gpuguard.wait_no_foreign(float(spec.get("gpu_gate_wait_s", 300)) * 2)
            if not ok_gpu:
                rep.setdefault("foreign_at_restore", []).append(gpuguard.describe(foreign))
            t_boot = time.time()
            gpuguard.set_busy(state.get("by") or lane, f"{wid} restore boot", ttl_s=1800, window=wid, phase="boot")
            res = unitrun.run(lane, f"{wid}-restore-{attempt}",
                              restart_argv(spec.get("by") or lane, f"window {wid} restore (attempt {attempt})", state.get("hold")),
                              timeout_s=1500,
                              out=os.path.join(results or "/tmp", "steps", f"restore-{attempt}.log"))
            healthy, _ = wait_health(1200 - min(900, time.time() - t_boot))
            pool = gpuguard.kv_pool_since(t_boot - 5)
            rep[f"restore_boot_{attempt}"] = {"unit": res.get("unit"), "rc": res.get("rc"), "healthy": healthy,
                                              "kv_pool": pool, "boot_s": round(time.time() - t_boot, 1)}
            low = bool(expected and pool and pool < expected * (1 - tol))
            if healthy and not low:
                break
            log(f"restore boot {attempt}: healthy={healthy} KV pool {pool} (expected >= {expected * (1 - tol):.0f})")
        if not healthy:
            bad("engine not healthy after restore")
        if low:
            bad(f"KV pool after restore {pool} < expected {expected} (-{tol:.1%}): a foreign GPU process or a config "
                f"change shrank it")
        rep["kv_pool"] = pool
    else:
        rep["kv_pool"] = gpuguard.kv_pool_since()
    rr = running_root()
    rep["running_root"] = rr
    if rr and snap.get("prod_root") and rr != snap["prod_root"]:
        bad(f"engine runs from {rr}, window started on {snap['prod_root']}")
    # 4. timers, exactly as found (pause_timers included)
    rep["timers"] = {}
    for t, was in (snap.get("timers") or {}).items():
        now = timer_state(t)
        if now != was and was in ("active", "inactive"):
            set_timer(t, was == "active")
            now = timer_state(t)
        rep["timers"][t] = {"was": was, "now": now}
        if was in ("active", "inactive") and now != was:
            bad(f"timer {t} is {now}, was {was}")
    # 5. production .so: unchanged, or put back from the JIT backup
    root = snap.get("prod_root")
    now_so = so_hashes(root)
    changed = sorted(k for k in set(now_so) | set(snap.get("so") or {}) if now_so.get(k) != (snap.get("so") or {}).get(k))
    if changed:
        for rel, backup in (snap.get("jit_backup") or {}).items():
            if any(c.startswith(rel + "/") for c in changed) and os.path.isdir(backup):
                tgt = os.path.join(root, rel)
                log(f"prod JIT dir {rel} changed during the window: restoring it from the snapshot backup")
                shutil.rmtree(tgt, ignore_errors=True)
                shutil.copytree(backup, tgt, symlinks=True)
        now_so = so_hashes(root)
        changed = sorted(k for k in set(now_so) | set(snap.get("so") or {}) if now_so.get(k) != (snap.get("so") or {}).get(k))
    rep["so_changed_after_restore"] = changed
    if changed:
        bad(f"production .so differ from the snapshot: {changed[:6]}")
    # 6. gateway window closed, busy signal + marker + dead-man cleared
    lease = state.get("lease")
    if lease:
        r = http_json(f"{GATEWAY}/gateway/offline", "DELETE", {"lease": lease})
        rep["lease_close"] = "closed" if not r.get("error") else ("already-gone" if r.get("http") == 409 else r.get("error"))
        try:
            go = gateway_offline_mod()
            go.drop_lease(lease)
        except Exception:  # noqa: BLE001
            pass
    off = http_json(f"{GATEWAY}/gateway/offline")
    rep["planned_offline_after"] = off.get("offline")
    if off.get("offline") and lease and off.get("by") == (spec.get("by") or lane):
        bad("gateway offline window still open after restore")
    if state.get("hold"):
        h = actuator_hold("release", "--lease", state["hold"])
        rep["hold_release"] = h
        state["hold"] = None
    gpuguard.clear_busy(wid)
    try:
        m = json.load(open(MARKER))
        if m.get("window") == wid:
            os.unlink(MARKER)
    except (OSError, ValueError):
        pass
    if state.get("deadman_unit"):
        subprocess.run(["systemctl", "--user", "stop", f"{state['deadman_unit']}.timer"], capture_output=True, timeout=30)
    rep["xid_count_after"] = len(xid_lines())
    rep["xid_new"] = rep["xid_count_after"] - int(snap.get("xid_count") or 0)
    rep["spend_after"] = spend_usd()
    rep["health_after"] = engine_health()
    if rep["health_after"] != 200:
        bad("engine health after restore is not 200")
    state["restored"] = True
    log(f"restore {'OK' if rep['ok'] else 'FAILED'}: {json.dumps({k: rep[k] for k in ('override_verbatim', 'kv_pool', 'timers', 'so_changed_after_restore', 'health_after')}, default=str)[:600]}")
    return rep


def deadman(state_path):
    """Dead-man timer tick: restore when the window process is gone; TERM it when it overran its deadline."""
    try:
        state = json.load(open(state_path))
    except (OSError, ValueError):
        return 0
    if state.get("restored"):
        return 0
    results = state["results"]
    alive = unitrun.proc_start(state["pid"]) == state.get("pid_start")
    over = time.time() > float(state.get("deadline") or 0) + 900
    if alive and not over:
        return 0
    if alive and over:
        os.kill(state["pid"], signal.SIGTERM)     # its finally restores; next tick escalates if it does not
        if time.time() > float(state.get("deadline") or 0) + 2700:
            os.kill(state["pid"], signal.SIGKILL)
        return 0
    snap = json.load(open(os.path.join(results, "snapshot.json")))
    spec = {}
    try:
        spec = json.load(open(os.path.join(results, "spec.resolved.json")))
    except (OSError, ValueError):
        pass

    def log(msg):
        with open(os.path.join(results, "window.log"), "a") as fh:
            fh.write(f"{now_iso()} [dead-man] {msg}\n")
    log(f"window process {state['pid']} is gone without restoring: restoring now")
    if state.get("hold"):
        log("its engine liveness hold was owner-pid scoped and is void now; restoring without it")
        state["hold"] = None
    rep = restore_from(snap, state, spec, log=log, results=results)
    state["restored"] = True
    with open(state_path, "w") as fh:
        json.dump(state, fh, indent=1)
    try:
        s = json.load(open(os.path.join(results, "summary.json")))
    except (OSError, ValueError):
        s = {}
    s.update(status=s.get("status") if s.get("status") not in (None, "running") else "killed", restore=rep,
             restored_by="dead-man", finished=now_iso())
    with open(os.path.join(results, "summary.json"), "w") as fh:
        json.dump(s, fh, indent=1, default=str)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("validate")
    p.add_argument("spec")
    p = sp.add_parser("run")
    p.add_argument("spec")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--results")
    p.add_argument("--set", action="append", default=[], help="VAR=VAL preset (overrides a capture; for dry runs)")
    p = sp.add_parser("submit")
    p.add_argument("spec")
    p.add_argument("--wait", action="store_true")
    sp.add_parser("status")
    p = sp.add_parser("deadman")
    p.add_argument("--state", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        try:
            spec = load_spec(a.spec)
            validate(spec)
        except (SpecError, OSError, ValueError) as e:
            print(json.dumps({"valid": False, "error": str(e)}))
            return 2
        print(json.dumps({"valid": True, "lane": spec["lane"], "window": spec["window"],
                          "steps": [{"name": s["name"], "kind": next(k for k in STEP_KINDS if k in s),
                                     "when": s.get("when")} for s in spec["steps"]]}, indent=1))
        return 0
    if a.cmd == "run":
        spec = load_spec(a.spec)
        w = Window(spec, os.path.abspath(a.spec), dry_run=a.dry_run, results=a.results,
                   preset=dict(kv.split("=", 1) for kv in a.set if "=" in kv))
        os.makedirs(w.path("steps"), exist_ok=True)
        with open(w.path("spec.resolved.json"), "w") as fh:
            json.dump(spec, fh, indent=1)
        s = w.run()
        print(json.dumps({"status": s["status"], "why": s.get("why"), "results": w.results,
                          "restore_ok": (s.get("restore") or {}).get("ok")}, default=str))
        return 0 if s["status"] in ("ok", "dry-run-ok") and (s.get("restore") or {}).get("ok", True) else 1
    if a.cmd == "submit":
        spec = load_spec(a.spec)
        validate(spec)
        unit = unitrun.unit_name(f"win-{spec['lane']}", spec["window"])
        argv = ["systemd-run", "--user", f"--unit={unit}", "--collect", "-p", "TimeoutStopSec=1800",
                "-p", "KillMode=mixed", f"--setenv=PATH={os.environ.get('PATH', '/usr/bin:/bin')}", f"--setenv=HOME={HOME}"]
        if a.wait:
            argv.append("--wait")
        # snaprun rule for the framework itself: it (and the dead-man it arms) runs from a read-only snapshot of its
        # code + the spec taken NOW, so merging/editing deploy/bin or the spec mid-window cannot change a running window.
        snap = os.path.join(unitrun.STATE_DIR, "windowctl-snap", f"{unit}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(snap, exist_ok=True)
        for f in CODE_FILES:
            if os.path.exists(os.path.join(HERE, f)):
                shutil.copy2(os.path.join(HERE, f), os.path.join(snap, f))
        spec_snap = os.path.join(snap, os.path.basename(a.spec))
        shutil.copy2(a.spec, spec_snap)
        for f in os.listdir(snap):
            os.chmod(os.path.join(snap, f), 0o444)
        argv += ["--", sys.executable, os.path.join(snap, "windowctl.py"), "run", spec_snap]
        r = subprocess.run(argv)
        print(json.dumps({"unit": unit, "rc": r.returncode, "code_snapshot": snap, "stop": f"systemctl --user stop {unit}  (restores first)"}))
        return r.returncode
    if a.cmd == "status":
        try:
            m = json.load(open(MARKER))
            m["owner_alive"] = unitrun.proc_start(m["pid"]) == m.get("pid_start")
        except (OSError, ValueError):
            m = {"active": False}
        print(json.dumps({"window": m, "gpu_busy": gpuguard.busy_state()}, indent=1, default=str))
        return 0
    if a.cmd == "deadman":
        return deadman(a.state)
    return 2


if __name__ == "__main__":
    sys.exit(main())
