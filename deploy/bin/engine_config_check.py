#!/usr/bin/env python3
"""Engine config: resolved chain, intended-vs-actual drift, and "where is knob X set?" -- lane CFG, 2026-10-03.

WHY
The engine's config is spread over a systemd unit, its EnvironmentFile(s), a dispatcher script, a pointer file, the
active serve script, files that script sources (v02.override.env) and the script's own export/unset lines. Wrong
answers this caused, all from reading the WRONG layer:
  * a ledger said VLLM_CUSTOM_ALLREDUCE_MAX_SIZE_MB=32 was live; the running engine's environ did not have it;
  * a worker's /proc/<pid>/environ was read as proof a knob was absent -- but setproctitle (VLLM::EngineCore,
    VLLM::Worker_TP*) overwrites the start of the child environ block, so worker environs show ~931 entries of
    which ~47 survive. Only the api_server (the unit's MainPID) environ is authoritative; children inherit it;
  * a production-bug claim inferred VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE from serve-profile-v02.sh, a
    file NOT in the production chain (production takes it from vllm-qwen27b.env, where it is off);
  * a serve script re-exported PYTHONPATH and silently dropped an override (lane RL owns that fix; this tool
    detects the class: "set by a sourced file, overwritten later in the script").
  * five knobs in vllm-qwen27b.env are read by nothing in the running v0.2 tree (dead knobs).

WHAT
  chain          unit -> ExecStart -> pointer -> serve script -> each sourced file, in order -> final env -> argv,
                 rendered by a DRY RUN of the serve script (its final `exec` is replaced by a dump; nothing is
                 launched; scripts containing process-control commands are refused). Plus config-looking files
                 that are NOT in the chain, and repo copies that differ from the deployed files.
  check          intended (rendered) env + argv vs ACTUAL api_server /proc environ + cmdline (read-only); pending-
                 restart detection (chain file newer than the engine); dropped overrides; env-file values the script
                 overrides or unsets; dead knobs (no code in the engine tree reads them); unknown knobs; --expect K=V
                 ledger claims. Exit 1 on drift or a failed claim.
  whereis-knob   for each NAME: production value (api_server), which chain file:line set or unset it, the value
                 the chain would produce now, argv flags it feeds, look-alike files that mention it but are NOT in
                 the chain, and the engine code that reads it (with its code default).
All reads are read-only (/proc, systemctl show, files). Secret-looking values are masked.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

UNIT = "vllm-qwen27b.service"
RUNTIME_DIR = Path("/home/kevin/.local/share/vllm-qwen27b")
DEFAULT_ENGINE_TREE = Path("/home/kevin/Desktop/wt-integrate")
REPO_COPIES = [Path("/home/kevin/Desktop/vLLM-2080Ti-Definitive/deploy"), DEFAULT_ENGINE_TREE / "deploy"]
RELEVANT = re.compile(r"^(VLLM_|V02_|PYTHON|CUDA|TORCH|TRITON|FLASH|NCCL|HF_|OMP_|PYTORCH_|NVCC|CC$|CXX$|CUDAHOSTCXX$|PATH$|LD_LIBRARY_PATH$)")
SECRET = re.compile(r"(_KEY|_SECRET|_PASSWORD|_PASSWD|_TOKEN|_AUTH)$|^HF_TOKEN$")
KNOB = re.compile(r"\b((?:VLLM|V02)_[A-Z0-9_]+)\b")
DENY = re.compile(r"\b(systemctl|nvidia-smi|kill|pkill|killall|rm|sudo|docker|curl|wget|nohup|reboot|shutdown|dd|mkfs|tee)\b")

# Deploy-layer knob schema. consumer: "script" = consumed by the serve script itself (becomes argv or a path);
# "engine" = read by the engine's Python code from its environment; "toolchain" = compilers / caches / CUDA.
# Code defaults for engine knobs are discovered from the engine tree at check time (envs.py / os.environ.get).
ENGINE_KNOBS = {
    "V02_STACK": ("script", "bool", "Default stack (text-only + int4 lm_head + int4 MTP); 0 = rollback."),
    "V02_ROOT": ("script", "path", "Engine tree the v0.2 serve script runs from."),
    "V02_RUNNER": ("script", "enum", "Model runner: v1 (default here) | v2."),
    "V02_ASYNC_FLAG": ("script", "str", "Async scheduling flag (default --no-async-scheduling)."),
    "V02_COMPILATION_CONFIG": ("script", "json", "--compilation-config override."),
    "V02_SPEC": ("script", "json", "--speculative-config override (default MTP-3)."),
    "V02_MAXLEN": ("script", "int", "--max-model-len (default 524288)."),
    "V02_MAXSEQS": ("script", "int", "--max-num-seqs (default 16)."),
    "V02_KV_DTYPE": ("script", "str", "--kv-cache-dtype (default turboquant_k3v4_nc)."),
    "V02_SSD_KV_DIR": ("script", "path", "Opt-in SSD prefix-KV persistence directory."),
    "V02_SSD_KV_CPU_BYTES": ("script", "int", "CPU bytes for SSD prefix-KV offload."),
    "V02_PROFILE_ARGS": ("script", "path", "serve-profile-v02.sh: saved upstream profile args file."),
    "V02_SPEC_SYNC": ("script", "str", "serve-profile-v02.sh: VLLM_SM75_SPEC_SYNC_MODE value."),
    "V02_FULL_GRAPH": ("script", "bool", "serve-profile-v02.sh: VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH value."),
    "VLLM_GPU_UTIL": ("script", "float", "--gpu-memory-utilization (default 0.84)."),
    "VLLM_MNBT": ("script", "int", "--max-num-batched-tokens (scheduler step budget)."),
    "VLLM_SERVE_EXTRA_ARGS": ("script", "str", "Extra api_server args appended to argv."),
    "VLLM_U2_INT4_HEAD": ("engine", "bool", "Load-time int4 lm_head (U2)."),
    "VLLM_U2_INT4_MTP": ("engine", "bool", "Load-time int4 MTP block (U2)."),
    "VLLM_USE_V2_MODEL_RUNNER": ("engine", "bool", "Model Runner V2 (0 = V1)."),
    "VLLM_TQ_GQA_CUDA": ("engine", "bool", "Grouped sm_75 TurboQuant decode attention kernel (S2); 0 = stock."),
    "VLLM_TQ_GQA_BUILD_DIR": ("engine", "path", "Prebuilt tq_gqa extension directory."),
    "VLLM_SCHED_SHORT_FIRST_RUN": ("engine", "int", "Consecutive short-first yield steps (EF2)."),
    "VLLM_SCHED_PREFILL_SHARE": ("engine", "float", "Prefill share of the step budget (EF2)."),
    "VLLM_SM75_SPEC_SYNC_MODE": ("engine", "enum", "SM75 speculative sync mode."),
    "VLLM_ENFORCE_STRICT_TOOL_CALLING": ("engine", "bool", "Strict tool-call parsing."),
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": ("engine", "bool", "Allow max_model_len past the model config."),
    "VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH": ("engine", "bool", "Full CUDA graphs for mamba+spec."),
    "VLLM_ROPE_MAX_POSITION": ("engine", "int", "Rope cos/sin cache size past the native window (EXP-017)."),
    "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE": ("engine", "int", "FlashInfer workspace bytes."),
    "VLLM_CUSTOM_ALLREDUCE_MAX_SIZE_MB": ("engine", "int", "Custom allreduce max size (0 = stock)."),
    "VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE": ("engine", "bool", "TurboQuant spec-decode CUDA-graph safety."),
    "VLLM_TURBOQUANT_FLASHINFER_BACKEND": ("engine", "str", "TurboQuant FlashInfer backend."),
    "VLLM_TURBOQUANT_FLASHINFER_PREFILL_PLAN_CACHE_MAXSIZE": ("engine", "int", "Prefill plan cache size."),
    "VLLM_TURBOQUANT_MAX_KV_SPLITS": ("engine", "int", "TurboQuant max KV splits."),
    "VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH": ("engine", "bool", "Spec continuation decode fast path."),
    "VLLM_TURBOQUANT_USE_FLASHINFER_PREFILL": ("engine", "bool", "FlashInfer prefill for TurboQuant."),
    "VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE": ("engine", "str", "Prefix-combine continuation branch (off = Xid-31 mitigation)."),
    "VLLM_TURBOQUANT_CONTINUATION_BOUNDS_CHECK": ("engine", "bool", "0.1.x continuation bounds check."),
    "VLLM_TURBOQUANT_CONTINUATION_SDPA_MAX_QK_CELLS": ("engine", "int", "0.1.x continuation SDPA tiling."),
    "VLLM_TURBOQUANT_CONTINUATION_SDPA_Q_CHUNK": ("engine", "int", "0.1.x continuation SDPA q chunk."),
    "VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS": ("engine", "int", "0.1.x continuation workspace reserve (unset by the v0.2 script)."),
    "VLLM_TURBOQUANT_STAGE1_QTILE": ("engine", "bool", "0.1.x stage-1 q-tile."),
    "VLLM_FLASHQLA_VARLEN_LOOP": ("engine", "bool", "0.1.x FlashQLA varlen loop."),
    "VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK": ("engine", "bool", "0.1.x MTP block retention (superseded by upstream #53388)."),
    "VLLM_PREFIX_CACHE_USE_RETAINED_MTP_BLOCK": ("engine", "bool", "0.1.x retained MTP block prefix use."),
    "VLLM_WORKER_MULTIPROC_METHOD": ("engine", "enum", "Worker start method (set by vLLM itself)."),
    "VLLM_DISABLE_TILELANG": ("engine", "bool", "Disable TileLang kernels."),
}


def mask(k, v):
    if v is None:
        return None
    return "<set, %d chars>" % len(v) if SECRET.search(k) else v


# ------------------------------------------------------------------ systemd
def unit_props(unit=UNIT):
    keys = ["FragmentPath", "DropInPaths", "EnvironmentFiles", "Environment", "ExecStart", "WorkingDirectory",
            "MainPID", "ExecMainStartTimestamp"]
    try:
        out = subprocess.run(["systemctl", "show", unit] + ["-p" + k for k in keys], capture_output=True, text=True,
                             timeout=15).stdout
    except Exception as e:
        return {"error": repr(e)}
    props = {"EnvironmentFiles": []}
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k == "EnvironmentFiles":
            m = re.match(r"(\S+) \(ignore_errors=(\w+)\)", v)
            if m:
                props["EnvironmentFiles"].append((m.group(1), m.group(2) == "yes"))
        else:
            props[k] = v
    m = re.search(r"argv\[\]=([^;]+);", props.get("ExecStart", ""))
    props["ExecArgv"] = shlex.split(m.group(1)) if m else []
    props["Environment"] = shlex.split(props.get("Environment", "") or "")
    return props


def parse_env_file(path):
    """systemd EnvironmentFile: KEY=VALUE, comments, quotes; returns [(lineno, key, value)]."""
    rows = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return None
    for n, line in enumerate(lines, 1):
        s = line.strip()
        if not s or s[0] in "#;" or "=" not in s:
            continue
        k, v = s.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        k = k.strip()
        if k.startswith("export "):
            k = k[7:].strip()
        rows.append((n, k, v))
    return rows


def shell_assignments(path):
    """KEY=VALUE / export KEY=VALUE lines of a sourced shell file: [(lineno, key, exported)]. A plain KEY=VALUE only
    sets a SHELL variable: it reaches the engine only if KEY was already exported (or is exported later)."""
    out = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return None
    exported_later = set()
    for line in lines:
        m = re.match(r"^\s*export\s+([A-Za-z_]\w*)\s*$", line)
        if m:
            exported_later.add(m.group(1))
    for n, line in enumerate(lines, 1):
        m = re.match(r"^\s*(export\s+)?([A-Za-z_]\w*)=", line)
        if m:
            out.append((n, m.group(2), bool(m.group(1)) or m.group(2) in exported_later))
    return out


# ------------------------------------------------------------------ dispatcher / script resolution
def resolve_target(exec_script: Path):
    """serve-active.sh reads a pointer file and execs one script per case arm. Returns (target, pointer, note)."""
    try:
        text = exec_script.read_text()
    except OSError as e:
        return None, None, "cannot read %s: %s" % (exec_script, e)
    pm = re.search(r'cat\s+"\$D/([\w.-]+)"', text)
    if not pm or "case" not in text:
        return exec_script, None, "ExecStart script is the serve script"
    pointer = exec_script.parent / pm.group(1)
    try:
        target_name = pointer.read_text().strip()
    except OSError:
        target_name = ""
    arms = re.findall(r'^\s*([^\s)#]+)\)\s*exec\s+bash\s+"\$D/([\w.-]+)"', text, re.M)
    chosen = None
    for pat, script in arms:
        if pat == target_name:
            chosen = script
    if chosen is None:
        for pat, script in arms:
            if pat == "*":
                chosen = script
        note = "pointer %r matches no case arm -> default arm" % target_name
    else:
        note = "pointer %s = %r" % (pointer.name, target_name)
    return (exec_script.parent / chosen if chosen else None), pointer, note


def sourced_files(script_text, env):
    """Files the script sources ('. file' / 'source file'), in order, with line numbers. $VAR expanded from env."""
    out = []
    for n, line in enumerate(script_text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for m in re.finditer(r'(?:^|[;&|]\s*|\s)(?:\.|source)\s+("?)([^\s";&|]+)\1', line):
            p = m.group(2)
            p = re.sub(r"\$\{?(\w+)\}?", lambda mm: env.get(mm.group(1), "$" + mm.group(1)), p)
            out.append((n, p))
    return out


# ------------------------------------------------------------------ dry run render
_DUMP_FN = r'''__cfg_dump() { __cfg_n=$((__cfg_n+1)); env -0 > "$__CFG_DIR/snap.$__cfg_n"; printf '%s\n' "$1" >> "$__CFG_DIR/steps"; }
__cfg_final() { __cfg_dump final; printf '%s\0' "$@" > "$__CFG_DIR/argv"; exit 0; }
__cfg_n=0
'''


def render(script: Path, base_env: dict, workdir: str, timeout=30):
    """Run the serve script's shell logic with its final exec replaced by a dump. Returns dict(steps=[(label, env)],
    argv=[...]) or dict(error=...). Refuses scripts with process-control commands."""
    try:
        text = script.read_text()
    except OSError as e:
        return {"error": "cannot read %s: %s" % (script, e)}
    lines = text.splitlines()
    bad = [(i + 1, l.strip()) for i, l in enumerate(lines) if not l.lstrip().startswith("#") and DENY.search(l)]
    if bad:
        return {"error": "refusing to dry-run: process-control command at line %d: %s" % bad[0]}
    exec_idx = [i for i, l in enumerate(lines) if re.match(r"^\s*exec\s", l)]
    eval_exec = [i for i, l in enumerate(lines) if re.match(r'^\s*eval\s+"exec', l)]
    if not exec_idx and not eval_exec:
        return {"error": "no final exec line found in %s" % script}
    out = []
    srcs = {n for n, _ in sourced_files(text, base_env)}
    last_exec = max(exec_idx + eval_exec)
    for i, l in enumerate(lines):
        l = l.replace("${BASH_SOURCE[0]}", str(script)).replace('"$0"', '"%s"' % script)
        if i == last_exec:
            if i in eval_exec:
                l = l.replace('eval "exec', 'eval "__cfg_final', 1)
            else:
                l = re.sub(r"^(\s*)exec\s", r"\1__cfg_final ", l, count=1)
        out.append(l)
        if (i + 1) in srcs:
            out.append('__cfg_dump "after line %d: %s"' % (i + 1, l.strip().replace('"', "'")[:160]))
        if i == 0 and l.startswith("#!"):
            out.append(_DUMP_FN + "__cfg_dump start")
    if not lines or not lines[0].startswith("#!"):
        out.insert(0, _DUMP_FN + "__cfg_dump start")
    with tempfile.TemporaryDirectory(prefix="engcfg-") as td:
        rs = Path(td) / "render.sh"
        rs.write_text("\n".join(out) + "\n")
        env = dict(base_env)
        env["__CFG_DIR"] = td
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        try:
            p = subprocess.run(["bash", str(rs)], env=env, cwd=workdir if os.path.isdir(workdir or "") else td,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"error": "dry run timed out"}
        steps_file = Path(td) / "steps"
        if not steps_file.exists():
            return {"error": "dry run produced no snapshots (rc=%d): %s" % (p.returncode, p.stderr[-400:])}
        labels = steps_file.read_text().splitlines()
        steps = []
        for idx, label in enumerate(labels, 1):
            raw = (Path(td) / ("snap.%d" % idx)).read_bytes().split(b"\0")
            e = {}
            for item in raw:
                if b"=" in item:
                    k, v = item.split(b"=", 1)
                    e[k.decode(errors="replace")] = v.decode(errors="replace")
            e.pop("__CFG_DIR", None)
            for k in [k for k in e if k.startswith("__cfg")]:
                e.pop(k)
            steps.append((label, e))
        argv_file = Path(td) / "argv"
        argv = [a.decode(errors="replace") for a in argv_file.read_bytes().split(b"\0")[:-1]] if argv_file.exists() else None
        if argv is None:
            return {"error": "script exited before its final exec (rc=%d): %s" % (p.returncode, p.stderr[-400:]),
                    "steps": steps}
        return {"steps": steps, "argv": argv, "stderr": p.stderr[-400:]}


# ------------------------------------------------------------------ actual process
def proc_env(pid):
    try:
        raw = Path("/proc/%d/environ" % pid).read_bytes().split(b"\0")
    except OSError as e:
        return None, str(e)
    env, empty = {}, 0
    for item in raw:
        if not item:
            empty += 1
            continue
        if b"=" in item:
            k, v = item.split(b"=", 1)
            env[k.decode(errors="replace")] = v.decode(errors="replace")
    return env, ("clobbered (setproctitle): %d of %d entries empty" % (empty, len(raw))) if empty > 8 else None


def proc_argv(pid):
    try:
        return [a.decode(errors="replace") for a in Path("/proc/%d/cmdline" % pid).read_bytes().split(b"\0")[:-1]]
    except OSError:
        return None


def proc_start_epoch(pid):
    try:
        st = Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()
        ticks = int(st[19])
        btime = int([l for l in Path("/proc/stat").read_text().splitlines() if l.startswith("btime")][0].split()[1])
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None


def descendants(pid):
    out, todo = [], [pid]
    while todo:
        kids = children(todo.pop())
        out.extend(kids)
        todo.extend(kids)
    return out


def children(pid):
    out = []
    try:
        for d in Path("/proc").iterdir():
            if d.name.isdigit():
                try:
                    ppid = int(d.joinpath("stat").read_text().rsplit(")", 1)[1].split()[1])
                except Exception:
                    continue
                if ppid == pid:
                    out.append(int(d.name))
    except OSError:
        pass
    return out


# ------------------------------------------------------------------ engine tree readers
_TREE_CACHE = {}


def tree_readers(tree: Path):
    """{KNOB: [relative files that mention it]} over the engine tree's code (.py/.cu/.cpp/.h)."""
    key = str(tree)
    if key in _TREE_CACHE:
        return _TREE_CACHE[key]
    found = {}
    roots = [tree / "vllm", tree / ".deps" / "FlashQLA-SM70-SM75", tree / "csrc"]
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".torch_extensions") and d not in ("__pycache__", "build")]
            for f in files:
                if not f.endswith((".py", ".cu", ".cuh", ".cpp", ".h", ".c")):
                    continue
                p = os.path.join(dirpath, f)
                try:
                    text = open(p, errors="replace").read()
                except OSError:
                    continue
                for k in set(KNOB.findall(text)):
                    found.setdefault(k, []).append(os.path.relpath(p, tree))
    _TREE_CACHE[key] = found
    return found


def code_default(tree: Path, name):
    """The engine's default for name from envs.py / a direct os.environ.get(name, d) read, if literal."""
    for rel in ("vllm/envs.py",):
        try:
            text = (tree / rel).read_text()
        except OSError:
            continue
        m = re.search(r'getenv\(\s*"%s"\s*,\s*"([^"]*)"' % re.escape(name), text)
        if m:
            return m.group(1), rel
        m = re.search(r'getenv\(\s*"%s"\s*\)\s*or\s*([\w."]+)' % re.escape(name), text)
        if m:
            return m.group(1), rel
    return None, None


# ------------------------------------------------------------------ the chain
def build_chain(unit=UNIT, engine_tree=None, render_script=True):
    props = unit_props(unit)
    chain = {"unit": unit, "fragment": props.get("FragmentPath"), "dropins": (props.get("DropInPaths") or "").split(),
             "workdir": props.get("WorkingDirectory"), "main_pid": int(props.get("MainPID") or 0),
             "started": props.get("ExecMainStartTimestamp"), "layers": [], "errors": []}
    base, origin = {}, {}
    for kv in props.get("Environment", []):
        if "=" in kv:
            k, v = kv.split("=", 1)
            base[k] = v
            origin[k] = "unit Environment="
    for path, ignore in props.get("EnvironmentFiles", []):
        rows = parse_env_file(path)
        layer = {"kind": "EnvironmentFile", "path": path, "optional": ignore, "exists": rows is not None, "sets": {}}
        for n, k, v in rows or []:
            base[k] = v
            origin[k] = "%s:%d" % (path, n)
            layer["sets"][k] = n
        chain["layers"].append(layer)
    exec_argv = props.get("ExecArgv") or []
    exec_script = Path(exec_argv[0]) if exec_argv else None
    chain["exec_start"] = str(exec_script) if exec_script else None
    target, pointer, note = resolve_target(exec_script) if exec_script else (None, None, "no ExecStart")
    chain["dispatch"] = {"pointer": str(pointer) if pointer else None, "note": note, "target": str(target) if target else None}
    chain["serve_script"] = str(target) if target else None
    rendered = None
    if target and render_script:
        text = target.read_text() if target.exists() else ""
        for n, p in sourced_files(text, base):
            rows = shell_assignments(p)
            chain["layers"].append({"kind": "sourced by %s:%d" % (target.name, n), "path": p, "exists": rows is not None,
                                    "sets": {k: ln for ln, k, _ in (rows or [])},
                                    "not_exported": {k: ln for ln, k, ex in (rows or []) if not ex}})
        chain["layers"].append({"kind": "serve script body", "path": str(target)})
        rendered = render(target, base, chain["workdir"] or "/")
        if rendered.get("error"):
            chain["errors"].append(rendered["error"])
    chain["base_env"], chain["base_origin"] = base, origin
    chain["rendered"] = rendered
    # attribution of every final key
    final, attrib, history = {}, {}, {}
    if rendered and rendered.get("steps"):
        steps = rendered["steps"]
        final = steps[-1][1]
        prev = base
        for label, env in steps:
            for k in set(prev) | set(env):
                if prev.get(k) != env.get(k):
                    where = ("serve script body (%s)" % target.name) if label in ("final",) else (
                        "systemd env (script start)" if label == "start" else label)
                    if label == "start":
                        where = origin.get(k, "inherited (not in unit env files)")
                    history.setdefault(k, []).append((where, env.get(k)))
            prev = env
        for k in final:
            attrib[k] = history[k][-1][0] if k in history else origin.get(k, "inherited (not in unit env files)")
    chain["final_env"], chain["attribution"], chain["history"] = final, attrib, history
    chain["argv"] = rendered.get("argv") if rendered else None
    # line numbers in the serve script for each knob it touches
    script_lines = {}
    if target and target.exists():
        for n, line in enumerate(target.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for k in set(re.findall(r"\b([A-Z][A-Z0-9_]+)\b", line)):
                script_lines.setdefault(k, []).append(n)
    chain["script_lines"] = script_lines
    chain["engine_tree"] = str(engine_tree or Path(final.get("V02_ROOT") or base.get("V02_ROOT") or DEFAULT_ENGINE_TREE))
    return chain


def chain_files(chain):
    files = [chain.get("fragment")] + chain.get("dropins", []) + [chain.get("exec_start"), chain["dispatch"].get("pointer"),
                                                                    chain.get("serve_script")]
    files += [l["path"] for l in chain["layers"] if l.get("path")]
    return [f for f in dict.fromkeys(files) if f]


def lookalikes(chain):
    """Config-looking files that are NOT in the chain: runtime-dir env/serve files and repo copies."""
    inchain = set(chain_files(chain))
    live, backups = [], 0
    for p in sorted(RUNTIME_DIR.glob("*")):
        if not p.is_file():
            continue
        n = p.name
        if not (n.endswith(".env") or ".env." in n or n.startswith("serve-")):
            continue
        if str(p) in inchain:
            continue
        if re.search(r"\.(bak|BANKED|pre|before)|\.bak-|BANKED", n):
            backups += 1
            continue
        live.append(str(p))
    divergent, repo_only, presence = [], [], []
    for f in chain_files(chain):
        if not f.startswith(str(RUNTIME_DIR)) or f.endswith("active-serve"):
            continue
        name = os.path.basename(f)
        row = {"deployed": f}
        for root in REPO_COPIES:
            cands = list(root.glob("bin/" + name)) + list(root.glob("env/" + name))
            state = "absent"
            for cand in cands:
                try:
                    same = Path(f).read_bytes() == cand.read_bytes()
                except OSError:
                    continue
                state = "same" if same else "DIFFERS"
                (repo_only if same else divergent).append({"deployed": f, "repo_copy": str(cand), "same": same})
            row[str(root)] = state
        presence.append(row)
    return {"not_in_chain": live, "backup_variants_not_in_chain": backups,
            "repo_copies_differ": [d for d in divergent], "repo_copies_same": repo_only, "repo_presence": presence}


# ------------------------------------------------------------------ drift check
def check(chain, expects=()):
    pid = chain["main_pid"]
    findings = []

    def add(level, code, key, msg):
        findings.append({"level": level, "code": code, "key": key, "msg": msg})
    actual, clob = proc_env(pid) if pid else (None, "no MainPID")
    if actual is None:
        add("error", "no-engine", "", "cannot read the engine environ: %s" % clob)
        actual = {}
    if clob:
        add("warn", "environ-clobbered", "", "MainPID %d environ looks clobbered (%s)" % (pid, clob))
    kids = []
    for c in descendants(pid) if pid else []:
        e, cl = proc_env(c)
        argv = proc_argv(c) or []
        kids.append({"pid": c, "argv0": (argv[0] if argv else "")[:60], "environ": "unreliable: " + cl if cl else "readable"})
    final = chain.get("final_env") or {}
    attrib = chain.get("attribution") or {}
    if not final:
        add("error", "no-render", "", "intended config could not be rendered: %s" % "; ".join(chain["errors"]))
    start = proc_start_epoch(pid) if pid else None
    newer = []
    for f in chain_files(chain):
        try:
            if start and os.path.getmtime(f) > start + 2:
                newer.append(f)
        except OSError:
            pass
    for f in newer:
        add("warn", "pending-restart", "", "%s changed after the engine started (%s); intended != actual below may be a "
            "change not yet applied, not a bug" % (f, time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(f)))))
    for k in sorted(set(final) | set(actual)):
        if not RELEVANT.match(k) or k == "PWD":
            continue
        a, i = actual.get(k), final.get(k)
        if a == i:
            continue
        if i is None:
            add("error", "extra-in-engine", k, "engine has %s=%s but the chain no longer sets it" % (k, mask(k, a)))
        elif a is None:
            add("error", "missing-in-engine", k, "chain sets %s=%s (by %s) but the running engine does not have it" % (
                k, mask(k, i), attrib.get(k)))
        else:
            add("error", "value-drift", k, "engine %s=%s, chain now gives %s (by %s)" % (k, mask(k, a), mask(k, i), attrib.get(k)))
    want_argv = chain.get("argv")
    have_argv = proc_argv(pid) if pid else None
    if want_argv is not None and have_argv is not None and want_argv != have_argv:
        wa, ha = _flags(want_argv), _flags(have_argv)
        for f in sorted(set(wa) | set(ha)):
            if wa.get(f) != ha.get(f):
                add("error", "argv-drift", f, "engine %s %s, chain now gives %s" % (f, ha.get(f, "<absent>"), wa.get(f, "<absent>")))
    # layer conflicts: a sourced file / env file value overwritten or unset later
    for k, hist in (chain.get("history") or {}).items():
        base_origin = chain["base_origin"].get(k)
        src_steps = [(w, v) for w, v in hist if w.startswith("after line")]
        fin = final.get(k)
        for w, v in src_steps:
            if v != fin:
                add("warn", "override-dropped", k, "%s set %s=%s but the serve script later changed it to %s -- the "
                    "override does not reach the engine" % (w, k, mask(k, v), mask(k, fin)))
        if base_origin and k in chain["base_env"] and fin != chain["base_env"][k] and not src_steps and RELEVANT.match(k):
            add("info", "envfile-unset" if fin is None else "envfile-overridden", k, "%s sets %s=%s; the serve script %s" % (
                base_origin, k, mask(k, chain["base_env"][k]), "unsets it" if fin is None else "replaces it with %s" % mask(k, fin)))
    for layer in chain["layers"]:
        for k, ln in (layer.get("not_exported") or {}).items():
            if k.startswith("V02_") or (ENGINE_KNOBS.get(k) or ("",))[0] == "script":
                continue                                   # consumed by the script as a shell variable: fine
            if k not in final:
                add("error", "override-not-exported", k, "%s:%d assigns %s without `export`, so it is only a shell "
                    "variable and never reaches the engine; write `export %s=...`" % (layer["path"], ln, k, k))
    tree = Path(chain["engine_tree"])
    readers = tree_readers(tree) if tree.is_dir() else {}
    for k in sorted(final):
        if not k.startswith("VLLM_") and not k.startswith("V02_"):
            continue
        meta = ENGINE_KNOBS.get(k)
        consumer = meta[0] if meta else None
        script_uses = k in (chain.get("script_lines") or {})
        if consumer == "script" or (k.startswith("V02_") and script_uses):
            continue
        if k not in readers:
            add("warn", "dead-knob", k, "%s=%s (by %s) is read by no code in %s -- it does nothing" % (
                k, mask(k, final[k]), attrib.get(k), tree))
        if meta is None and k not in readers:
            add("warn", "unknown-knob", k, "not in the engine knob schema and not read by the engine (typo?)")
    for e in expects:
        if "=" not in e:
            continue
        k, v = e.split("=", 1)
        a = actual.get(k)
        ok = (a is None and v in ("", "<absent>")) or a == v
        add("info" if ok else "error", "claim-%s" % ("ok" if ok else "false"), k,
            "claim %s=%s vs running engine (pid %d): %s" % (k, v or "<absent>", pid, "<absent>" if a is None else mask(k, a)))
    return {"pid": pid, "children": kids, "findings": findings,
            "drift": any(f["level"] == "error" for f in findings)}


def _flags(argv):
    out, i = {}, 0
    pos = []
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            vals = []
            j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                vals.append(argv[j])
                j += 1
            out[a] = " ".join(vals)
            i = j
        else:
            pos.append(a)
            i += 1
    out["<positional>"] = " ".join(pos)
    return out


# ------------------------------------------------------------------ whereis-knob
def whereis(chain, names):
    pid = chain["main_pid"]
    actual, clob = proc_env(pid) if pid else ({}, None)
    actual = actual or {}
    argv = proc_argv(pid) or []
    tree = Path(chain["engine_tree"])
    readers = tree_readers(tree) if tree.is_dir() else {}
    looks = lookalikes(chain)
    other_files = looks["not_in_chain"] + [d["repo_copy"] for d in looks["repo_copies_differ"]]
    for root in REPO_COPIES:
        for p in list(root.glob("bin/serve-*.sh")) + list(root.glob("env/*.env")):
            other_files.append(str(p))
    other_files = [f for f in dict.fromkeys(other_files) if f not in chain_files(chain)]
    same_copies = {d["repo_copy"] for d in looks["repo_copies_same"]}
    out = []
    for name in names:
        r = {"knob": name, "production_pid": pid}
        r["production_value"] = mask(name, actual.get(name)) if name in actual else "<absent>"
        r["production_source"] = "api_server /proc/%d/environ (authoritative; workers inherit it)" % pid
        r["chain_value"] = mask(name, (chain.get("final_env") or {}).get(name)) if name in (chain.get("final_env") or {}) else "<absent>"
        r["set_by"] = [{"where": w, "value": mask(name, v) if v is not None else "<unset>"} for w, v in (chain.get("history") or {}).get(name, [])]
        if not r["set_by"] and name in chain["base_origin"]:
            r["set_by"] = [{"where": chain["base_origin"][name], "value": mask(name, chain["base_env"][name])}]
        mentions = []
        for f in chain_files(chain):
            try:
                for n, line in enumerate(Path(f).read_text(errors="replace").splitlines(), 1):
                    if re.search(r"\b%s\b" % re.escape(name), line):
                        mentions.append("%s:%d: %s" % (f, n, line.strip()[:140]))
            except (OSError, UnicodeError):
                pass
        r["chain_mentions"] = mentions
        r["argv_flags"] = [a for a in _flags(chain.get("argv") or []).items()
                           if name in " ".join(l for l in mentions if "--" in l)][:0]  # filled below
        sl = (chain.get("script_lines") or {}).get(name, [])
        if chain.get("serve_script") and sl:
            text = Path(chain["serve_script"]).read_text().splitlines()
            flags = set()
            for n in sl:
                flags.update(re.findall(r"(--[\w-]+)\s+\"?\$\{?%s" % re.escape(name), text[n - 1]))
            r["argv_flags"] = [{"flag": f, "production": _flags(argv).get(f)} for f in sorted(flags)]
        not_chain = []
        for f in other_files:
            try:
                for n, line in enumerate(Path(f).read_text(errors="replace").splitlines(), 1):
                    if re.search(r"\b%s\b" % re.escape(name), line) and not line.lstrip().startswith("#"):
                        not_chain.append("%s:%d%s: %s" % (f, n, " (identical repo copy of a chain file)" if f in same_copies else "",
                                                          line.strip()[:120]))
            except (OSError, UnicodeError):
                pass
        r["mentioned_outside_chain"] = not_chain[:20]
        r["engine_readers"] = readers.get(name, [])[:8]
        d, rel = code_default(tree, name)
        r["code_default"] = {"value": d, "file": rel} if d is not None else None
        meta = ENGINE_KNOBS.get(name)
        r["schema"] = {"consumer": meta[0], "type": meta[1], "desc": meta[2]} if meta else None
        if not r["engine_readers"] and (not meta or meta[0] == "engine"):
            r["verdict"] = "DEAD: no code in %s reads it" % tree
        out.append(r)
    return out


# ------------------------------------------------------------------ text rendering
def print_chain(chain, looks):
    print("unit           %s  (%s)" % (chain["unit"], chain["fragment"]))
    for d in chain["dropins"]:
        print("  drop-in      %s" % d)
    print("main pid       %s  started %s" % (chain["main_pid"], chain["started"]))
    print("workdir        %s" % chain["workdir"])
    print("ExecStart      %s" % chain["exec_start"])
    print("dispatch       %s -> %s" % (chain["dispatch"]["note"], chain["dispatch"]["target"]))
    print("layers, in order:")
    for i, l in enumerate(chain["layers"], 1):
        extra = ""
        if "sets" in l:
            extra = "  sets %d keys%s" % (len(l["sets"]), "" if l.get("exists", True) else "  (MISSING%s)" % (", optional" if l.get("optional") else ""))
        print("  %d. %-28s %s%s" % (i, l["kind"], l["path"], extra))
    print("engine tree    %s" % chain["engine_tree"])
    if chain["errors"]:
        print("errors         %s" % "; ".join(chain["errors"]))
    fin = chain.get("final_env") or {}
    print("final engine-relevant env (%d keys):" % sum(1 for k in fin if RELEVANT.match(k)))
    for k in sorted(fin):
        if RELEVANT.match(k) and k != "PATH":
            print("  %-56s %-28s <- %s" % (k, str(mask(k, fin[k]))[:28], chain["attribution"].get(k)))
    if chain.get("argv"):
        print("argv: " + " ".join(shlex.quote(a) for a in chain["argv"])[:2000])
    print("config-looking files NOT in the chain (%d; plus %d backup variants):" % (len(looks["not_in_chain"]), looks["backup_variants_not_in_chain"]))
    for f in looks["not_in_chain"]:
        print("  " + f)
    print("deployed chain files vs repo copies (same / DIFFERS / absent):")
    roots = [str(r) for r in REPO_COPIES]
    print("  %-62s %s" % ("deployed file", "   ".join(roots)))
    for row in looks["repo_presence"]:
        print("  %-62s %s" % (row["deployed"], "   ".join("%-*s" % (len(r), row.get(r, "?")) for r in roots)))


def _main(argv):
    ap = argparse.ArgumentParser(description="engine config chain / drift / whereis-knob (read-only)")
    ap.add_argument("--unit", default=UNIT)
    ap.add_argument("--engine-tree", default=None)
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chain")
    c = sub.add_parser("check")
    c.add_argument("--expect", action="append", default=[], help="ledger claim KEY=VALUE (KEY= for absent)")
    w = sub.add_parser("whereis-knob")
    w.add_argument("names", nargs="+")
    a = ap.parse_args(argv)
    chain = build_chain(a.unit, Path(a.engine_tree) if a.engine_tree else None)
    if a.cmd == "chain":
        looks = lookalikes(chain)
        if a.json:
            d = {k: v for k, v in chain.items() if k not in ("rendered", "base_env", "history")}
            d["final_env"] = {k: mask(k, v) for k, v in chain["final_env"].items() if RELEVANT.match(k)}
            d["lookalikes"] = looks
            print(json.dumps(d, indent=1, default=str))
        else:
            print_chain(chain, looks)
        return 0 if not chain["errors"] else 2
    if a.cmd == "whereis-knob":
        res = whereis(chain, a.names)
        if a.json:
            print(json.dumps(res, indent=1))
        else:
            for r in res:
                print("== %s" % r["knob"])
                print("  production (pid %s): %s" % (r["production_pid"], r["production_value"]))
                print("  chain now gives:    %s" % r["chain_value"])
                for s in r["set_by"]:
                    print("  set by:             %s -> %s" % (s["where"], s["value"]))
                for f in r["argv_flags"]:
                    print("  feeds argv:         %s (production: %s)" % (f["flag"], f["production"]))
                for m in r["chain_mentions"]:
                    print("  in chain:           %s" % m)
                for m in r["mentioned_outside_chain"]:
                    print("  NOT in chain:       %s" % m)
                print("  engine readers:     %s" % (", ".join(r["engine_readers"]) or "NONE"))
                if r["code_default"]:
                    print("  code default:       %s (%s)" % (r["code_default"]["value"], r["code_default"]["file"]))
                if r["schema"]:
                    print("  schema:             %s/%s: %s" % (r["schema"]["consumer"], r["schema"]["type"], r["schema"]["desc"]))
                if r.get("verdict"):
                    print("  VERDICT:            %s" % r["verdict"])
        return 0
    res = check(chain, a.expect)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print("engine pid %s; serve script %s" % (res["pid"], chain.get("serve_script")))
        for k in res["children"]:
            print("  child %-7d %-24s environ %s" % (k["pid"], k["argv0"], k["environ"]))
        for f in res["findings"]:
            print("%-5s %-19s %-48s %s" % (f["level"].upper(), f["code"], f["key"], f["msg"]))
        print("DRIFT" if res["drift"] else "no drift between the chain and the running engine")
    return 1 if res["drift"] else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
