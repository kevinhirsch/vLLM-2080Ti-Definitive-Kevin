#!/usr/bin/env python3
"""release.py -- immutable engine releases (lane RL, 2026-10-03, lead L146).

Production used to boot the mutable dev worktree ~/Desktop/wt-integrate, where people also develop. Near-misses on
2026-10-03: an uncommitted fix sat in the prod tree; lane trees symlink .venv/.deps into it, so a lane boot could
JIT-rebuild tq_gqa / FlashQLA .so files INSIDE the prod tree (torch.utils.cpp_extension keys build.ninja on the source
path); a PYTHONPATH override was silently dropped. A release fixes all of that by construction:

  ~/.local/share/vllm-releases/                  (outside every git worktree, same disk as the trees, ~0.5 GB/release)
    <id>/                       = V02_ROOT. Read-only except its own caches and JIT build dirs (torch needs a lock there).
      vllm/ tools/ deploy/ ...   `git archive <sha>`: exactly the commit, nothing uncommitted can be in it
      vllm/*.so _version.py third_party/triton_kernels   compiled artifacts copied from --from TREE (csrc drift-checked)
      .deps/FlashQLA-SM70-SM75/  FlashQLA sources (vendored dep, patch hash recorded) + its prebuilt JIT .so
      .deps/tq_gqa_build/        prebuilt tq_gqa .so, built at THIS path (so a boot is a ninja no-op, verified)
      .venv -> ../venvs/<fp>     hard-linked snapshot of the source venv (no data copied; immune to pip in the dev tree)
      triton-cache/ .cache/flashinfer/ torchinductor-cache/   seeded caches (content-addressed; writable)
      RELEASE.json               manifest: sha, dirty=false, sha256 of every .so, venv fingerprint, build env, created_at
      RELEASE.files.sha256       sha256 of every immutable file (what `verify` re-checks)
    venvs/<fp>/                  one venv snapshot per distinct source-venv state, shared by releases
    current -> <id>              what production boots (serve-hauhaucs-v02.sh resolves it ONCE at boot)
    history.jsonl                every activation {ts, from, to, by, reason}; rollback = back to the previous `from`

Commands:
  release.py build <sha> [--label k5] [--from TREE] [--no-jit] [--no-seed-caches] [--allow-dirty-source] [--allow-stale-so]
  release.py list
  release.py verify [<id>|current] [--running]
  release.py activate <id> --reason R [--by X] [--no-restart] [--no-auto-rollback] [--drain-s N] [--gpu-wait-s N]
  release.py rollback --reason R [--by X] [--no-restart]
  release.py env <id>          # the override line a lane window uses: V02_ROOT=<release dir>
  release.py rm <id>           # refuses current, the rollback target and the running engine's release

Activation = verify -> GPU gate (no foreign compute app) -> flip `current` -> drained restart through
engine-actuator.py (offline window) in its own systemd unit -> health -> the engine's cwd must be the release -> verify
again (proves the boot rebuilt nothing) -> KV pool recorded. A failed activation flips back and restarts (auto-rollback).
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import fnmatch
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpuguard  # noqa: E402
import unitrun  # noqa: E402

HOME = os.path.expanduser("~")
ROOT = os.environ.get("VLLM_RELEASES_ROOT", f"{HOME}/.local/share/vllm-releases")
REPO = os.environ.get("VLLM_FORK_REPO", f"{HOME}/Desktop/vLLM-2080Ti-Definitive")
DEFAULT_FROM = os.environ.get("VLLM_RELEASE_FROM", f"{HOME}/Desktop/wt-integrate")
ENGINE_BASE = os.environ.get("ENGINE_BASE", f"{HOME}/.local/share/vllm-qwen27b")
ACTUATOR = os.environ.get("ENGINE_ACTUATOR", f"{ENGINE_BASE}/engine-actuator.py")
ENGINE_ENV_FILE = os.environ.get("ENGINE_ENV_FILE", f"{ENGINE_BASE}/vllm-qwen27b.env")
ENGINE_UNIT = os.environ.get("ENGINE_UNIT", "vllm-qwen27b")
ENGINE_URL = os.environ.get("ENGINE_URL", "http://127.0.0.1:8001")
SHIM_GCC = f"{HOME}/.local/share/shim-gcc15"
CUDA = "/usr/local/cuda-13"
MIN_FREE_BYTES = int(os.environ.get("VLLM_RELEASE_MIN_FREE_BYTES", str(5 << 30)))

# compiled artifacts that are not in git (built by setup.py into the tree); copied from --from TREE
ARTIFACT_GLOBS = ["vllm/*.so", "vllm/_version.py"]
ARTIFACT_DIRS = ["vllm/third_party/triton_kernels"]
# what the vllm/*.so are compiled from: a release whose sha differs from TREE's HEAD here would get stale .so
COMPILED_SOURCES = ["csrc", "CMakeLists.txt", "setup.py", "cmake"]
FLASHQLA = ".deps/FlashQLA-SM70-SM75"
FLASHQLA_EXT = f"{FLASHQLA}/.torch_extensions_vllm_flashqla_legacy"
TQ_BUILD = ".deps/tq_gqa_build"
# writable inside an otherwise read-only release (caches + JIT build dirs: torch takes a 'lock' file there)
WRITABLE_DIRS = ["triton-cache", "torchinductor-cache", ".cache", TQ_BUILD, f"{FLASHQLA_EXT}/flash_qla_legacy_gdn"]
# never hashed: mutable caches, lock/ninja bookkeeping, the manifest itself
HASH_SKIP_TOP = {".venv", "triton-cache", "torchinductor-cache", ".cache", "RELEASE.json", "RELEASE.files.sha256"}
HASH_SKIP_NAMES = {"lock", ".ninja_log", ".ninja_deps"}
SEED_CACHES = ["triton-cache", ".cache/flashinfer"]


class ReleaseError(Exception):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sh(argv, cwd=None, timeout=600, check=True, env=None, input_bytes=None):
    r = subprocess.run(argv, cwd=cwd, capture_output=True, timeout=timeout, env=env, input=input_bytes)
    if check and r.returncode != 0:
        raise ReleaseError(f"{' '.join(map(str, argv))[:200]} failed rc={r.returncode}: {r.stderr.decode('utf-8', 'replace')[-600:]}")
    return r


def git(*args, repo=None, **kw) -> str:
    return sh(["git", "-C", repo or REPO, *args], **kw).stdout.decode().strip()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def locked(root=None):
    root = root or ROOT
    os.makedirs(root, exist_ok=True)
    fh = open(os.path.join(root, ".lock"), "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise ReleaseError("another release.py build/activate/rm is running (lock held)")
    try:
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


# ---------------------------------------------------------------- venv

def site_packages(venv: str) -> str | None:
    lib = os.path.join(venv, "lib")
    try:
        for d in sorted(os.listdir(lib)):
            sp = os.path.join(lib, d, "site-packages")
            if d.startswith("python") and os.path.isdir(sp):
                return sp
    except OSError:
        pass
    return None


def venv_fingerprint(venv: str) -> dict:
    """Cheap, change-sensitive identity of a venv: pyvenv.cfg, the resolved interpreter, every dist-info RECORD, every .pth."""
    h = hashlib.sha256()
    try:
        h.update(open(os.path.join(venv, "pyvenv.cfg"), "rb").read())
    except OSError:
        pass
    py = os.path.realpath(os.path.join(venv, "bin", "python"))
    h.update(py.encode())
    sp = site_packages(venv)
    dists = 0
    if sp:
        for name in sorted(os.listdir(sp)):
            p = os.path.join(sp, name)
            if name.endswith(".dist-info"):
                dists += 1
                h.update(name.encode())
                rec = os.path.join(p, "RECORD")
                if os.path.exists(rec):
                    h.update(sha256_file(rec).encode())
            elif name.endswith(".pth") or (name.startswith("__editable__") and name.endswith(".py")):
                h.update(name.encode())
                h.update(open(p, "rb").read())
    return {"fingerprint": h.hexdigest()[:16], "python": py, "dists": dists, "site_packages": sp}


EDITABLE_PATTERNS = ("__editable__*vllm*", "__editable___vllm*", "__editable__*flash_qla*", "__editable___flash_qla*")


def snapshot_venv(src: str, root: str) -> dict:
    """Hard-linked, read-only-directory snapshot of `src` under <root>/venvs/<fp>/ (reused when it already exists).
    Hard links copy no file data (~70k entries -> a few MB of directories). The snapshot drops the editable-install
    finders (which point at the dev tree: a release must never fall back to importing it) and rewrites bin/ shebangs."""
    fp = venv_fingerprint(src)
    dest = os.path.join(root, "venvs", fp["fingerprint"])
    meta_path = os.path.join(dest, ".snapshot.json")
    if os.path.isdir(dest) and os.path.exists(meta_path):
        return json.load(open(meta_path))
    os.makedirs(os.path.join(root, "venvs"), exist_ok=True)
    tmp = dest + f".tmp{os.getpid()}"
    if os.path.exists(tmp):
        _rmtree(tmp)
    sh(["cp", "-al", "--", os.path.realpath(src), tmp], timeout=1800)
    sp = site_packages(tmp)
    dropped = []
    if sp:
        for name in sorted(os.listdir(sp)):
            if any(fnmatch.fnmatch(name, pat) for pat in EDITABLE_PATTERNS):
                os.unlink(os.path.join(sp, name))      # unlinking a hard link never touches the source venv
                dropped.append(name)
    old_py, new_py = os.path.join(os.path.realpath(src), "bin", "python"), os.path.join(dest, "bin", "python")
    rewritten = 0
    bindir = os.path.join(tmp, "bin")
    for name in sorted(os.listdir(bindir)):
        p = os.path.join(bindir, name)
        if os.path.islink(p) or not os.path.isfile(p):
            continue
        with open(p, "rb") as fh:
            head = fh.read(2)
            if head != b"#!":
                continue
            data = head + fh.read()
        first, _, rest = data.partition(b"\n")
        if old_py.encode() not in first:
            continue
        mode = os.stat(p).st_mode
        os.unlink(p)                                   # break the hard link before writing
        with open(p, "wb") as fh:
            fh.write(first.replace(old_py.encode(), new_py.encode()) + b"\n" + rest)
        os.chmod(p, stat.S_IMODE(mode))
        rewritten += 1
    meta = {"fingerprint": fp["fingerprint"], "source": os.path.realpath(src), "source_python": fp["python"],
            "dists": fp["dists"], "created_at": now_iso(), "dropped_editable": dropped, "shebangs_rewritten": rewritten,
            "path": dest, "snapshot_fingerprint": venv_fingerprint(tmp)["fingerprint"]}
    with open(os.path.join(tmp, ".snapshot.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    # read-only DIRECTORIES only: chmod on a hard-linked FILE would change the source venv's file too
    for d, dirs, _files in os.walk(tmp, topdown=False):
        if not os.path.islink(d):
            os.chmod(d, 0o555)
    os.rename(tmp, dest)
    meta["path"] = dest
    return meta


# ---------------------------------------------------------------- build helpers

def _rmtree(path: str) -> None:
    for d, dirs, files in os.walk(path):
        for x in dirs:
            p = os.path.join(d, x)
            if not os.path.islink(p):
                try:
                    os.chmod(p, 0o755)
                except OSError:
                    pass
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass
    shutil.rmtree(path)


def tree_dirty(tree: str) -> list[str]:
    out = sh(["git", "-C", tree, "status", "--porcelain=v1", "--untracked-files=no"], check=False).stdout.decode()
    return [line for line in out.splitlines() if line.strip()]


def so_drift(tree: str, sha: str) -> dict:
    """Are the compiled .so in TREE valid for `sha`? (a) TREE HEAD and sha agree on every compiled source path, and
    (b) the .so are not older than the last commit (at TREE HEAD) that touched a compiled source."""
    head = git("rev-parse", "HEAD", repo=tree)
    diff = sh(["git", "-C", REPO, "diff", "--name-only", head, sha, "--", *COMPILED_SOURCES], check=False).stdout.decode().split()
    last = sh(["git", "-C", tree, "log", "-1", "--format=%ct", "HEAD", "--", *COMPILED_SOURCES], check=False).stdout.decode().strip()
    sos = [os.path.join(tree, "vllm", f) for f in os.listdir(os.path.join(tree, "vllm")) if f.endswith(".so")] if os.path.isdir(os.path.join(tree, "vllm")) else []
    oldest = min((os.stat(p).st_mtime for p in sos), default=None)
    stale = bool(last and oldest is not None and oldest + 60 < int(last))
    return {"tree_head": head, "compiled_source_diff": diff, "last_compiled_source_commit": int(last) if last else None,
            "oldest_so_mtime": oldest, "so_older_than_sources": stale}


def copy_artifacts(tree: str, dest: str) -> list[str]:
    copied = []
    for pat in ARTIFACT_GLOBS:
        d, base = os.path.split(pat)
        src_dir = os.path.join(tree, d)
        if not os.path.isdir(src_dir):
            continue
        for name in sorted(os.listdir(src_dir)):
            if fnmatch.fnmatch(name, base):
                os.makedirs(os.path.join(dest, d), exist_ok=True)
                shutil.copy2(os.path.join(src_dir, name), os.path.join(dest, d, name))   # follows lane-tree symlinks
                copied.append(os.path.join(d, name))
    for rel in ARTIFACT_DIRS:
        src = os.path.join(tree, rel)
        if os.path.isdir(src) and not os.path.exists(os.path.join(dest, rel)):
            shutil.copytree(src, os.path.join(dest, rel), symlinks=False,
                            ignore=shutil.ignore_patterns("__pycache__"))
            copied.append(rel + "/")
    return copied


def copy_flashqla(tree: str, dest: str) -> dict | None:
    src = os.path.realpath(os.path.join(tree, FLASHQLA))
    if not os.path.isdir(src):
        return None
    info = {"source": src}
    if os.path.isdir(os.path.join(src, ".git")):
        info["head"] = sh(["git", "-C", src, "rev-parse", "HEAD"], check=False).stdout.decode().strip()
        patch = sh(["git", "-C", src, "diff", "HEAD"], check=False).stdout
        info["dirty"] = bool(patch.strip())
        info["patch_sha256"] = hashlib.sha256(patch).hexdigest()
    shutil.copytree(src, os.path.join(dest, FLASHQLA), symlinks=False,
                    ignore=shutil.ignore_patterns(".git", ".torch_extensions*", "__pycache__", "*.egg-info", "build"))
    return info


def engine_env_value(key: str, default: str | None = None) -> str | None:
    try:
        for line in open(ENGINE_ENV_FILE):
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return default


def build_env(release: str, venv: str) -> dict:
    """The JIT-relevant environment serve-hauhaucs-v02.sh gives the engine, but with NO GPU visible (a release build
    must never touch the production GPUs) and the arch pinned (torch would otherwise query a device)."""
    env = {
        "HOME": HOME, "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PATH": f"{SHIM_GCC}:{venv}/bin:{CUDA}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/snap/bin",
        "CUDA_HOME": CUDA, "CUDA_PATH": CUDA, "CC": "/usr/bin/gcc-15", "CXX": "/usr/bin/g++-15",
        "CUDAHOSTCXX": "/usr/bin/g++-15", "NVCC_CCBIN": "/usr/bin/g++-15",
        "TORCH_CUDA_ARCH_LIST": engine_env_value("TORCH_CUDA_ARCH_LIST", "7.5"),
        "TORCH_EXTENSIONS_DIR": f"{release}/{FLASHQLA_EXT}", "FLASHQLA_ROOT": f"{release}/{FLASHQLA}",
        "VLLM_TQ_GQA_BUILD_DIR": f"{release}/{TQ_BUILD}", "PYTHONPATH": f"{release}:{release}/{FLASHQLA}",
        "PYTHONSAFEPATH": "1", "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "", "MAX_JOBS": "8", "R": release,
    }
    return env


JIT_SCRIPT = r"""
import hashlib, importlib.util, json, os, sys, time
R = os.environ["R"]
import torch
out = {}
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
src = os.path.join(R, "vllm/v1/attention/ops/tq_gqa_cuda.py")
if os.path.exists(src):
    t = time.time()
    spec = importlib.util.spec_from_file_location("rl_tq_gqa_cuda", src)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m._load()
    so = os.path.join(os.environ["VLLM_TQ_GQA_BUILD_DIR"], "tq_gqa_sm75.so")
    out["tq_gqa"] = {"so": os.path.relpath(so, R), "sha256": sha(so), "s": round(time.time() - t, 1)}
src = os.path.join(os.environ["FLASHQLA_ROOT"], "flash_qla/ops/gated_delta_rule/legacy/sm_legacy.py")
if os.path.exists(src):
    t = time.time()
    torch.cuda.is_available = lambda: True      # build only: no GPU is visible on purpose; nothing runs on a device
    spec = importlib.util.spec_from_file_location("vllm_flashqla_sm75_legacy", src)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m._load_ext()
    import glob
    hits = sorted(glob.glob(os.path.join(os.environ["TORCH_EXTENSIONS_DIR"], "**", "flash_qla_legacy_gdn.so"), recursive=True))
    so = hits[0] if hits else os.path.join(os.environ["TORCH_EXTENSIONS_DIR"], "flash_qla_legacy_gdn", "flash_qla_legacy_gdn.so")
    out["flashqla_legacy_gdn"] = {"so": os.path.relpath(so, R), "sha256": sha(so), "s": round(time.time() - t, 1)}
for ext in json.loads(os.environ.get("RL_EXTRA_JIT") or "[]"):
    # lane extensions: {"module": rel path of the python wrapper, "env": build-dir env var, "dir": rel build dir}
    t = time.time()
    os.environ[ext["env"]] = os.path.join(R, ext["dir"])
    os.makedirs(os.environ[ext["env"]], exist_ok=True)
    spec = importlib.util.spec_from_file_location("rl_extra_" + ext["env"].lower(), os.path.join(R, ext["module"]))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    getattr(m, ext.get("loader", "_load"))()
    import glob
    hits = sorted(glob.glob(os.path.join(R, ext["dir"], "*.so")))
    out["extra:" + ext["env"]] = {"so": os.path.relpath(hits[0], R) if hits else None, "sha256": sha(hits[0]) if hits else None,
                                  "s": round(time.time() - t, 1), "env": ext["env"], "dir": ext["dir"]}
print("JIT-RESULT " + json.dumps(out))
"""


def parse_extra_jit(specs) -> list[dict]:
    """--jit-ext MODULE:ENV[:DIR] (repeatable): a lane's torch cpp_extension wrapper (a .py with _load()) whose build dir
    comes from ENV. Prebuilt inside the release at DIR (default .deps/<basename of the module dir>_build)."""
    out = []
    for s in specs or []:
        parts = s.split(":")
        if len(parts) not in (2, 3) or not parts[0].endswith(".py") or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", parts[1]):
            raise ReleaseError(f"--jit-ext wants MODULE.py:ENV_VAR[:DIR], got {s!r}")
        mod = parts[0]
        d = parts[2] if len(parts) == 3 else f".deps/{os.path.basename(os.path.dirname(mod)) or 'ext'}_build"
        if os.path.isabs(d) or ".." in d.split("/"):
            raise ReleaseError(f"--jit-ext DIR must be relative inside the release: {d!r}")
        out.append({"module": mod, "env": parts[1], "dir": d})
    return out


def prebuild_jit(release: str, venv: str, extra=None) -> dict:
    env = build_env(release, venv)
    env["RL_EXTRA_JIT"] = json.dumps(extra or [])
    py = os.path.join(venv, "bin", "python")
    res = {}
    for rnd in ("build", "noop-check"):
        r = sh(["nice", "-n", "15", py, "-c", JIT_SCRIPT], cwd=release, env=env, timeout=3600, check=False)
        line = [x for x in r.stdout.decode("utf-8", "replace").splitlines() if x.startswith("JIT-RESULT ")]
        if r.returncode != 0 or not line:
            raise ReleaseError(f"JIT prebuild ({rnd}) failed rc={r.returncode}: {r.stderr.decode('utf-8', 'replace')[-1500:]}")
        res[rnd] = json.loads(line[-1][len("JIT-RESULT "):])
    # the second load must be a ninja no-op: same .so bytes. A boot with the same env is then a no-op as well.
    for k, v in res["build"].items():
        if res["noop-check"].get(k, {}).get("sha256") != v.get("sha256"):
            raise ReleaseError(f"JIT extension {k} rebuilt on a second load: the boot would rebuild it too")
    return {k: {**v, "noop_check_s": res["noop-check"][k]["s"]} for k, v in res["build"].items()}


def compile_pyc(release: str, venv: str) -> None:
    """Pre-compile .pyc: the release directories are read-only, so Python could not cache bytecode at boot."""
    py = os.path.join(venv, "bin", "python")
    targets = [p for p in (os.path.join(release, "vllm"), os.path.join(release, FLASHQLA, "flash_qla")) if os.path.isdir(p)]
    if targets:
        sh([py, "-m", "compileall", "-q", "-j", "8", *targets], timeout=1800, check=False,
           env={"PATH": "/usr/bin:/bin", "HOME": HOME})


def hash_tree(release: str) -> list[tuple[str, str]]:
    rows = []
    for d, dirs, files in os.walk(release):
        rel_d = os.path.relpath(d, release)
        if rel_d == ".":
            dirs[:] = [x for x in dirs if x not in HASH_SKIP_TOP]
        dirs.sort()
        for x in dirs:
            p = os.path.join(d, x)
            if os.path.islink(p):
                rows.append((os.path.relpath(p, release), "link:" + os.readlink(p)))
        for f in sorted(files):
            if rel_d == "." and f in HASH_SKIP_TOP:
                continue
            if f in HASH_SKIP_NAMES or f.startswith("lock."):
                continue
            p = os.path.join(d, f)
            if os.path.islink(p):
                rows.append((os.path.relpath(p, release), "link:" + os.readlink(p)))
            elif os.path.isfile(p):
                rows.append((os.path.relpath(p, release), sha256_file(p)))
    return rows


def freeze(release: str, extra_writable=()) -> None:
    writable = {os.path.join(release, w) for w in list(WRITABLE_DIRS) + list(extra_writable)}
    for d, dirs, files in os.walk(release, topdown=True):
        if any(d == w or d.startswith(w + os.sep) for w in writable):
            # caches: keep writable; JIT build dirs: dir writable (torch's lock), the .so itself read-only
            for f in files:
                if f.endswith(".so"):
                    p = os.path.join(d, f)
                    os.chmod(p, stat.S_IMODE(os.stat(p).st_mode) & ~0o222)
            if os.path.basename(d) in ("triton-cache", "torchinductor-cache", ".cache"):
                dirs[:] = []
            continue
        if os.path.islink(d):
            continue
        for f in files:
            p = os.path.join(d, f)
            if not os.path.islink(p):
                os.chmod(p, stat.S_IMODE(os.stat(p).st_mode) & ~0o222)
    for d, dirs, _files in os.walk(release, topdown=False):
        if os.path.islink(d) or any(d == w or d.startswith(w + os.sep) for w in writable):
            continue
        os.chmod(d, stat.S_IMODE(os.stat(d).st_mode) & ~0o222)


def du_bytes(path: str) -> int:
    seen, total = set(), 0
    for d, dirs, files in os.walk(path):
        for f in files:
            p = os.path.join(d, f)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if (st.st_dev, st.st_ino) in seen:
                continue
            seen.add((st.st_dev, st.st_ino))
            total += st.st_blocks * 512
    return total


def build(sha: str, *, label: str | None = None, from_tree: str = DEFAULT_FROM, jit: bool = True, seed_caches: bool = True,
          allow_dirty_source: bool = False, allow_stale_so: bool = False, root: str | None = None, by: str = "cli",
          log=print, jit_ext=None) -> dict:
    root = root or ROOT
    extra = parse_extra_jit(jit_ext)
    full = git("rev-parse", "--verify", f"{sha}^{{commit}}")
    os.makedirs(root, exist_ok=True)
    st = os.statvfs(root)
    free = st.f_bavail * st.f_frsize
    if free < MIN_FREE_BYTES:
        raise ReleaseError(f"only {free >> 30} GiB free on the release disk (< {MIN_FREE_BYTES >> 30} GiB): refusing")
    from_tree = os.path.abspath(from_tree)
    dirty = tree_dirty(from_tree)
    if dirty and not allow_dirty_source:
        raise ReleaseError(f"--from tree {from_tree} has uncommitted changes to tracked files ({len(dirty)}: "
                           f"{'; '.join(dirty[:5])}). The release is built from {full[:10]} and would NOT contain them, so it "
                           f"would differ from anything booted from that tree. Commit them (or pass --allow-dirty-source).")
    drift = so_drift(from_tree, full)
    if (drift["compiled_source_diff"] or drift["so_older_than_sources"]) and not allow_stale_so:
        raise ReleaseError(f"compiled .so in {from_tree} do not match {full[:10]}: compiled sources differ "
                           f"{drift['compiled_source_diff'][:8]} / .so older than the last csrc commit={drift['so_older_than_sources']}. "
                           f"Rebuild them in that tree first (or pass --allow-stale-so).")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    rid = f"{label}-{full[:10]}-{stamp}" if label else f"{full[:10]}-{stamp}"
    rid = unitrun._NAME_OK.sub("-", rid)
    with locked(root):
        dest = os.path.join(root, rid)
        if os.path.exists(dest):
            raise ReleaseError(f"release {rid} already exists")
        stage = os.path.join(root, f".staging-{rid}")
        if os.path.exists(stage):
            _rmtree(stage)
        os.makedirs(stage)
        t0 = time.time()
        try:
            log(f"[1/7] git archive {full[:10]} -> {stage}")
            ga = subprocess.Popen(["git", "-C", REPO, "archive", "--format=tar", full], stdout=subprocess.PIPE)
            with tarfile.open(fileobj=ga.stdout, mode="r|") as tf:
                tf.extractall(stage, filter="tar")
            if ga.wait() != 0:
                raise ReleaseError(f"git archive {full} failed rc={ga.returncode}")
            log(f"[2/7] compiled artifacts from {from_tree}")
            artifacts = copy_artifacts(from_tree, stage)
            flashqla = copy_flashqla(from_tree, stage)
            log("[3/7] venv snapshot (hard links)")
            src_venv = os.path.realpath(os.path.join(from_tree, ".venv"))
            vmeta = snapshot_venv(src_venv, root)
            os.symlink(os.path.relpath(vmeta["path"], dest), os.path.join(stage, ".venv"))
            for d in ("triton-cache", "torchinductor-cache", ".cache", TQ_BUILD):
                os.makedirs(os.path.join(stage, d), exist_ok=True)
            seeded = []
            if seed_caches:
                log("[4/7] seed content-addressed caches")
                for rel in SEED_CACHES:
                    src = os.path.join(from_tree, rel)
                    if os.path.isdir(src):
                        tgt = os.path.join(stage, rel)
                        if os.path.isdir(tgt) and not os.listdir(tgt):
                            os.rmdir(tgt)
                        shutil.copytree(src, tgt, symlinks=True, dirs_exist_ok=True)
                        seeded.append(rel)
            # the JIT build must happen at the FINAL path (ninja files embed it); build in place after the rename
            os.rename(stage, dest)
            stage = None
            jit_res = {}
            # The interpreter is invoked as <release>/.venv/bin/python, exactly as serve-hauhaucs-v02.sh does: sys.prefix,
            # the torch include paths in build.ninja and FlashInfer's JIT paths all derive from that spelling.
            rel_venv = os.path.join(dest, ".venv")
            if jit:
                log("[5/7] JIT prebuild at the final path (no GPU visible) + no-op re-load check")
                for e in extra:
                    if not os.path.exists(os.path.join(dest, e["module"])):
                        raise ReleaseError(f"--jit-ext module {e['module']} is not in {full[:10]}")
                jit_res = prebuild_jit(dest, rel_venv, extra)
            log("[6/7] pre-compile bytecode, hash, freeze")
            compile_pyc(dest, rel_venv)
            rows = hash_tree(dest)
            with open(os.path.join(dest, "RELEASE.files.sha256"), "w") as fh:
                for rel, h in rows:
                    fh.write(f"{h}  {rel}\n")
            sos = {rel: h for rel, h in rows if rel.endswith(".so")}
            nvcc = sh([f"{CUDA}/bin/nvcc", "--version"], check=False).stdout.decode().strip().splitlines()[-1:] if os.path.exists(f"{CUDA}/bin/nvcc") else []
            gcc = sh(["/usr/bin/gcc-15", "--version"], check=False).stdout.decode().splitlines()[:1] if os.path.exists("/usr/bin/gcc-15") else []
            benv = build_env(dest, rel_venv)
            manifest = {
                "id": rid, "sha": full, "label": label, "dirty": False, "created_at": now_iso(), "created_by": by,
                "repo": REPO, "subject": git("log", "-1", "--format=%s", full),
                "source": {"from_tree": from_tree, "from_tree_head": drift["tree_head"], "from_tree_dirty": dirty,
                           "allow_dirty_source": allow_dirty_source, "allow_stale_so": allow_stale_so, "so_drift": drift},
                "artifacts": artifacts, "deps": {"flashqla": flashqla}, "jit": jit_res, "seeded_caches": seeded,
                "so": sos, "venv": vmeta,
                "build_env": {k: benv[k] for k in ("CUDA_HOME", "CC", "CXX", "NVCC_CCBIN", "TORCH_CUDA_ARCH_LIST")} | {
                    "nvcc": nvcc[0] if nvcc else None, "gcc": gcc[0] if gcc else None},
                "files": len(rows), "files_sha256": sha256_file(os.path.join(dest, "RELEASE.files.sha256")),
                "writable_dirs": list(WRITABLE_DIRS) + [e["dir"] for e in extra],
                "extra_jit": extra,
                "boot_env": {e["env"]: os.path.join(dest, e["dir"]) for e in extra},
                "size_bytes": du_bytes(dest),
            }
            with open(os.path.join(dest, "RELEASE.json"), "w") as fh:
                json.dump(manifest, fh, indent=1)
            freeze(dest, [e["dir"] for e in extra])
            log(f"[7/7] release {rid} built in {round(time.time() - t0)} s, {manifest['size_bytes'] >> 20} MiB "
                f"(venv shared: {vmeta['path']})")
            return manifest
        except BaseException:
            for p in (stage,):
                if p and os.path.exists(p):
                    _rmtree(p)
            if os.path.exists(dest) and not os.path.exists(os.path.join(dest, "RELEASE.json")):
                _rmtree(dest)
            raise


# ---------------------------------------------------------------- query

def release_dir(rid: str, root: str | None = None) -> str:
    root = root or ROOT
    if rid in ("current", "previous"):
        p = os.path.join(root, "current")
        if rid == "previous":
            prev = rollback_target(root)
            if not prev:
                raise ReleaseError("no previous release in history")
            return os.path.join(root, prev)
        if not os.path.islink(p):
            raise ReleaseError("no current release (production still boots its legacy tree)")
        return os.path.realpath(p)
    p = os.path.join(root, rid)
    if not os.path.exists(os.path.join(p, "RELEASE.json")):
        raise ReleaseError(f"no such release: {rid}")
    return p


def manifest_of(rid: str, root: str | None = None) -> dict:
    return json.load(open(os.path.join(release_dir(rid, root), "RELEASE.json")))


def current_id(root: str | None = None) -> str | None:
    p = os.path.join(root or ROOT, "current")
    return os.path.basename(os.path.realpath(p)) if os.path.islink(p) else None


def history(root: str | None = None) -> list[dict]:
    rows = []
    try:
        for line in open(os.path.join(root or ROOT, "history.jsonl")):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return rows


def rollback_target(root: str | None = None) -> str | None:
    """The release that was current before the current one (None = the legacy tree / nothing)."""
    cur = current_id(root)
    for row in reversed(history(root)):
        if row.get("to") == cur and row.get("kind") in ("activate", "rollback"):
            return row.get("from")
    return None


def list_releases(root: str | None = None) -> list[dict]:
    root = root or ROOT
    cur, prev = current_id(root), rollback_target(root)
    run_root = running_root()
    out = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        names = []
    for n in names:
        mf = os.path.join(root, n, "RELEASE.json")
        if not os.path.exists(mf):
            continue
        m = json.load(open(mf))
        out.append({"id": n, "sha": m["sha"][:10], "label": m.get("label"), "created_at": m.get("created_at"),
                    "subject": (m.get("subject") or "")[:70], "size_mib": (m["size_bytes"] >> 20) if m.get("size_bytes") else None,
                    "current": n == cur, "rollback_target": n == prev,
                    "running": bool(run_root and os.path.realpath(os.path.join(root, n)) == run_root)})
    return out


def engine_main_pid() -> int | None:
    r = subprocess.run(["systemctl", "show", "-p", "MainPID", "--value", ENGINE_UNIT], capture_output=True, text=True)
    try:
        pid = int((r.stdout or "0").strip())
    except ValueError:
        return None
    return pid or None


def running_root() -> str | None:
    """The directory the RUNNING engine booted from (its cwd: the serve script cd's into V02_ROOT)."""
    pid = engine_main_pid()
    if not pid:
        return None
    try:
        return os.path.realpath(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        return None


def verify(rid: str = "current", root: str | None = None, running: bool = False) -> dict:
    d = release_dir(rid, root)
    m = json.load(open(os.path.join(d, "RELEASE.json")))
    want = {}
    for line in open(os.path.join(d, "RELEASE.files.sha256")):
        h, _, rel = line.rstrip("\n").partition("  ")
        want[rel] = h
    have = dict(hash_tree(d))
    mism = sorted(k for k in want if have.get(k) != want[k])
    missing = sorted(k for k in want if k not in have)
    extra = sorted(k for k in have if k not in want)
    so_bad = sorted(k for k, h in (m.get("so") or {}).items() if have.get(k) != h)
    files_ok = sha256_file(os.path.join(d, "RELEASE.files.sha256")) == m.get("files_sha256")
    venv = os.path.realpath(os.path.join(d, ".venv"))
    vfp = venv_fingerprint(venv)["fingerprint"] if os.path.isdir(venv) else None
    venv_ok = vfp is not None and vfp == (m.get("venv") or {}).get("snapshot_fingerprint")
    res = {"id": os.path.basename(d), "ok": not (mism or missing or extra or so_bad) and files_ok and venv_ok,
           "changed": [k for k in mism if k not in missing][:50], "missing": missing[:50], "unexpected": extra[:50],
           "so_changed": so_bad, "manifest_hashes_ok": files_ok, "venv_ok": venv_ok, "venv_fingerprint": vfp,
           "files": len(want)}
    if running:
        rr = running_root()
        res["running_root"] = rr
        res["engine_runs_this_release"] = bool(rr and rr == os.path.realpath(d))
    return res


# ---------------------------------------------------------------- activate / rollback

def _flip(root: str, target: str | None) -> None:
    link = os.path.join(root, "current")
    if target is None:
        if os.path.islink(link):
            os.unlink(link)
        return
    tmp = os.path.join(root, f".current.tmp{os.getpid()}")
    if os.path.lexists(tmp):
        os.unlink(tmp)
    os.symlink(target, tmp)
    os.replace(tmp, link)


def _log_history(root: str, row: dict) -> None:
    with open(os.path.join(root, "history.jsonl"), "a") as fh:
        fh.write(json.dumps(row) + "\n")


def engine_healthy(timeout=3) -> bool:
    try:
        with urllib.request.urlopen(f"{ENGINE_URL}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def restart_engine(reason: str, by: str, drain_s: int, tag: str) -> dict:
    out = os.path.join(ROOT, "logs", f"{tag}.log")
    res = unitrun.run("release", tag, ["/usr/bin/python3", ACTUATOR, "restart", "--reason", reason, "--by", by,
                                       "--foreground", "--drain-s", str(drain_s)],
                      timeout_s=2400, out=out, env={"PYTHONUNBUFFERED": "1"})
    res["healthy"] = engine_healthy()
    return res


def activate(rid: str, *, reason: str, by: str = "cli", restart: bool = True, auto_rollback: bool = True,
             drain_s: int = 120, gpu_wait_s: int = 300, kv_tolerance: float = 0.005, root: str | None = None,
             _kind: str = "activate", restart_fn=None) -> dict:
    root = root or ROOT
    restart_fn = restart_fn or restart_engine
    target = os.path.basename(release_dir(rid, root)) if rid != "legacy" else None
    if len(reason.strip()) < 8:
        raise ReleaseError("--reason is required (what is being switched and why)")
    with locked(root):
        if target:
            v = verify(target, root)
            if not v["ok"]:
                raise ReleaseError(f"release {target} fails verification, refusing to activate: {json.dumps(v)[:600]}")
        prev = current_id(root)
        res = {"from": prev, "to": target, "kind": _kind, "by": by, "reason": reason, "ts": now_iso()}
        if not restart:
            _flip(root, target)
            _log_history(root, {**res, "restart": False})
            return {**res, "restart": False, "note": "pointer flipped; takes effect at the next engine start"}
        ok, foreign = gpuguard.wait_no_foreign(gpu_wait_s)
        if not ok:
            raise ReleaseError(f"foreign compute apps on the GPUs would shrink the KV pool of the new boot: {gpuguard.describe(foreign)}")
        pool_before = gpuguard.kv_pool_since()
        gpuguard.set_busy(by, f"release activate {target}", ttl_s=3600, window=f"release-{target}", phase="boot")
        try:
            _flip(root, target)
            _log_history(root, {**res, "restart": True})
            t_boot = time.time()
            rr = restart_fn(f"release activate {target or 'legacy'} (from {prev or 'legacy'}): {reason}", by, drain_s,
                            f"activate-{(target or 'legacy')[:60]}")
            res["restart_result"] = rr
            res["running_root"] = running_root()
            res["kv_pool_before"] = pool_before
            res["kv_pool_after"] = gpuguard.kv_pool_since(t_boot - 5)
            want_root = os.path.realpath(os.path.join(root, target)) if target else None
            problems = []
            if not rr.get("healthy"):
                problems.append("engine not healthy after restart")
            if target and res["running_root"] != want_root:
                problems.append(f"engine runs from {res['running_root']}, expected {want_root}")
            if target:
                v2 = verify(target, root)
                res["verify_after_boot"] = {k: v2[k] for k in ("ok", "changed", "so_changed")}
                if not v2["ok"]:
                    problems.append("release changed during boot (JIT rebuild or writes into the release)")
            if pool_before and res["kv_pool_after"] and res["kv_pool_after"] < pool_before * (1 - kv_tolerance):
                res["kv_pool_degraded"] = True   # reported; a smaller pool alone is not a reason to roll back code
            res["problems"] = problems
            res["ok"] = not problems
            if problems and auto_rollback:
                _flip(root, prev)
                _log_history(root, {"from": target, "to": prev, "kind": "auto-rollback", "by": by,
                                    "reason": "; ".join(problems), "ts": now_iso(), "restart": True})
                rb = restart_fn(f"auto-rollback of {target}: {'; '.join(problems)}"[:200], by, drain_s,
                                f"rollback-{(prev or 'legacy')[:60]}")
                res["auto_rollback"] = {"to": prev, "restart_result": rb, "running_root": running_root()}
            return res
        finally:
            gpuguard.clear_busy(f"release-{target}")


def rollback(*, reason: str, by: str = "cli", restart: bool = True, root: str | None = None, **kw) -> dict:
    root = root or ROOT
    if not current_id(root):
        raise ReleaseError("no current release: nothing to roll back")
    prev = rollback_target(root)
    return activate(prev or "legacy", reason=reason, by=by, restart=restart, root=root, _kind="rollback", **kw)


def rm(rid: str, root: str | None = None) -> dict:
    root = root or ROOT
    d = release_dir(rid, root)
    name = os.path.basename(d)
    with locked(root):
        if name == current_id(root):
            raise ReleaseError(f"{name} is current")
        if name == rollback_target(root):
            raise ReleaseError(f"{name} is the rollback target")
        rr = running_root()
        if rr and rr == os.path.realpath(d):
            raise ReleaseError(f"the running engine boots from {name}")
        _rmtree(d)
    return {"removed": name}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("build")
    p.add_argument("sha")
    p.add_argument("--label")
    p.add_argument("--from", dest="from_tree", default=DEFAULT_FROM)
    p.add_argument("--no-jit", action="store_true")
    p.add_argument("--no-seed-caches", action="store_true")
    p.add_argument("--allow-dirty-source", action="store_true")
    p.add_argument("--allow-stale-so", action="store_true")
    p.add_argument("--jit-ext", action="append", default=[], metavar="MODULE.py:ENV[:DIR]",
                   help="also prebuild a lane JIT extension inside the release (repeatable)")
    p.add_argument("--by", default=os.environ.get("USER", "cli"))
    sp.add_parser("list")
    p = sp.add_parser("verify")
    p.add_argument("id", nargs="?", default="current")
    p.add_argument("--running", action="store_true")
    for name in ("activate", "rollback"):
        p = sp.add_parser(name)
        if name == "activate":
            p.add_argument("id")
        p.add_argument("--reason", required=True)
        p.add_argument("--by", default=os.environ.get("USER", "cli"))
        p.add_argument("--no-restart", action="store_true")
        p.add_argument("--no-auto-rollback", action="store_true")
        p.add_argument("--drain-s", type=int, default=120)
        p.add_argument("--gpu-wait-s", type=int, default=300)
    p = sp.add_parser("env")
    p.add_argument("id")
    p = sp.add_parser("rm")
    p.add_argument("id")
    a = ap.parse_args(argv)
    if a.cmd in ("activate", "rollback", "rm", "build") and os.environ.get("PYTEST_CURRENT_TEST") \
            and os.path.realpath(ROOT) == os.path.realpath(f"{__import__('pwd').getpwuid(os.getuid()).pw_dir}/.local/share/vllm-releases"):
        print(json.dumps({"error": f"release {a.cmd} on the LIVE release root from a test run (PYTEST_CURRENT_TEST set)"}), file=sys.stderr)
        return 4
    try:
        if a.cmd == "build":
            m = build(a.sha, label=a.label, from_tree=a.from_tree, jit=not a.no_jit, seed_caches=not a.no_seed_caches,
                      allow_dirty_source=a.allow_dirty_source, allow_stale_so=a.allow_stale_so, by=a.by, jit_ext=a.jit_ext,
                      log=lambda s: print(s, file=sys.stderr, flush=True))
            print(json.dumps({"id": m["id"], "sha": m["sha"], "dir": os.path.join(ROOT, m["id"]), "so": m["so"],
                              "boot_env": m.get("boot_env") or {},
                              "jit": m["jit"], "size_mib": m["size_bytes"] >> 20}, indent=1))
        elif a.cmd == "list":
            print(json.dumps({"current": current_id(), "rollback_target": rollback_target(), "running_root": running_root(),
                              "releases": list_releases()}, indent=1))
        elif a.cmd == "verify":
            v = verify(a.id, running=a.running)
            print(json.dumps(v, indent=1))
            return 0 if v["ok"] else 1
        elif a.cmd == "activate":
            r = activate(a.id, reason=a.reason, by=a.by, restart=not a.no_restart, auto_rollback=not a.no_auto_rollback,
                         drain_s=a.drain_s, gpu_wait_s=a.gpu_wait_s)
            print(json.dumps(r, indent=1, default=str))
            return 0 if r.get("ok", True) else 1
        elif a.cmd == "rollback":
            r = rollback(reason=a.reason, by=a.by, restart=not a.no_restart, auto_rollback=not a.no_auto_rollback,
                         drain_s=a.drain_s, gpu_wait_s=a.gpu_wait_s)
            print(json.dumps(r, indent=1, default=str))
            return 0 if r.get("ok", True) else 1
        elif a.cmd == "env":
            print(f"V02_ROOT={release_dir(a.id)}")
        elif a.cmd == "rm":
            print(json.dumps(rm(a.id)))
    except ReleaseError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
