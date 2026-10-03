"""RL (L104): immutable engine releases -- build from a sha, freeze, verify, activate via the pointer, roll back."""
import json
import os
import stat
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpuguard  # noqa: E402
import release  # noqa: E402


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def world(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "vllm").mkdir(parents=True)
    (repo / "csrc").mkdir()
    (repo / "vllm" / "__init__.py").write_text("X = 1\n")
    (repo / "vllm" / "mod.py").write_text("def f():\n    return 2\n")
    (repo / "csrc" / "a.cu").write_text("// kernel\n")
    (repo / "setup.py").write_text("# setup\n")
    (repo / ".gitignore").write_text("*.so\n_version.py\n.venv\n.deps\n")
    git(repo, "init", "-q")
    git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "c1")
    sha = git(repo, "rev-parse", "HEAD")
    tree = tmp_path / "tree"
    git(repo, "worktree", "add", "-q", "--detach", str(tree), sha)
    (tree / "vllm" / "_C.abi3.so").write_bytes(b"\x7fELF compiled")
    (tree / "vllm" / "_version.py").write_text("__version__ = '0.2.2'\n")
    fq = tree / ".deps" / "FlashQLA-SM70-SM75" / "flash_qla"
    fq.mkdir(parents=True)
    (fq / "__init__.py").write_text("")
    venv = tree / ".venv"
    sp = venv / "lib" / "python3.12" / "site-packages"
    (sp / "pkg-1.0.dist-info").mkdir(parents=True)
    (sp / "pkg-1.0.dist-info" / "RECORD").write_text("pkg/__init__.py,sha256=x,1\n")
    (sp / "pkg").mkdir()
    (sp / "pkg" / "__init__.py").write_text("V = 1\n")
    (sp / "__editable__.vllm-0.2.pth").write_text("import __editable___vllm_finder\n")
    (sp / "__editable___vllm_finder.py").write_text(f"MAPPING = {{'vllm': '{tree}/vllm'}}\n")
    (venv / "bin").mkdir()
    (venv / "pyvenv.cfg").write_text(f"home = {os.path.dirname(sys.executable)}\n")
    os.symlink(sys.executable, venv / "bin" / "python")
    (venv / "bin" / "tool").write_text(f"#!{os.path.realpath(venv)}/bin/python\nprint(1)\n")
    (venv / "bin" / "tool").chmod(0o755)
    (tree / "triton-cache" / "ABC").mkdir(parents=True)
    (tree / "triton-cache" / "ABC" / "k.cubin").write_bytes(b"cubin")
    root = tmp_path / "releases"
    monkeypatch.setattr(release, "REPO", str(repo))
    monkeypatch.setattr(release, "ROOT", str(root))
    monkeypatch.setattr(release, "MIN_FREE_BYTES", 0)
    monkeypatch.setattr(gpuguard, "BUSY_FILE", str(tmp_path / "busy.json"))
    monkeypatch.setattr(gpuguard, "wait_no_foreign", lambda *a, **k: (True, []))
    monkeypatch.setattr(gpuguard, "kv_pool_since", lambda *a, **k: 900000)
    monkeypatch.setattr(release, "running_root", lambda: os.path.realpath(root / "current") if (root / "current").exists() else "/legacy")
    return {"repo": repo, "tree": tree, "sha": sha, "root": root, "venv": venv}


def _build(w, **kw):
    kw.setdefault("from_tree", str(w["tree"]))
    kw.setdefault("jit", False)
    return release.build(kw.pop("sha", w["sha"]), root=str(w["root"]), log=lambda s: None, **kw)


def test_build_freezes_an_exact_commit_with_manifest(world):
    m = _build(world)
    d = world["root"] / m["id"]
    assert m["sha"] == world["sha"] and m["dirty"] is False
    assert (d / "vllm" / "mod.py").read_text() == "def f():\n    return 2\n"
    assert m["so"]["vllm/_C.abi3.so"] == release.sha256_file(str(world["tree"] / "vllm" / "_C.abi3.so"))
    assert "vllm/_version.py" in m["artifacts"]
    assert m["created_at"] and m["build_env"]["CUDA_HOME"] and m["files"] > 0
    # read-only code, writable caches
    assert not os.access(d / "vllm" / "mod.py", os.W_OK)
    assert not os.access(d / "vllm", os.W_OK)
    assert os.access(d / "triton-cache", os.W_OK) and os.access(d / ".deps" / "tq_gqa_build", os.W_OK)
    assert (d / "triton-cache" / "ABC" / "k.cubin").read_bytes() == b"cubin"       # seeded
    json.loads((d / "RELEASE.json").read_text())
    assert release.verify(m["id"], root=str(world["root"]))["ok"]


def test_venv_snapshot_is_hardlinked_without_the_dev_tree_finders(world):
    m = _build(world)
    v = os.path.realpath(world["root"] / m["id"] / ".venv")
    assert v.startswith(str(world["root"] / "venvs"))
    src = world["venv"] / "lib" / "python3.12" / "site-packages" / "pkg" / "__init__.py"
    dst = os.path.join(v, "lib", "python3.12", "site-packages", "pkg", "__init__.py")
    assert os.stat(src).st_ino == os.stat(dst).st_ino            # hard link: no data copied
    assert not os.path.exists(os.path.join(v, "lib", "python3.12", "site-packages", "__editable__.vllm-0.2.pth"))
    assert (world["venv"] / "lib" / "python3.12" / "site-packages" / "__editable__.vllm-0.2.pth").exists()   # source untouched
    assert open(os.path.join(v, "bin", "tool")).readline().strip() == f"#!{v}/bin/python"
    assert (world["venv"] / "bin" / "tool").read_text().startswith(f"#!{os.path.realpath(world['venv'])}/bin/python")
    assert not os.access(os.path.join(v, "lib"), os.W_OK)        # no pip install into a release venv
    assert os.access(src, os.W_OK)                                # but the dev venv's files keep their modes
    # a second release on the same venv state reuses the snapshot
    m2 = _build(world, label="again")
    assert os.path.realpath(world["root"] / m2["id"] / ".venv") == v


def test_verify_catches_a_modified_file_and_a_replaced_so(world):
    m = _build(world)
    d = world["root"] / m["id"]
    p = d / "vllm" / "mod.py"
    os.chmod(d / "vllm", 0o755)
    os.chmod(p, 0o644)
    p.write_text("def f():\n    return 3  # hot fix in prod\n")
    so = d / "vllm" / "_C.abi3.so"
    os.chmod(so, 0o644)
    so.write_bytes(b"rebuilt")
    v = release.verify(m["id"], root=str(world["root"]))
    assert not v["ok"] and "vllm/mod.py" in v["changed"] and v["so_changed"] == ["vllm/_C.abi3.so"]


def test_dirty_source_tree_is_refused(world):
    (world["tree"] / "vllm" / "mod.py").write_text("uncommitted fix\n")
    with pytest.raises(release.ReleaseError, match="uncommitted"):
        _build(world)
    m = _build(world, allow_dirty_source=True)
    assert m["source"]["from_tree_dirty"] and (world["root"] / m["id"] / "vllm" / "mod.py").read_text() == "def f():\n    return 2\n"


def test_compiled_source_drift_is_refused(world):
    (world["repo"] / "csrc" / "a.cu").write_text("// changed kernel\n")
    git(world["repo"], "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "c2")
    c2 = git(world["repo"], "rev-parse", "HEAD")
    with pytest.raises(release.ReleaseError, match="do not match"):
        _build(world, sha=c2)
    assert _build(world, sha=c2, allow_stale_so=True)["source"]["so_drift"]["compiled_source_diff"] == ["csrc/a.cu"]


def test_activate_flip_history_rollback_and_rm_guards(world):
    r = str(world["root"])
    m1 = _build(world, label="a")
    m2 = _build(world, label="b")
    calls = []

    def fake_restart(reason, by, drain_s, tag):
        calls.append(tag)
        return {"healthy": True, "rc": 0}
    res = release.activate(m1["id"], reason="first switch to releases", restart=False, root=r)
    assert res["to"] == m1["id"] and release.current_id(r) == m1["id"] and not calls
    res = release.activate(m2["id"], reason="test activate b", root=r, restart_fn=fake_restart)
    assert res["ok"], res
    assert release.current_id(r) == m2["id"] and release.rollback_target(r) == m1["id"]
    assert res["kv_pool_after"] == 900000 and res["verify_after_boot"]["ok"]
    with pytest.raises(release.ReleaseError, match="rollback target"):
        release.rm(m1["id"], root=r)
    with pytest.raises(release.ReleaseError, match="is current"):
        release.rm(m2["id"], root=r)
    rb = release.rollback(reason="test roll back to a", root=r, restart_fn=fake_restart)
    assert rb["ok"] and release.current_id(r) == m1["id"]
    rows = release.list_releases(r)
    assert {x["id"]: x["current"] for x in rows}[m1["id"]] is True


def test_failed_activation_auto_rolls_back(world):
    r = str(world["root"])
    m1 = _build(world, label="a")
    m2 = _build(world, label="b")
    release.activate(m1["id"], reason="start on release a", restart=False, root=r)
    seen = []

    def flaky_restart(reason, by, drain_s, tag):
        seen.append(tag)
        return {"healthy": tag.startswith("rollback-")}
    res = release.activate(m2["id"], reason="this boot fails", root=r, restart_fn=flaky_restart)
    assert not res["ok"] and "engine not healthy after restart" in res["problems"]
    assert res["auto_rollback"]["to"] == m1["id"] and release.current_id(r) == m1["id"]
    assert seen[0].startswith("activate-") and seen[1].startswith("rollback-")
    kinds = [h["kind"] for h in release.history(r)]
    assert kinds[-1] == "auto-rollback"


def test_activate_refuses_a_tampered_release_and_a_busy_gpu(world, monkeypatch):
    r = str(world["root"])
    m = _build(world)
    d = world["root"] / m["id"]
    os.chmod(d / "vllm", 0o755)
    os.chmod(d / "vllm" / "mod.py", stat.S_IWUSR | stat.S_IRUSR)
    (d / "vllm" / "mod.py").write_text("tampered\n")
    with pytest.raises(release.ReleaseError, match="fails verification"):
        release.activate(m["id"], reason="should not happen", root=r, restart_fn=lambda *a: {"healthy": True})
    m2 = _build(world, label="clean")
    monkeypatch.setattr(gpuguard, "wait_no_foreign", lambda *a, **k: (False, [{"pid": 5, "gpu": 0, "used_mib": 202, "unit": "lp-smoke.service"}]))
    with pytest.raises(release.ReleaseError, match="lp-smoke.service"):
        release.activate(m2["id"], reason="gpu has a bench on it", root=r, restart_fn=lambda *a: {"healthy": True})
    assert release.current_id(r) is None                       # nothing flipped


def test_reason_is_required(world):
    m = _build(world)
    with pytest.raises(release.ReleaseError, match="reason"):
        release.activate(m["id"], reason="x", root=str(world["root"]), restart=False)
