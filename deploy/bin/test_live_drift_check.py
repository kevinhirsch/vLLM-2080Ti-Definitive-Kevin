#!/usr/bin/env python3
"""EB-103: live_drift_check classifies every live file correctly, fails on drift and untracked files, and never writes.

A throwaway git repo with install.sh's layout stands in for the fork, and a temp directory for the live layer; nothing
under the real ~/.local/share/vllm-qwen27b is read or written (conftest's live_guard enforces the writes part).
Run:  python -m pytest -q test_live_drift_check.py
"""
import os
import subprocess

import pytest

import live_drift_check as ldc

ALLOW = """# test allowlist
ignore *.lock
local clients.map    # private host config: only an example is in git
local shim.env       # secret carrier
report knobs.env
"""


def sh(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout


@pytest.fixture
def layout(tmp_path):
    repo, live = tmp_path / "repo", tmp_path / "live"
    files = {
        "deploy/bin/a.sh": "echo a\n",
        "deploy/bin/test_a.py": "def test(): pass\n",
        "deploy/bin/tool.py": "print('not deployed')\n",
        "deploy/templates/t.jinja": "{{ x }}\n",
        "deploy/templates/notes.md": "not a template\n",
        "deploy/env/vllm-qwen27b.env": "A=1\n",
        "deploy/env/shim.env.example": "SHIM_REMOTE_KEY=\n",
        "deploy/env/host-local/clients.map.example": "192.0.2.10 = example-agent-host\n",
        "deploy/env/host-local/README.md": "docs\n",
        "deploy/env/live-layer-allow.txt": ALLOW,
    }
    for rel, body in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    sh(repo, "init", "-q", "-b", "main")
    sh(repo, "add", "-A")
    sh(repo, "commit", "-q", "-m", "layout")
    live.mkdir()
    (live / "a.sh").write_text("echo a\n")
    (live / "t.jinja").write_text("{{ x }}\n")
    (live / "vllm-qwen27b.env").write_text("A=2\n")          # edited in place
    (live / "clients.map").write_text("192.0.2.1 = box\n")    # host-local: never compared
    (live / "restart.lock").write_text("")
    (live / "shim.env").write_text("SHIM_REMOTE_KEY=secret\n")
    (live / "knobs.env").write_text("V02_X=1\n")
    (live / "extra.sh").write_text("echo hand-made\n")       # never committed
    (live / "subdir").mkdir()                                 # top-level files only
    return repo, live


def test_classification(layout):
    repo, live = layout
    r = ldc.check(str(repo), "main", str(live))
    assert sorted(r["ok"]) == ["a.sh", "t.jinja"]
    assert [d["name"] for d in r["drift"]] == ["vllm-qwen27b.env"]
    assert r["drift"][0]["source"] == "deploy/env/vllm-qwen27b.env"
    assert r["untracked"] == ["extra.sh"]
    assert r["untracked_reported"] == ["knobs.env"]
    assert sorted(r["host_local"]) == ["clients.map", "shim.env"]
    assert r["ignored"] == ["restart.lock"]
    assert r["not_deployed"] == ["tool.py"]                   # tests, README, examples and non-jinja never count
    assert r["clean"] is False


def test_exit_codes(layout, capsys):
    repo, live = layout
    args = ["--repo", str(repo), "--ref", "main", "--live", str(live)]
    assert ldc.main(args) == 1
    out = capsys.readouterr().out
    assert "DRIFT      vllm-qwen27b.env" in out and "UNTRACKED  extra.sh" in out and "2 host-local" in out
    (live / "vllm-qwen27b.env").write_text("A=1\n")
    (live / "extra.sh").unlink()
    assert ldc.main(args) == 0                               # reported and host-local files alone never fail


def test_examples_are_never_installed_names(layout):
    repo, live = layout
    (live / "clients.map.example").write_text("192.0.2.10 = example-agent-host\n")
    r = ldc.check(str(repo), "main", str(live))
    assert r["untracked"] == ["clients.map.example", "extra.sh"]   # an example copied live is not a source


def test_report_verb_covers_drift_of_a_tracked_file(layout):
    repo, live = layout
    (repo / "deploy/bin/knobs.env").write_text("V02_X=0\n")
    sh(repo, "add", "-A")
    sh(repo, "commit", "-q", "-m", "track the knobs")
    r = ldc.check(str(repo), "main", str(live))
    assert [d["name"] for d in r["drift_reported"]] == ["knobs.env"]
    assert "knobs.env" not in [d["name"] for d in r["drift"]]


def test_allow_file_overrides_the_ref(layout, tmp_path):
    repo, live = layout
    alt = tmp_path / "allow.txt"
    alt.write_text(ALLOW + "ignore extra.sh\n")
    args = ["--repo", str(repo), "--ref", "main", "--live", str(live), "--allow-file", str(alt), "--json"]
    (live / "vllm-qwen27b.env").write_text("A=1\n")
    assert ldc.main(args) == 0


def test_bad_allowlist_and_name_clash_exit_2(layout, tmp_path):
    repo, live = layout
    bad = tmp_path / "bad.txt"
    bad.write_text("keep *.lock\n")
    assert ldc.main(["--repo", str(repo), "--ref", "main", "--live", str(live), "--allow-file", str(bad)]) == 2
    (repo / "deploy/bin/t.jinja").write_text("dup\n")           # two sources would install to one live name
    sh(repo, "add", "-A")
    sh(repo, "commit", "-q", "-m", "clash")
    assert ldc.main(["--repo", str(repo), "--ref", "main", "--live", str(live)]) == 2
    assert ldc.main(["--repo", str(repo), "--ref", "no-such-ref", "--live", str(live)]) == 2


def test_read_only(layout):
    repo, live = layout
    before_live = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in live.iterdir() if p.is_file()}
    before_objects = sh(repo, "count-objects", "-v")
    ldc.check(str(repo), "main", str(live))
    after_live = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in live.iterdir() if p.is_file()}
    assert after_live == before_live
    assert sh(repo, "count-objects", "-v") == before_objects   # hash-object ran without -w
