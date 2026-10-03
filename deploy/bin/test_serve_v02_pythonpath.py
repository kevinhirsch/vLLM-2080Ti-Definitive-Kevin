"""RL (L107 + L146): serve-hauhaucs-v02.sh / serve-profile-v02.sh honor an override PYTHONPATH (it used to be re-exported
after sourcing v02.override.env and silently dropped) and boot the release `current` pointer, resolved once."""
import os
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def fake_root(path):
    (path / ".venv" / "bin").mkdir(parents=True)
    py = path / ".venv" / "bin" / "python"
    py.write_text('#!/bin/bash\necho "PYTHONPATH=$PYTHONPATH"\necho "PWD=$PWD"\necho "TQ=$VLLM_TQ_GQA_BUILD_DIR"\n'
                  'echo "TXD=$TORCH_EXTENSIONS_DIR"\necho "ARGS=$*"\n')
    py.chmod(0o755)
    (path / ".deps" / "FlashQLA-SM70-SM75").mkdir(parents=True)
    return path


def run(script, tmp_path, override="", env_extra=None):
    ov = tmp_path / "override.env"
    ov.write_text(override)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "V02_OVERRIDE_ENV": str(ov),
           "V02_RELEASES": str(tmp_path / "releases"), **(env_extra or {})}
    if script == "serve-profile-v02.sh":
        args = tmp_path / "args"
        args.write_text("-m vllm.entrypoints.openai.api_server --port 1")
        env["V02_PROFILE_ARGS"] = str(args)
    r = subprocess.run(["bash", os.path.join(HERE, script)], capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 0, r.stderr
    return dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line), r.stderr


@pytest.mark.parametrize("script", ["serve-hauhaucs-v02.sh", "serve-profile-v02.sh"])
def test_override_pythonpath_is_honored_and_prepended(tmp_path, script):
    root = fake_root(tmp_path / "lane")
    out, err = run(script, tmp_path, f"export V02_ROOT={root}\nexport PYTHONPATH=/home/kevin/Desktop/wt-k3\n")
    assert out["PYTHONPATH"] == f"/home/kevin/Desktop/wt-k3:{root}:{root}/.deps/FlashQLA-SM70-SM75"
    assert "PYTHONPATH-override=/home/kevin/Desktop/wt-k3" in err


@pytest.mark.parametrize("script", ["serve-hauhaucs-v02.sh", "serve-profile-v02.sh"])
def test_no_override_keeps_the_exact_production_pythonpath(tmp_path, script):
    root = fake_root(tmp_path / "lane")
    out, _ = run(script, tmp_path, f"export V02_ROOT={root}\n")
    assert out["PYTHONPATH"] == f"{root}:{root}/.deps/FlashQLA-SM70-SM75"
    assert out["PWD"] == str(root)


def test_inherited_pythonpath_from_the_unit_env_is_not_mistaken_for_an_override(tmp_path):
    root = fake_root(tmp_path / "lane")
    out, _ = run("serve-hauhaucs-v02.sh", tmp_path, f"export V02_ROOT={root}\n", {"PYTHONPATH": "/old/0.1.x/tree"})
    assert out["PYTHONPATH"] == f"{root}:{root}/.deps/FlashQLA-SM70-SM75"


def test_current_release_pointer_is_booted_and_resolved_once(tmp_path):
    rel = fake_root(tmp_path / "releases" / "abc123-20261003T1501")
    (tmp_path / "releases" / "current").symlink_to("abc123-20261003T1501")
    (rel / "RELEASE.json").write_text('{"id": "abc123-20261003T1501", "sha": "abc123abc123abc123"}')
    out, err = run("serve-hauhaucs-v02.sh", tmp_path, "")
    assert out["PWD"] == str(rel)                      # the concrete release dir, not .../current
    assert out["TQ"] == f"{rel}/.deps/tq_gqa_build"
    assert out["TXD"] == f"{rel}/.deps/FlashQLA-SM70-SM75/.torch_extensions_vllm_flashqla_legacy"
    assert "abc123-20261003T1501 sha=abc123abc123" in err
    assert "--model" in out["ARGS"] and "--port 8001" in out["ARGS"]


def test_symlinked_jit_dirs_are_flagged(tmp_path):
    prod = fake_root(tmp_path / "prod")
    (prod / ".deps" / "tq_gqa_build").mkdir()
    lane = fake_root(tmp_path / "lane")
    (lane / ".deps" / "tq_gqa_build").symlink_to(prod / ".deps" / "tq_gqa_build")
    _, err = run("serve-hauhaucs-v02.sh", tmp_path, f"export V02_ROOT={lane}\n")
    assert "WARNING" in err and "tq_gqa_build is a symlink" in err


def test_venv_is_invoked_through_the_root_path_not_resolved(tmp_path):
    """sys.prefix (and so every torch JIT path) derives from the spelling of the interpreter path: it must be
    $V02_ROOT/.venv/bin/python, the same path release.py prebuilds through, even when .venv is a symlink."""
    real = fake_root(tmp_path / "venvhome")
    lane = tmp_path / "lane2"
    (lane / ".deps" / "FlashQLA-SM70-SM75").mkdir(parents=True)
    (lane / ".venv").symlink_to(real / ".venv")
    out, err = run("serve-hauhaucs-v02.sh", tmp_path, f"export V02_ROOT={lane}\n")
    assert f"venv={lane}/.venv -> {real}/.venv" in err
    assert out["PWD"] == str(lane)


def test_uvicorn_access_log_off_by_default_and_reenabled_by_override(tmp_path):
    root = fake_root(tmp_path / "lane")
    out, _ = run("serve-hauhaucs-v02.sh", tmp_path, f"export V02_ROOT={root}\n")
    assert "--disable-uvicorn-access-log" in out["ARGS"]
    out, _ = run("serve-hauhaucs-v02.sh", tmp_path, f"export V02_ROOT={root}\nexport V02_ACCESS_LOG=1\n")
    assert "--disable-uvicorn-access-log" not in out["ARGS"]
