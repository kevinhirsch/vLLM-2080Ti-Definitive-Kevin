"""Capture launcher argv/fingerprint without importing vLLM or touching live caches."""
import json
import os
from pathlib import Path
import subprocess

import pytest

import model_switch as switch


@pytest.fixture
def launcher(tmp_path):
    root = tmp_path / "private lane"
    (root / ".venv/bin").mkdir(parents=True)
    python = root / ".venv/bin/python"
    python.write_text("""#!/usr/bin/python3
import json, os, sys
if sys.argv[1].endswith('checkpoint_fingerprint.py'):
    with open(os.path.join(os.environ['V02_ROOT'], 'fingerprint.json'), 'w') as out:
        json.dump(sys.argv[1:], out)
    print('private-fingerprint')
else:
    print(json.dumps({'args':sys.argv[1:], 'head':os.environ.get('VLLM_U2_INT4_HEAD'), 'mtp':os.environ.get('VLLM_U2_INT4_MTP')}))
""")
    python.chmod(0o755)
    override = tmp_path / "override.env"
    override.write_text("")
    config = tmp_path / "selection.json"
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "V02_ROOT": str(root),
           "V02_OVERRIDE_ENV": str(override), "V02_RELEASES": str(tmp_path / "releases"),
           "V02_MODEL_CONFIG": str(config)}
    def run(selection=None, ssd=False):
        if selection is None:
            config.unlink(missing_ok=True)
        else:
            config.write_text(json.dumps(selection))
        extra = {"V02_SSD_KV_DIR": str(tmp_path / "ssd cache")} if ssd else {}
        result = subprocess.run(["bash", str(Path(__file__).with_name(switch.LAUNCHER))], env=dict(env, **extra),
                                text=True, capture_output=True, timeout=15)
        return result, json.loads(result.stdout) if result.returncode == 0 else None
    return run, root


def runtime_args(args):
    result = list(args)
    index = result.index("--served-model-name")
    end = result.index("--dtype", index)
    del result[index:end]
    index = result.index("--model")
    del result[index:index + 2]
    return result


def test_default_model_selection_preserves_exact_existing_arguments(launcher):
    run, _ = launcher
    result, baseline = run()
    assert result.returncode == 0, result.stderr
    model = baseline["args"][baseline["args"].index("--model") + 1]
    result, selected = run({"model": model, "served_model_name": switch.ALIASES})
    assert result.returncode == 0, result.stderr
    assert selected == baseline


def test_trained_checkpoint_keeps_base_runtime_and_uses_its_own_fingerprint(launcher, tmp_path):
    run, root = launcher
    base = str(tmp_path / switch.CHECKPOINTS["hauhaucs"])
    # Characters that would execute or split if JSON were sourced/interpolated into shell code.
    trained = str(tmp_path / "model space ' $(touch SHOULD_NOT_EXIST)" / switch.CHECKPOINTS["dft1"])
    result, baseline = run({"model": base, "served_model_name": switch.ALIASES}, ssd=True)
    assert result.returncode == 0, result.stderr
    result, selected = run({"model": trained, "served_model_name": switch.ALIASES + [switch.CHECKPOINTS["dft1"].lower()]}, ssd=True)
    assert result.returncode == 0, result.stderr
    assert selected["args"][selected["args"].index("--model") + 1] == trained
    assert runtime_args(selected["args"]) == runtime_args(baseline["args"])
    assert selected["head"] == selected["mtp"] == "1"
    fingerprint = json.loads((root / "fingerprint.json").read_text())
    assert fingerprint == [str(root / "tools/checkpoint_fingerprint.py"), trained, ""]
    assert not (root / "SHOULD_NOT_EXIST").exists()
    assert "--language-model-only" in selected["args"]
    assert selected["args"][selected["args"].index("--max-model-len") + 1] == "524288"


@pytest.mark.parametrize("selection", [{"model": "relative/path"}, {"model": "/model\n--bad"},
                                         {"model": "/model", "served_model_name": ["--injected"]}])
def test_invalid_json_selection_fails_before_boot(launcher, selection):
    run, root = launcher
    result, _ = run(selection, ssd=True)
    assert result.returncode != 0
    assert "invalid" in result.stderr
    assert not (root / "fingerprint.json").exists()
