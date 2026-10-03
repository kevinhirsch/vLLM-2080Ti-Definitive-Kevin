"""RL: window specs parse without PyYAML (the test gate's venv has none) and exactly as PyYAML would."""
import glob
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mini_yaml  # noqa: E402

SPECS = sorted(glob.glob(os.path.join(HERE, "..", "windows", "*.yaml")))

CASES = [
    ("a: 1\nb: [x, 'y z', \"q\\\\S\"]\nc: {k: v, n: [1, 2]}\n", {"a": 1, "b": ["x", "y z", "q\\S"], "c": {"k": "v", "n": [1, 2]}}),
    ("- a\n- b: 1\n  c: 2\n- [1, 2]\n", ["a", {"b": 1, "c": 2}, [1, 2]]),
    ("k: |\n  line1\n    ind\n  line3\n\nz: 3\n", {"k": "line1\n  ind\nline3\n", "z": 3}),
    ("k: |-\n  x\n  y\nm: >\n  folded\n  text\n\n  para\n", {"k": "x\ny", "m": "folded text\npara\n"}),
    ("a:\n- 1\n- 2\nb: null\nc: ~\nd: yes\ne: 'it''s'\nf: 1.5\ng: x # comment\nh: 'a # not comment'\n",
     {"a": [1, 2], "b": None, "c": None, "d": True, "e": "it's", "f": 1.5, "g": "x", "h": "a # not comment"}),
    ("x: \"tab\\there\"\ny: http://a.b/c\nz: a:b\n", {"x": "tab\there", "y": "http://a.b/c", "z": "a:b"}),
    ("steps:\n  - name: a\n    run: \"echo {{X}}\"\n    capture:\n      V: {json: \"{{L}}/t.json\", key: pass, default: \"False\"}\n",
     {"steps": [{"name": "a", "run": "echo {{X}}", "capture": {"V": {"json": "{{L}}/t.json", "key": "pass", "default": "False"}}}]}),
]


@pytest.mark.parametrize("text,want", CASES)
def test_subset(text, want):
    assert mini_yaml.load(text) == want


@pytest.mark.parametrize("bad", ["a: &x 1\n", "a: 1\n---\nb: 2\n", "a: 1\na: 2\n", "a:\n\t- 1\n", "a: [1, 2\n"])
def test_unsupported_or_broken_yaml_is_an_error_not_a_guess(bad):
    with pytest.raises(mini_yaml.MiniYAMLError):
        mini_yaml.load(bad)


@pytest.mark.parametrize("path", SPECS, ids=os.path.basename)
def test_every_window_spec_parses_and_matches_pyyaml_when_available(path):
    text = open(path).read()
    got = mini_yaml.load(text)
    assert isinstance(got, dict) and got.get("steps")
    yaml = pytest.importorskip("yaml")
    assert got == yaml.safe_load(text)
