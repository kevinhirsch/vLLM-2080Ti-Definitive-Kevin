"""RL: the identity gate that decides whether production may switch to a release."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import release_ab_probe as ab  # noqa: E402


def write(d, name, obj):
    with open(os.path.join(d, name), "w") as fh:
        fh.write(obj if isinstance(obj, str) else json.dumps(obj))


def probe(texts):
    return {"prompts": [{"i": i, "texts": t, "completion_tokens": [1]*len(t), "generation_verified": True, "self_consistent": len(set(t)) == 1} for i, t in enumerate(texts)]}


def setup(d, kv=(922358, 922358), base_texts=None, cand_texts=None, tok=(80.0, 80.0), ev=("15/15 passed", "15/15 passed")):
    write(d, "summary.json", {"boots": [{"label": "base", "kv_pool": kv[0]}, {"label": "release", "kv_pool": kv[1]}]})
    base_texts = base_texts or [["a", "a"], ["b", "b"], ["c", "c"], ["d", "x"]]
    cand_texts = cand_texts or [["a", "a"], ["b", "b"], ["c", "c"], ["q", "q"]]
    write(d, "probe_base.json", probe(base_texts))
    write(d, "probe_release.json", probe(cand_texts))
    write(d, "quick_base.json", {"natural_tok_s": [tok[0]] * 3})
    write(d, "quick_release.json", {"natural_tok_s": [tok[1]] * 3})
    write(d, "evalkit_base.out", f" {ev[0]}, 200s total\n")
    write(d, "evalkit_release.out", f" {ev[1]}, 200s total\n")


def test_identical_release_passes_and_skips_prompts_the_base_did_not_reproduce(tmp_path):
    setup(str(tmp_path))
    r = ab.compare(str(tmp_path))
    assert r["ok"], r
    assert r["facts"]["greedy"]["base_not_self_consistent"] == [3] and r["facts"]["greedy"]["comparable"] == 3
    assert json.load(open(tmp_path / "compare.json"))["ok"]


def test_each_gate_fails_on_its_own(tmp_path):
    for kw, gate in (({"kv": (922358, 899704)}, "kv_pool_equal"),
                     ({"cand_texts": [["a", "a"], ["DIFF", "DIFF"], ["c", "c"], ["q", "q"]]}, "greedy_identical"),
                     ({"tok": (80.0, 70.0)}, "decode_not_slower"),
                     ({"ev": ("15/15 passed", "14/15 passed")}, "evalkit_not_worse")):
        d = tmp_path / gate
        d.mkdir()
        setup(str(d), **kw)
        r = ab.compare(str(d))
        assert not r["ok"] and r["gates"][gate] is False, (gate, r)
        assert all(v for k, v in r["gates"].items() if k != gate), (gate, r)


def test_too_few_comparable_prompts_is_not_a_pass(tmp_path):
    setup(str(tmp_path), base_texts=[["a", "b"], ["c", "d"], ["e", "e"], ["f", "g"]])
    assert ab.compare(str(tmp_path))["gates"]["greedy_identical"] is False


def test_zero_tokens_and_legacy_unverified_probe_fail(tmp_path):
    setup(str(tmp_path))
    data = json.load(open(tmp_path / 'probe_release.json'))
    data['prompts'][0]['completion_tokens'] = [0, 0]
    write(str(tmp_path), 'probe_release.json', data)
    assert not ab.compare(str(tmp_path))['gates']['generation_verified']
    for row in data['prompts']:
        row.pop('generation_verified')
    write(str(tmp_path), 'probe_release.json', data)
    assert not ab.compare(str(tmp_path))['ok']


def test_candidate_second_rep_and_missing_fixture_are_gated(tmp_path):
    setup(str(tmp_path), cand_texts=[['a', 'BAD'], ['b', 'b'], ['c', 'c'], ['q', 'q']])
    assert not ab.compare(str(tmp_path))['gates']['greedy_identical']
    setup(str(tmp_path))
    data = json.load(open(tmp_path / 'probe_release.json'))
    data['prompts'].pop()
    write(str(tmp_path), 'probe_release.json', data)
    assert not ab.compare(str(tmp_path))['gates']['greedy_identical']


def test_post_rejects_http400_and_empty_generation(monkeypatch):
    from unittest.mock import Mock
    import pytest
    response = Mock(status=400)
    connection = Mock()
    connection.getresponse.return_value = response
    monkeypatch.setattr(ab.http.client, 'HTTPConnection', lambda *a, **k: connection)
    with pytest.raises(ValueError, match='HTTP 400'):
        ab._post('/v1/chat/completions', {})
    response.status = 200
    response.read.return_value = json.dumps({'usage': {'completion_tokens': 0}, 'choices': [{'message': {'content': ''}, 'finish_reason': 'stop'}]}).encode()
    with pytest.raises(ValueError, match='zero'):
        ab._post('/v1/chat/completions', {})
