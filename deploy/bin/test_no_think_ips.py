"""Config list/set round-trip: the no-think IP policy must survive being saved.

WHY THIS FILE EXISTS (2026-09-09 RCA).

`GET /gateway/config` was serving this as no_think_ips:

    {{'''"{'10.0.1.10'"}'}'}'''}'}'}|{'{'{'''{'{'{"'10.0.1.250'}"'''}'}'''

`h_chat` decides the no-think policy with `request.remote in NO_THINK_IPS`. NO_THINK_IPS was
built by splitting that string on commas, so it held ONE member -- the whole garbage run --
and neither 10.0.1.10 (Hermes) nor 10.0.1.250 (the scheduler peer) could ever match. The
policy was dead, and dead silently: nothing logs a set that simply never matches.

The cause is in this file, not in a shell. It is a lossy round trip between two halves of
the SAME module:

  reader  (module level + the _CFG caster)  splits on ","
  writer  (_persist_config)                 joins  with "|"

so the first dashboard save that touched any field re-wrote a good value into one the reader
could not parse. An older _persist_config was worse still: it wrote `str(g[gname])` for every
type, so a set went to disk as its Python repr, braces and quotes included. Each save then
re-serialised the previous parse of the previous repr and the punctuation COMPOUNDED. The
env-file backups in this directory record the ratchet exactly:

    08-14  SHIM_NO_THINK_IPS={'10.0.1.10', '10.0.1.250'}
    08-16  SHIM_NO_THINK_IPS={"'10.0.1.250'}", "{'10.0.1.10'"}
    08-24  SHIM_NO_THINK_IPS={'{\'\'\'"{\'10.0.1.10\'"}\'}\'}\'', ...
    09-05  SHIM_NO_THINK_IPS={{'''"{'10.0.1.10'"}'}'}'''}'}'}|{'{'{...

It has the shape the vault files under [[Workflow Interpolation Footgun]], but no shell was
involved -- the shim corrupted its own config, unattended, over three weeks.

So these tests assert BOTH halves, because fixing only the parser leaves the writer free to
lay down the next bad value:

  * the parser self-heals a value already corrupt on disk (test_live_corrupted_value_*),
  * the writer emits something its own reader can read back (test_round_trip_*).

SHIM_BG_XCLIENTS had the identical defect from the identical cause and is covered here too;
SHIM_BG_MARKERS is pipe-on-both-sides and was always fine, so it is pinned against the fix
regressing it.

Run:  /home/kevin/.venv-tests/bin/pytest -q /home/kevin/.local/share/vllm-qwen27b/test_no_think_ips.py
"""
import importlib.util
import json
import os
import sys

import pytest

SHIM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keepalive-shim.py")

# The exact string the live gateway was serving on 2026-09-09, copied from
#   curl -s http://127.0.0.1:8000/gateway/config
LIVE_CORRUPTED = """{{'''"{'10.0.1.10'"}'}'}'''}'}'}|{'{'{'''{'{'{"'10.0.1.250'}"'''}'}'''"""

_counter = [0]


def load_shim(monkeypatch, tmp_path, env=None):
    """Import a fresh copy of the shim.

    Fresh, not cached: the module keeps its config in module-level globals, so a second test
    would otherwise inherit the first one's NO_THINK_IPS. Env vars go in BEFORE exec because
    the shim reads them at import time. SHIM_ENV_FILE is redirected into tmp_path so
    _persist_config can never touch the real /home/kevin/.local/share/vllm-qwen27b/shim.env
    that the running gateway reads.
    """
    base_env = {
        "SHIM_UPSTREAM": "http://127.0.0.1:1",
        "SHIM_REMOTE_BASE": "",
        "SHIM_REMOTE_KEY": "",
        "SHIM_ENV_FILE": str(tmp_path / "shim.env"),
        "SHIM_STATS_FILE": str(tmp_path / "gateway-stats.json"),
        "SHIM_LOG_REQUESTS": "0",
        "SHIM_NO_THINK_IPS": "",
    }
    base_env.update(env or {})
    for k, v in base_env.items():
        monkeypatch.setenv(k, v)

    _counter[0] += 1
    name = "shim_under_test_%d" % _counter[0]
    spec = importlib.util.spec_from_file_location(name, SHIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@pytest.fixture
def shim(monkeypatch, tmp_path):
    return load_shim(monkeypatch, tmp_path)


# --------------------------------------------------------------------------- parser


def test_live_corrupted_value_self_heals(shim):
    """The exact garbage the gateway served must parse back to the two real IPs.

    This is the whole point: an operator should not have to hand-repair shim.env. A value
    already corrupt on disk has to come back to life on the next read, or the fix only helps
    estates that never hit the bug.
    """
    assert set(shim._parse_seq(LIVE_CORRUPTED, ",")) == {"10.0.1.10", "10.0.1.250"}


def test_live_corrupted_value_matches_a_request_ip(monkeypatch, tmp_path):
    """End to end at import: the policy set must actually contain the IPs it gates on.

    `request.remote in NO_THINK_IPS` is the real predicate. Asserting the parsed set is what
    the defect report asked to be proven, so assert membership the way the code asks it.
    """
    mod = load_shim(monkeypatch, tmp_path, {"SHIM_NO_THINK_IPS": LIVE_CORRUPTED})
    assert "10.0.1.10" in mod.NO_THINK_IPS
    assert "10.0.1.250" in mod.NO_THINK_IPS
    assert mod.NO_THINK_IPS == {"10.0.1.10", "10.0.1.250"}


def test_comma_separated(shim):
    assert set(shim._parse_seq("10.0.1.10,10.0.1.250", ",")) == {"10.0.1.10", "10.0.1.250"}


def test_pipe_separated(shim):
    """A pipe-joined value is what _persist_config used to write, so it must still read."""
    assert set(shim._parse_seq("10.0.1.10|10.0.1.250", ",")) == {"10.0.1.10", "10.0.1.250"}


def test_whitespace_is_stripped(shim):
    assert set(shim._parse_seq("  10.0.1.10 ,\t10.0.1.250  ", ",")) == {"10.0.1.10", "10.0.1.250"}


def test_empty_value_is_empty_not_a_blank_member(shim):
    """An empty policy must be empty. A set holding "" would make `remote in` odd, and a set
    holding a stray separator would silently gate nothing while looking configured."""
    for blank in ("", "   ", ",", "|", " , | , "):
        assert shim._parse_seq(blank, ",") == [], repr(blank)


def test_python_repr_of_a_set_parses(shim):
    """The 08-14 on-disk form: str({'10.0.1.10', '10.0.1.250'}) written verbatim."""
    assert set(shim._parse_seq("{'10.0.1.10', '10.0.1.250'}", ",")) == {"10.0.1.10", "10.0.1.250"}


def test_duplicates_collapse_and_order_is_kept(shim):
    assert shim._parse_seq("b,a,b", ",") == ["b", "a"]


def test_a_real_set_or_list_passes_through(shim):
    """apply_config casters get whatever the dashboard POSTs, which is not always a string."""
    assert set(shim._parse_seq({"10.0.1.10", "10.0.1.250"}, ",")) == {"10.0.1.10", "10.0.1.250"}
    assert shim._parse_seq(["10.0.1.10"], ",") == ["10.0.1.10"]
    assert shim._parse_seq(None, ",") == []


# --------------------------------------------------------------------------- round trip


def _persisted(mod, key):
    for line in open(os.environ["SHIM_ENV_FILE"]):
        if line.split("=", 1)[0].strip() == key:
            return line.split("=", 1)[1].rstrip("\n")
    raise AssertionError("%s not persisted" % key)


def test_round_trip_no_think_ips_survives_a_save(monkeypatch, tmp_path):
    """THE regression. Save the config, reload the shim from the file it just wrote, and the
    policy must still match both IPs.

    Before the fix this failed on the reload: the writer put "10.0.1.10|10.0.1.250" on disk
    and the reader comma-split it into the single member "10.0.1.10|10.0.1.250".
    """
    mod = load_shim(monkeypatch, tmp_path, {"SHIM_NO_THINK_IPS": "10.0.1.10,10.0.1.250"})
    assert mod.NO_THINK_IPS == {"10.0.1.10", "10.0.1.250"}

    # Any save re-writes every tunable, so touching one unrelated knob was enough to kill it.
    mod.apply_config({"local_budget": "14"})

    reloaded = load_shim(monkeypatch, tmp_path,
                         {"SHIM_NO_THINK_IPS": _persisted(mod, "SHIM_NO_THINK_IPS")})
    assert reloaded.NO_THINK_IPS == {"10.0.1.10", "10.0.1.250"}


def test_round_trip_repairs_an_already_corrupt_value(monkeypatch, tmp_path):
    """Boot corrupt, save once, and what lands on disk must be the clean canonical form --
    otherwise the next operator still sees garbage in the dashboard."""
    mod = load_shim(monkeypatch, tmp_path, {"SHIM_NO_THINK_IPS": LIVE_CORRUPTED})
    mod.apply_config({"local_budget": "14"})
    assert _persisted(mod, "SHIM_NO_THINK_IPS") == "10.0.1.10,10.0.1.250"


def test_dashboard_shows_a_value_it_can_read_back(monkeypatch, tmp_path):
    """current_config() feeds the dashboard form, and the form POSTs straight back into
    apply_config. If the two disagree on the separator, merely opening the page and pressing
    save corrupts the policy."""
    mod = load_shim(monkeypatch, tmp_path, {"SHIM_NO_THINK_IPS": LIVE_CORRUPTED})
    shown = mod.current_config()["no_think_ips"]
    assert shown == "10.0.1.10,10.0.1.250"
    mod.apply_config({"no_think_ips": shown})
    assert mod.NO_THINK_IPS == {"10.0.1.10", "10.0.1.250"}


def test_round_trip_bg_xclients_survives_a_save(monkeypatch, tmp_path):
    """Same defect, same cause, second field: read on "," and written on "|"."""
    mod = load_shim(monkeypatch, tmp_path, {"SHIM_BG_XCLIENTS": "cron,batch,workflow-bg"})
    assert mod.BG_XCLIENTS == ["cron", "batch", "workflow-bg"]

    mod.apply_config({"local_budget": "14"})
    reloaded = load_shim(monkeypatch, tmp_path,
                         {"SHIM_BG_XCLIENTS": _persisted(mod, "SHIM_BG_XCLIENTS")})
    assert reloaded.BG_XCLIENTS == ["cron", "batch", "workflow-bg"]


def test_bg_markers_keep_the_pipe_separator(monkeypatch, tmp_path):
    """BG_MARKERS was never broken -- it is pipe on both sides -- and must not be 'fixed'.

    Its members are free-text phrases matched against prompt bodies. A phrase may legitimately
    contain a comma, so this field must NOT gain comma splitting.
    """
    markers = "scheduled cron job|a phrase, with a comma|local-lane-runner batch job"
    mod = load_shim(monkeypatch, tmp_path, {"SHIM_BG_MARKERS": markers})
    assert mod.BG_MARKERS == ["scheduled cron job", "a phrase, with a comma",
                             "local-lane-runner batch job"]

    mod.apply_config({"local_budget": "14"})
    assert _persisted(mod, "SHIM_BG_MARKERS") == markers

    reloaded = load_shim(monkeypatch, tmp_path,
                         {"SHIM_BG_MARKERS": _persisted(mod, "SHIM_BG_MARKERS")})
    assert reloaded.BG_MARKERS == mod.BG_MARKERS


# ------------------------------------------------------- the policy actually being applied
#
# Repairing the string revived the policy on the MAIN local lane and nowhere else. Measured
# on the live gateway on 2026-09-09, same prompt, same second:
#
#   max_tokens=2000 (main lane) from 10.0.1.10   -> reasoning_content None       policy ON
#   max_tokens=2000 (main lane) from 127.0.0.1   -> reasoning_content 114 chars   policy OFF
#   max_tokens=220  (TINY lane) from 10.0.1.10   -> reasoning_content 114 chars   policy OFF  <-- bug
#
# The TINY fast-lane built its own body chain and simply never called strip_thinking, so it
# ignored both no-think triggers. That lane takes every call where prompt + max_tokens <=
# TINY_TOKENS (1500) -- which is most of what NO_THINK_IPS exists to cover, since the listed
# hosts are the ones firing short status turns. The lane's own comment says it "needs the same
# thinking guard as the main path"; the guard it copied was thinking_budget_guard, not this.
#
# The cause is duplication: two copies of one body chain, and only one of them was updated.
# So the fix is not a third copy -- it is a single _prepare_local_body() both lanes call, and
# a test below that fails if the chain is ever pasted a second time.


class _FakeRequest:
    """Just enough of aiohttp's Request for the policy: it reads .remote and nothing else."""

    def __init__(self, remote):
        self.remote = remote


def test_no_think_policy_matches_a_listed_ip(monkeypatch, tmp_path):
    mod = load_shim(monkeypatch, tmp_path, {"SHIM_NO_THINK_IPS": "10.0.1.10,10.0.1.250"})
    assert mod._no_think_policy(_FakeRequest("10.0.1.10"), False) is True
    assert mod._no_think_policy(_FakeRequest("10.0.1.250"), False) is True


def test_no_think_policy_ignores_an_unlisted_ip(monkeypatch, tmp_path):
    mod = load_shim(monkeypatch, tmp_path, {"SHIM_NO_THINK_IPS": "10.0.1.10,10.0.1.250"})
    assert mod._no_think_policy(_FakeRequest("127.0.0.1"), False) is False
    assert mod._no_think_policy(_FakeRequest(None), False) is False


def test_no_think_policy_still_honours_the_background_trigger(monkeypatch, tmp_path):
    """The other half of the same predicate: background + BG_NO_THINK. Pinned so the refactor
    that unified the two lanes cannot quietly drop it."""
    mod = load_shim(monkeypatch, tmp_path, {"SHIM_NO_THINK_IPS": "", "SHIM_BG_NO_THINK": "1"})
    assert mod._no_think_policy(_FakeRequest("127.0.0.1"), True) is True
    mod.BG_NO_THINK = False
    assert mod._no_think_policy(_FakeRequest("127.0.0.1"), True) is False


def _thinking_flag(body):
    return json.loads(body).get("chat_template_kwargs", {}).get("enable_thinking")


def test_prepare_local_body_strips_thinking_for_a_listed_ip(monkeypatch, tmp_path):
    """This is the assertion the TINY lane would have failed: the body handed to the engine
    must carry enable_thinking=False for a listed client."""
    mod = load_shim(monkeypatch, tmp_path, {"SHIM_NO_THINK_IPS": "10.0.1.10"})
    body = json.dumps({"model": "qwen-local", "max_tokens": 200,
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    out = mod._prepare_local_body(_FakeRequest("10.0.1.10"), body, False)
    assert _thinking_flag(out) is False


def test_prepare_local_body_leaves_thinking_on_for_an_unlisted_ip(monkeypatch, tmp_path):
    mod = load_shim(monkeypatch, tmp_path, {"SHIM_NO_THINK_IPS": "10.0.1.10"})
    body = json.dumps({"model": "qwen-local", "max_tokens": 200,
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    out = mod._prepare_local_body(_FakeRequest("127.0.0.1"), body, False)
    assert _thinking_flag(out) is not False


def test_the_local_body_chain_is_written_exactly_once():
    """Structural guard on the CAUSE, not the symptom.

    The tiny lane diverged because the guard chain was copy-pasted and one copy was never
    updated. Nothing failed when that happened -- the lane just quietly stopped honouring a
    policy. Pin the chain to a single occurrence so the next paste fails here instead.
    """
    src = open(SHIM_PATH).read()
    assert src.count("repetition_guard(nonthinking_sampling_profile(") == 1, (
        "the local-body guard chain is duplicated again -- route the second caller through "
        "_prepare_local_body() instead, or the two copies will drift like the tiny lane did")


def test_every_sequence_field_round_trips(monkeypatch, tmp_path):
    """Guard the CLASS of bug, not the two instances of it.

    A future tunable that is a list or a set gets the same lossy round trip for free unless
    someone pins it. This walks _CFG, saves, reloads and demands equality, so the next such
    field fails here instead of dying silently in production three weeks later.
    """
    mod = load_shim(monkeypatch, tmp_path, {
        "SHIM_NO_THINK_IPS": "10.0.1.10,10.0.1.250",
        "SHIM_BG_XCLIENTS": "cron,batch",
        "SHIM_BG_MARKERS": "scheduled cron job|another marker",
    })
    seq_fields = {env: gname for env, (gname, _) in mod._CFG.items()
                  if isinstance(getattr(mod, gname), (set, list, tuple))}
    assert "SHIM_NO_THINK_IPS" in seq_fields and "SHIM_BG_XCLIENTS" in seq_fields

    before = {env: getattr(mod, gname) for env, gname in seq_fields.items()}
    mod.apply_config({"local_budget": "14"})

    reloaded = load_shim(monkeypatch, tmp_path,
                         {env: _persisted(mod, env) for env in seq_fields})
    for env, gname in seq_fields.items():
        assert getattr(reloaded, gname) == before[env], env
