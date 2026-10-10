# SPDX-License-Identifier: Apache-2.0
"""Private CPU races; no gateway/tool/systemd/GPU execution permitted."""
import ast
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import types

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import offline_lease_guard as lg
import windowctl as wc

PATCHER = Path(__file__).parents[2] / 'tools' / 'patch-current-gateway-offline.py'
spec = importlib.util.spec_from_file_location('gateway_source_patch', PATCHER)
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)
CURRENT_SOURCE = Path('/home/kevin/.local/share/vllm-qwen27b/keepalive-shim.py')
LEASE = 'a' * 32


@pytest.fixture(autouse=True)
def no_external(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError('external call in private CPU race contract')
    monkeypatch.setattr(wc.subprocess, 'run', forbidden)
    monkeypatch.setattr(wc.urllib.request, 'urlopen', forbidden)


class Gateway:
    """The exact patched installed handler AST, with private fake globals only."""
    def __init__(self, active=True):
        data = patcher.patch_source(CURRENT_SOURCE.read_bytes())
        tree = ast.parse(data)
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name in ('_offline_status', 'gateway_offline')]
        self.now = 100
        self.persisted = 0
        self.random_calls = 0
        self.draining = False
        self.state = {'until': 300 if active else 0, 'lease': LEASE if active else None,
                      'reason': 'owned', 'by': 'HaloFrontier', 't0': 1 if active else None,
                      'ttl_s': 200 if active else None, 'refused': 0}
        def random(n):
            self.random_calls += 1
            return bytes([self.random_calls]) * n
        def persist():
            self.persisted += 1
        ns = {'_OFFLINE': self.state, 'OFFLINE_MAX_TTL_S': 3600,
              'OFFLINE_REFUSED_PUBLISH': 'publish-in-flight',
              '_admin_ok': lambda request: True, '_offline_reap': lambda *a: None,
              '_local_offline': lambda now=None: self.state['until'] > self.now,
              '_offline_persist': persist, '_flow_event': lambda row: None,
              'flow_note_mode': lambda now: None, '_draining': lambda: self.draining,
              'time': types.SimpleNamespace(time=lambda: self.now),
              'os': types.SimpleNamespace(urandom=random),
              '_ACTIVE': {}, '_inflight': 0, 'flow_mode_now': lambda: {'mode': 'local-offline'},
              'web': types.SimpleNamespace(json_response=lambda row, **kwargs: {'body': row, **kwargs})}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<patched-installed-handler-only>', 'exec'), ns)
        self.handler = ns['gateway_offline']

    def request(self, method, payload=None, wait=None):
        async def body():
            if wait is not None:
                await wait.wait()
            return payload
        return types.SimpleNamespace(method=method, json=body, headers={})

    def call(self, method, payload=None):
        return asyncio.run(self.handler(self.request(method, payload)))


def test_server_patcher_preserves_all_unrelated_ast():
    before = ast.parse(CURRENT_SOURCE.read_bytes())
    after = ast.parse(patcher.patch_source(CURRENT_SOURCE.read_bytes()))
    strip = lambda tree: ast.dump(ast.Module(body=[n for n in tree.body if not
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) or n.name not in
        ('_offline_status', 'gateway_offline')], type_ignores=[]))
    assert strip(before) == strip(after)


def test_server_patcher_refuses_different_base():
    with pytest.raises(ValueError, match='pinned base'):
        patcher.patch_source(CURRENT_SOURCE.read_bytes() + b'\n')


def test_server_advertises_semantics_even_without_window():
    gw = Gateway(False)
    assert gw.call('GET')['body']['offline_lease_renewal'] == lg.PROTOCOL


def test_server_normal_open_unchanged():
    gw = Gateway(False)
    result = gw.call('POST', {'ttl_s': 60, 'by': 'first', 'reason': 'normal open'})
    assert result['body']['offline'] and gw.random_calls == 1
    assert gw.state['by'] == 'first' and gw.state['reason'] == 'normal open'


@pytest.mark.parametrize('lease', [None, '', 'foreign', 'b' * 32, False])
def test_server_supplied_foreign_lease_never_opens(lease):
    gw = Gateway(False)
    original = copy.deepcopy(gw.state)
    result = gw.call('POST', {'lease': lease, 'ttl_s': 60})
    assert result['status'] == 409 and gw.state == original and not gw.random_calls


def test_server_expired_exact_lease_never_reopens():
    gw = Gateway()
    gw.now = 301
    result = gw.call('POST', {'lease': LEASE, 'ttl_s': 60})
    assert result['status'] == 409 and not gw.random_calls and gw.state['until'] == 300


def test_server_exact_renew_preserves_owner_and_allows_drain():
    gw = Gateway()
    gw.draining = True
    result = gw.call('POST', {'lease': LEASE, 'ttl_s': 900, 'by': 'spoof', 'reason': 'spoof'})
    assert result['body']['lease'] == LEASE and gw.state['until'] == 1000
    assert gw.state['by'] == 'HaloFrontier' and gw.state['reason'] == 'owned'


def test_server_delete_then_late_inflight_post_is_409():
    gw = Gateway()
    async def race():
        gate = asyncio.Event()
        post = asyncio.create_task(gw.handler(gw.request('POST', {'lease': LEASE, 'ttl_s': 900}, gate)))
        await asyncio.sleep(0)
        deleted = await gw.handler(gw.request('DELETE', {'lease': LEASE}))
        gate.set()
        renewed = await post
        return deleted, renewed
    deleted, renewed = asyncio.run(race())
    assert deleted['body']['offline'] is False
    assert renewed['status'] == 409 and not gw.random_calls and not gw.state['lease']


def test_server_post_then_delete_leaves_no_window():
    gw = Gateway()
    assert gw.call('POST', {'lease': LEASE, 'ttl_s': 900})['body']['lease'] == LEASE
    assert gw.call('DELETE', {'lease': LEASE})['body']['offline'] is False
    assert gw.call('POST', {'lease': LEASE, 'ttl_s': 900})['status'] == 409
    assert gw.random_calls == 0


class Authority:
    def __init__(self):
        self.active = True
        self.supported = True
        self.lease = LEASE
        self.events = []
        self.fail = False
        self.post_hook = None
    def __call__(self, url, method='GET', payload=None):
        self.events.append(method)
        if method == 'GET':
            return {'offline': self.active, 'offline_lease_renewal': lg.PROTOCOL if self.supported else 'legacy'}
        if method == 'POST':
            if self.post_hook is not None:
                self.post_hook()
            if self.fail:
                raise TimeoutError('private error must not be copied')
            if not self.active or payload['lease'] != self.lease:
                return {'error': 'expired/mismatch', 'http': 409, 'offline': self.active,
                        'offline_lease_renewal': lg.PROTOCOL}
            return {'offline': True, 'offline_lease_renewal': lg.PROTOCOL, 'lease': self.lease,
                    'remaining_s': payload['ttl_s']}
        assert method == 'DELETE' and payload['lease'] == self.lease
        self.active = False
        return {'offline': False, 'offline_lease_renewal': lg.PROTOCOL}


@pytest.fixture
def owned(tmp_path):
    state = {'results': str(tmp_path), 'window': 'commission', 'lane': 'halo', 'by': 'HaloFrontier',
             'lease': LEASE, 'pid': 999999, 'pid_start': '42', 'restored': False}
    (tmp_path / 'state.json').write_text(json.dumps(state))
    return state, {'ttl_s': 3600, 'reason': 'CPU fixture'}, Authority()


def guardian(owned, **kwargs):
    return lg.RestorationLease(*owned, 'http://CPU-only', **kwargs)


def test_client_missing_capability_has_no_post(owned):
    state, spec, http = owned
    http.supported = False
    with pytest.raises(lg.Refused, match='advertise'):
        lg.extend(state, spec, http, 'http://CPU-only')
    assert http.events == ['GET'] and state['lease'] == LEASE


def test_client_expiry_does_not_open_or_adopt(owned):
    state, spec, http = owned
    http.active = False
    with pytest.raises(lg.Refused, match='absent'):
        lg.extend(state, spec, http, 'http://CPU-only')
    assert http.events == ['GET'] and state['lease'] == LEASE


def test_keeper_extends_during_blocked_restore_and_deferral(owned):
    state, _, http = owned
    with guardian(owned, cadence_s=.01) as guard:
        deadline = time.monotonic() + 1
        while guard.renewals < 3 and time.monotonic() < deadline:
            time.sleep(.005)
        assert guard.renewals >= 3 and state['lease'] == LEASE and http.active
        state['restored'] = False
    assert guard.quiesced and http.active and state['lease'] == LEASE
    assert json.loads((Path(state['results']) / 'state.json').read_text())['lease'] == LEASE


def test_close_joins_and_no_post_occurs_after_delete(owned):
    state, _, http = owned
    with guardian(owned, cadence_s=.005) as guard:
        guard.close()
        n = len(http.events)
        time.sleep(.015)
        assert len(http.events) == n
        assert http.events[-2:] == ['DELETE', 'GET']
        assert guard.quiesced and guard.closed and state['lease'] is None
        state['restored'] = True
    assert not http.active and json.loads((Path(state['results']) / 'state.json').read_text())['restored']


def test_inflight_keeper_close_refuses_delete_until_joined(owned):
    _, _, http = owned
    entered, release = threading.Event(), threading.Event()
    with guardian(owned, cadence_s=.005, join_s=.01) as guard:
        def block():
            entered.set()
            assert release.wait(1)
        http.post_hook = block
        assert entered.wait(1)
        with pytest.raises(lg.Refused, match='quiesced'):
            guard.close()
        assert 'DELETE' not in http.events and http.active
        http.post_hook = None
        release.set()
    assert guard.quiesced and 'DELETE' not in http.events


def test_keeper_failure_retains_lease_and_calls_owned_defer(owned):
    state, _, http = owned
    paused = []
    with guardian(owned, cadence_s=.005, on_defer=lambda: paused.append('owned') or {'owned': True}) as guard:
        http.fail = True
        deadline = time.monotonic() + 1
        while guard.failure is None and time.monotonic() < deadline:
            time.sleep(.005)
        with pytest.raises(lg.Refused):
            guard.close()
        guard.defer()
    assert paused and http.active and state['lease'] == LEASE and not state['restored']
    assert 'DELETE' not in http.events and 'private error' not in json.dumps(guard.audit)


def test_unknown_409_never_counts_as_successful_close(owned):
    state, _, http = owned
    original = http.__call__
    def transport(url, method='GET', payload=None):
        if method == 'DELETE':
            return {'error': 'mismatch', 'http': 409, 'offline': False}
        return original(url, method, payload)
    with guardian((state, owned[1], transport)) as guard:
        with pytest.raises(lg.Refused, match='409'):
            guard.close()
    assert not guard.closed and state['lease'] == LEASE


def test_crossprocess_restore_lock_refuses_without_actions(owned):
    state, _, http = owned
    paused = []
    with guardian(owned) as first:
        n = len(http.events)
        with pytest.raises(lg.OwnedElsewhere):
            with guardian(owned, on_defer=lambda: paused.append(True)):
                pytest.fail('foreign recovery entered')
        assert len(http.events) == n and not paused and state['lease'] == LEASE


def test_stale_deadman_authority_refuses_without_actions(owned):
    state, _, http = owned
    disk = {**state, 'lease': 'b' * 32}
    (Path(state['results']) / 'state.json').write_text(json.dumps(disk))
    paused = []
    with pytest.raises(lg.OwnedElsewhere):
        with guardian(owned, on_defer=lambda: paused.append(True)):
            pytest.fail('stale recovery entered')
    assert not http.events and not paused
    assert json.loads((Path(state['results']) / 'state.json').read_text()) == disk


def test_guard_state_published_before_owner_released(owned):
    state, _, _ = owned
    with guardian(owned) as first:
        first.close()
        state['restored'] = True
    stale = {**state, 'lease': LEASE, 'restored': False}
    with pytest.raises(lg.OwnedElsewhere):
        with guardian((stale, owned[1], owned[2])):
            pytest.fail('closed state replayed')


def test_deadman_fresh_owner_can_retry_same_lease_after_deferral(owned):
    state, spec, http = owned
    with guardian(owned):
        state['restored'] = False
    persisted = json.loads((Path(state['results']) / 'state.json').read_text())
    with guardian((persisted, spec, http)) as deadman:
        deadman.close()
        persisted['restored'] = True
    assert persisted['lease'] is None and not http.active


def test_unreadable_or_missing_state_is_not_empty_ownership(owned):
    state, _, http = owned
    (Path(state['results']) / 'state.json').unlink()
    with pytest.raises(lg.OwnedElsewhere):
        with guardian(owned):
            pytest.fail('missing state entered')
    assert not http.events


def test_default_off_restore_does_not_use_guard(monkeypatch):
    monkeypatch.setattr(wc, '_restore_from', lambda *a, **k: {'ok': True, 'legacy': True})
    monkeypatch.setattr(lg, 'RestorationLease', lambda *a, **k: pytest.fail('default-off guardian'))
    assert wc.restore_from({}, {}, {}) == {'ok': True, 'legacy': True}


def test_instrumented_state_without_spec_cannot_fall_back_to_legacy(owned, monkeypatch):
    state, _, http = owned
    state['profiler_configured'] = True
    monkeypatch.setattr(wc, '_restore_from', lambda *a, **k: pytest.fail('missing opt-in spec used legacy restore'))
    report = wc.restore_from({}, state, {})
    assert report['no_actions'] and report['owner_busy'] and report['deferred'] and not report['ok']
    assert state['lease'] == LEASE and not http.events


def test_source_patcher_preserves_mode_and_never_overwrites(tmp_path):
    source, target = tmp_path / 'current.py', tmp_path / 'candidate.py'
    source.write_bytes(CURRENT_SOURCE.read_bytes())
    source.chmod(0o755)
    patcher.main(['--source', str(source), '--out', str(target)])
    assert target.stat().st_mode & 0o777 == 0o755
    assert target.read_bytes() == patcher.patch_source(source.read_bytes())
    before = target.read_bytes()
    with pytest.raises(ValueError, match='fresh separate'):
        patcher.main(['--source', str(source), '--out', str(target)])
    assert target.read_bytes() == before and source.read_bytes() == CURRENT_SOURCE.read_bytes()


def test_wrapper_owns_keeper_through_cpu_restore_and_defers(owned, monkeypatch):
    state, spec, http = owned
    spec['owned_profiler'] = {}
    monkeypatch.setattr(wc, 'http_json', http)
    actions = []
    monkeypatch.setattr(wc, 'pause_recovery_producers', lambda *a: actions.append('declared') or {})
    def body(snap, current, spec, **kwargs):
        assert kwargs['lease_guard'].renewals >= 1 and current['lease'] == LEASE
        actions.append('original-collector-cleanup-deferred')
        current['restored'] = False
        return {'ok': False, 'profiler_cleanup': {'ok': False, 'deferred': True}}
    monkeypatch.setattr(wc, '_restore_from', body)
    report = wc.restore_from({}, state, spec)
    assert not report['ok'] and http.active and state['lease'] == LEASE
    assert actions == ['original-collector-cleanup-deferred', 'declared']
    assert report['offline_lease_guard']['quiesced'] and report['offline_lease_guard']['renewals'] >= 2


def test_wrapper_stale_owner_does_not_pause_or_write(owned, monkeypatch):
    state, spec, http = owned
    spec['owned_profiler'] = {}
    monkeypatch.setattr(wc, 'http_json', http)
    monkeypatch.setattr(wc, 'pause_recovery_producers', lambda *a: pytest.fail('foreign pause'))
    with guardian(owned):
        report = wc.restore_from({}, state, spec)
        assert report['owner_busy'] and report['no_actions'] and not report['ok']


def test_controller_inflight_normal_renewal_defers_without_restoring(tmp_path, monkeypatch):
    from test_windowctl import World, spec as window_spec
    import owned_profiler as op
    world = World(tmp_path / 'world', monkeypatch)
    tool = tmp_path / 'nsys'
    lib = tmp_path / 'libInjection.so'
    tool.write_bytes(b'CPU only')
    lib.write_bytes(b'CPU only')
    config = {'tool': str(tool), 'tool_sha256': op.digest(tool),
              'libraries': {str(lib): op.digest(lib)}}
    w = wc.Window(window_spec(tmp_path, owned_profiler=config,
        steps=[{'name': 'start', 'profiler': 'start'}]))
    monkeypatch.setattr(w, 'preflight', lambda: [])
    monkeypatch.setattr(op, 'baseline', lambda *a: {})
    monkeypatch.setattr(w, '_remote_watch_loop', lambda: None)
    joins = []
    class Inflight:
        def join(self, bound):
            joins.append(bound)
        def is_alive(self):
            return True
    def opened():
        w.state['lease'] = LEASE
        w._renew_thread = Inflight()
        world.offline = True
        w.save_state()
    monkeypatch.setattr(w, 'open_gateway', opened)
    monkeypatch.setattr(w, 'run_steps', lambda: (_ for _ in ()).throw(wc.WindowAbort('failed-step', 'CPU fault')))
    monkeypatch.setattr(w, 'restore', lambda: pytest.fail('in-flight renewal must not begin restore'))
    report = w.run()
    assert joins == [lg.JOIN_TIMEOUT_S]
    assert not report['restore']['ok'] and report['restore']['deferred']
    assert w.state['lease'] == LEASE and world.offline and not world.lease_closed
    assert (Path(w.results) / 'state.json').stat().st_mode & 0o777 == 0o600


def test_deadman_busy_owner_does_not_clobber_state_or_summary(owned, monkeypatch):
    state, _, _ = owned
    directory = Path(state['results'])
    (directory / 'snapshot.json').write_text('{}')
    (directory / 'spec.resolved.json').write_text('{}')
    state_path = directory / 'state.json'
    before = state_path.read_bytes()
    monkeypatch.setattr(wc.unitrun, 'proc_start', lambda pid: None)
    monkeypatch.setattr(wc, 'restore_from', lambda *a, **k: {
        'ok': False, 'deferred': True, 'owner_busy': True, 'no_actions': True})
    assert wc.deadman(str(state_path)) == 0
    assert state_path.read_bytes() == before and not (directory / 'summary.json').exists()


@pytest.mark.parametrize('task_state', ['Z', 'X', 'x'])
def test_instrumented_zombie_or_dead_owner_enters_restore(owned, tmp_path, monkeypatch, task_state):
    state, _, _ = owned
    state['profiler_configured'] = True
    state['hold'] = 'b' * 32
    state['deadline'] = time.time() + 3600
    directory = Path(state['results'])
    (directory / 'state.json').write_text(json.dumps(state))
    (directory / 'snapshot.json').write_text('{}')
    (directory / 'spec.resolved.json').write_text('{"owned_profiler": {}}')
    proc = tmp_path / 'proc'
    stat = proc / str(state['pid']) / 'stat'
    stat.parent.mkdir(parents=True)
    stat.write_text(str(state['pid']) + ' (fake ) controller) ' +
                    ' '.join([task_state] + ['0'] * 18 + [state['pid_start']]))
    monkeypatch.setattr(wc.unitrun, 'PROC', str(proc))
    monkeypatch.setattr(wc.os, 'kill', lambda *a: pytest.fail('dead task must not be signalled'))
    restored = []
    def restored_under_owner(snap, current, spec, **kwargs):
        restored.append((current['lease'], current['hold']))
        current['restored'] = True
        (directory / 'state.json').write_text(json.dumps(current))
        return {'ok': True}
    monkeypatch.setattr(wc, 'restore_from', restored_under_owner)
    assert wc.deadman(str(directory / 'state.json')) == 0
    assert restored == [(LEASE, 'b' * 32)]
    assert json.loads((directory / 'state.json').read_text())['restored'] is True


def test_instrumented_owner_stat_reuse_missing_and_live(tmp_path):
    stat = tmp_path / '1234' / 'stat'
    assert lg.owner_status(1234, '42', tmp_path) == 'gone'
    stat.parent.mkdir()
    stat.write_text('1234 (CPU fake) ' + ' '.join(['S'] + ['0'] * 18 + ['42']))
    assert lg.owner_status(1234, '42', tmp_path) == 'live'
    assert lg.owner_status(1234, '99', tmp_path) == 'gone'


def test_instrumented_unknown_owner_stat_refuses_without_actions(owned, tmp_path, monkeypatch):
    state, _, _ = owned
    state['profiler_configured'] = True
    directory = Path(state['results'])
    path = directory / 'state.json'
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    proc = tmp_path / 'proc'
    stat = proc / str(state['pid']) / 'stat'
    stat.parent.mkdir(parents=True)
    stat.write_text('malformed CPU-only stat')
    monkeypatch.setattr(wc.unitrun, 'PROC', str(proc))
    monkeypatch.setattr(wc.os, 'kill', lambda *a: pytest.fail('unknown task must not be signalled'))
    monkeypatch.setattr(wc, 'restore_from', lambda *a, **k: pytest.fail('unknown task must not restore'))
    assert wc.deadman(str(path)) == 1
    assert path.read_bytes() == before and not (directory / 'summary.json').exists()


def test_window_restore_does_not_rewrite_guardian_state_after_unlock(owned, monkeypatch):
    state, spec, _ = owned
    window = object.__new__(wc.Window)
    window.snapshot, window.state = {}, state
    window.spec = {**spec, 'owned_profiler': {}}
    window.results, window.log = state['results'], lambda *a: None
    monkeypatch.setattr(wc, 'restore_from', lambda *a, **k: {'ok': False, 'deferred': True})
    monkeypatch.setattr(window, 'save_state', lambda: pytest.fail('unlocked instrumented state write'))
    assert window.restore()['deferred']


def test_enter_renewal_failure_publishes_failed_state_while_locked(owned):
    state, spec, http = owned
    state['restored'] = False
    http.supported = False
    guard = guardian(owned)
    with pytest.raises(lg.Refused):
        with guard:
            pytest.fail('unsupported authority entered')
    assert guard.authority_verified and guard.lockfd is None
    assert json.loads((Path(state['results']) / 'state.json').read_text())['restored'] is False


def test_ambiguous_delete_transport_never_reacquires_or_claims_restore(owned):
    state, spec, http = owned
    def transport(url, method='GET', payload=None):
        row = http(url, method, payload)
        if method == 'DELETE':
            raise TimeoutError('DELETE acknowledgement unknown')
        return row
    paused = []
    with pytest.raises(lg.Refused):
        with guardian((state, spec, transport), on_defer=lambda: paused.append('declared') or {}) as guard:
            guard.close()
    assert not http.active and state['lease'] == LEASE and not state['restored']
    assert not guard.closed and paused
    assert http.events.count('DELETE') == 1
    assert http.events[http.events.index('DELETE') + 1:] == ['GET']


def test_failed_postdelete_readback_remains_unqualified(owned):
    state, spec, http = owned
    def transport(url, method='GET', payload=None):
        if method == 'GET' and not http.active:
            raise TimeoutError('readback unknown')
        return http(url, method, payload)
    paused = []
    with pytest.raises(lg.Refused):
        with guardian((state, spec, transport), on_defer=lambda: paused.append('declared') or {}) as guard:
            guard.close()
    assert guard.closed and state['lease'] is None and not state['restored'] and paused
    assert http.events[-1] == 'DELETE' and http.events.count('DELETE') == 1
