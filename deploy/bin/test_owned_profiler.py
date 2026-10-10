# SPDX-License-Identifier: Apache-2.0
"""Private CPU contracts: no Nsight, systemd, GPU, network or driver calls."""
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import owned_profiler as op
import windowctl as wc
import unitrun
from test_windowctl import World, spec as window_spec


@pytest.fixture(autouse=True)
def prohibit_external(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('external tool/service invocation in CPU contract')
    monkeypatch.setattr(op.subprocess, 'run', forbidden)
    for name in ('run', 'show', 'stop', 'stop_lane'):
        monkeypatch.setattr(unitrun, name, forbidden)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    tool = tmp_path / 'nsys'
    lib = tmp_path / 'libOtherInjection.so'
    tool.write_bytes(b'CPU fake tool')
    lib.write_bytes(b'CPU fake injection')
    config = {'tool': str(tool), 'tool_sha256': op.digest(tool),
              'libraries': {str(lib): op.digest(lib)}}
    results = tmp_path / 'results'
    results.mkdir()
    state = {'lane': 'rea', 'window': 'trace', 'results': str(results), 'lease': 'owned-window'}
    inherited = {'id': 'old-id', 'name': 'Unrelated', 'state': 'running'}
    baseline = {'sessions': [inherited], 'tool_processes': [],
                'engine': {'env_sha256': {}, 'libraries': {}, 'processes': []}}
    record = op.reserve(config, state, baseline)
    manifest = op.load(record, state)
    owner = {'pid': 12, 'start': '123', 'uid': os.getuid(), 'unit': manifest['unit'], 'argv_sha256': 'mock'}
    owned = {'id': 'own-id', 'name': manifest['session'], 'state': 'running'}
    data = {'config': config, 'state': state, 'baseline': baseline, 'record': record,
            'manifest': manifest, 'owner': owner, 'rows': [inherited, owned], 'events': []}
    monkeypatch.setattr(op, 'inspect_owner', lambda *a: data['owner'])
    monkeypatch.setattr(op, 'sessions', lambda *a: list(data['rows']))
    monkeypatch.setattr(op, 'tool_processes', lambda *a: [])
    def command(config, argv):
        data['events'].append(argv)
        data['rows'] = [r for r in data['rows'] if r['name'] != manifest['session']]
        return ''
    def stop(unit):
        data['events'].append(['unit-stop', unit])
        data['owner'] = None
        return True
    monkeypatch.setattr(op, 'command', command)
    monkeypatch.setattr(unitrun, 'stop', stop)
    return data


def ready_receipt(f, **overrides):
    receipt = {'manifest_sha256': f['record']['sha256'], 'session': f['rows'][-1],
               'owner': f['owner'], 'monotonic': op.time.monotonic()}
    receipt.update(overrides)
    op.atomic(Path(f['manifest']['directory']) / 'ready.json', receipt)
    return receipt


def test_bounded_no_application_command_uses_kill_none(fixture):
    args = op.argv(fixture['manifest'], fixture['record'])
    assert '--kill=none' in args and '--duration=120' in args and '--trace=cuda-sw' in args
    assert '--cuda-trace-scope=system-wide' in args and '--sample=none' in args
    assert args[-1].startswith('--output=')
    assert sum(a.startswith('--after-collection-start=') for a in args) == 1
    assert Path(fixture['record']['manifest']).stat().st_mode & 0o077 == 0


@pytest.mark.parametrize('delta', [ {'collection_s': 0}, {'collection_s': 301}, {'collection_s': True},
    {'cleanup_timeout_s': 61}, {'injection_env': {'LD_PRELOAD': 'anything'}}, {'unknown': 1} ])
def test_config_rejects_unbounded_or_implicit_injection(fixture, delta):
    with pytest.raises(op.Refused):
        op.validate_config({**fixture['config'], **delta})


@pytest.mark.parametrize('raw', ['{}', 'null', '[{"id":"x","ID":"y","name":"n","state":"s"}]',
    '[{"id":"x","name":"n"}]', '[{"id":"x","name":"n","state":"s"},{"id":"x","name":"m","state":"s"}]'])
def test_unknown_session_schema_never_means_empty(fixture, monkeypatch, raw):
    # Restore the real adapter while keeping its command fully mocked.
    monkeypatch.setattr(op, 'command', lambda *a: raw)
    with pytest.raises((op.Refused, ValueError)):
        op._sessions_for_test(fixture['config'])


def test_exact_owned_cleanup_and_idempotence_preserve_inherited(fixture):
    ready_receipt(fixture)
    result = op.cleanup(fixture['record'], fixture['state'])
    assert result['ok']
    assert fixture['events'] == [['shutdown', '--session=own-id', '--kill=none'],
                                 ['unit-stop', fixture['manifest']['unit']]]
    assert fixture['rows'] == fixture['baseline']['sessions']
    fixture['events'].clear()
    assert op.cleanup(fixture['record'], fixture['state'])['ok']
    assert not fixture['events']


def test_pre_callback_recovery_of_still_owned_collector(fixture):
    assert op.cleanup(fixture['record'], fixture['state'])['ok']
    assert fixture['events'][0] == ['shutdown', '--session=own-id', '--kill=none']


def test_completed_collector_with_no_session_does_not_stop_other_sessions(fixture):
    fixture['owner'] = None
    fixture['rows'] = fixture['baseline']['sessions']
    assert op.cleanup(fixture['record'], fixture['state'])['ok']
    assert not fixture['events']


@pytest.mark.parametrize('case', ['owner_dead', 'id_changed', 'inherited_id', 'escaped', 'empty_ready', 'foreign_window', 'tampered_manifest', 'public_receipt'])
def test_ambiguous_ownership_refuses_before_side_effects(fixture, monkeypatch, case):
    ready_receipt(fixture)
    if case == 'owner_dead':
        fixture['owner'] = None
    elif case == 'id_changed':
        fixture['rows'][-1] = {**fixture['rows'][-1], 'id': 'new-id'}
    elif case == 'inherited_id':
        fixture['rows'][-1] = {**fixture['rows'][-1], 'id': 'old-id'}
    elif case == 'escaped':
        monkeypatch.setattr(op, 'tool_processes', lambda *a: [{'pid': 99, 'start': '9', 'unit': 'unrelated.service'}])
    elif case == 'empty_ready':
        op.atomic(Path(fixture['manifest']['directory']) / 'ready.json', {})
    elif case == 'foreign_window':
        fixture['state']['window'] = 'other'
    elif case == 'tampered_manifest':
        Path(fixture['record']['manifest']).write_text('{}')
    else:
        (Path(fixture['manifest']['directory']) / 'ready.json').chmod(0o644)
    result = op.cleanup(fixture['record'], fixture['state'])
    assert not result['ok'] and result['deferred']
    assert not fixture['events']


def test_shutdown_failure_is_retryable_without_unit_stop(fixture, monkeypatch):
    saved = op.command
    monkeypatch.setattr(op, 'command', lambda *a: (_ for _ in ()).throw(op.Refused('failed')))
    assert not op.cleanup(fixture['record'], fixture['state'])['ok']
    assert not fixture['events']
    monkeypatch.setattr(op, 'command', saved)
    assert op.cleanup(fixture['record'], fixture['state'])['ok']


def test_shutdown_timeout_preserves_unit(fixture, monkeypatch):
    monkeypatch.setattr(op, 'command', lambda c, a: fixture['events'].append(a))
    ticks = iter([0, 31])
    assert not op.cleanup(fixture['record'], fixture['state'], clock=lambda: next(ticks), sleep=lambda _: None)['ok']
    assert fixture['events'] == [['shutdown', '--session=own-id', '--kill=none']]


def test_unit_stop_failure_retries_after_session_shutdown(fixture, monkeypatch):
    monkeypatch.setattr(unitrun, 'stop', lambda _: False)
    assert not op.cleanup(fixture['record'], fixture['state'])['ok']
    assert fixture['rows'] == fixture['baseline']['sessions']
    fixture['owner'] = None  # collector subsequently completed naturally
    assert op.cleanup(fixture['record'], fixture['state'])['ok']


def test_start_timeout_keeps_durable_manifest_for_cleanup(fixture, monkeypatch):
    monkeypatch.setattr(unitrun, 'run', lambda *a, **k: {'result': 'started', 'unit': fixture['manifest']['unit']})
    ticks = iter([0, 31])
    fixture['rows'] = fixture['baseline']['sessions']
    with pytest.raises(op.Refused, match='timed out'):
        op.start(fixture['record'], fixture['state'], clock=lambda: next(ticks), sleep=lambda _: None)
    assert op.load(fixture['record'], fixture['state'])['session'] == fixture['manifest']['session']
    assert op.cleanup(fixture['record'], fixture['state'])['ok']


def test_collection_boot_gate_checks_generation_and_duration(fixture):
    ready_receipt(fixture)
    assert op.collection_active(fixture['record'], fixture['state'])['owner'] == fixture['owner']
    fixture['owner'] = {**fixture['owner'], 'start': 'reused'}
    with pytest.raises(op.Refused):
        op.collection_active(fixture['record'], fixture['state'])


def test_inspect_owner_checks_exact_argv_start_uid_unit(fixture, monkeypatch):
    owner = fixture['owner']
    expected = hashlib.sha256(b'\0'.join(a.encode() for a in op.argv(fixture['manifest'], fixture['record'])) + b'\0').hexdigest()
    owner = {**owner, 'argv_sha256': expected}
    monkeypatch.setattr(unitrun, 'show', lambda *a: {'LoadState': 'loaded', 'ActiveState': 'active', 'MainPID': '12'})
    monkeypatch.setattr(op, 'proc_info', lambda _: owner)
    # Use real function, fixture replaces it only to keep other cases simple.
    original = sys.modules['owned_profiler'].__dict__['_inspect_owner_for_test']
    assert original(fixture['manifest'], fixture['record']) == owner
    owner['unit'] = fixture['manifest']['unit'] + '2'
    with pytest.raises(op.Refused):
        original(fixture['manifest'], fixture['record'])


def test_declared_injection_mapping_and_child_env_are_detected(fixture, tmp_path, monkeypatch):
    proc = tmp_path / 'proc'
    lib = Path(next(iter(fixture['config']['libraries'])))
    stat = lib.stat()
    for pid, children, env, maps in [
        (1, '2', b'', ''),
        (2, '', b'CUDA_INJECTION_SHM_ALLOWED=TRUE\0NSYS_FOO=secret\0',
         f'1000-2000 r-xp 0000 {os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x} {stat.st_ino} {lib}\n')]:
        directory = proc / str(pid)
        (directory / 'task' / str(pid)).mkdir(parents=True)
        (directory / 'task' / str(pid) / 'children').write_text(children)
        (directory / 'environ').write_bytes(env)
        (directory / 'maps').write_text(maps)
    monkeypatch.setattr(op, 'PROC', proc)
    monkeypatch.setattr(op, 'proc_info', lambda pid: {'start': str(pid), 'uid': os.getuid()})
    observed = op.engine_state({'engine_pid': 1, 'pid_start': '1'}, fixture['config'])
    assert observed['libraries'] == fixture['config']['libraries']
    assert set(observed['env_sha256']) == {'CUDA_INJECTION_SHM_ALLOWED', 'NSYS_FOO'}
    assert 'secret' not in json.dumps(observed)
    # A deleted mapping cannot be treated as absent or clean.
    (proc / '2' / 'maps').write_text((proc / '2' / 'maps').read_text().strip() + ' (deleted)\n')
    with pytest.raises(op.Refused, match='deleted'):
        op.engine_state({'engine_pid': 1, 'pid_start': '1'}, fixture['config'])


@pytest.mark.parametrize('mismatch', ['env', 'library', 'process', 'session'])
def test_restoration_rejects_lingering_instrumentation(fixture, monkeypatch, mismatch):
    current = {'env_sha256': {}, 'libraries': {}, 'processes': [{'pid': 50, 'start': '50'}]}
    fixture['rows'] = fixture['baseline']['sessions']
    if mismatch == 'env':
        current['env_sha256'] = {'CUDA_INJECTION_SHM_ALLOWED': ['new']}
    elif mismatch == 'library':
        current['libraries'] = fixture['config']['libraries']
    elif mismatch == 'process':
        monkeypatch.setattr(op, 'tool_processes', lambda *a: [{'pid': 99, 'start': '9'}])
    else:
        fixture['rows'] = fixture['rows'] + [{'id': 'new', 'name': 'new', 'state': 'running'}]
    monkeypatch.setattr(op, 'engine_state', lambda *a: current)
    assert not op.restored(fixture['baseline'], {}, fixture['config'])['ok']


def test_unchanged_inherited_cupti_is_allowed(fixture, monkeypatch):
    original = fixture['baseline']
    original['engine']['libraries'] = {'/original/libcupti.so': 'old'}
    fixture['rows'] = original['sessions']
    monkeypatch.setattr(op, 'engine_state', lambda *a: {**original['engine'], 'processes': [{'pid': 100, 'start': '100'}]})
    assert op.restored(original, {}, fixture['config'])['ok']


def test_spec_default_off_and_instrumented_restrictions(fixture, tmp_path):
    assert wc.validate(window_spec(tmp_path))
    for extra in [ {}, {'owned_profiler': fixture['config'], 'promote': {'release': 'x'}} ]:
        with pytest.raises(wc.SpecError):
            wc.validate(window_spec(tmp_path, steps=[{'name': 'start', 'profiler': 'start'}], **extra))
    assert wc.validate(window_spec(tmp_path, owned_profiler=fixture['config'],
        steps=[{'name': 'start', 'profiler': 'start'}, {'name': 'stop', 'profiler': 'stop'}]))


def test_window_manifest_is_saved_before_launch(fixture, tmp_path, monkeypatch):
    World(tmp_path / 'world', monkeypatch)
    w = wc.Window(window_spec(tmp_path, owned_profiler=fixture['config'], steps=[{'name': 'start', 'profiler': 'start'}]))
    w.state['lease'] = 'owned'
    w.snapshot['owned_profiler'] = fixture['baseline']
    events = []
    record = fixture['record']
    monkeypatch.setattr(op, 'reserve', lambda *a: record)
    monkeypatch.setattr(w, 'save_state', lambda: events.append(('save', dict(w.state['instrumentation']))))
    monkeypatch.setattr(op, 'start', lambda r, s: events.append(('launch', dict(s['instrumentation']))) or {})
    assert w.step_profiler({'profiler': 'start'})['rc'] == 0
    assert [e[0] for e in events] == ['save', 'launch']


@pytest.mark.parametrize('failure', ['cleanup', 'restore', 'generation', 'post-generation', None])
def test_restore_order_and_fail_closed_recovery(fixture, tmp_path, monkeypatch, failure):
    world = World(tmp_path / 'world', monkeypatch)
    w = wc.Window(window_spec(tmp_path, owned_profiler=fixture['config'], steps=[{'name': 'start', 'profiler': 'start'}]))
    Path(w.results).mkdir(exist_ok=True)
    monkeypatch.setattr(op, 'baseline', lambda *a: fixture['baseline'])
    w.take_snapshot()
    w.state.update(lease='owned', hold='oldhold', instrumentation=fixture['record'])
    world.offline = True
    world.timers['vllm-qwen27b-watchdog.timer'] = 'inactive'
    events = []
    monkeypatch.setattr(op, 'cleanup', lambda *a: events.append('cleanup') or {'ok': failure != 'cleanup'})
    def stop_jobs(*a, **k):
        assert k['exclude'] == wc.abort_exclusions(w.state)
        events.append('jobs')
        return ['mock-owned-probe.service']
    monkeypatch.setattr(unitrun, 'stop_lane', stop_jobs)
    def run(*a, **k):
        events.append('boot')
        world.actuator_calls.append({'cmd': a[2]})
        world.pid += 1
        return {'rc': 0, 'unit': unitrun.unit_name(a[0], a[1])}
    monkeypatch.setattr(unitrun, 'run', run)
    def restore_gate(*a):
        second = events.count('maps-env') == 1
        events.append('maps-env')
        return {'ok': failure != 'restore' and not (second and failure == 'post-generation')}
    monkeypatch.setattr(op, 'restored', restore_gate)
    monkeypatch.setattr(wc, 'restoration_generation', lambda *a: events.append('generation') or {'ok': failure != 'generation'})
    monkeypatch.setattr(wc, 'set_timer', lambda t, on: events.append('timer') or world.timers.__setitem__(t, 'active' if on else 'inactive') or True)
    result = wc.restore_from(w.snapshot, w.state, w.spec, log=lambda _: None, results=w.results)
    assert result['ok'] == (failure is None)
    assert events[:2] == ['jobs', 'cleanup']
    if failure == 'cleanup':
        assert events == ['jobs', 'cleanup'] and not world.actuator_calls
        assert result['stopped_units'] == ['mock-owned-probe.service']
    else:
        assert events[:4] == ['jobs', 'cleanup', 'boot', 'maps-env']
        assert world.pid != w.snapshot['runtime_identity']['engine_pid']
    if failure:
        assert not world.lease_closed and world.offline and w.state['hold'] == 'oldhold'
        assert 'timer' not in events and not w.state['restored']
    else:
        assert events.index('generation') < events.index('timer')
        assert world.lease_closed == ['owned']


def test_broad_abort_excludes_only_exact_recorded_collector():
    state = {'lane': 'rea', 'window': 'trace'}
    deadman = unitrun.unit_name('rea', 'trace-deadman') + '.service'
    collector = unitrun.unit_name('rea', 'trace-profiler') + '.service'
    assert wc.abort_exclusions(state) == (deadman,)
    state['instrumentation'] = {'manifest': 'irrelevant-foreign-name'}
    assert wc.abort_exclusions(state) == (deadman, collector)


@pytest.mark.parametrize('delta', [{'lane': 'x' * 210}, {'window': 'w' * 210},
    {'steps': [{'name': 'start', 'profiler': 'start'}, {'name': 'profiler', 'run': 'x'}]},
    {'steps': [{'name': 'start', 'profiler': 'start'}, {'name': 'deadman', 'run': 'x'}]},
    {'steps': [{'name': 'start', 'profiler': 'start'}, {'name': 'restore-1', 'engine': 'stop'}]}])
def test_instrumented_reserved_namespace_rejects_collisions(fixture, tmp_path, delta):
    s = window_spec(tmp_path, owned_profiler=fixture['config'], steps=[{'name': 'start', 'profiler': 'start'}])
    s.update(delta)
    with pytest.raises(wc.SpecError):
        wc.validate(s)


def test_ready_callback_binds_session_and_unit(fixture, monkeypatch):
    monkeypatch.setenv('NSYS_SESSION_NAME', fixture['manifest']['session'])
    monkeypatch.setenv('NSYS_CALLBACK_NAME', 'after-collection-start')
    monkeypatch.setattr(op, 'proc_info', lambda pid: {'unit': fixture['manifest']['unit'], 'start': str(pid)})
    op.ready(fixture['record'])
    receipt = op.read_receipt(Path(fixture['manifest']['directory']) / 'ready.json')
    assert receipt['owner'] == fixture['owner'] and receipt['session']['id'] == 'own-id'
    with pytest.raises(op.Refused, match='duplicate'):
        op.ready(fixture['record'])


def test_wrong_callback_environment_does_not_issue_receipt(fixture, monkeypatch):
    monkeypatch.setenv('NSYS_SESSION_NAME', 'foreign')
    monkeypatch.setenv('NSYS_CALLBACK_NAME', 'after-collection-start')
    with pytest.raises(op.Refused, match='callback'):
        op.ready(fixture['record'])
    assert not (Path(fixture['manifest']['directory']) / 'ready.json').exists()


def test_start_receives_owned_readiness_callback(fixture, monkeypatch):
    owned = fixture['rows'][-1]
    fixture['rows'] = fixture['baseline']['sessions']
    def launch(*a, **k):
        assert k['wait'] is False and k['timeout_s'] == 240
        fixture['rows'] = [*fixture['rows'], owned]
        ready_receipt(fixture)
        return {'result': 'started', 'unit': fixture['manifest']['unit']}
    monkeypatch.setattr(unitrun, 'run', launch)
    assert op.start(fixture['record'], fixture['state'])['session']['id'] == 'own-id'


def test_boot_needs_explicit_injection_permission(fixture, tmp_path, monkeypatch):
    World(tmp_path / 'world', monkeypatch)
    config = {**fixture['config'], 'injection_env': {'CUDA_INJECTION_SHM_ALLOWED': 'TRUE'}}
    w = wc.Window(window_spec(tmp_path, owned_profiler=config, steps=[{'name': 'start', 'profiler': 'start'}]))
    w.state['instrumentation'] = fixture['record']
    monkeypatch.setattr(op, 'collection_active', lambda *a: {})
    with pytest.raises(wc.WindowAbort, match='explicit'):
        w.step_boot({'name': 'boot'}, {'boot': {'default': True}})


def test_concurrent_cleanup_defers_without_session_or_unit_side_effects(fixture):
    with op.cleanup_lock(fixture['manifest']):
        result = op.cleanup(fixture['record'], fixture['state'])
    assert not result['ok'] and 'in progress' in result['error']
    assert not fixture['events']
    assert op.cleanup(fixture['record'], fixture['state'])['ok']


def test_default_off_restore_does_not_consult_profiler(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    w = wc.Window(window_spec(tmp_path))
    Path(w.results).mkdir()
    w.take_snapshot()
    w.state['lease'] = 'owned'
    world.offline = True
    def forbidden(*a, **k):
        raise AssertionError('profiler touched in default-off path')
    for name in ('baseline', 'cleanup', 'restored', 'sessions', 'tool_processes'):
        monkeypatch.setattr(op, name, forbidden)
    assert wc.restore_from(w.snapshot, w.state, w.spec, log=lambda _: None, results=w.results)['ok']
    assert not world.actuator_calls and world.lease_closed == ['owned']


def test_instrumented_window_boundary_preserves_neighbor_units(monkeypatch):
    state = {'lane': 'rea', 'window': 'trace', 'instrumentation': {'manifest': 'recorded'}}
    units = [unitrun.unit_name('rea', name) + '.service' for name in
             ('trace-probe', 'trace-profiler', 'trace-deadman', 'trace2-profiler')]
    monkeypatch.setattr(unitrun, 'list_units', lambda _: [{'unit': u, 'active': 'active'} for u in units])
    stopped = []
    monkeypatch.setattr(unitrun, 'stop', lambda u: stopped.append(u) or True)
    actual = sys.modules['owned_profiler'].__dict__['_stop_lane_for_test']
    monkeypatch.setattr(unitrun, 'stop_lane', actual)
    assert wc.stop_window_units(state) == [units[0]] and stopped == [units[0]]


def test_malformed_record_is_not_empty_ownership(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    w = wc.Window(window_spec(tmp_path))
    Path(w.results).mkdir()
    w.take_snapshot()
    w.state.update(lease='owned', instrumentation={})
    world.offline = True
    captured = []
    monkeypatch.setattr(unitrun, 'stop_lane', lambda *a, **k: captured.append(k) or [])
    result = wc.restore_from(w.snapshot, w.state, w.spec, log=lambda _: None, results=w.results)
    assert not result['ok'] and not w.state['restored'] and world.offline
    assert not world.lease_closed and not world.actuator_calls
    assert captured[0]['prefix_boundary'] and len(captured[0]['exclude']) == 2


def test_known_collector_namespace_is_protected_before_manifest_reservation():
    state = {'lane': 'rea', 'window': 'trace', 'profiler_configured': True}
    assert unitrun.unit_name('rea', 'trace-profiler') + '.service' in wc.abort_exclusions(state)


# Preserve the actual implementation before fixture monkeypatches are applied.
op._inspect_owner_for_test = op.inspect_owner
op._sessions_for_test = op.sessions
op._stop_lane_for_test = unitrun.stop_lane
