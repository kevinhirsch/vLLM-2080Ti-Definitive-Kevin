#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Optional, fail-closed Nsight owner for windowctl; imports no CUDA packages.

Only an exact current-window unit/session may be stopped. CLI JSON shape and
automatic injection still need offline qualification on the installed tool.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

import unitrun

PROC = Path('/proc')
INJECTION_KEYS = ('CUDA_INJECTION_SHM_ALLOWED', 'CUDA_INJECTION64_PATH',
                  'CUDA_INJECTION32_PATH', 'NVTX_INJECTION64_PATH', 'LD_PRELOAD')


class Refused(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise Refused(reason)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with open(tmp, 'x', encoding='utf-8') as out:
        os.chmod(tmp, 0o600)
        json.dump(data, out, sort_keys=True)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def read_receipt(path):
    path = Path(path)
    require(path.resolve() == path and path.stat().st_uid == os.getuid()
            and path.stat().st_mode & 0o077 == 0, 'receipt owner/path/mode changed')
    return json.loads(path.read_text())


def validate_config(config):
    require(isinstance(config, dict), 'owned_profiler must be a mapping')
    require(set(config) <= {'tool', 'tool_sha256', 'libraries', 'collection_s', 'ready_timeout_s',
                            'cleanup_timeout_s', 'graph', 'injection_env'}, 'unknown owned_profiler field')
    tool = config.get('tool')
    require(isinstance(tool, str) and Path(tool).is_absolute() and Path(tool).name == 'nsys', 'explicit absolute nsys path required')
    require(isinstance(config.get('tool_sha256'), str) and re.fullmatch('[0-9a-f]{64}', config['tool_sha256']), 'nsys SHA256 required')
    libs = config.get('libraries')
    require(isinstance(libs, dict) and libs, 'explicit profiler library hashes required')
    for path, value in libs.items():
        require(isinstance(path, str) and Path(path).is_absolute() and Path(path).parent == Path(tool).parent
                and Path(path).name.startswith('lib') and '.so' in Path(path).name
                and isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'invalid profiler library hash')
    for key, default, limit in (('collection_s', 120, 300), ('ready_timeout_s', 30, 60), ('cleanup_timeout_s', 30, 60)):
        value = config.get(key, default)
        require(type(value) is int and 1 <= value <= limit, f'{key} must be1..{limit}')
    require(config.get('graph', 'graph:host-only') in ('graph:host-only', 'node:host-only'), 'unsupported graph mode')
    require(config.get('injection_env', {}) in ({}, {'CUDA_INJECTION_SHM_ALLOWED': 'TRUE'}), 'only explicit SHM permission arm is allowed')
    return config


def check_files(config):
    validate_config(config)
    for path, expected in {config['tool']: config['tool_sha256'], **config['libraries']}.items():
        require(Path(path).resolve() == Path(path) and digest(path) == expected, 'profiler file/path changed')


def command(config, argv):
    check_files(config)
    result = subprocess.run([config['tool'], *argv], capture_output=True, text=True,
                            timeout=config.get('cleanup_timeout_s', 30))
    require(result.returncode == 0, 'Nsight command failed')
    return result.stdout


def sessions(config):
    """Strict adapter: unknown schemas fail closed, never mean no sessions."""
    rows = json.loads(command(config, ['sessions', 'list', '--output-format=json']))
    if isinstance(rows, dict) and set(rows) == {'sessions'}:
        rows = rows['sessions']
    require(isinstance(rows, list), 'unknown Nsight session JSON shape')
    out = []
    for row in rows:
        require(isinstance(row, dict), 'invalid session row')
        parsed = []
        for aliases in (('id', 'ID', 'Session ID'), ('name', 'Name', 'Session Name'), ('state', 'State')):
            found = [row[key] for key in aliases if key in row]
            require(len(found) == 1 and isinstance(found[0], (str, int)) and not isinstance(found[0], bool), 'unknown/contradictory session fields')
            parsed.append(str(found[0]))
        require(all(parsed) and re.fullmatch('[A-Za-z0-9_.:-]+', parsed[0]), 'invalid session identifier')
        out.append(dict(zip(('id', 'name', 'state'), parsed)))
    require(len({r['id'] for r in out}) == len(out) and len({r['name'] for r in out}) == len(out), 'duplicate session identity')
    return out


def proc_info(pid):
    p = PROC / str(pid)
    start = p.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
    return {'pid': int(pid), 'start': start, 'uid': p.stat().st_uid,
            'unit': unitrun.unit_of_cgroup(p.joinpath('cgroup').read_text()),
            'argv_sha256': hashlib.sha256(p.joinpath('cmdline').read_bytes()).hexdigest()}


def tool_processes(config):
    found = []
    for p in PROC.iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            executable = p.joinpath('exe').resolve(strict=True)
            if executable.is_relative_to(Path(config['tool']).parent):
                found.append(proc_info(int(p.name)))
        except FileNotFoundError:
            if p.exists():
                raise Refused('cannot identify same-UID profiler process')
    return found


def process_tree(pid):
    pending, found = [pid], {}
    while pending:
        current = pending.pop()
        if current in found:
            continue
        info = proc_info(current)
        require(info['uid'] == os.getuid(), 'engine target is not same UID')
        found[current] = {'pid': current, 'start': info['start']}
        for task in (PROC / str(current) / 'task').iterdir():
            pending.extend(int(v) for v in task.joinpath('children').read_text().split())
        require(proc_info(current)['start'] == info['start'], 'engine descendant changed')
    return [found[k] for k in sorted(found)]


def engine_state(identity, config):
    """Hash only instrumentation env values and mapped ELF files; no raw env."""
    pid = identity['engine_pid']
    require(proc_info(pid)['start'] == str(identity['pid_start']), 'engine generation changed')
    processes = process_tree(pid)
    envs, libs = {}, {}
    for info in processes:
        current = info['pid']
        p = PROC / str(current)
        for raw in p.joinpath('environ').read_bytes().split(b'\0'):
            if b'=' not in raw:
                continue
            key, value = raw.split(b'=', 1)
            key = key.decode('utf-8', errors='strict')
            if key in INJECTION_KEYS or key.startswith(('NSYS_', 'CUPTI_')):
                envs.setdefault(key, set()).add(hashlib.sha256(value).hexdigest())
        for line in p.joinpath('maps').read_text().splitlines():
            fields = line.split(None, 5)
            if len(fields) < 6:
                continue
            path = fields[5].replace('\\040', ' ')
            mapped = Path(path.removesuffix(' (deleted)'))
            name = mapped.name
            declared = str(mapped) in config['libraries'] or mapped.is_relative_to(Path(config['tool']).parent)
            if not declared and not name.startswith(('libToolsInjection', 'libcupti', 'libNsight', 'libNvtxInjection')):
                continue
            require(not path.endswith(' (deleted)'), 'deleted profiler mapping')
            target = Path(path)
            stat = target.stat()
            major, minor = (int(v, 16) for v in fields[3].split(':'))
            require(stat.st_ino == int(fields[4]) and os.major(stat.st_dev) == major and os.minor(stat.st_dev) == minor, 'profiler mapped inode changed')
            resolved = str(target.resolve())
            if resolved not in libs:
                libs[resolved] = digest(target)
        require(proc_info(current)['start'] == info['start'], 'engine descendant changed')
    require(proc_info(pid)['start'] == str(identity['pid_start']), 'engine generation changed')
    require(process_tree(pid) == processes, 'engine descendant tree changed')
    return {'env_sha256': {k: sorted(v) for k, v in envs.items()}, 'libraries': libs, 'processes': processes}


def baseline(config, identity):
    check_files(config)
    result = {'sessions': sessions(config), 'tool_processes': tool_processes(config), 'engine': engine_state(identity, config)}
    require(all(Path(p).name.startswith('libcupti') for p in result['engine']['libraries']), 'original engine already has profiler injection')
    return result


def reserve(config, state, original):
    require(state.get('lease'), 'profiler requires owned offline window')
    check_files(config)
    directory = Path(state['results']) / 'owned-profiler'
    require(not directory.exists(), 'profiler directory already exists')
    directory.mkdir(mode=0o700)
    name = 'Halo_' + uuid.uuid4().hex
    helper = Path(__file__).resolve()
    manifest = {'schema': 1, 'window': state['window'], 'lane': state['lane'], 'uid': os.getuid(),
                'unit': unitrun.unit_name(state['lane'], state['window'] + '-profiler') + '.service',
                'session': name, 'config': config, 'baseline': original,
                'helper': str(helper), 'helper_sha256': digest(helper),
                'directory': str(directory), 'created': time.time()}
    record = {'manifest': str(directory / 'manifest.json')}
    atomic(record['manifest'], manifest)
    record['sha256'] = digest(record['manifest'])
    return record


def load(record, state=None):
    path = Path(record['manifest'])
    require(path.is_absolute() and path.resolve() == path and digest(path) == record['sha256'], 'instrumentation manifest changed')
    require(path.stat().st_uid == os.getuid() and path.stat().st_mode & 0o077 == 0, 'manifest owner/mode changed')
    manifest = json.loads(path.read_text())
    require(manifest.get('schema') == 1 and manifest.get('uid') == os.getuid(), 'manifest schema/UID mismatch')
    require(Path(manifest['directory']) == path.parent and path.parent.name == 'owned-profiler', 'manifest directory mismatch')
    require(path.parent.stat().st_mode & 0o077 == 0 and path.parent.stat().st_uid == os.getuid(), 'private directory changed')
    require(manifest['unit'] == unitrun.unit_name(manifest['lane'], manifest['window'] + '-profiler') + '.service'
            and re.fullmatch('Halo_[0-9a-f]{32}', manifest['session']), 'ownership namespace mismatch')
    if state is not None:
        require(manifest['window'] == state['window'] and manifest['lane'] == state['lane']
                and path.parent.parent == Path(state['results']), 'foreign window manifest')
    require(Path(manifest['helper']).resolve() == Path(__file__).resolve()
            and digest(manifest['helper']) == manifest['helper_sha256'], 'callback helper changed')
    check_files(manifest['config'])
    return manifest


def argv(manifest, record):
    c = manifest['config']
    callback = shlex.join(['/usr/bin/python3', manifest['helper'], 'ready', '--manifest', record['manifest'], '--sha', record['sha256']])
    return [c['tool'], 'profile', '--session-new=' + manifest['session'], '--trace=cuda-sw',
            '--cuda-trace-scope=system-wide', '--sample=none', '--cpuctxsw=none', '--backtrace=none',
            '--cudabacktrace=none', '--cuda-memory-usage=false', '--cuda-trace-all-apis=false',
            '--cuda-graph-trace=' + c.get('graph', 'graph:host-only'),
            '--duration=' + str(c.get('collection_s', 120)), '--stop-on-exit=false', '--kill=none',
            '--force-overwrite=false', '--after-collection-start=' + callback,
            '--output=' + str(Path(manifest['directory']) / 'trace')]


def inspect_owner(manifest, record):
    props = unitrun.show(manifest['unit'], 'LoadState', 'ActiveState', 'MainPID')
    require(props.get('LoadState') in ('loaded', 'not-found') and props.get('ActiveState') in ('active', 'activating', 'inactive', 'failed'), 'collector unit status unknown')
    pid = props.get('MainPID')
    require(isinstance(pid, str) and pid.isdigit(), 'collector MainPID unreadable')
    if int(pid) == 0:
        require(props['ActiveState'] in ('inactive', 'failed'), 'collector active without MainPID')
        return None
    info = proc_info(int(pid))
    expected = hashlib.sha256(b'\0'.join(v.encode() for v in argv(manifest, record)) + b'\0').hexdigest()
    require(info['uid'] == manifest['uid'] and info['unit'] == manifest['unit'] and info['argv_sha256'] == expected, 'collector ownership changed')
    return info


def ready(record):
    manifest = load(record)
    require(os.environ.get('NSYS_SESSION_NAME') == manifest['session'] and os.environ.get('NSYS_CALLBACK_NAME') == 'after-collection-start', 'callback session mismatch')
    owner = inspect_owner(manifest, record)
    require(owner and proc_info(os.getpid())['unit'] == manifest['unit'], 'callback escaped owned collector')
    matches = [r for r in sessions(manifest['config']) if r['name'] == manifest['session']]
    require(len(matches) == 1 and not any(r['id'] == matches[0]['id'] for r in manifest['baseline']['sessions']), 'session ownership contradicted')
    receipt = {'manifest_sha256': record['sha256'], 'session': matches[0], 'owner': owner,
               'callback_pid': os.getpid(), 'callback_start': proc_info(os.getpid())['start'],
               'started': time.time(), 'monotonic': time.monotonic()}
    path = Path(manifest['directory']) / 'ready.json'
    require(not path.exists(), 'duplicate collection callback')
    atomic(path, receipt)


def start(record, state, *, clock=time.monotonic, sleep=time.sleep):
    manifest = load(record, state)
    require(not any(r['name'] == manifest['session'] for r in sessions(manifest['config'])), 'session already exists')
    result = unitrun.run(state['lane'], state['window'] + '-profiler', argv(manifest, record),
                         timeout_s=manifest['config'].get('collection_s', 120) + 120,
                         out=str(Path(manifest['directory']) / 'collector.log'), wait=False)
    require(result.get('result') == 'started' and result.get('unit', '').removesuffix('.service') == manifest['unit'].removesuffix('.service'), 'collector launch failed')
    end = clock() + manifest['config'].get('ready_timeout_s', 30)
    path = Path(manifest['directory']) / 'ready.json'
    while not path.exists():
        require(inspect_owner(manifest, record) is not None, 'collector exited before readiness')
        require(clock() < end, 'collector readiness timed out')
        sleep(.1)
    receipt = read_receipt(path)
    require(receipt.get('manifest_sha256') == record['sha256'] and receipt.get('owner') == inspect_owner(manifest, record)
            and receipt['session']['name'] == manifest['session'], 'readiness identity mismatch')
    return receipt


def collection_active(record, state):
    """A later boot must occur after this exact callback, while capture is alive."""
    manifest = load(record, state)
    receipt = read_receipt(Path(manifest['directory']) / 'ready.json')
    owner = inspect_owner(manifest, record)
    require(owner is not None and owner == receipt.get('owner')
            and receipt.get('manifest_sha256') == record['sha256'], 'collector readiness no longer owned')
    rows = [r for r in sessions(manifest['config']) if r['name'] == manifest['session']]
    require(len(rows) == 1 and rows[0] == receipt['session']
            and 0 <= time.monotonic() - receipt['monotonic'] < manifest['config'].get('collection_s', 120), 'collection session no longer active')
    return receipt


@contextmanager
def cleanup_lock(manifest):
    """Concurrent controller/deadman recovery never races session/unit teardown."""
    path = Path(manifest['directory']) / 'cleanup.lock'
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        require(info.st_uid == os.getuid() and info.st_mode & 0o077 == 0, 'cleanup lock owner/mode changed')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Refused('another owned cleanup is in progress') from error
        yield
    finally:
        os.close(fd)


def cleanup(record, state, *, clock=time.monotonic, sleep=time.sleep):
    """Restartable on success/abort/deadman; shutdown cancels an active report."""
    try:
        manifest = load(record, state)
        with cleanup_lock(manifest):
            return _cleanup(record, manifest, clock=clock, sleep=sleep)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        return {'ok': False, 'deferred': True, 'error': str(error)[:200]}


def _cleanup(record, manifest, *, clock, sleep):
    try:
        config = manifest['config']
        owner = inspect_owner(manifest, record)
        rows = sessions(config)
        matches = [r for r in rows if r['name'] == manifest['session']]
        ready_path = Path(manifest['directory']) / 'ready.json'
        receipt = read_receipt(ready_path) if ready_path.exists() else None
        if receipt is not None:
            require(receipt.get('manifest_sha256') == record['sha256'] and receipt['session']['name'] == manifest['session'], 'callback provenance changed')
            require(not owner or owner == receipt['owner'], 'collector PID/start changed')
            require(not matches or matches[0]['id'] == receipt['session']['id'], 'session ID changed')
        if matches:
            require(owner is not None, 'owned session survived collector; ownership deferred')
            require(not any(r['id'] == matches[0]['id'] for r in manifest['baseline']['sessions']), 'session inherited/foreign')
        inherited = {(p['pid'], p['start']) for p in manifest['baseline']['tool_processes']}
        require(all((p['pid'], p['start']) in inherited or p['unit'] == manifest['unit'] for p in tool_processes(config)), 'new profiler helper escaped ownership')
        if matches:
            command(config, ['shutdown', '--session=' + matches[0]['id'], '--kill=none'])
        end = clock() + config.get('cleanup_timeout_s', 30)
        while any(r['name'] == manifest['session'] for r in sessions(config)):
            require(clock() < end, 'owned session shutdown timed out')
            sleep(.1)
        if owner:
            current = inspect_owner(manifest, record)
            require(current is None or current == owner, 'collector changed during cleanup')
            if current:
                require(unitrun.stop(manifest['unit']), 'owned collector unit stop failed')
        require(inspect_owner(manifest, record) is None, 'collector remained active')
        require(all((p['pid'], p['start']) in inherited for p in tool_processes(config)), 'profiler helper survived cleanup')
        result = {'ok': True, 'session': manifest['session'], 'unit': manifest['unit'],
                  'manifest_sha256': record['sha256'], 'cleaned': time.time(),
                  'report_scope': 'shutdown may cancel an active collection; no report-completeness claim'}
        atomic(Path(manifest['directory']) / 'cleanup.json', result)
        return result
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        return {'ok': False, 'deferred': True, 'error': str(error)[:200]}


def restored(original, identity, config):
    try:
        check_files(config)
        current = engine_state(identity, config)
        require(current['env_sha256'] == original['engine']['env_sha256']
                and current['libraries'] == original['engine']['libraries'], 'restored engine instrumentation differs')
        inherited = {(p['pid'], p['start']) for p in original['tool_processes']}
        require(all((p['pid'], p['start']) in inherited for p in tool_processes(config)), 'unexpected profiler process after restoration')
        require({(r['id'], r['name']) for r in sessions(config)}
                == {(r['id'], r['name']) for r in original['sessions']}, 'session inventory differs after restoration')
        return {'ok': True, 'engine': current}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        return {'ok': False, 'deferred': True, 'error': str(error)[:200]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['ready'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--sha', required=True)
    args = parser.parse_args()
    ready({'manifest': args.manifest, 'sha256': args.sha})
