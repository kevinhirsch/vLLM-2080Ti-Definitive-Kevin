# SPDX-License-Identifier: Apache-2.0
"""Exact existing gateway lease recovery for optional owned-profiler windows.

Uses the existing planned-offline API/authority, never opens a new lease. Based
on the vLLM 2080 Ti Definitive Edition control framework (github.com/weicj).
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time


PROTOCOL = 'extend-only-v1'
HTTP_TIMEOUT_S = 8
JOIN_TIMEOUT_S = HTTP_TIMEOUT_S + 2


class Refused(RuntimeError):
    pass


class OwnedElsewhere(Refused):
    """No state/unit/HTTP mutation is permitted from this stale recovery call."""


def require(ok, message):
    if not ok:
        raise Refused(message)


def require_supported(row):
    require(isinstance(row, dict) and row.get('offline_lease_renewal') == PROTOCOL,
            'gateway does not advertise exact extend-only lease renewal')


def call(http, *args):
    try:
        row = http(*args)
    except Exception as error:
        raise Refused('gateway transport failed: ' + type(error).__name__) from error
    require(isinstance(row, dict), 'gateway response schema changed')
    return row


def owner_status(pid, start_ticks, proc_root):
    """Bind one process-stat read; a zombie/dead task cannot run its finally.

    Missing/reused PIDs are gone. Unreadable or malformed authority is unknown,
    so the caller must defer without signalling or beginning restoration.
    """
    require(type(pid) is int and pid > 0 and isinstance(start_ticks, str)
            and start_ticks.isdigit(), 'controller process authority malformed')
    try:
        text = (Path(proc_root) / str(pid) / 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return 'gone'
    except OSError as error:
        raise Refused('controller process authority unreadable') from error
    try:
        tail = text.rsplit(')', 1)[1].split()
        require(len(tail) >= 20 and tail[19].isdigit() and len(tail[0]) == 1,
                'controller process stat malformed')
    except (IndexError, ValueError) as error:
        raise Refused('controller process stat malformed') from error
    if tail[19] != start_ticks:
        return 'gone'
    return 'dead-retained' if tail[0] in ('Z', 'X', 'x') else 'live'


def extend(state, spec, http, gateway):
    """The server precondition is mandatory; a client-side GET cannot replace it."""
    lease = state.get('lease')
    require(isinstance(lease, str) and re.fullmatch('[0-9a-f]{32}', lease), 'owned offline lease missing')
    status = call(http, gateway + '/gateway/offline')
    require_supported(status)
    require(status.get('offline') is True, 'owned offline fence absent')
    ttl = int(spec.get('ttl_s', 1800))
    require(30 <= ttl <= 3600, 'invalid offline lease TTL')
    row = call(http, gateway + '/gateway/offline', 'POST', {
        'lease': lease, 'ttl_s': ttl, 'by': state.get('by') or state['lane'],
        'reason': (str(state['window']) + ': ' + str(spec.get('reason', 'owned recovery')))[:120]})
    require_supported(row)
    require(not row.get('error') and row.get('lease') == lease and row.get('offline') is True,
            'exact offline lease extension refused or changed ownership')
    require(type(row.get('remaining_s')) is int and row['remaining_s'] >= ttl - 2,
            'offline extension lifetime unqualified')
    return row


class RestorationLease:
    """Cross-process restore owner with an exact-lease keeper and joined close.

    Failure never closes or replaces a lease. The standard dead-man can retry
    after this owner releases the restore lock; its first extension refreshes
    the same authority. No field can bypass a gateway protocol refusal.
    """
    def __init__(self, state, spec, http, gateway, *, cadence_s=None, join_s=JOIN_TIMEOUT_S, on_defer=None):
        self.state, self.spec, self.http, self.gateway = state, spec, http, gateway
        self.cadence_s = min(60, max(1, int(spec.get('ttl_s', 1800)) / 3)) if cadence_s is None else cadence_s
        self.join_s = join_s
        self.lease = state.get('lease')
        self.directory = Path(state['results'])
        self.stop = threading.Event()
        self.operation_lock = threading.RLock()
        self.thread = None
        self.lockfd = None
        self.failure = None
        self.closed = False
        self.quiesced = False
        self.renewals = 0
        self.last_renewed = None
        self.audit = None
        self.authority_verified = False
        self.on_defer = on_defer
        self.repaused = {}

    def receipt(self, phase):
        row = {'protocol': PROTOCOL, 'phase': phase, 'owner_pid': os.getpid(),
               'lease_sha256': hashlib.sha256(str(self.lease).encode()).hexdigest(),
               'renewals': self.renewals, 'last_renewed': self.last_renewed,
               'failed': self.failure, 'quiesced': self.quiesced, 'closed': self.closed,
               'scope': 'lease ownership/lifecycle only; not profiler or engine qualification'}
        path = self.directory / 'offline-lease-guard.json'
        tmp = self.directory / ('offline-lease-guard.tmp-' + str(os.getpid()))
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(row, stream, sort_keys=True, indent=1)
        os.replace(tmp, path)
        self.audit = row
        return row

    def defer(self):
        self.state['restored'] = False
        if self.authority_verified and self.on_defer is not None:
            try:
                self.repaused = self.on_defer()
            except Exception as error:
                self.repaused = {'unqualified_error_type': type(error).__name__}

    def publish_state(self):
        """Call only while this owner holds the recovery lock."""
        require(self.lockfd is not None, 'state publication has no recovery owner')
        path = self.directory / ('offline-state.tmp-' + str(os.getpid()))
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(self.state, stream, indent=1)
        os.replace(path, self.directory / 'state.json')

    def renew(self):
        with self.operation_lock:
            require(not self.closed and self.state.get('lease') == self.lease, 'offline authority changed during restore')
            require(self.failure is None, self.failure or 'lease keeper failed')
            try:
                row = extend(self.state, self.spec, self.http, self.gateway)
            except Exception as error:
                # Do not copy arbitrary response/error bodies or literal leases.
                self.failure = 'exact lease renewal failed: ' + type(error).__name__
                self.receipt('deferred')
                raise Refused(self.failure) from error
            self.renewals += 1
            self.last_renewed = time.time()
            self.receipt('restoring')
            return row

    def loop(self):
        while not self.stop.wait(self.cadence_s):
            try:
                self.renew()
            except Exception:
                return

    def __enter__(self):
        try:
            require(self.directory.is_absolute() and self.directory.resolve() == self.directory
                    and self.directory.stat().st_uid == os.getuid(), 'restore results ownership changed')
            fd = os.open(self.directory / 'offline-restore.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        except Exception as error:
            raise OwnedElsewhere('window recovery path/ownership unqualified') from error
        try:
            info = os.fstat(fd)
            require(info.st_uid == os.getuid() and info.st_mode & 0o077 == 0, 'restore lock owner/mode changed')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise OwnedElsewhere('another process owns instrumented recovery') from error
            self.lockfd = fd
            path = self.directory / 'state.json'
            require(path.resolve() == path and path.stat().st_uid == os.getuid(), 'window state authority missing/changed')
            current = json.loads(path.read_text())
            if current.get('restored') or any(current.get(key) != self.state.get(key)
                    for key in ('window', 'lane', 'lease', 'pid', 'pid_start')):
                raise OwnedElsewhere('window recovery authority changed; caller is stale')
            self.authority_verified = True
            self.renew()  # refresh before any cleanup/restore action
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()
            return self
        except Exception as error:
            try:
                if not isinstance(error, OwnedElsewhere):
                    self.defer()
                    if self.authority_verified:
                        self.publish_state()
            finally:
                os.close(fd)
                self.lockfd = None
            if not self.authority_verified and not isinstance(error, OwnedElsewhere):
                raise OwnedElsewhere('window state/lock authority unreadable or unqualified') from error
            raise

    def check(self):
        require(self.failure is None and not self.closed and self.state.get('lease') == self.lease,
                self.failure or 'offline restoration authority changed')

    def join(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(self.join_s)
            require(not self.thread.is_alive(), 'offline keeper request has not quiesced')
        self.quiesced = True

    def close(self):
        """Join first; final acknowledged extension; exact DELETE; no more POST."""
        self.join()
        with self.operation_lock:
            self.check()
            self.renew()
            row = call(self.http, self.gateway + '/gateway/offline', 'DELETE', {'lease': self.lease})
            require(not row.get('error') and row.get('offline') is False,
                    'exact offline close unqualified; 409 is not uninterrupted fencing proof')
            self.closed = True
            self.state['lease'] = None
            after = call(self.http, self.gateway + '/gateway/offline')
            require_supported(after)
            require(after.get('offline') is False, 'another/offline fence present after exact close')
            self.receipt('closed')
            return row

    def __exit__(self, kind, error, traceback):
        try:
            if kind is not None:
                self.defer()
            self.join()
            # A deferred original helper keeps genuine recovery evidence. Give
            # the next 120s dead-man tick a newly acknowledged TTL when possible.
            if not self.closed and self.failure is None:
                self.renew()
            self.receipt('closed' if self.closed else 'deferred')
        except Exception:
            self.defer()
            raise
        finally:
            if self.lockfd is not None:
                # Publish the authoritative state before releasing ownership,
                # so another dead-man cannot replay a stale pre-close lease.
                try:
                    self.publish_state()
                finally:
                    os.close(self.lockfd)
                    self.lockfd = None
        return False
