#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Source-only narrow patch of the pinned installed gateway; no deployment.

Keeps all unrelated deployed gateway bytes, rather than downgrading them to
historical repository source. Based on vLLM 2080 Ti Definitive Edition/weicj.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import stat

BASE_SHA = '84efdf5beb10df20a941f911901885e89a4c77f46f923148588ad705def97b34'


def patch_source(data):
    if hashlib.sha256(data).hexdigest() != BASE_SHA:
        raise ValueError('installed gateway source differs from pinned base; re-audit instead of patching')
    text = data.decode()
    old = 'return {"offline": on, "until": _OFFLINE["until"] if on else None, "reason": _OFFLINE["reason"] if on else None,'
    new = 'return {"offline_lease_renewal": "extend-only-v1", "offline": on, "until": _OFFLINE["until"] if on else None, "reason": _OFFLINE["reason"] if on else None,'
    if text.count(old) != 1:
        raise ValueError('offline status anchor changed')
    text = text.replace(old, new, 1)
    old = '    if _draining() and not (body.get("lease") and body.get("lease") == _OFFLINE["lease"]):\n'
    new = '''    # A supplied lease is always an extension, never permission to open a
    # replacement after expiry/DELETE. The async handler has no await between
    # this server-side precondition and the mutation, so client GET races do
    # not reopen admission fencing with a fresh, unrecorded authority.
    if "lease" in body:
        if (not isinstance(body["lease"], str) or not _local_offline(now)
                or body["lease"] != _OFFLINE["lease"]):
            return web.json_response({"error": "offline lease expired or mismatched",
                                      "code": "offline-lease-expired-or-mismatch",
                                      **_offline_status()}, status=409)
        _OFFLINE["until"] = now + ttl
        _offline_persist()
        return web.json_response({**_offline_status(), "lease": _OFFLINE["lease"]})
    if _draining() and not (body.get("lease") and body.get("lease") == _OFFLINE["lease"]):
'''
    if text.count(old) != 1:
        raise ValueError('offline extension anchor changed')
    text = text.replace(old, new, 1)
    ast.parse(text)
    return text.encode()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    source, target = Path(args.source), Path(args.out)
    if source.resolve() == target.resolve() or target.exists():
        raise ValueError('a fresh separate output is required; never overwrite installed source')
    payload = patch_source(source.read_bytes())
    with target.open('xb') as stream:
        stream.write(payload)
    target.chmod(stat.S_IMODE(source.stat().st_mode))


if __name__ == '__main__':
    main()
