#!/usr/bin/env python3
"""bench_guard.py -- refuse to send inference to the ENGINE (:8001) unless a gateway offline window is open.

FX2 (2026-10-03). A bench aimed straight at :8001 bypasses the gateway's routing, spend ledger and drain; the 23:18 planned
restart cut Lane DFT's live_ab.py mid-request. Standalone and stdlib-only so it can be copied next to any lane script:

    python3 bench_guard.py [--gateway URL]      # shell:  python3 bench_guard.py || exit 3
    import bench_guard; bench_guard.require()   # python: SystemExit(3) with a clear message when refused

Allowed when ANY of: BENCH_DIRECT_OK=1 (explicit, noted on stderr); a window is open on the gateway (GET /gateway/offline
says offline=true); the gateway is unreachable (nothing to protect). Otherwise exit 3 with the exact command to use:
    gateway-offline.py run --reason R --by WHO -- <your command>
"""
import json
import os
import sys
import urllib.request

GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8000")
WRAPPER = "/home/kevin/Desktop/vLLM-2080Ti-Definitive/deploy/bin/gateway-offline.py"


def window_state(gateway=None, timeout=3):
    """(True, who) window open | (False, None) closed | (None, why) gateway unreadable."""
    try:
        d = json.loads(urllib.request.urlopen(f"{gateway or GATEWAY}/gateway/offline", timeout=timeout).read())
    except Exception as e:  # noqa: BLE001
        return None, repr(e)[:80]
    return (True, d.get("by")) if d.get("offline") else (False, None)


def check(gateway=None, env=None):
    """Returns (ok, message). Pure apart from the one GET."""
    env = os.environ if env is None else env
    if env.get("BENCH_DIRECT_OK"):
        return True, "bench_guard: BENCH_DIRECT_OK set -- sending inference straight to the engine without a gateway window"
    state, info = window_state(gateway)
    if state is True:
        return True, f"bench_guard: gateway offline window open (by {info}) -- direct engine bench allowed"
    if state is None:
        return True, f"bench_guard: gateway unreachable ({info}) -- nothing to protect, allowed"
    return False, ("bench_guard: REFUSED. This sends inference straight to the engine (:8001), bypassing the gateway "
                   "(routing, spend ledger, drain), and no gateway offline window is open, so estate traffic shares the "
                   "engine and a planned restart can cut you mid-request.\n"
                   f"  run it inside a window:  {WRAPPER} run --reason \"<why>\" --by <LANE> -- <your command>\n"
                   "  or, if you know the engine is yours alone, set BENCH_DIRECT_OK=1 for this run.")


def require(gateway=None, quiet=False):
    ok, msg = check(gateway)
    if not ok:
        print(msg, file=sys.stderr)
        raise SystemExit(3)
    if not quiet:
        print(msg, file=sys.stderr)


if __name__ == "__main__":
    ap = __import__("argparse").ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gateway"); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    require(a.gateway, a.quiet)
