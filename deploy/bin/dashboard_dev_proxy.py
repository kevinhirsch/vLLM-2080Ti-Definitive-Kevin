#!/usr/bin/env python3
"""Preview gateway_dashboard.html against a REAL gateway without publishing anything.

  python3 dashboard_dev_proxy.py [--port 8099] [--upstream http://127.0.0.1:8000] [--file gateway_dashboard.html] [--mock-capacity-model]

Serves the dashboard file at /gateway/dashboard (re-read on every request, so edit-and-reload works) and proxies every
other request to the upstream gateway, so the page is same-origin and needs no CORS. GET only: this tool refuses POST so a
preview can never change gateway settings. --mock-capacity-model injects a capacity_model block (shape published by the
live-capacity-model commit) into /gateway/capacity and /gateway/stats, for previewing a gateway that predates it.
"""
import argparse
import http.server
import json
import socketserver
import urllib.error
import urllib.request
from pathlib import Path


def mock_capacity_model(stats):
    pool = 922358
    return {"enabled": True, "kill_switch": "SHIM_CAPACITY_LIVE=0",
            "principle": "capacity numbers come from the running engine; configured values are fallbacks or explicit overrides",
            "kv_pool_tokens": {"effective": pool, "source": "live", "live": pool, "configured": 637560, "configured_stale": True,
                               "read_from": "vllm:cache_config_info kv_cache_size_tokens", "age_s": 1.8, "engine_reachable": True},
            "token_budget": {"effective": 500000, "source": "override", "detail": "explicit SHIM_TOKEN_BUDGET=500000 (set it to 'auto' to follow the engine)",
                             "derived_from_live": 611000, "fraction_of_pool": 0.6629, "ceiling": 680000, "ceiling_applied": False,
                             "override": 500000, "calibration": "500000 reserved tokens proven against a 754068-token pool", "halo_control_reserve": 62500},
            "prefill_tok_s": {"effective": 1270.0, "source": "measured", "detail": "p75 of per-minute pure per-request rates, 14 valid minutes",
                              "configured": 1100.0, "measured_p75": 1270.0, "measured_range": [980.0, 1410.0], "valid_minutes": 14, "min_minutes": 5,
                              "status": "ok", "window_start": 0},
            "prefix_align_tokens": {"effective": 3568, "source": "live", "configured": 3568, "read_from": "vllm:cache_config_info block_size"},
            "engine_generation": {"start": 0, "source": "process_start_time_seconds", "age_s": 9800.0, "settled": True,
                                  "recent_changes": [{"at": 0, "event": "engine restarted (mock)"}]},
            "warnings": ["SHIM_POOL_TOKENS=637560 is stale (the engine reports 922358); the configured value is only a fallback",
                         "SHIM_TOKEN_BUDGET=500000 pins the budget -18% away from the live-derived 611000; set SHIM_TOKEN_BUDGET=auto to follow the engine"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--file", default=str(Path(__file__).with_name("gateway_dashboard.html")))
    ap.add_argument("--mock-capacity-model", action="store_true")
    args = ap.parse_args()

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/gateway/dashboard", "/"):
                return self._send(200, Path(args.file).read_bytes(), "text/html; charset=utf-8")
            try:
                with urllib.request.urlopen(args.upstream + self.path, timeout=15) as r:
                    body, ctype = r.read(), r.headers.get("Content-Type", "application/json")
            except urllib.error.HTTPError as e:
                return self._send(e.code, e.read(), e.headers.get("Content-Type", "text/plain"))
            except Exception as e:  # noqa: BLE001
                return self._send(502, str(e).encode(), "text/plain")
            if args.mock_capacity_model and path in ("/gateway/capacity", "/gateway/stats"):
                d = json.loads(body)
                if "capacity_model" not in d:
                    d["capacity_model"] = mock_capacity_model(d)
                    if path == "/gateway/capacity":
                        d.setdefault("throughput", {})["prefill_effective_tok_s"] = 1270.0
                        d["throughput"]["prefill_effective_basis"] = "measured"
                    body = json.dumps(d).encode()
            self._send(200, body, ctype)

        def do_POST(self):
            self._send(405, b"preview proxy is read-only", "text/plain")

        do_PUT = do_DELETE = do_POST

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), H) as srv:
        print("dashboard preview: http://127.0.0.1:%d/gateway/dashboard  ->  %s" % (args.port, args.upstream))
        srv.serve_forever()


if __name__ == "__main__":
    main()
