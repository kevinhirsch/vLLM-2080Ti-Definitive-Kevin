#!/usr/bin/env python3
"""Extend THIS window's own gateway-offline lease (the shim caps one TTL at 3600 s but supports `extend` with the lease id) every 20 min
until killed. Only extends the lease recorded by our own `gateway-offline.py run` (by == DFT)."""
import importlib.util, json, os, sys, time
spec = importlib.util.spec_from_file_location("go", "/home/kevin/Desktop/vLLM-2080Ti-Definitive/deploy/bin/gateway-offline.py")
go = importlib.util.module_from_spec(spec); spec.loader.exec_module(go)
hard_end = time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 14400)
while time.time() < hard_end:
    time.sleep(1200)
    cur = go.load_lease()
    if not cur or cur.get("by") != "DFT": sys.exit(0)
    r = go.http("/gateway/offline", "POST", {"ttl_s": 3000, "lease": cur["lease"]})
    print(time.strftime("%T"), "extended", json.dumps(r)[:120], flush=True)
