#!/usr/bin/env python3
"""L100: 1 h ABAB A/B of SHIM_WARM_PRIORITY through the live (hot-reloadable) gateway config. Gateway-only, no restart.

  run      alternate OFF/ON/OFF/ON (--arm-min, default 15) via POST /gateway/config {"warm_priority": 0|1}; records arm
           windows + engine preemption counter; ABORTS to OFF at once if vllm:num_preemptions_total increases in an ON
           arm, if a planned-offline window starts, or on any exit (finally: OFF).
  analyze  per arm from the request log: remote share of warm turns (chain_prev_route=local & pm_credit >= 4096),
           Halo (halo-hermes) TTFT p50/p95, cold local TTFT p95 (pm_credit < 4096), warm-priority marks, preemptions.
  Keep ON only if the warm remote share falls, Halo TTFT p50/p95 regress <= 10 %, cold p95 regresses <= 15 %, preemptions 0.
"""
import argparse, glob, json, os, sys, time, urllib.request

GW = "http://127.0.0.1:8000"
TOKEN_FILE = os.path.expanduser("~/.local/share/vllm-qwen27b/admin.token")
TDIR = os.path.expanduser("~/.local/share/vllm-qwen27b/telemetry")


def http(path, body=None, base=GW, timeout=10):
    h = {"Content-Type": "application/json"}
    if body is not None:
        h["X-Admin-Token"] = open(TOKEN_FILE).read().strip()
    req = urllib.request.Request(base + path, None if body is None else json.dumps(body).encode(), h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode()


def preemptions():
    txt = urllib.request.urlopen("http://127.0.0.1:8001/metrics", timeout=10).read().decode()
    return sum(float(l.rsplit(" ", 1)[1]) for l in txt.splitlines() if l.startswith("vllm:num_preemptions_total"))


def set_arm(on):
    http("/gateway/config", {"warm_priority": 1 if on else 0})
    cur = http("/gateway/config")
    assert int(cur.get("warm_priority")) == (1 if on else 0), cur.get("warm_priority")


def run(a):
    log = open(a.out, "a")
    note = lambda **k: (log.write(json.dumps(dict(t=round(time.time(), 1), **k)) + "\n"), log.flush(), print(k, flush=True))
    try:
        for i, on in enumerate([False, True, False, True][: a.arms]):
            if http("/gateway/capacity").get("planned_offline"):
                note(event="abort", why="planned-offline window started"); break
            set_arm(on); p0 = preemptions(); t0 = time.time()
            note(event="arm_start", arm=i, on=on, preemptions=p0)
            while time.time() - t0 < a.arm_min * 60:
                time.sleep(15)
                p = preemptions()
                if on and p > p0:
                    note(event="abort", why="engine preemption during ON arm", delta=p - p0); set_arm(False); return 2
                if http("/gateway/capacity").get("planned_offline"):
                    note(event="abort", why="planned-offline window started"); return 3
            note(event="arm_end", arm=i, on=on, preemptions=preemptions())
    finally:
        try:
            set_arm(False); note(event="restored", warm_priority=0)
        except Exception as e:
            note(event="RESTORE FAILED", err=str(e)); raise
    return 0


def q(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))], 2) if xs else None


def analyze(a):
    ev = [json.loads(l) for l in open(a.out)]
    arms, cur = [], None
    for e in ev:
        if e.get("event") == "arm_start":
            cur = dict(arm=e["arm"], on=e["on"], t0=e["t"], p0=e["preemptions"])
        elif e.get("event") in ("arm_end", "abort") and cur:
            cur.update(t1=e["t"], p1=e.get("preemptions", cur["p0"])); arms.append(cur); cur = None
    rows = []
    for f in sorted(glob.glob(os.path.join(TDIR, "requests-*.jsonl")))[-2:]:
        for l in open(f, errors="replace"):
            try:
                rows.append(json.loads(l))
            except Exception:
                pass
    out = []
    for arm in arms:
        rs = [r for r in rows if arm["t0"] <= r.get("t", 0) < arm["t1"]]
        warm = [r for r in rs if r.get("chain_prev_route") == "local" and (r.get("pm_credit") or 0) >= 4096
                and r.get("route") in ("local", "remote")]
        halo = [r["ttft"] for r in rs if r.get("client") == "halo-hermes" and r.get("route") == "local" and r.get("ttft") is not None]
        cold = [r["ttft"] for r in rs if r.get("route") == "local" and r.get("ttft") is not None and (r.get("pm_credit") or 0) < 4096]
        out.append(dict(arm=arm["arm"], on=arm["on"], minutes=round((arm["t1"] - arm["t0"]) / 60, 1), requests=len(rs),
                        warm_turns=len(warm), warm_remote_share=round(sum(r["route"] == "remote" for r in warm) / len(warm), 3) if warm else None,
                        warm_priority_marked=sum(1 for r in rs if r.get("warm_priority")),
                        halo_ttft_p50=q(halo, .5), halo_ttft_p95=q(halo, .95), halo_n=len(halo),
                        cold_ttft_p95=q(cold, .95), cold_n=len(cold), preemptions=arm["p1"] - arm["p0"]))
    for o in out:
        print(json.dumps(o))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "analyze"))
    ap.add_argument("--out", default="/home/kevin/projects/lanes/gw2/l100/ab.jsonl")
    ap.add_argument("--arm-min", type=float, default=15)
    ap.add_argument("--arms", type=int, default=4)
    a = ap.parse_args()
    sys.exit(run(a) if a.cmd == "run" else (analyze(a) and 0))
