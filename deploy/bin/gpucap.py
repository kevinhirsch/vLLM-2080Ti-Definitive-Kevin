#!/usr/bin/env python3
"""gpucap.py -- launch a lane GPU process with an ENFORCED memory cap (lane RL, 2026-10-03).

Why: gpuok.sh only checks at launch. At 10:16 K7's run passed it with 1000 MiB free, then grew to ~846 MiB and left
GPU1 with ~40 MiB beside production. A cap must hold for the whole run:
  1. in-process: GPU_CAP_MIB + gpucap_site/ first on PYTHONPATH -> its sitecustomize caps torch's caching allocator at
     the first CUDA init (torch.cuda.set_per_process_memory_fraction); past the cap the PROCESS gets the OOM;
  2. out-of-process: the job runs in its own systemd unit (unitrun.py) and a watcher sums nvidia-smi used_memory of
     every pid in that unit's cgroup every poll; above cap + margin the UNIT is stopped (then SIGKILLed). This also
     covers non-torch CUDA programs (raw ./bench binaries, Triton, CuPy).

  gpucap.py run --lane K7 --job kbench --gpu 1 --cap 600 [--margin 300] [--out F] [--timeout S] [--no-gate] -- CMD...
    exit = the command's rc; 137 + a 'gpucap: KILLED' line when the cap was enforced; 3 when the gate refused.
  gpuok.sh --run ... is the same thing behind the shared gate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpuguard  # noqa: E402
import unitrun  # noqa: E402

SITE = os.path.join(HERE, "gpucap_site")
GPUOK = os.environ.get("GPUOK_SH", os.path.expanduser("~/projects/lanes/windows/gpuok.sh"))


def unit_gpu_mib(cgroup: str | None, apps: list[dict], gpu=None) -> tuple[int, list[dict]]:
    """MiB the unit's processes hold (on `gpu` when given), from nvidia-smi compute apps."""
    if not cgroup:
        return 0, []
    pids = unitrun.cgroup_pids(cgroup)
    mine = [a for a in apps if a["pid"] in pids and (gpu is None or a.get("gpu") == gpu)]
    return sum(int(a.get("used_mib") or 0) for a in mine), mine


class Watcher:
    def __init__(self, unit: str, cap_mib: int, margin_mib: int, gpu=None, poll_s: float = 2.0, log=None):
        self.unit, self.limit, self.gpu, self.poll_s = unit, cap_mib + margin_mib, gpu, poll_s
        self.cap = cap_mib
        self.peak = 0
        self.killed = None
        self.stop = threading.Event()
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))

    def tick(self):
        cg = unitrun.show(self.unit, "ControlGroup").get("ControlGroup")
        used, rows = unit_gpu_mib(cg, gpuguard.compute_apps(), self.gpu)
        self.peak = max(self.peak, used)
        if used > self.limit and not self.killed:
            self.killed = {"used_mib": used, "limit_mib": self.limit, "cap_mib": self.cap, "pids": [r["pid"] for r in rows],
                           "ts": round(time.time(), 1)}
            self.log(f"gpucap: KILLED {self.unit}: {used} MiB on gpu{self.gpu} > cap {self.cap} + margin = {self.limit} MiB")
            unitrun.stop(self.unit)
            time.sleep(1)
            if unitrun.is_active(self.unit):
                unitrun.kill(self.unit, "KILL")
        return used

    def loop(self):
        while not self.stop.wait(self.poll_s):
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001
                self.log(f"gpucap: watcher error {e!r}")


def env_for(cap_mib: int, gpu, env=None) -> dict:
    out = dict(env or {})
    pp = os.environ.get("PYTHONPATH", "")
    out["PYTHONPATH"] = SITE + (os.pathsep + pp if pp else "")
    out["GPU_CAP_MIB"] = str(int(cap_mib))
    if gpu is not None and "CUDA_VISIBLE_DEVICES" not in out:
        out["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k in ("WINDOW_ID", "K2_IN_WINDOW", "GPU_IN_WINDOW", "CUDA_DEVICE_ORDER", "LD_LIBRARY_PATH", "VIRTUAL_ENV"):
        if k in os.environ and k not in out:
            out[k] = os.environ[k]
    out.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    return out


def run(lane, job, cmd, *, gpu, cap_mib, margin_mib=300, out=None, timeout_s=None, cwd=None, gate=True, poll_s=2.0,
        env=None):
    if gate:
        r = subprocess.run(["bash", GPUOK, str(gpu), str(int(cap_mib) + int(margin_mib))], capture_output=True, text=True)
        if r.returncode != 0:
            return {"rc": 3, "result": "gate-refused", "why": (r.stdout or r.stderr).strip()[-300:]}
    unit = unitrun.unit_name(lane, job)
    w = Watcher(unit, int(cap_mib), int(margin_mib), gpu, poll_s)
    th = threading.Thread(target=w.loop, daemon=True)
    th.start()
    try:
        res = unitrun.run(lane, job, cmd, timeout_s=timeout_s, env=env_for(cap_mib, gpu, env), cwd=cwd or os.getcwd(),
                          out=out, meta={"gpu_cap_mib": int(cap_mib), "gpu": gpu})
    finally:
        w.stop.set()
        th.join(5)
    res["gpu_peak_mib"] = w.peak
    if w.killed:
        res.update(result="gpu-cap-killed", gpu_cap=w.killed, rc=137)
    return res


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    tail = []
    if "--" in argv:
        i = argv.index("--")
        argv, tail = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run")
    p.add_argument("--lane", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--cap", type=int, required=True, help="MiB the process may allocate")
    p.add_argument("--margin", type=int, default=300, help="CUDA context + slack before the watcher kills (MiB)")
    p.add_argument("--out")
    p.add_argument("--timeout", type=int)
    p.add_argument("--no-gate", action="store_true")
    a = ap.parse_args(argv)
    if not tail:
        print(json.dumps({"error": "run needs a command after --"}))
        return 2
    res = run(a.lane, a.job, tail, gpu=a.gpu, cap_mib=a.cap, margin_mib=a.margin,
              out=os.path.abspath(a.out) if a.out else None, timeout_s=a.timeout, gate=not a.no_gate)
    print(json.dumps({k: v for k, v in res.items() if k != "pids"}), file=sys.stderr)
    return int(res["rc"]) if isinstance(res.get("rc"), int) else 1


if __name__ == "__main__":
    sys.exit(main())
