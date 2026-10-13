#!/usr/bin/env python3
"""Registers / stack spill of every dense Marlin kernel instantiation in a _C_stable_libtorch .so (CPU only).

Usage: resource_usage.py [<so>] [--btype uint4] [--group-blocks 8] [--json out.json]
Prints one line per instantiation: threads, thread_m_blocks, thread_n_blocks x thread_k_blocks (tile = 16*n x 16*k),
m_block_size_8, group_blocks, REG, STACK (bytes of local spill).
"""
import argparse
import json
import re
import subprocess
import sys

FUNC_RE = re.compile(r"Function (\S+):\s*\n\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)")
# _ZN6marlin6MarlinILl<a>ELl<b>ELl<c>ELl<s>ELi<threads>ELi<mb>ELi<nb>ELi<kb>ELb<m8>ELi<stages>ELi<gb>ELb<zpf>EEEv...
MANGLED = re.compile(r"_ZN6marlin6MarlinILl(\d+)ELl(\d+)ELl(\d+)ELl(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELb([01])"
                     r"ELi(\d+)ELin?(\d+)ELb([01])E")


def type_names():
    try:
        from vllm.scalar_type import scalar_types
        return {getattr(scalar_types, n).id: n for n in dir(scalar_types)
                if not n.startswith("_") and hasattr(getattr(scalar_types, n), "id")}
    except Exception:  # noqa: BLE001
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("so", nargs="?", default="vllm/_C_stable_libtorch.abi3.so")
    ap.add_argument("--btype", default=None, help="filter b type name, e.g. uint4 / uint4b8")
    ap.add_argument("--group-blocks", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    txt = subprocess.run(["/usr/local/cuda-13/bin/cuobjdump", "--dump-resource-usage", a.so],
                         capture_output=True, text=True, check=True).stdout
    names = type_names()
    rows = []
    for f in FUNC_RE.finditer(txt):
        m = MANGLED.match(f.group(1))
        if not m:
            continue
        g = m.groups()
        gb = -int(g[10]) if "ELin" + g[10] in f.group(1) else int(g[10])
        row = {"a": names.get(int(g[0]), g[0]), "b": names.get(int(g[1]), g[1]), "threads": int(g[4]),
               "mb": int(g[5]), "nb": int(g[6]), "kb": int(g[7]), "m8": g[8] == "1", "stages": int(g[9]),
               "group_blocks": gb, "zp_float": g[11] == "1", "reg": int(f.group(2)), "stack": int(f.group(3))}
        if a.btype and row["b"] != a.btype:
            continue
        if a.group_blocks is not None and row["group_blocks"] != a.group_blocks:
            continue
        rows.append(row)
    rows.sort(key=lambda r: (r["a"], r["b"], r["group_blocks"], r["mb"], r["threads"], r["nb"], r["m8"]))
    for r in rows:
        print(f"{r['a']:8s} {r['b']:8s} g{r['group_blocks']:<3d} mb{r['mb']} m8={int(r['m8'])} th{r['threads']:<3d} "
              f"tile n{16 * r['nb']:<3d} k{16 * r['kb']:<3d} REG {r['reg']:3d} STACK {r['stack']:3d}"
              f"{'  <-- spills' if r['stack'] else ''}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
