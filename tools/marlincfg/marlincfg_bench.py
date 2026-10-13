#!/usr/bin/env python3
"""LP-MARLINCFG microbench + table builder (one GPU, engine stopped). Single script, JSON out.

What it answers: for every production Marlin GEMM shape (TP=2 rank of Qwen3.8-27B, uint4 + zero points, group 128) and
prefill-sized M (17..4096), which (thread config, max thread_m_blocks, blocks/SM, max m-slices per launch) is fastest,
and is it numerically as good as the upstream choice? Upstream picks 64x256/256 threads at thread_m_blocks 4 for every
M > 16; on SM75 that variant is REG 255 + 32-40 B stack spill (cuobjdump), thread_m_blocks 3 variants do not spill, and
128-thread variants can run 2 blocks/SM.

How: the release's own _C_stable_libtorch (commit "marlin: VLLM_MARLIN_CFG") re-reads VLLM_MARLIN_CFG on every call, so
"prod" = env unset (upstream dispatch, unchanged code path) and each candidate = a one-rule table for exactly this
(n, k, M). VLLM_MARLIN_CFG_DEBUG=1 makes the op print every launch decision; the bench records that dispatch signature
(what really ran, incl. silent fallbacks) and times each distinct signature once per round. Timing: CUDA graph of R
back-to-back GEMMs (~6 ms per replay), 5 replays, median; 2 rounds in opposite order; min of the two round medians.
Numerics: relative Frobenius error vs an fp32 reference x @ W_deq (W_deq = Marlin applied to the identity, exact per
element) must be <= 1.10 x prod's + 1e-5; a candidate whose signature equals prod's must be bit-identical to prod.

Table: per shape, consecutive M points whose best passing candidate beats prod by >= --min-gain are merged into
"N,K,MLO,MHI:TK,TN,TH,MB,BPS,PAR" rules (ranges split at midpoints between measured M). Then the whole table is set and
every measured point is re-dispatched + re-timed against prod (verify phase): the table must hit the intended
signature everywhere, stay numerically within the gate, and no point may be slower than --max-loss.

Output JSON: verdict PASS (table non-empty, verified, positive weighted prefill gain) | NOWIN (no rule survives) | FAIL;
"table" (the VLLM_MARLIN_CFG value for the engine), per-point rows, weighted estimates per M.

Usage (window, engine stopped; release venv + PYTHONPATH=<release>):
  CUDA_VISIBLE_DEVICES=0 python tools/marlincfg/marlincfg_bench.py --shapes ~/projects/lanes/coexec/shapes_zp.pt \
      --out RESULTS/marlincfg_bench.json [--budget-s 1200] [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time

ENV = "VLLM_MARLIN_CFG"
DBG = "VLLM_MARLIN_CFG_DEBUG"

# TP=2 rank of Qwen3.8-27B: 64 layers = 16 full attention (qkv, o_proj) + 48 Gated DeltaNet (in_proj_qkvz, out_proj),
# every layer an MLP (gate_up, down).
LAYER_COUNTS = {"qkv": 16, "o_proj": 16, "gdn_in": 48, "gdn_out": 48, "gate_up": 64, "down": 64}
# VLLM_LP_W8X=1 (production) moves these to int8 IMMA at M >= 2048; Marlin only sees them below.
W8X_SHAPES = {"gate_up", "gdn_in"}
W8X_MIN_M = 2048
M_GRID = [17, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2047, 2048, 2560, 3072, 3632, 4096]
M_QUICK = [32, 64, 256, 1024, 2047, 3632]
# measurement priority (the engine's prefill chunk is 3632; a 2K prompt is one ~2000-row chunk)
M_PRIORITY = [3632, 2047, 1024, 4096, 2048, 512, 3072, 256, 2560, 1536, 768, 128, 64, 32, 384, 192, 96, 48, 24, 17]
M_MAX_RULE = 1 << 20

THREAD_CFGS = [(64, 256, 256), (64, 128, 128), (128, 64, 128)]  # (thread_k, thread_n, threads) = large_batch table


def candidates():
    out = []
    for mb in (4, 3, 2):
        for tk, tn, th in THREAD_CFGS:
            for bps in ((1,) if th == 256 else (1, 2)):
                for par in (0, 64):
                    out.append((tk, tn, th, mb, bps, par))
    return out


def rule_str(n, k, lo, hi, cand):
    tk, tn, th, mb, bps, par = cand
    return f"{n},{k},{lo},{hi}:{tk},{tn},{th},{mb},{bps},{par}"


RULE_RE = re.compile(r"^(\d+),(\d+),(\d+),(\d+):(\d+),(\d+),(\d+),([234]),([12]),(\d+)$")


def parse_table(s):
    """Python mirror of marlin_cfg_rules() (marlin.cu) for tests and the verify phase."""
    rules = []
    for item in (s or "").split(";"):
        if not item.strip():
            continue
        m = RULE_RE.match(item.strip())
        if not m:
            raise ValueError(f"malformed rule {item!r}")
        v = [int(x) for x in m.groups()]
        if v[2] > v[3]:
            raise ValueError(f"empty M range in {item!r}")
        rules.append({"n": v[0], "k": v[1], "lo": v[2], "hi": v[3], "cand": tuple(v[4:])})
    return rules


def match(rules, m, n, k):
    for r in rules:
        if (r["n"] in (0, n)) and (r["k"] in (0, k)) and r["lo"] <= m <= r["hi"]:
            return r
    return None


def build_table(points, min_gain=0.03, near=0.02):
    """points: {shape: {"n", "k", "pts": {M: {"prod_ms", "cands": {cand_tuple: {"ms", "ok"}}}}}} -> (table str, rules).
    At each M the 'good' set = numerics-passing candidates with speedup >= 1 + min_gain and within `near` of the
    best. Greedy over sorted M: pick the good candidate covering the longest run of consecutive measured points
    (tie: best mean speedup), emit one rule for that run, continue. Points with no good candidate keep upstream."""
    rules = []
    for shape in sorted(points):
        sp = points[shape]
        n, k, pts = sp["n"], sp["k"], sp["pts"]
        ms = sorted(pts)
        if not ms:
            continue
        good = []
        for m in ms:
            p = pts[m]
            ok = {c: v["ms"] for c, v in p["cands"].items() if v.get("ok") and v.get("ms")}
            if not ok or not p.get("prod_ms"):
                good.append({})
                continue
            best = min(ok.values())
            good.append({c: p["prod_ms"] / t for c, t in ok.items()
                         if p["prod_ms"] / t >= 1 + min_gain and t <= best * (1 + near)})
        i = 0
        while i < len(ms):
            if not good[i]:
                i += 1
                continue
            best_c, best_len, best_mean = None, 0, 0.0
            for c in good[i]:
                j = i
                while j + 1 < len(ms) and c in good[j + 1]:
                    j += 1
                run = j - i + 1
                mean = sum(good[t][c] for t in range(i, j + 1)) / run
                if run > best_len or (run == best_len and mean > best_mean):
                    best_c, best_len, best_mean = c, run, mean
            j = i + best_len - 1
            lo = ms[i] if i == 0 else (ms[i - 1] + ms[i]) // 2 + 1
            if j == len(ms) - 1:
                hi = M_MAX_RULE if not sp.get("cap_m") else ms[j]
            else:
                hi = (ms[j] + ms[j + 1]) // 2
            rules.append({"shape": shape, "n": n, "k": k, "lo": lo, "hi": hi, "cand": best_c,
                          "points": ms[i:j + 1], "mean_speedup": round(best_mean, 4)})
            i = j + 1
    table = ";".join(rule_str(r["n"], r["k"], r["lo"], r["hi"], r["cand"]) for r in rules)
    return table, rules


def weighted(points, ms_key):
    """Per M: sum over shapes of LAYER_COUNTS x per-GEMM ms (Marlin share of one forward over M tokens)."""
    out = {}
    all_m = sorted({m for sp in points.values() for m in sp["pts"]})
    for m in all_m:
        prod = cand = 0.0
        complete = True
        for shape, sp in points.items():
            if shape in W8X_SHAPES and m >= W8X_MIN_M:
                continue
            p = sp["pts"].get(m)
            if not p or not p.get("prod_ms") or not p.get(ms_key):
                complete = False
                continue
            prod += LAYER_COUNTS[shape] * p["prod_ms"]
            cand += LAYER_COUNTS[shape] * p[ms_key]
        if prod:
            out[m] = {"prod_ms": round(prod, 3), "table_ms": round(cand, 3), "speedup": round(prod / cand, 4),
                      "complete": complete}
    return out


# ------------------------------------------------------------------------------------------------ GPU part

def _so_info():
    import hashlib

    import vllm
    p = os.path.join(os.path.dirname(vllm.__file__), "_C_stable_libtorch.abi3.so")
    try:
        return {"path": os.path.realpath(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    except OSError as ex:
        return {"path": p, "error": str(ex)}


class StderrCapture:
    """Capture fd 2 (the C++ op's fprintf) into a temp file."""

    def __enter__(self):
        sys.stderr.flush()
        self.tmp = tempfile.TemporaryFile(mode="w+b")
        self.saved = os.dup(2)
        os.dup2(self.tmp.fileno(), 2)
        return self

    def __exit__(self, *a):
        sys.stderr.flush()
        os.dup2(self.saved, 2)
        os.close(self.saved)
        self.tmp.seek(0)
        self.text = self.tmp.read().decode(errors="replace")
        self.tmp.close()


SIG_RE = re.compile(r"\[marlin-cfg\] m=(\d+) n=(\d+) k=(\d+) split=(\d+) mb=(\d+) cfg=(-?\d+),(-?\d+),(-?\d+) bps=(\d+)")


def set_table(s):
    if s:
        os.environ[ENV] = s
    else:
        os.environ.pop(ENV, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=os.path.expanduser("~/projects/lanes/coexec/shapes_zp.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-s", type=float, default=1200)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--min-gain", type=float, default=0.03)
    ap.add_argument("--max-loss", type=float, default=0.02)
    ap.add_argument("--target-ms", type=float, default=6.0)
    args = ap.parse_args()

    import torch
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types

    t_start = time.time()
    set_table(None)
    os.environ.pop(DBG, None)
    dev = torch.device("cuda", 0)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    wtype = scalar_types.uint4
    workspace = torch.zeros(sms * 2, dtype=torch.int, device=dev)  # production dense path: 2 lock slots per SM
    res = {"gpu": torch.cuda.get_device_name(0), "sms": sms, "torch": torch.__version__,
           "ops_so": _so_info(),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(args), "rows": [], "errors": []}

    def gemm(x, w, m):
        return ops.marlin_gemm(x, None, w["q"], None, w["s"], None, None, w["zp"], workspace, wtype, size_m=m,
                               size_n=w["n"], size_k=w["k"], use_atomic_add=False, use_fp32_reduce=True,
                               is_zp_float=False)

    def signature(x, w, m):
        os.environ[DBG] = "1"
        try:
            with StderrCapture() as cap:
                out = gemm(x, w, m)
                torch.cuda.synchronize()
        finally:
            os.environ.pop(DBG, None)
        sig = tuple((int(g[3]), int(g[4]), (int(g[5]), int(g[6]), int(g[7])), int(g[8]))
                    for g in SIG_RE.findall(cap.text))
        return out, sig

    def time_graph(x, w, m, reps):
        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            gemm(x, w, m)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(reps):
                gemm(x, w, m)
        g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(5):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) / reps)
        del g
        return statistics.median(ts)

    def rel_err(out, ref):
        d = out.float() - ref
        if not torch.isfinite(out).all():
            return float("inf")
        return (d.norm() / ref.norm()).item()

    shapes = torch.load(args.shapes, map_location="cpu")
    weights = {}
    for name in LAYER_COUNTS:
        c = shapes[name][0]
        weights[name] = {"q": c["q"].to(dev), "s": c["s"].to(dev), "zp": c["zp"].to(dev), "k": c["k"], "n": c["n"]}
    del shapes
    # exact dequantized weights: Marlin(identity) (one nonzero product per output element)
    wdeq = {}
    for name, w in weights.items():
        eye = torch.eye(w["k"], device=dev, dtype=torch.float16)
        wdeq[name] = gemm(eye, w, w["k"]).float()
        del eye
    torch.cuda.empty_cache()

    cands = candidates()
    m_grid = M_QUICK if args.quick else M_GRID
    order = [m for m in M_PRIORITY if m in m_grid]
    points = {name: {"n": w["n"], "k": w["k"], "pts": {}, "cap_m": name in W8X_SHAPES}
              for name, w in weights.items()}
    xs = {}
    budget_hit = False

    for m in order:
        for name, w in weights.items():
            if name in W8X_SHAPES and m >= W8X_MIN_M:
                continue
            if time.time() - t_start > args.budget_s * 0.75:
                budget_hit = True
                break
            gen = torch.Generator(device=dev).manual_seed(1000 + m)
            x = torch.randn(m, w["k"], device=dev, dtype=torch.float16, generator=gen)
            ref = x.float() @ wdeq[name]
            row = {"shape": name, "n": w["n"], "k": w["k"], "m": m, "cands": []}
            try:
                set_table(None)
                out_p, sig_p = signature(x, w, m)
                if not sig_p:
                    raise RuntimeError("no [marlin-cfg] dispatch line: this _C_stable_libtorch lacks VLLM_MARLIN_CFG")
                err_p = rel_err(out_p, ref)
                # reps so one replay is ~target_ms
                t0 = time_graph(x, w, m, 2)
                reps = max(2, min(64, int(round(args.target_ms / max(t0, 1e-3)))))
                by_sig = {sig_p: {"cands": ["prod"], "out_equal_prod": True, "rel_err": err_p, "ok": True}}
                cand_sig = {}
                for c in cands:
                    set_table(rule_str(w["n"], w["k"], m, m, c))
                    out_c, sig_c = signature(x, w, m)
                    cand_sig[c] = sig_c
                    if sig_c in by_sig:
                        e = by_sig[sig_c]
                        e["cands"].append(list(c))
                        if sig_c == sig_p and not torch.equal(out_c, out_p):
                            e["out_equal_prod"] = False
                            e["ok"] = False
                            res["errors"].append(f"{name} M={m} {c}: same dispatch as prod but output differs")
                        continue
                    err_c = rel_err(out_c, ref)
                    by_sig[sig_c] = {"cands": [list(c)], "rel_err": err_c,
                                     "out_equal_prod": bool(torch.equal(out_c, out_p)),
                                     "max_abs_diff_prod": (out_c.float() - out_p.float()).abs().max().item(),
                                     "ok": err_c <= err_p * 1.10 + 1e-5}
                    del out_c
                # timing: 2 rounds, opposite order, distinct signatures only; prod in every round
                sigs = list(by_sig)
                times = {s: [] for s in sigs}
                for rnd in range(2):
                    for s in (sigs if rnd == 0 else sigs[::-1]):
                        first = by_sig[s]["cands"][0]
                        set_table(None if first == "prod" else rule_str(w["n"], w["k"], m, m, tuple(first)))
                        times[s].append(time_graph(x, w, m, reps))
                set_table(None)
                prod_ms = min(times[sig_p])
                pt = {"prod_ms": prod_ms, "prod_err": err_p, "cands": {}}
                for c in cands:
                    e = by_sig[cand_sig[c]]
                    pt["cands"][c] = {"ms": min(times[cand_sig[c]]), "ok": e["ok"]}
                points[name]["pts"][m] = pt
                row.update(reps=reps, prod_ms=prod_ms, prod_err=err_p, prod_sig=[list(map(str, t)) for t in sig_p],
                           prod_round_ms=times[sig_p],
                           sigs=[{"sig": [list(map(str, t)) for t in s], "ms": min(times[s]), "round_ms": times[s],
                                  "speedup": prod_ms / min(times[s]), **by_sig[s]} for s in sigs])
                best = min(sigs, key=lambda s: (not by_sig[s]["ok"], min(times[s])))
                print(f"{name:8s} M={m:5d} prod {prod_ms * 1e3:8.1f} us  best x{prod_ms / min(times[best]):.3f} "
                      f"{by_sig[best]['cands'][0]} ({len(sigs)} sigs, err {err_p:.2e})", flush=True)
            except Exception as ex:  # noqa: BLE001
                res["errors"].append(f"{name} M={m}: {type(ex).__name__}: {ex}")
                print(f"ERROR {name} M={m}: {ex}", flush=True)
            finally:
                set_table(None)
            res["rows"].append(row)
            del x, ref
            torch.cuda.empty_cache()
        if budget_hit:
            break
    res["budget_hit"] = budget_hit

    table, rules = build_table(points, args.min_gain)
    res["table"] = table
    res["rules"] = [{**r, "cand": list(r["cand"])} for r in rules]
    res["table_len"] = len(table)

    # verify: the whole table set at once; every measured point dispatches as intended, numerics ok, no loss
    verify = []
    parsed = parse_table(table)
    for name, sp in points.items():
        w = weights[name]
        for m in sorted(sp["pts"]):
            if time.time() - t_start > args.budget_s:
                res["budget_hit"] = True
                break
            r = match(parsed, m, w["n"], w["k"])
            gen = torch.Generator(device=dev).manual_seed(1000 + m)
            x = torch.randn(m, w["k"], device=dev, dtype=torch.float16, generator=gen)
            ref = x.float() @ wdeq[name]
            try:
                set_table(None)
                out_p, sig_p = signature(x, w, m)
                set_table(rule_str(w["n"], w["k"], m, m, r["cand"]) if r else None)
                _, sig_want = signature(x, w, m)
                set_table(table)
                out_t, sig_t = signature(x, w, m)
                err_p, err_t = rel_err(out_p, ref), rel_err(out_t, ref)
                reps = next((row["reps"] for row in res["rows"] if row["shape"] == name and row["m"] == m and
                             "reps" in row), 4)
                tp, tt = [], []
                for rnd in range(2):
                    for which in ((0, 1) if rnd == 0 else (1, 0)):
                        set_table(table if which else None)
                        (tt if which else tp).append(time_graph(x, w, m, reps))
                set_table(None)
                v = {"shape": name, "m": m, "rule": rule_str(w["n"], w["k"], r["lo"], r["hi"], r["cand"]) if r else None,
                     "sig_ok": sig_t == sig_want, "unchanged": sig_t == sig_p, "rel_err_prod": err_p,
                     "rel_err_table": err_t, "num_ok": err_t <= err_p * 1.10 + 1e-5,
                     "prod_ms": min(tp), "table_ms": min(tt), "speedup": min(tp) / min(tt)}
                if v["unchanged"] and not torch.equal(out_t, out_p):
                    v["num_ok"] = False
                verify.append(v)
                sp["pts"][m]["table_ms"] = min(tt)
                sp["pts"][m]["prod_ms_verify"] = min(tp)
            except Exception as ex:  # noqa: BLE001
                res["errors"].append(f"verify {name} M={m}: {type(ex).__name__}: {ex}")
            finally:
                set_table(None)
            del x, ref
    res["verify"] = verify
    # weighted Marlin time per forward of M tokens (prod vs table), from the verify round (same-round comparison)
    for sp in points.values():
        for p in sp["pts"].values():
            if "prod_ms_verify" in p:
                p["prod_ms"] = p["prod_ms_verify"]
    res["weighted"] = {str(m): v for m, v in weighted(points, "table_ms").items()}

    bad_sig = [v for v in verify if not v["sig_ok"]]
    bad_num = [v for v in verify if not v["num_ok"]]
    loss = [v for v in verify if v["speedup"] < 1 - args.max_loss]
    chunk = res["weighted"].get("3632", {})
    res["checks"] = {"bad_sig": len(bad_sig), "bad_num": len(bad_num), "loss_points": len(loss),
                     "loss_detail": [(v["shape"], v["m"], round(v["speedup"], 3)) for v in loss],
                     "chunk3632_speedup": chunk.get("speedup"), "errors": len(res["errors"])}
    if res["errors"] or bad_num or bad_sig:
        res["verdict"] = "FAIL"
    elif not table:
        res["verdict"] = "NOWIN"
    elif loss or (chunk.get("speedup") or 1.0) < 1.0:
        res["verdict"] = "NOWIN"  # the table does not hold up in the same-round re-measurement
    else:
        res["verdict"] = "PASS"
    res["seconds"] = round(time.time() - t_start, 1)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    with open(os.path.splitext(args.out)[0] + "_verdict.json", "w") as fh:
        json.dump({"verdict": res["verdict"], "table": table or "none", "checks": res["checks"],
                   "weighted": res["weighted"]}, fh, indent=1)
    print(json.dumps({"verdict": res["verdict"], "table": table, "checks": res["checks"],
                      "weighted": res["weighted"]}, indent=1), flush=True)
    return 0 if res["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
