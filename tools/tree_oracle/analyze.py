#!/usr/bin/env python3
"""E16/P0 tree-drafting oracle analysis (CPU only, stdlib + optional numpy-free).

Inputs (written by vllm/v1/spec_decode/tree_oracle.py, env-gated in the engine):
  value rows  {"t":"v", req, seq, ctx, temp, drafts[D], ids[D][k], p[D][k], out[a+1], acc}
  step rows   {"t":"s", B, q, uni, dt_ms, draft_ms, ctx_max}

What it computes (EXACT unless marked approx):
  * chain K' for every K' <= D_run (the run's draft depth): tokens/step = min(a, K') + 1.
  * chain + leaves trees (D_t <= D_run, w_d leaves at depth d, leaf ranks 2..w_d+1 of the drafter at that depth, parent
    = chain token d-1): a step that rejects the chain at depth j = a+1 <= D_t gains exactly +1 token iff the emitted
    token out[a] is one of those leaves. Exact because the chain's verification is lossless: out[a] is a draw from the
    target conditioned on rejecting the chain draft, which is the distribution the leaves are verified against
    (recursive rejection over one-hot drafts == draw y ~ p, accept the sibling equal to y).
  * confidence-gated leaves (leaf only where its draft prob >= tau): exact, with the mean leaf rows actually used.
  * branch shapes (a rank-2 sibling at depth j WITH an own child chain of length L): APPROX (upper bound). The child
    chain's acceptance is taken from the same request's NEXT step (whose drafter ran from the target's hidden state
    at the rescued token, better than the MTP-chained state the tree would use).
  * rescue rate per depth (P(out[a] in ranks 2..w | rejected at depth a+1)), by category, temperature class, context.
  * cost oracle: per (K, B) the median uniform-decode step period and drafter GPU time -> verify cost per row count
    M = K+1, then predicted tok/s for each tree = tokens/step / (verify(M_tree) + drafter(D_tree)).

Usage:
  analyze.py --results <window results dir>           # finds oracle_k*.jsonl / cost_k*.jsonl there
  analyze.py --value a.jsonl [--cost 3=k3.jsonl 5=k5.jsonl 7=k7.jsonl] [--out r.json] [--md r.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import statistics as st
import sys
from collections import defaultdict

CAT_RE = re.compile(r"tro-([A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)-")
CTX_BUCKETS = ((0, 4096, "<4K"), (4096, 16384, "4-16K"), (16384, 40960, "16-40K"), (40960, 1 << 40, ">40K"))


# ---------------------------------------------------------------------------------------------------- loading

def load_rows(paths):
    """Yield parsed JSONL rows from one or more files; a truncated last line (engine killed) is skipped."""
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def category(req: str, manifest=None, ts=None) -> str:
    m = CAT_RE.search(req or "")
    if m:
        return m.group(1)
    if manifest and ts is not None:
        for ph in manifest:
            if ph["t0"] <= ts <= ph["t1"]:
                return ph["cat"]
    return "other"


def ctx_bucket(ctx) -> str:
    if ctx is None:
        return "?"
    for lo, hi, name in CTX_BUCKETS:
        if lo <= ctx < hi:
            return name
    return "?"


def temp_class(t) -> str:
    if t is None:
        return "?"
    return "greedy" if t < 1e-5 else "sampled"


class Step:
    """One verified step of one request, normalized for the tree math."""
    __slots__ = ("req", "seq", "ctx", "temp", "drafts", "ids", "p", "out", "a", "D", "cat", "next_a", "next_D",
                 "top1_match")

    def __init__(self, r, manifest=None):
        self.req = r["req"]
        self.seq = r.get("seq", 0)
        self.ctx = r.get("ctx")
        self.temp = r.get("temp")
        self.drafts = list(r["drafts"])
        self.ids = r["ids"]
        self.p = r.get("p") or [[None] * len(x) for x in self.ids]
        self.out = list(r["out"])
        self.a = int(r.get("acc", len(self.out) - 1))
        # the depth verified this step: scheduled drafts (may be trimmed near max_tokens) capped by logged depths
        self.D = min(len(self.drafts), len(self.ids))
        self.cat = category(self.req, manifest, r.get("ts"))
        self.next_a = None
        self.next_D = None
        self.top1_match = all(self.ids[d][0] == self.drafts[d] for d in range(self.D))

    def leaf_candidates(self, d: int, w: int, tau: float = 0.0):
        """Up to w drafter tokens at depth d (0-based) other than the chain draft, best rank first; with tau, only
        those whose draft probability >= tau."""
        out = []
        for tok, pr in zip(self.ids[d], self.p[d]):
            if tok == self.drafts[d]:
                continue
            if len(out) >= w:
                break
            if tau > 0 and (pr is None or pr < tau):
                break  # ranks are sorted by probability: nothing below can pass
            out.append(tok)
        return out

    def rescue_rank(self):
        """1-based rank among the non-chain candidates of the token emitted after the first rejection, None if the
        chain was fully accepted, 0 if it is not in the logged top-k."""
        if self.a >= self.D:
            return None
        y = self.out[self.a]
        k = 0
        for tok in self.ids[self.a]:
            if tok == self.drafts[self.a]:
                continue
            k += 1
            if tok == y:
                return k
        return 0


def link_next(steps):
    by_req = defaultdict(list)
    for s in steps:
        by_req[s.req].append(s)
    for lst in by_req.values():
        lst.sort(key=lambda s: s.seq)
        for s, n in zip(lst, lst[1:]):
            if n.seq == s.seq + 1:
                s.next_a, s.next_D = n.a, n.D


# ---------------------------------------------------------------------------------------------------- tree math

class Tree:
    """Chain depth D plus, per depth d (1-based), `leaves[d-1]` sibling leaves (ranks 2..w+1) whose parent is chain
    token d-1. `branch` = {depth: L}: the rank-2 sibling at that depth also gets an own chain of L children (approx).
    `tau`: leaves only where their draft prob >= tau (dynamic; rows reported as the mean actually used)."""

    def __init__(self, name, D, leaves=(), branch=None, tau=0.0, note=""):
        self.name, self.D, self.tau, self.note = name, D, tau, note
        self.leaves = tuple(leaves) + (0,) * (D - len(leaves))
        self.branch = dict(branch or {})
        self.approx = bool(self.branch)

    @property
    def rows(self) -> int:
        """Verify rows per request M = root + chain + leaves + branch children (static shape)."""
        return 1 + self.D + sum(self.leaves) + sum(self.branch.values())

    def evaluate(self, s: Step):
        """(tokens emitted this step, dynamic leaf rows used) or None if the step cannot evaluate this tree
        (verified depth < tree depth, or an approx branch without a next step)."""
        if s.D < self.D:
            return None
        if self.tau > 0:
            used = sum(len(s.leaf_candidates(d, self.leaves[d], self.tau)) for d in range(self.D))
        else:
            used = sum(self.leaves)
        if s.a >= self.D:
            return self.D + 1, used
        d = s.a
        y = s.out[d]
        w = self.leaves[d]
        if w and y in s.leaf_candidates(d, w, self.tau):
            L = self.branch.get(d + 1, 0)
            if L and y in s.leaf_candidates(d, 1, self.tau):
                if s.next_a is None or s.next_D is None or s.next_D < L:
                    return None
                return d + 2 + min(s.next_a, L), used
            return d + 2, used
        return d + 1, used


def standard_trees(D_run: int):
    trees = []
    for K in range(1, D_run + 1):
        trees.append(Tree(f"chain K={K}", K))
    if D_run >= 3:
        trees += [
            Tree("K3 + r2 leaf @d1 (M=5)", 3, (1,)),
            Tree("K3 + r2,r3 leaves @d1 (M=6)", 3, (2,)),
            Tree("K3 + r2..r4 leaves @d1 (M=7)", 3, (3,)),
            Tree("K3 + r2 leaf @d1,d2 (M=6)", 3, (1, 1)),
            Tree("K3 + r2 leaf @each depth (M=7) [design proto]", 3, (1, 1, 1)),
            Tree("K3 + r2,r3 @d1 + r2 @d2,d3 (M=8)", 3, (2, 1, 1)),
            Tree("K3 + r2,r3 @each depth (M=10)", 3, (2, 2, 2)),
            Tree("K3 + r2 leaf @each depth, p2>=0.05 (dyn)", 3, (1, 1, 1), tau=0.05),
            Tree("K3 + r2 leaf @each depth, p2>=0.10 (dyn)", 3, (1, 1, 1), tau=0.10),
            Tree("K3 + r2 leaf @each depth, p2>=0.20 (dyn)", 3, (1, 1, 1), tau=0.20),
            Tree("K3 + r2@d1 with 1 child (M=6) [approx]", 3, (1,), branch={1: 1}),
            Tree("K3 + r2@d1 with 2-chain (M=7) [approx]", 3, (1,), branch={1: 2}),
            Tree("K3 + r2@d1 2-chain + r2@d2,d3 (M=9) [approx]", 3, (1, 1, 1), branch={1: 2}),
        ]
    if D_run >= 4:
        trees += [
            Tree("K4 + r2 leaf @d1 (M=6)", 4, (1,)),
            Tree("K4 + r2 leaf @each depth (M=9)", 4, (1, 1, 1, 1)),
        ]
    if D_run >= 5:
        trees += [Tree("K5 + r2 leaf @each depth (M=11)", 5, (1, 1, 1, 1, 1))]
    return trees


def mean_ci(values, groups=None, n_boot=200, seed=0):
    """Mean and a request-clustered bootstrap 95% CI (steps of one request are correlated)."""
    if not values:
        return None, None, None
    m = sum(values) / len(values)
    if groups is None or len(set(groups)) < 3:
        sd = st.pstdev(values) if len(values) > 1 else 0.0
        h = 1.96 * sd / math.sqrt(len(values))
        return m, m - h, m + h
    by = defaultdict(list)
    for v, g in zip(values, groups):
        by[g].append(v)
    sums = [(sum(v), len(v)) for v in by.values()]
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        tot = cnt = 0
        for _ in range(len(sums)):
            s_, c_ = sums[rng.randrange(len(sums))]
            tot += s_
            cnt += c_
        boots.append(tot / cnt)
    boots.sort()
    return m, boots[int(0.025 * n_boot)], boots[min(n_boot - 1, int(0.975 * n_boot))]


def evaluate_trees(steps, trees, base_K=3):
    """Per tree: tokens/step on the steps where BOTH it and the production chain K=base_K are evaluable (paired), the
    gain vs the chain, the CI, and the excluded-step count."""
    base = Tree(f"chain K={base_K}", base_K)
    res = []
    for t in trees:
        vals, bvals, groups, used = [], [], [], []
        excl = 0
        for s in steps:
            r = t.evaluate(s)
            b = base.evaluate(s)
            if r is None or b is None:
                excl += 1
                continue
            vals.append(r[0])
            used.append(r[1])
            bvals.append(b[0])
            groups.append(s.req)
        if not vals:
            res.append({"tree": t.name, "n": 0, "excluded": excl})
            continue
        m, lo, hi = mean_ci(vals, groups)
        bm = sum(bvals) / len(bvals)
        diffs = [v - b for v, b in zip(vals, bvals)]
        dm, dlo, dhi = mean_ci(diffs, groups)
        dyn_rows = 1 + t.D + (sum(used) / len(used)) + sum(t.branch.values())
        res.append({"tree": t.name, "D": t.D, "rows": t.rows if t.tau <= 0 else round(dyn_rows, 2),
                    "static_rows": t.rows, "approx": t.approx, "n": len(vals), "excluded": excl,
                    "tok_per_step": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                    "base_tok_per_step": round(bm, 4), "gain_tok": round(dm, 4),
                    "gain_ci95": [round(dlo, 4), round(dhi, 4)], "gain_pct": round(100 * dm / bm, 2)})
    return res


def rescue_table(steps, widths=(1, 2, 3, 7)):
    """Rescue rate per (0-based) rejection depth: P(emitted token in the top-w non-chain candidates | chain rejected
    there). Also the share of rejections whose emitted token is outside the logged top-k."""
    by_d = defaultdict(list)
    for s in steps:
        rk = s.rescue_rank()
        if rk is not None:
            by_d[s.a].append(rk)
    out = {}
    for d in sorted(by_d):
        ranks = by_d[d]
        n = len(ranks)
        out[f"d{d + 1}"] = {"rejections": n,
                            **{f"r_top{w + 1}": round(sum(1 for r in ranks if 1 <= r <= w) / n, 4) for w in widths},
                            "outside_topk": round(sum(1 for r in ranks if r == 0) / n, 4)}
    return out


def survival(steps, D_run):
    n = len(steps)
    if not n:
        return {}
    return {f"P(acc>={d})": round(sum(1 for s in steps if s.a >= d and s.D >= d) / n, 4) for d in range(1, D_run + 1)}


def p2_calibration(steps, edges=(0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01)):
    """Rescue rate of the rank-2 leaf binned by its draft probability (does the drafter's own confidence tell us
    where a leaf pays? -> dynamic trees)."""
    bins = defaultdict(lambda: [0, 0])
    for s in steps:
        if s.a >= s.D:
            continue
        cands = [(t, p) for t, p in zip(s.ids[s.a], s.p[s.a]) if t != s.drafts[s.a]]
        if not cands or cands[0][1] is None:
            continue
        tok, p2 = cands[0]
        for lo, hi in zip(edges, edges[1:]):
            if lo <= p2 < hi:
                b = bins[f"[{lo},{hi})"]
                b[0] += 1
                b[1] += int(s.out[s.a] == tok)
                break
    return {k: {"n": v[0], "rescue": round(v[1] / v[0], 4) if v[0] else None} for k, v in sorted(bins.items())}


# ---------------------------------------------------------------------------------------------------- cost oracle

def cost_curve(step_rows):
    """Median uniform-decode step period and drafter GPU time per (B, ctx bucket). A row counts when it and the
    previous row are uniform decode with the same B (the period then spans exactly one decode step)."""
    groups = defaultdict(lambda: {"dt": [], "draft": []})
    prev = None
    for r in step_rows:
        if r.get("t") != "s":
            continue
        if (prev is not None and r.get("uni") and prev.get("uni") and r.get("B") == prev.get("B")
                and r.get("dt_ms") is not None and r["dt_ms"] < 2000):
            key = (r["B"], ctx_bucket(r.get("ctx_max")))
            groups[key]["dt"].append(r["dt_ms"])
            if r.get("draft_ms") is not None:
                groups[key]["draft"].append(r["draft_ms"])
        prev = r
    out = {}
    for (B, cb), g in sorted(groups.items()):
        if len(g["dt"]) < 20:
            continue
        out[f"B{B}|{cb}"] = {"B": B, "ctx": cb, "n": len(g["dt"]), "step_ms_p50": round(st.median(g["dt"]), 3),
                             "draft_ms_p50": round(st.median(g["draft"]), 3) if g["draft"] else None,
                             "verify_ms_p50": round(st.median(g["dt"]) - st.median(g["draft"]), 3)
                             if g["draft"] else None}
    return out


def interp(xs_ys, x):
    pts = sorted(xs_ys)
    if not pts:
        return None
    if x <= pts[0][0]:
        if len(pts) == 1:
            return pts[0][1]
        (x0, y0), (x1, y1) = pts[0], pts[1]
    elif x >= pts[-1][0]:
        if len(pts) == 1:
            return pts[-1][1]
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
    else:
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                break
    return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def predict(tree_res, costs, base_K=3):
    """costs: {K: cost_curve}. For each B / ctx bucket with all arms: verify(M) interpolated over M = K+1, drafter(D)
    interpolated over K; predicted tok/s of each tree vs the production chain K=base_K (both from the same model)."""
    if not costs:
        return []
    keys = set.intersection(*[set(c.keys()) for c in costs.values()]) if len(costs) > 1 else set()
    out = []
    for key in sorted(keys, key=lambda k: (int(k.split("|")[0][1:]), k)):
        ver = [(K + 1, c[key]["verify_ms_p50"]) for K, c in costs.items() if c[key].get("verify_ms_p50") is not None]
        dra = [(K, c[key]["draft_ms_p50"]) for K, c in costs.items() if c[key].get("draft_ms_p50") is not None]
        if len(ver) < 2 or len(dra) < 2:
            continue
        base_ms = interp(ver, base_K + 1) + interp(dra, base_K)
        row = {"key": key, "verify_ms_by_M": dict(sorted(ver)), "draft_ms_by_K": dict(sorted(dra)),
               "base_step_ms": round(base_ms, 3), "trees": []}
        for t in tree_res:
            if not t.get("n"):
                continue
            ms = interp(ver, t["rows"]) + interp(dra, t["D"])
            tps = t["tok_per_step"] / ms * 1e3
            base_tps = t["base_tok_per_step"] / base_ms * 1e3
            row["trees"].append({"tree": t["tree"], "rows": t["rows"], "step_ms": round(ms, 3),
                                 "tok_s_per_req": round(tps, 2), "base_tok_s_per_req": round(base_tps, 2),
                                 "net_pct": round(100 * (tps / base_tps - 1), 2), "approx": t.get("approx", False)})
        out.append(row)
    return out


# ---------------------------------------------------------------------------------------------------- driver

def analyze(value_paths, cost_paths=None, manifest=None, base_K=3):
    meta, steps, step_rows = [], [], []
    clean_cost = defaultdict(list)   # step rows logged while value logging was paused: no top-k overhead
    bad_top1 = 0
    for path in value_paths:
        m = re.search(r"oracle_k(\d+)", os.path.basename(path))
        k_file = int(m.group(1)) if m else None
        paused = False
        for r in load_rows([path]):
            t = r.get("t")
            if t == "v":
                if not r.get("drafts") or not r.get("ids"):
                    continue
                s = Step(r, manifest)
                bad_top1 += int(not s.top1_match)
                steps.append(s)
                if k_file is None:
                    k_file = len(s.ids)
            elif t == "s":
                (clean_cost[k_file] if paused else step_rows).append(r)
            elif t == "meta":
                meta.append(r)
                paused = False
            elif t == "pause":
                paused = bool(r.get("paused"))
    link_next(steps)
    D_run = max((len(s.ids) for s in steps), default=0)
    trees = standard_trees(D_run)
    rep = {"inputs": {"value": value_paths, "cost": cost_paths or {}}, "D_run": D_run, "steps": len(steps),
           "requests": len({s.req for s in steps}), "drafts_not_top1": bad_top1,
           "truncated_steps": sum(1 for s in steps if s.D < D_run), "meta": meta[:3],
           "notes": [
               "chain K'<=D_run and chain+leaves trees are EXACT (lossless verification: the emitted token after a "
               "rejection is a draw from the target conditioned on rejecting the chain draft).",
               "[approx] rows: branch children use the NEXT step's acceptance (drafter from the target's hidden state, "
               "not the MTP-chained state): an upper bound.",
               "Positions differ between a K=D_run run and a K=3 run (steps land at other offsets); per-token "
               "acceptance statistics are unaffected to first order.",
               "Trees are compared PAIRED against the chain K=%d on the same steps; steps whose verified depth is "
               "below the tree depth (trimmed near max_tokens) are excluded and counted." % base_K,
               "Cost: verify(M) is measured with chain rows (which write GDN states); leaves would not, so the tree's "
               "GDN cost is slightly over-estimated. Sampler cost over M rows is included in the measured step.",
           ]}
    rep["overall"] = {"survival": survival(steps, D_run), "rescue": rescue_table(steps),
                      "p2_calibration": p2_calibration(steps), "trees": evaluate_trees(steps, trees, base_K)}
    slices = defaultdict(list)
    for s in steps:
        slices[f"cat={s.cat}"].append(s)
        slices[f"temp={temp_class(s.temp)}"].append(s)
        slices[f"ctx={ctx_bucket(s.ctx)}"].append(s)
        slices[f"cat={s.cat}|temp={temp_class(s.temp)}"].append(s)
    rep["slices"] = {}
    for k in sorted(slices):
        ss = slices[k]
        if len(ss) < 200:
            continue
        main = [t for t in trees if t.name.startswith(("chain", "K3 + r2 leaf @each depth (M=7)",
                                                         "K3 + r2,r3 leaves @d1", "K3 + r2 leaf @d1 (M=5)"))]
        rep["slices"][k] = {"steps": len(ss), "survival": survival(ss, D_run), "rescue": rescue_table(ss),
                            "trees": evaluate_trees(ss, main, base_K)}
    costs = {}
    for K, paths in (cost_paths or {}).items():
        c = cost_curve(r for r in load_rows(paths) if r.get("t") == "s")
        if c:
            costs[K] = c
    for K, rows in clean_cost.items():
        if K is not None and K not in costs:
            c = cost_curve(rows)
            if c:
                costs[K] = c
    if step_rows:
        rep["value_run_cost_with_topk_overhead"] = cost_curve(step_rows)
    rep["cost"] = {str(K): c for K, c in sorted(costs.items())}
    rep["prediction"] = predict(rep["overall"]["trees"], costs, base_K)
    rep["go_no_go"] = go_no_go(rep["prediction"])
    return rep


def go_no_go(pred):
    """Design gate: best exact tree >= +4% predicted tok/s at B=1 and no regression at B=2 (short context)."""
    def best(key_prefix, name=None):
        for row in pred:
            if row["key"].startswith(key_prefix):
                cands = [t for t in row["trees"] if not t["approx"] and not t["tree"].startswith("chain")
                         and (name is None or t["tree"] == name)]
                if cands:
                    return max(cands, key=lambda t: t["net_pct"])
        return None
    b1 = best("B1|<4K") or best("B1|")
    if b1 is None:
        return {"verdict": "no cost data", "rule": "B=1 >= +4% and B=2 >= 0%"}
    b2 = best("B2|<4K", b1["tree"]) or best("B2|", b1["tree"])
    ok = b1["net_pct"] >= 4.0 and (b2 is None or b2["net_pct"] >= 0.0)
    return {"verdict": "GO" if ok else "NO-GO", "rule": "best exact tree: B=1 >= +4% and B=2 >= 0% (gated tree)",
            "best_B1": b1, "same_tree_B2": b2}


def to_markdown(rep) -> str:
    L = [f"# E16 tree-drafting oracle report", "",
         f"steps {rep['steps']}, requests {rep['requests']}, D_run {rep['D_run']}, drafts != drafter top-1: "
         f"{rep['drafts_not_top1']}, truncated steps {rep['truncated_steps']}", ""]
    ov = rep["overall"]
    L += ["## Chain survival", "", " ".join(f"{k}={v}" for k, v in ov["survival"].items()), "",
          "## Rescue rate by rejection depth (overall)", "", "| depth | rejections | top2 | top3 | top4 | top8 | outside top-k |",
          "|---|---|---|---|---|---|---|"]
    for d, r in ov["rescue"].items():
        L.append(f"| {d} | {r['rejections']} | {r['r_top2']} | {r['r_top3']} | {r['r_top4']} | {r['r_top8']} | "
                 f"{r['outside_topk']} |")
    L += ["", "## Trees (paired vs chain K=3)", "", "| tree | rows | n | tok/step | 95% CI | gain tok | gain % |",
          "|---|---|---|---|---|---|---|"]
    for t in ov["trees"]:
        if t.get("n"):
            L.append(f"| {t['tree']} | {t['rows']} | {t['n']} | {t['tok_per_step']} | {t['ci95']} | {t['gain_tok']} "
                     f"| {t['gain_pct']} |")
    L += ["", "## Rescue rate (rank 2) by slice", "", "| slice | steps | d1 | d2 | d3 | M=7 gain % |", "|---|---|---|---|---|---|"]
    for k, s in rep["slices"].items():
        r = s["rescue"]
        g = next((t.get("gain_pct") for t in s["trees"] if t["tree"].startswith("K3 + r2 leaf @each depth (M=7)")), None)
        L.append(f"| {k} | {s['steps']} | {r.get('d1', {}).get('r_top2')} | {r.get('d2', {}).get('r_top2')} | "
                 f"{r.get('d3', {}).get('r_top2')} | {g} |")
    L += ["", "## Rank-2 rescue vs its draft probability", ""]
    L += [f"- p2 in {k}: n={v['n']} rescue={v['rescue']}" for k, v in ov["p2_calibration"].items()]
    if rep.get("cost"):
        L += ["", "## Cost oracle (median uniform-decode step)", "", "| K | key | n | step ms | drafter ms | verify ms |",
              "|---|---|---|---|---|---|"]
        for K, c in rep["cost"].items():
            for key, v in c.items():
                L.append(f"| {K} | {key} | {v['n']} | {v['step_ms_p50']} | {v['draft_ms_p50']} | {v['verify_ms_p50']} |")
    for row in rep.get("prediction", []):
        L += ["", f"### Predicted per-request tok/s, {row['key']} (chain K=3 step {row['base_step_ms']} ms)", "",
              "| tree | rows | step ms | tok/s | chain tok/s | net % |", "|---|---|---|---|---|---|"]
        for t in row["trees"]:
            L.append(f"| {t['tree']} | {t['rows']} | {t['step_ms']} | {t['tok_s_per_req']} | {t['base_tok_s_per_req']} "
                     f"| {t['net_pct']} |")
    g = rep.get("go_no_go", {})
    L += ["", f"## Go/no-go: **{g.get('verdict')}** ({g.get('rule')})", "", "```", json.dumps(g, indent=1), "```", "",
          "## Notes", ""] + [f"- {n}" for n in rep["notes"]]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", help="window results dir (oracle_k*.jsonl, cost_k*.jsonl, traffic_manifest.json)")
    ap.add_argument("--value", nargs="*", default=[])
    ap.add_argument("--cost", nargs="*", default=[], help="K=path ...")
    ap.add_argument("--manifest")
    ap.add_argument("--base-k", type=int, default=3)
    ap.add_argument("--out")
    ap.add_argument("--md")
    a = ap.parse_args(argv)
    value = list(a.value)
    cost = defaultdict(list)
    for kv in a.cost:
        k, p = kv.split("=", 1)
        cost[int(k)].append(p)
    manifest = json.load(open(a.manifest))["phases"] if a.manifest else None
    if a.results:
        value += sorted(glob.glob(os.path.join(a.results, "oracle_k*.jsonl")))
        for p in sorted(glob.glob(os.path.join(a.results, "cost_k*.jsonl"))):
            cost[int(re.search(r"cost_k(\d+)", p).group(1))].append(p)
        mf = os.path.join(a.results, "traffic_manifest.json")
        if manifest is None and os.path.exists(mf):
            manifest = json.load(open(mf)).get("phases")
        a.out = a.out or os.path.join(a.results, "tree_oracle_report.json")
        a.md = a.md or os.path.join(a.results, "tree_oracle_report.md")
    if not value:
        ap.error("no value-oracle JSONL (oracle_k*.jsonl) found")
    rep = analyze(value, dict(cost), manifest, a.base_k)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
    md = to_markdown(rep)
    if a.md:
        open(a.md, "w").write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
