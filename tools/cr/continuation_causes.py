#!/usr/bin/env python3
"""Lane CR: why do tool-loop continuations lose their prefix cache?  (read-only)

Joins the gateway request telemetry (requests-*.jsonl) with the engine-generation timeline
(journalctl: "Setting attention block size" / "GPU KV cache size") and splits every LOCAL
continuation's recomputed prompt tokens into causes:

  ok      : cached >= the predecessor's block-aligned prefix (minus slack)  -> only the tail is paid
  d_tail  : the structural tail, prefix - floor(prefix/B)*B, paid by EVERY continuation on hybrid GDN
  b_remote: the predecessor turn was served REMOTE (local cache never saw it)
  e_restart: the predecessor ran on an older engine generation (restart / config swap)
  a_evict : predecessor local + same generation, gateway's content-addressed model matched the
            messages (pm_credit), engine still missed  -> evicted (or a token-level divergence)
  c_diverge: predecessor local + same generation, gateway's message-hash chain did NOT match
            -> the client rewrote history (or the pairing is wrong)
"lost" tokens of a request = E - cached where E = floor((pred_ptok-4)/B)*B; the request's NEW
tokens (cur - pred) are inherent and never counted as lost.

Pairing: same (client, ip); predecessor finished before this request started; <= GAP s gap;
0 <= growth <= MAXGROW tokens; predecessor prompt >= 2 blocks. Ambiguity (several candidates)
is resolved toward the largest prefix, as the engine would.

usage: continuation_causes.py --since 'YYYY-MM-DD HH:MM' [--until ...] [--client X] [--json]
       [--events gen_events.txt]   (journalctl -u vllm-qwen27b -o short-unix | grep ...)
"""
import argparse, bisect, collections, datetime, glob, json, os, re, sys

TELEM = os.path.expanduser("~/.local/share/vllm-qwen27b/telemetry/requests-*.jsonl")


def load_generations(path):
    """[(ready_t, block, pool)] sorted.  A generation starts when its KV pool is sized."""
    gens, block = [], None
    for line in open(path):
        try:
            t = float(line.split()[0])
        except Exception:
            continue
        m = re.search(r"attention block size to (\d+) tokens", line)
        if m:
            block = int(m.group(1)); continue
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", line)
        if m:
            gens.append((t, block or 3568, int(m.group(1).replace(",", ""))))
    gens.sort()
    return gens


def gen_of(gens, starts, t):
    i = bisect.bisect_right(starts, t) - 1
    return i


def load_rows(lo, hi):
    rows = []
    for f in sorted(glob.glob(TELEM)):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            t = r.get("t")
            if t is None or (lo and t < lo) or (hi and t >= hi):
                continue
            if not r.get("ptok"):
                continue
            r["t0"] = t - float(r.get("duration") or 0.0)
            rows.append(r)
    rows.sort(key=lambda r: r["t0"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since"); ap.add_argument("--until"); ap.add_argument("--client")
    ap.add_argument("--events", default=os.path.join(os.path.dirname(__file__), "gen_events.txt"))
    ap.add_argument("--gap", type=float, default=300.0)
    ap.add_argument("--maxgrow", type=int, default=64000)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--pairs-out")
    ap.add_argument("--strict", action="store_true", help="skip continuations whose newest candidate is not also the largest")
    a = ap.parse_args()
    ts = lambda s: datetime.datetime.strptime(s, "%Y-%m-%d %H:%M").timestamp() if s else None
    lo, hi = ts(a.since), ts(a.until)
    gens = load_generations(a.events)
    starts = [g[0] for g in gens]
    rows = load_rows(lo - 3600 if lo else None, hi)

    # predecessor index: per (client, ip), finished requests (any route, any status 200)
    fin = collections.defaultdict(list)          # key -> [(t_end, ptok, row)]
    by_end = sorted([r for r in rows if r.get("status") == 200], key=lambda r: r["t"])
    ei = 0
    tot = collections.Counter(); n = collections.Counter(); per_client = collections.defaultdict(collections.Counter)
    tail_tok = 0; tail_n = 0; computed_all = 0; local_n = 0
    pairs = []
    for r in rows:
        while ei < len(by_end) and by_end[ei]["t"] <= r["t0"]:
            q = by_end[ei]; k = (q.get("client"), q.get("ip"))
            fin[k].append(q); ei += 1
            if len(fin[k]) > 400:
                fin[k] = fin[k][-200:]
        if lo and r["t0"] < lo:
            continue
        if a.client and r.get("client") != a.client:
            continue
        if not (r.get("route") == "local" and r.get("status") == 200 and r.get("cached_actual") is not None):
            continue
        local_n += 1
        p, ca = int(r["ptok"]), int(r["cached_actual"])
        comp = max(0, p - ca); computed_all += comp
        gi = gen_of(gens, starts, r["t0"])
        B = gens[gi][1] if gi >= 0 else 3568
        k = (r.get("client"), r.get("ip"))
        cands = [q for q in fin.get(k, []) if r["t0"] - q["t"] <= a.gap and q["ptok"] <= p
                 and p - q["ptok"] <= a.maxgrow and q["ptok"] > 2 * B]
        c = r.get("client") or "?"
        if not cands:
            tot["cold"] += comp; n["cold"] += 1; per_client[c]["cold"] += comp
            continue
        if a.strict and len({x["ptok"] // B for x in cands}) > 1 and not (cands and max(cands, key=lambda x: x["t"])["ptok"] == max(x["ptok"] for x in cands)):
            tot["ambiguous"] += comp; n["ambiguous"] += 1
            continue
        q = max(cands, key=lambda q: (q["ptok"], q["t"]))
        pp = int(q["ptok"])
        E = ((pp - 4) // B) * B
        new = p - pp
        lost = max(0, E - ca)
        # structural tail: the predecessor's prompt tokens past its last block boundary
        tail = max(0, min(pp, p) - max(ca, E)) if ca >= E - B // 4 else max(0, pp - E)
        if q.get("route") != "local":
            cause = "b_remote"
        elif gen_of(gens, starts, q["t0"]) != gi:
            cause = "e_restart"
        elif ca >= E - B // 4:
            cause = "ok"
        elif ca >= E - B - B // 4:
            cause = "f_one_block_short"   # pred's last boundary state not published (align split / MTP back-off)
        else:
            pm = r.get("pm_credit")
            if pm is not None and pm >= max(B, 0.5 * E) and pm > ca + B:
                cause = "a_evict"
            elif pm is None:
                cause = "a_or_c_unknown"
            else:
                cause = "c_diverge"
        if cause == "ok":
            lost = 0
        tot[cause] += lost; n[cause] += 1; per_client[c][cause] += lost
        tot["new"] += new; tot["d_tail"] += tail; tail_n += 1
        per_client[c]["d_tail"] += tail; per_client[c]["new"] += new
        if a.pairs_out:
            pairs.append({"t0": r["t0"], "client": c, "p": p, "pp": pp, "ca": ca, "E": E, "B": B, "cause": cause,
                          "pm": r.get("pm_credit"), "q_route": q.get("route"), "gap": r["t0"] - q["t"],
                          "sha": r.get("request_body_sha256"), "q_sha": q.get("request_body_sha256"),
                          "prev": (r.get("preview") or "")[:40], "gen": gi, "q_gen": gen_of(gens, starts, q["t0"])})
    lost_total = sum(v for k2, v in tot.items() if k2 not in ("cold", "new", "d_tail", "ambiguous"))
    out = {"window": [a.since, a.until], "local_requests": local_n, "computed_M": round(computed_all / 1e6, 2),
           "continuations": tail_n, "cold_or_first_turn_M": round(tot["cold"] / 1e6, 2),
           "new_tokens_in_continuations_M": round(tot["new"] / 1e6, 2),
           "lost_M": round(lost_total / 1e6, 2), "lost_pct_of_computed": round(100 * lost_total / max(1, computed_all), 1),
           "d_tail_M": round(tot["d_tail"] / 1e6, 2), "d_tail_pct": round(100 * tot["d_tail"] / max(1, computed_all), 1),
           "by_cause_M": {k2: round(v / 1e6, 3) for k2, v in tot.items() if k2 not in ("cold", "new", "d_tail", "ambiguous")},
           "by_cause_n": dict(n),
           "per_client_M": {c: {k2: round(v / 1e6, 3) for k2, v in cc.items()} for c, cc in
                            sorted(per_client.items(), key=lambda kv: -sum(kv[1].values()))[:8]}}
    if a.pairs_out:
        with open(a.pairs_out, "w") as f:
            for x in pairs:
                f.write(json.dumps(x) + "\n")
    print(json.dumps(out, indent=None if a.json else 1))


if __name__ == "__main__":
    main()
