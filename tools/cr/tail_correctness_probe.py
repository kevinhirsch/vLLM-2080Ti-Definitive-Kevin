#!/usr/bin/env python3
"""Lane CR GPU correctness gate: tail-reused generations vs cold generations.

Runs the window probe's traffic shape (concurrent growing sessions, a fresh tool-result
tail each round, then sibling bursts) against the live engine. Every warm request (which
reuses the predecessor's prompt-tail state) is paired with a COLD twin: the same messages
with a unique ``cache_salt`` so nothing is reused and the whole prompt is computed from
scratch. Both run at temperature 0 with top-2 logprobs, concurrently with the other
sessions (concurrency + pool turnover is what exposed L98's freed-running-block defect).

Verdict (last stdout line and <out>/tail_correctness.json, exit 0 only on PASS):
  * degenerate == 0      no warm output collapses (8+ identical tokens in a row, '!!!!',
                         or empty) unless its cold twin does the same;
  * hard_div <= 1        a warm output may only differ from its cold twin at a near-tie
                         (cold top-1 vs top-2 logprob gap < --tie-nats at the first
                         differing token); any other divergence is "hard";
  * reuse                at least --min-tail warm requests actually reused a sub-block tail
                         (cached_tokens past the predecessor's last whole block), so the
                         gate exercised the path it guards.
usage: tail_correctness_probe.py --out DIR [--url http://127.0.0.1:8001] [--block 3568]
"""
import argparse, concurrent.futures as cf, json, os, random, sys, time, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8001")
ap.add_argument("--model", default="qwen-local")
ap.add_argument("--sizes", default="20000,35000,50000")
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--siblings", type=int, default=4)
ap.add_argument("--block", type=int, default=3568)
ap.add_argument("--max-tokens", type=int, default=48)
ap.add_argument("--tie-nats", type=float, default=0.5)
ap.add_argument("--max-hard", type=int, default=1)
ap.add_argument("--min-tail", type=int, default=6)
ap.add_argument("--label", default="tailcopy")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
WORDS = ("amber basil cobalt delta ember fjord granite harbor indigo juniper kestrel lumen meadow nectar "
         "onyx pepper quartz raven sierra tundra umber violet willow xenon yarrow zephyr").split()


def base_text(seed, tokens):
    r = random.Random(seed)
    facts, lines, i = {}, [], 0
    while sum(len(x) for x in lines) < tokens * 3.6:
        i += 1
        code = f"{r.choice(WORDS)}-{r.randrange(100000, 999999)}"
        facts[i] = code
        lines.append(f"Fact #{i}: the access code for locker {i} is {code}. "
                     f"It was filed under {r.choice(WORDS)} by the {r.choice(WORDS)} team.\n")
    return "".join(lines), facts


def call(messages, salt=None, max_tokens=None):
    body = {"model": a.model, "messages": messages, "max_tokens": max_tokens or a.max_tokens,
            "temperature": 0, "logprobs": True, "top_logprobs": 2,
            "chat_template_kwargs": {"enable_thinking": False}}
    if salt:
        body["cache_salt"] = salt
    req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        j = json.load(resp)
    ch = j["choices"][0]
    u = j.get("usage") or {}
    lp = ((ch.get("logprobs") or {}).get("content")) or []
    toks = [x.get("token") for x in lp]
    gaps = []
    for x in lp:
        top = x.get("top_logprobs") or []
        gaps.append(top[0]["logprob"] - top[1]["logprob"] if len(top) >= 2 else 99.0)
    return {"text": ch["message"].get("content") or "", "toks": toks, "gaps": gaps,
            "ptok": u.get("prompt_tokens"),
            "cached": ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
            "s": round(time.time() - t0, 2)}


def degenerate(r):
    t = r["toks"]
    if not r["text"].strip() or "!!!!" in r["text"]:
        return True
    run = 1
    for i in range(1, len(t)):
        run = run + 1 if t[i] == t[i - 1] else 1
        if run >= 8:
            return True
    return False


def compare(w, c):
    """-> (kind, index): exact | tie | hard, first differing token index."""
    tw, tc = w["toks"], c["toks"]
    n = min(len(tw), len(tc))
    for i in range(n):
        if tw[i] != tc[i]:
            return ("tie" if c["gaps"][i] < a.tie_nats else "hard"), i
    if len(tw) != len(tc):
        return ("tie" if n < len(c["gaps"]) and c["gaps"][n] < a.tie_nats else "hard"), n
    return "exact", -1


def pair(messages, tag):
    """Warm and cold twins concurrently."""
    with cf.ThreadPoolExecutor(2) as ex:
        fw = ex.submit(call, messages)
        fc = ex.submit(call, messages, f"cold-{tag}-{uuid.uuid4().hex}")
        return fw.result(), fc.result()


def session(si, size):
    r = random.Random(1000 + si)
    text, facts = base_text(si, size)
    msgs = [{"role": "system", "content": "You are a precise records clerk. Answer with the exact codes only."},
            {"role": "user", "content": "Here is the locker registry.\n\n" + text + "\nAcknowledge with OK."}]
    res = call(msgs, max_tokens=8)
    msgs.append({"role": "assistant", "content": res["text"]})
    prev, out = res["ptok"], []
    for rd in range(a.rounds):
        marker = f"MK{si}{rd}-{r.randrange(100000, 999999)}"
        k = r.randrange(1, max(2, len(facts) // 3))
        msgs.append({"role": "user", "content": f"<tool_result>scan {rd} complete; marker {marker}; "
                     f"{'filler ' * r.randrange(20, 400)}</tool_result>\nReply exactly: '<marker> <code of locker {k}>'"})
        w, c = pair(msgs, f"s{si}r{rd}")
        kind, idx = compare(w, c)
        E = (prev - 4) // a.block * a.block if prev else 0
        out.append({"case": f"s{si}r{rd}", "ptok": w["ptok"], "cached": w["cached"], "cold_cached": c["cached"],
                    "tail_recovered": w["cached"] > E, "kind": kind, "first_diff": idx,
                    "degenerate": degenerate(w) and not degenerate(c),
                    "warm_ok": marker in w["text"] and facts[k] in w["text"],
                    "cold_ok": marker in c["text"] and facts[k] in c["text"],
                    "warm": w["text"][:80], "cold": c["text"][:80]})
        msgs.append({"role": "assistant", "content": w["text"]})
        prev = w["ptok"]
    return out, msgs, facts


t0 = time.time()
sizes = [int(x) for x in a.sizes.split(",")]
with cf.ThreadPoolExecutor(len(sizes)) as ex:
    results = list(ex.map(lambda p: session(*p), enumerate(sizes)))
rows = [x for o, _, _ in results for x in o]
_, msgs, facts = results[-1]
shared = msgs[:-3]


def sibling(burst, j):
    r = random.Random(77 + j)
    k = r.randrange(1, len(facts))
    m = shared + [{"role": "user", "content": f"Probe {j}: reply with only the code of locker {k}."}]
    w, c = pair(m, f"sib{burst}{j}")
    kind, idx = compare(w, c)
    return {"case": f"sib{burst}{j}", "ptok": w["ptok"], "cached": w["cached"], "cold_cached": c["cached"],
            "tail_recovered": w["cached"] % a.block != 0, "kind": kind, "first_diff": idx,
            "degenerate": degenerate(w) and not degenerate(c),
            "warm_ok": facts[k] in w["text"], "cold_ok": facts[k] in c["text"],
            "warm": w["text"][:80], "cold": c["text"][:80]}


for burst in range(2):
    with cf.ThreadPoolExecutor(a.siblings) as ex:
        rows += list(ex.map(lambda j: sibling(burst, j), range(a.siblings)))

n = len(rows)
summary = {
    "label": a.label, "cases": n, "wall_s": round(time.time() - t0, 1),
    "exact": sum(r["kind"] == "exact" for r in rows),
    "tie_div": sum(r["kind"] == "tie" for r in rows),
    "hard_div": sum(r["kind"] == "hard" for r in rows),
    "degenerate": sum(r["degenerate"] for r in rows),
    "tail_recovered": sum(r["tail_recovered"] for r in rows),
    "cold_leaks": sum(r["cold_cached"] > 0 for r in rows),
    "warm_ok": sum(r["warm_ok"] for r in rows), "cold_ok": sum(r["cold_ok"] for r in rows),
}
reasons = []
if summary["degenerate"]:
    reasons.append(f"{summary['degenerate']} degenerate warm outputs")
if summary["hard_div"] > a.max_hard:
    reasons.append(f"{summary['hard_div']} hard divergences > {a.max_hard}")
if summary["tail_recovered"] < a.min_tail:
    reasons.append(f"only {summary['tail_recovered']} tail reuses < {a.min_tail}")
if summary["warm_ok"] < summary["cold_ok"] - 1:
    reasons.append(f"warm task accuracy {summary['warm_ok']} < cold {summary['cold_ok']} - 1")
summary["verdict"] = "FAIL" if reasons else "PASS"
summary["reasons"] = reasons
json.dump({"summary": summary, "rows": rows}, open(f"{a.out}/tail_correctness_rows.json", "w"), indent=1)
json.dump(summary, open(f"{a.out}/tail_correctness.json", "w"))
for r in rows:
    if r["kind"] != "exact" or r["degenerate"]:
        print(json.dumps(r))
print(json.dumps(summary))
sys.exit(0 if summary["verdict"] == "PASS" else 1)
