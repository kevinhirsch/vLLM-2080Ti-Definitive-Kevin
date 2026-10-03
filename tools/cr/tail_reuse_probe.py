#!/usr/bin/env python3
"""Lane CR window probe: correctness + reuse of prompt-tail prefix caching on the live engine.

Shape (after weicj #240's heavy repro): S growing sessions (default 20K/35K/50K tokens of
numbered facts) x R rounds, run concurrently. Every round appends a tool-result-like message
carrying a fresh MARKER and asks for (a) that marker and (b) a needle fact from deep in the
base. Then a sibling burst: K concurrent requests that share one session's prefix up to a
mid point and differ only in the last message (concurrent partial-tail hits).

Per request it records prompt_tokens and cached_tokens from usage, and checks the answer.
A continuation is "tail-recovered" when cached > floor(prev_prompt/B)*B, and "one-block-short"
when cached < floor(prev_prompt/B)*B - B/4 (cause f).
usage: tail_reuse_probe.py --label base|cr [--url http://127.0.0.1:8001] [--out DIR]
"""
import argparse, concurrent.futures as cf, json, os, random, re, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--url", default="http://127.0.0.1:8001")
ap.add_argument("--model", default="estate-hauhaucs")
ap.add_argument("--sizes", default="20000,35000,50000")
ap.add_argument("--rounds", type=int, default=12)
ap.add_argument("--siblings", type=int, default=4)
ap.add_argument("--block", type=int, default=1856)
ap.add_argument("--out", default="/home/kevin/projects/lanes/cr/win")
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


def call(messages, max_tokens=48):
    body = json.dumps({"model": a.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(a.url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        j = json.load(resp)
    u = j.get("usage") or {}
    return {"text": j["choices"][0]["message"].get("content") or "", "ptok": u.get("prompt_tokens"),
            "cached": ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0), "s": round(time.time() - t0, 2)}


def session(si, size):
    r = random.Random(1000 + si)
    text, facts = base_text(si, size)
    msgs = [{"role": "system", "content": "You are a precise records clerk. Answer with the exact codes only."},
            {"role": "user", "content": "Here is the locker registry.\n\n" + text + "\nAcknowledge with OK."}]
    out, prev = [], None
    res = call(msgs, 8); msgs.append({"role": "assistant", "content": res["text"]})
    prev = res["ptok"]
    for rd in range(a.rounds):
        marker = f"MK{si}{rd}-{r.randrange(100000, 999999)}"
        k = r.randrange(1, max(2, len(facts) // 3))      # deep needle
        msgs.append({"role": "user", "content": f"<tool_result>scan {rd} complete; marker {marker}; "
                     f"{'filler ' * r.randrange(20, 400)}</tool_result>\nReply exactly: '<marker> <code of locker {k}>'"})
        res = call(msgs)
        ok_marker = marker in res["text"]; ok_needle = facts[k] in res["text"]
        E = (prev - 4) // a.block * a.block if prev else 0
        out.append({"session": si, "round": rd, "ptok": res["ptok"], "cached": res["cached"], "prev": prev, "E": E,
                    "tail_recovered": res["cached"] > E, "one_block_short": res["cached"] < E - a.block // 4,
                    "ok_marker": ok_marker, "ok_needle": ok_needle, "s": res["s"], "text": res["text"][:80]})
        msgs.append({"role": "assistant", "content": res["text"]})
        prev = res["ptok"]
    return out, msgs, facts


t0 = time.time()
sizes = [int(x) for x in a.sizes.split(",")]
with cf.ThreadPoolExecutor(len(sizes)) as ex:
    results = list(ex.map(lambda p: session(*p), enumerate(sizes)))
rows = [x for o, _, _ in results for x in o]
# sibling burst on the largest session: concurrent requests sharing its prefix minus the last 3 messages
_, msgs, facts = results[-1]
shared = msgs[:-3]
sib = []
def sibling(j):
    r = random.Random(77 + j); k = r.randrange(1, len(facts))
    res = call(shared + [{"role": "user", "content": f"Probe {j}: reply with only the code of locker {k}."}])
    return {"sibling": j, "ptok": res["ptok"], "cached": res["cached"], "ok_needle": facts[k] in res["text"], "s": res["s"]}
with cf.ThreadPoolExecutor(a.siblings) as ex:
    sib = list(ex.map(sibling, range(a.siblings)))
# replay the sibling burst once more (all should now hit deep)
with cf.ThreadPoolExecutor(a.siblings) as ex:
    sib2 = list(ex.map(sibling, range(a.siblings)))
n = len(rows)
summary = {"label": a.label, "requests": n + 2 * len(sib), "wall_s": round(time.time() - t0, 1),
           "marker_ok": sum(x["ok_marker"] for x in rows), "needle_ok": sum(x["ok_needle"] for x in rows), "of": n,
           "sibling_ok": sum(x["ok_needle"] for x in sib + sib2), "sibling_of": 2 * len(sib),
           "tail_recovered": sum(x["tail_recovered"] for x in rows), "one_block_short": sum(x["one_block_short"] for x in rows),
           "computed_tokens": sum(x["ptok"] - x["cached"] for x in rows),
           "cached_tokens": sum(x["cached"] for x in rows),
           "mean_s": round(sum(x["s"] for x in rows) / max(1, n), 2)}
json.dump({"summary": summary, "rows": rows, "siblings": sib + sib2}, open(f"{a.out}/probe_{a.label}.json", "w"), indent=1)
print(json.dumps(summary))
