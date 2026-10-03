#!/usr/bin/env python3
"""Lane CR2 window probe (L101): does a warm continuation overtake a running cold prefill?

Per trial: a warm session (default 24K tokens) has its previous turn cached; a COLD prefill
(default 30K fresh tokens) starts; DELAY s later the session's next turn arrives (previous
prompt + one ~300-token tool result, i.e. ~300-4K uncached tokens). Measures the warm turn's
latency to its first token (max_tokens=1, non-streaming => total == TTFT), its cached tokens,
the cold request's total time, and checks the warm answer (a marker carried in the new message).

usage: warm_overtake_probe.py --label base|cr2 [--url http://127.0.0.1:8001] [--trials 6]
Expected: base -> warm waits out the cold prefill (~cold_tokens/1279 s); cr2 (with
VLLM_SCHED_SHORT_FIRST_PREFIX_AWARE=1) -> warm ~ uncached/1279 s + one step; cold +<=1 step.
"""
import argparse, json, os, random, statistics, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--url", default="http://127.0.0.1:8001")
ap.add_argument("--model", default="estate-hauhaucs")
ap.add_argument("--warm", type=int, default=24000)
ap.add_argument("--cold", type=int, default=30000)
ap.add_argument("--trials", type=int, default=6)
ap.add_argument("--delay", type=float, default=2.0)
ap.add_argument("--block", type=int, default=3568)
ap.add_argument("--out", default="/home/kevin/projects/lanes/cr2/win")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
W = ("amber basil cobalt delta ember fjord granite harbor indigo juniper kestrel lumen meadow nectar "
     "onyx pepper quartz raven sierra tundra umber violet willow xenon yarrow zephyr").split()


def text(seed, tokens):
    r = random.Random(seed)
    out = []
    while sum(len(x) for x in out) < tokens * 3.6:
        out.append(f"Record {len(out)}: {r.choice(W)}-{r.randrange(10**5, 10**6)} filed by {r.choice(W)}.\n")
    return "".join(out)


def call(messages, max_tokens=1):
    body = json.dumps({"model": a.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(a.url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        j = json.load(resp)
    u = j.get("usage") or {}
    return {"text": j["choices"][0]["message"].get("content") or "", "ptok": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0, "s": round(time.time() - t0, 2)}


sys_msg = {"role": "system", "content": "You are a terse clerk. Reply with the marker only."}
hist = [sys_msg, {"role": "user", "content": "Registry:\n" + text(1, a.warm) + "\nReply OK."}]
first = call(hist, 4)
hist.append({"role": "assistant", "content": first["text"] or "OK"})
rows = []
for t in range(a.trials):
    marker = f"M{random.Random(t).randrange(10**6, 10**7)}"
    cold_res = {}
    th = threading.Thread(target=lambda: cold_res.update(call(
        [sys_msg, {"role": "user", "content": text(9000 + t + int(time.time()), a.cold) + "\nReply OK."}], 1)))
    th.start(); time.sleep(a.delay)
    turn = hist + [{"role": "user", "content": text(500 + t, 300) + f"\nThe marker is {marker}. Reply with the marker only."}]
    w = call(turn, 12)
    th.join()
    ok = marker in w["text"]
    rows.append({"trial": t, "warm_s": w["s"], "warm_ptok": w["ptok"], "warm_cached": w["cached"],
                 "warm_uncached": (w["ptok"] or 0) - w["cached"], "marker_ok": ok, "cold_s": cold_res.get("s")})
    hist = turn + [{"role": "assistant", "content": w["text"]}]
    print(json.dumps(rows[-1]), flush=True)
el = [r for r in rows if r["warm_uncached"] <= a.block] or rows   # short by uncached tokens
summ = {"label": a.label, "eligible_trials": len(el), "warm_p50_s": statistics.median(r["warm_s"] for r in el),
        "warm_max_s": max(r["warm_s"] for r in el), "cold_p50_s": statistics.median(r["cold_s"] for r in rows if r["cold_s"]),
        "warm_uncached_p50": statistics.median(r["warm_uncached"] for r in rows),
        "markers_ok": sum(r["marker_ok"] for r in rows), "trials": len(rows)}
print(json.dumps(summ))
json.dump({"summary": summ, "rows": rows}, open(os.path.join(a.out, f"overtake_{a.label}.json"), "w"), indent=1)
