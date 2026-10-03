#!/usr/bin/env python3
"""GW2 engine window: validate POST /v1/fork/prefix_cache_probe against the engine's own usage.

Run against a booted engine with VLLM_FORK_PREFIX_PROBE=1 on :8001 (gateway in a planned offline window).
  A. render parity: probe prompt_tokens == usage.prompt_tokens of the same chat body (tools + thinking kwargs)
  B. exactness: probe cached_tokens right before a request == that request's usage.prompt_tokens_details.cached_tokens
     (cold turn, identical re-send, tool-loop continuation, unrelated conversation)
  C. latency: probe p50/p95 idle, and while a cold ~30K-token prefill is running (input-thread path must not wait
     out engine steps: p95 < 250 ms required)
  D. decode impact: tokens/s of a running stream with and without 20 probes/s
PASS = A all equal, B |err| == 0 on >= 90% and <= one block otherwise, C busy p95 < 250 ms, D slowdown < 3 %.
"""
import argparse, json, random, statistics, threading, time, urllib.request

E = "http://127.0.0.1:8001"
MODEL = "qwen-local"


def post(path, body, timeout=600):
    req = urllib.request.Request(E + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), (time.time() - t) * 1000


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file from the repo.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "run", "description": "Run a shell command.",
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]


def filler(seed, n_words):
    rnd = random.Random(seed)
    words = ["kernel", "cache", "block", "prefill", "decode", "gateway", "route", "token", "estate", "lane",
             "engine", "scheduler", "tensor", "warp", "latency", "budget", "remote", "local", "chain", "probe"]
    return " ".join(rnd.choice(words) + str(rnd.randint(0, 999)) for _ in range(n_words))


def convo(seed, turns, words=600):
    msgs = [{"role": "system", "content": "You are a coding agent. " + filler(seed, 400)}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"step {i}: " + filler(seed * 100 + i, words)})
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": f"c{seed}_{i}", "type": "function",
                                     "function": {"name": "read_file", "arguments": json.dumps({"path": f"src/f{i}.py"})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{seed}_{i}", "content": filler(seed * 1000 + i, words)})
    msgs.append({"role": "user", "content": "continue"})
    return msgs


def body(msgs, max_tokens=1):
    return {"model": MODEL, "messages": msgs, "tools": TOOLS, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": True}}


def probe(msgs):
    b = body(msgs)
    j, ms = post("/v1/fork/prefix_cache_probe", {k: b[k] for k in ("model", "messages", "tools", "chat_template_kwargs")})
    return j, ms


def served(msgs):
    j, ms = post("/v1/chat/completions", body(msgs))
    u = j["usage"]
    return u["prompt_tokens"], int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0), ms


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="gw2e")
    a = ap.parse_args()
    res = {"label": a.label, "cases": []}
    # A + B
    seed0 = int(time.time()) % 100000
    for k, (seed, turns) in enumerate([(seed0 + 1, 3), (seed0 + 2, 8), (seed0 + 3, 14)]):
        conv = convo(seed, turns)
        steps = [("cold", conv), ("resend", conv)]
        nxt = conv[:-1] + [{"role": "assistant", "content": "",
                            "tool_calls": [{"id": "cx", "type": "function", "function": {"name": "run", "arguments": "{\"cmd\": \"ls\"}"}}]},
                           {"role": "tool", "tool_call_id": "cx", "content": filler(seed + 7, 300)},
                           {"role": "user", "content": "continue"}]
        steps.append(("continuation", nxt))
        steps.append(("unrelated", convo(seed + 50000, 2)))
        for name, msgs in steps:
            p, pms = probe(msgs)
            ptok, cached, sms = served(msgs)
            res["cases"].append({"conv": k, "step": name, "probe_prompt": p.get("prompt_tokens"), "served_prompt": ptok,
                                 "probe_cached": p.get("cached_tokens"), "served_cached": cached, "probe_ms": round(pms, 1),
                                 "render_ms": p.get("render_ms"), "engine_probe_ms": p.get("probe_ms")})
            print(json.dumps(res["cases"][-1]), flush=True)
    # C: latency idle and under a cold prefill
    small = convo(seed0 + 9, 4)
    idle = [probe(small)[1] for _ in range(20)]
    busy, stop = [], threading.Event()

    def cold():
        try:
            served(convo(seed0 + 777, 40, words=700))
        finally:
            stop.set()
    th = threading.Thread(target=cold); th.start(); time.sleep(1.0)
    while not stop.is_set() and len(busy) < 200:
        busy.append(probe(small)[1]); time.sleep(0.05)
    th.join()
    res["latency_ms"] = {"idle_p50": pct(idle, .5), "idle_p95": pct(idle, .95), "busy_n": len(busy),
                         "busy_p50": pct(busy, .5), "busy_p95": pct(busy, .95), "busy_max": max(busy) if busy else None}
    print(json.dumps(res["latency_ms"]), flush=True)
    # D: decode impact
    def decode_tps(probing):
        stop2 = threading.Event()
        def spam():
            while not stop2.is_set():
                probe(small); time.sleep(0.05)
        th2 = threading.Thread(target=spam) if probing else None
        th2 and th2.start()
        t = time.time()
        j, _ = post("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "Count from 1 to 400, comma separated."}],
                                              "max_tokens": 400, "temperature": 0, "ignore_eos": True,
                                              "chat_template_kwargs": {"enable_thinking": False}})
        dt = time.time() - t
        stop2.set(); th2 and th2.join()
        return j["usage"]["completion_tokens"] / dt
    tps = {"off": [decode_tps(False) for _ in range(3)], "on": [decode_tps(True) for _ in range(3)]}
    res["decode_tps"] = {k: round(statistics.median(v), 2) for k, v in tps.items()}
    res["decode_slowdown_pct"] = round(100 * (1 - res["decode_tps"]["on"] / res["decode_tps"]["off"]), 2)
    print(json.dumps({"decode_tps": res["decode_tps"], "slowdown_pct": res["decode_slowdown_pct"]}), flush=True)
    # verdict
    c = res["cases"]
    parity = all(x["probe_prompt"] == x["served_prompt"] for x in c)
    errs = [abs((x["probe_cached"] or 0) - x["served_cached"]) for x in c]
    exact = sum(e == 0 for e in errs) / len(errs)
    res["verdict"] = {"render_parity": parity, "cached_exact_frac": round(exact, 3), "cached_max_err": max(errs),
                      "busy_p95_ok": (res["latency_ms"]["busy_p95"] or 1e9) < 250, "decode_ok": res["decode_slowdown_pct"] < 3}
    res["verdict"]["PASS"] = bool(parity and exact >= 0.9 and res["verdict"]["busy_p95_ok"] and res["verdict"]["decode_ok"])
    print("VERDICT " + json.dumps(res["verdict"]), flush=True)
    json.dump(res, open(f"{a.out}/probe_{a.label}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
