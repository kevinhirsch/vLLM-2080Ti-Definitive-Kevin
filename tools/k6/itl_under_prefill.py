#!/usr/bin/env python3
"""itl_under_prefill.py [--port 8001] [--prefill 24000] -- what a decoding stream feels when a cold prefill
arrives (the P/D-interference number). One streaming decode (600 tokens, temp 0); after its first 60 tokens a
cold prefill of --prefill tokens (max_tokens=1) is submitted. Reports chunk-gap ITL before/during/after the
prefill and decode tok/s in each phase. One JSON line."""
import argparse, json, http.client, random, threading, time
from prefill_probe import model, text
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8001); ap.add_argument("--prefill", type=int, default=24000)
    a = ap.parse_args(); M = model(a.port); rng = random.Random(time.time_ns())
    marks = {}; stamps = []  # (t, ntok)
    def prefill():
        p = f"[salt {rng.getrandbits(64):x}] " + text(a.prefill, rng)
        c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=1800); marks["pf0"] = time.time()
        c.request("POST", "/v1/completions", json.dumps(dict(model=M, prompt=p, max_tokens=1, temperature=0)), {"Content-Type": "application/json"})
        r = json.loads(c.getresponse().read()); marks["pf1"] = time.time(); marks["pf_tokens"] = r.get("usage", {}).get("prompt_tokens")
    c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=1800)
    body = dict(model=M, messages=[{"role": "user", "content": f"Write a long detailed essay (topic {rng.randint(0,10**6)}) about lighthouses, with numbered sections."}],
                max_tokens=600, min_tokens=600, temperature=0, stream=True, chat_template_kwargs={"enable_thinking": False})
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = c.getresponse(); n = 0; th = None
    while True:
        line = resp.readline()
        if not line: break
        line = line.strip()
        if not line.startswith(b"data:"): continue
        d = line[5:].strip()
        if d == b"[DONE]": break
        try: j = json.loads(d)
        except Exception: continue
        ch = j.get("choices") or [{}]
        if (ch[0].get("delta") or {}).get("content"):
            n += 1; stamps.append(time.time())
            if n == 60 and th is None:
                th = threading.Thread(target=prefill); th.start()
    if th: th.join()
    pf0, pf1 = marks.get("pf0"), marks.get("pf1")
    def phase(lo, hi):
        s = [t for t in stamps if lo <= t < hi]
        gaps = [b - a for a, b in zip(s, s[1:])]
        return dict(chunks=len(s), max_gap_s=round(max(gaps), 3) if gaps else None,
                    chunk_rate_s=round(len(s) / (hi - lo), 1) if hi > lo else None)
    print(json.dumps(dict(port=a.port, prefill_tokens=marks.get("pf_tokens"), prefill_wall=round(pf1 - pf0, 2) if pf0 and pf1 else None,
                          before=phase(stamps[0], pf0) if pf0 else None, during=phase(pf0, pf1) if pf0 else None,
                          after=phase(pf1, stamps[-1] + 1e-6) if pf1 else None,
                          note="chunk = one streamed delta (MTP may pack several tokens)")))
if __name__ == "__main__":
    main()
