#!/usr/bin/env python3
"""prefill_probe.py [--port 8001] [--sizes 8000 16000 30000] -- cold (cache-proof, random salt) prefill
rate: prompt_tokens / wall for max_tokens=1. Prints one JSON line."""
import argparse, json, http.client, random, time
WORDS = ("alpha river stone market garden window signal rocket orange silver planet engine forest bridge "
         "candle shadow winter summer copper harbor violet thunder meadow castle anchor pepper lantern "
         "falcon marble velvet canyon island prism quartz saddle timber walnut zephyr ember glacier").split()
def model(port):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10); c.request("GET", "/v1/models")
    return json.loads(c.getresponse().read())["data"][0]["id"]
def text(n, rng):
    return " ".join(rng.choice(WORDS) + str(rng.randint(0, 99)) for _ in range(n // 3))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8000, 16000, 30000]); ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args(); M = model(a.port); rng = random.Random(time.time_ns()); out = []
    for n in a.sizes:
        for _ in range(a.reps):
            p = f"[salt {rng.getrandbits(64):x}] " + text(n, rng)
            c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=1800); t0 = time.time()
            c.request("POST", "/v1/completions", json.dumps(dict(model=M, prompt=p, max_tokens=1, temperature=0)),
                      {"Content-Type": "application/json"})
            r = json.loads(c.getresponse().read()); dt = time.time() - t0
            pt = r.get("usage", {}).get("prompt_tokens", 0)
            out.append(dict(target=n, prompt_tokens=pt, wall=round(dt, 2), tok_s=round(pt / dt, 1)))
    print(json.dumps(dict(port=a.port, prefill=out)))
if __name__ == "__main__":
    main()
