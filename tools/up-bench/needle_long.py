#!/usr/bin/env python3
"""Long-context needle: unique ~N-token filler with a 6-digit code at a given depth fraction, temp 0, direct :8001."""
import argparse, json, random, time, http.client
W=("amber basalt cedar delta ember fjord garnet harbor indigo jasper kelp lumen marble nectar onyx pewter quartz russet sable tundra umber velvet willow xenon yarrow zephyr").split()
ap=argparse.ArgumentParser(); ap.add_argument("--tokens",type=int,default=300000); ap.add_argument("--depth",type=float,default=0.85); ap.add_argument("--seed",type=int,default=11); a=ap.parse_args()
r=random.Random(a.seed); code=str(r.randint(100000,999999)); n=a.tokens//4; pos=int(n*a.depth); body=[]
for i in range(n):
    if i==pos: body.append(f"IMPORTANT FACT: the secret code is {code}.")
    body.append(" ".join(r.choice(W)+str(r.randint(0,999)) for _ in range(3))+".")
P=" ".join(body)+"\n\nQuestion: what is the secret code? Answer with the 6-digit number only."
c=http.client.HTTPConnection("127.0.0.1",8001,timeout=3600); t0=time.time()
c.request("POST","/v1/chat/completions",json.dumps(dict(model="qwen-local",messages=[{"role":"user","content":P}],max_tokens=24,temperature=0,chat_template_kwargs={"enable_thinking":False})),{"Content-Type":"application/json","X-Client":"bench-up"})
d=json.loads(c.getresponse().read()); txt=d["choices"][0]["message"].get("content") or ""
print(json.dumps(dict(tokens=d["usage"]["prompt_tokens"],depth=a.depth,ok=code in txt,answer=txt.strip()[:40],secs=round(time.time()-t0,1))))
