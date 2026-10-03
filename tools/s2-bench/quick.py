#!/usr/bin/env python3
"""quick decode probe on :8001: natural-text 400-token decode x3 (temp 0), ref-lane-like ' the' decode x2; prints tok/s and spec acceptance."""
import json,http.client,time,sys
def req(path,body):
    c=http.client.HTTPConnection("127.0.0.1",8001,timeout=600); t0=time.time()
    c.request("POST",path,json.dumps(body),{"Content-Type":"application/json"}); r=json.loads(c.getresponse().read()); return r,time.time()-t0
def met():
    c=http.client.HTTPConnection("127.0.0.1",8001,timeout=10); c.request("GET","/metrics"); t=c.getresponse().read().decode()
    g=lambda k: sum(float(x.split()[-1]) for x in t.splitlines() if x.startswith(k) and not x.startswith("#"))
    return g("vllm:spec_decode_num_accepted_tokens_total"),g("vllm:spec_decode_num_draft_tokens_total")
m=json.loads(http.client.HTTPConnection("127.0.0.1",8001).request("GET","/v1/models") or "{}") if False else None
c=http.client.HTTPConnection("127.0.0.1",8001,timeout=10); c.request("GET","/v1/models"); MODEL=json.loads(c.getresponse().read())["data"][0]["id"]
res={}
for w in range(3):
    req('/v1/chat/completions',dict(model=MODEL,messages=[{'role':'user','content':f'Warm-up {w}: explain how a bicycle works in 150 words.'}],max_tokens=150,min_tokens=150,temperature=0,chat_template_kwargs={'enable_thinking':False}))
nat=[]; a0=met(); accl=[]
for i in range(3):
    r,dt=req("/v1/chat/completions",dict(model=MODEL,messages=[{"role":"user","content":f"Write a detailed essay about the history of topic number {i}: the printing press, with numbered sections."}],max_tokens=400,min_tokens=400,temperature=0,chat_template_kwargs={"enable_thinking":False}))
    n=r["usage"]["completion_tokens"]; nat.append(n/dt)
    sd=((r.get('metrics') or {}).get('speculative_decoding') or {})
    if sd: accl.append((round(sd.get('mean_acceptance_length',0),2),round(sd.get('draft_acceptance_rate',0),3)))
a1=met(); res["per_req_accept_len_rate"]=accl; res["natural_tok_s"]=[round(x,1) for x in nat]; res["natural_accept"]=round((a1[0]-a0[0])/max(a1[1]-a0[1],1),3)
rf=[]
for i in range(2):
    r,dt=req("/v1/completions",dict(model=MODEL,prompt=("the "*1500)+str(i),max_tokens=256,temperature=0,ignore_eos=True))
    rf.append(r.get("usage",{}).get("completion_tokens",0)/dt)
res["the_lane_tok_s"]=[round(x,1) for x in rf]
print(json.dumps(res))
