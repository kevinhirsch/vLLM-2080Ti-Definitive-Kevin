#!/usr/bin/env python3
"""Single-stream engine bench (direct :8001), temp 0, N reps, reports spread.
Per rep: short TTFT (cold unique ~300tok), decode tok/s (512 out), long cold TTFT (unique ~LONG tok) + needle,
repeat-prompt TTFT (prefix-cache hit), spec acceptance from /metrics delta."""
import argparse, json, random, re, statistics, time, http.client, sys
HOST=("127.0.0.1",8001)
W=("amber basalt cedar delta ember fjord garnet harbor indigo jasper kelp lumen marble nectar onyx pewter quartz russet sable tundra umber velvet willow xenon yarrow zephyr").split()
def para(r,n): return " ".join(r.choice(W)+str(r.randint(0,999)) for _ in range(n))
def stream(body):
    c=http.client.HTTPConnection(*HOST,timeout=900); t0=time.time()
    c.request("POST","/v1/chat/completions",json.dumps(body),{"Content-Type":"application/json","X-Client":"bench-up"})
    r=c.getresponse(); ttft=None; n=0; text=""; buf=b""; tlast=t0; usage=None
    while True:
        ch=r.read1(65536)
        if not ch: break
        buf+=ch
        while b"\n\n" in buf:
            ev,buf=buf.split(b"\n\n",1)
            for ln in ev.split(b"\n"):
                if ln.startswith(b"data: ") and ln!=b"data: [DONE]":
                    try: d=json.loads(ln[6:])
                    except Exception: continue
                    if d.get("usage"): usage=d["usage"]
                    for chn in d.get("choices",[]):
                        dl=chn.get("delta",{}); s=(dl.get("content") or "")+(dl.get("reasoning_content") or dl.get("reasoning") or "")
                        if s:
                            if ttft is None: ttft=time.time()-t0; tfirst=time.time()
                            n+=1; text+=dl.get("content") or ""; tlast=time.time()
    c.close()
    ct=(usage or {}).get('completion_tokens') or n
    dec=(ct-1)/(tlast-tfirst) if ttft is not None and ct>1 and tlast>tfirst else None
    return dict(ttft=ttft,chunks=n,dec_tps=dec,text=text,usage=usage)
def metrics():
    c=http.client.HTTPConnection(*HOST,timeout=10); c.request("GET","/metrics"); t=c.getresponse().read().decode()
    g=lambda k: sum(float(x.split()[-1]) for x in t.splitlines() if x.startswith(k) and not x.startswith("#"))
    return g("vllm:spec_decode_num_accepted_tokens_total"),g("vllm:spec_decode_num_draft_tokens_total"),g("vllm:prefix_cache_hits_total"),g("vllm:prefix_cache_queries_total")
def model():
    c=http.client.HTTPConnection(*HOST,timeout=10); c.request("GET","/v1/models"); return json.loads(c.getresponse().read())["data"][0]["id"]
def req(m,content,max_tokens,**kw):
    return dict(model=m,messages=[{"role":"user","content":content}],max_tokens=max_tokens,temperature=0,stream=True,
                stream_options={"include_usage":True},chat_template_kwargs={"enable_thinking":False},**kw)
def spread(v): v=[x for x in v if x is not None]; return dict(n=len(v),med=round(statistics.median(v),3),min=round(min(v),3),max=round(max(v),3)) if v else None
ap=argparse.ArgumentParser(); ap.add_argument("--reps",type=int,default=3); ap.add_argument("--long",type=int,default=20000); ap.add_argument("--seed",type=int,default=7); ap.add_argument("--out"); a=ap.parse_args()
m=model(); out=dict(model=m,reps=[])
# warm
stream(req(m,"Say hi.",16))
for i in range(a.reps):
    r=random.Random(a.seed*100+i); rep={}
    m0=metrics()
    s=stream(req(m,para(r,75)+"\nSummarize the above in one sentence.",64)); rep["short_ttft"]=s["ttft"]
    d=stream(req(m,"Write a detailed essay about the history of the printing press, with numbered sections.",512,min_tokens=512)); rep["dec_tps"]=d["dec_tps"]; rep["dec_ttft"]=d["ttft"]; rep["dec_chunks"]=d["chunks"]
    code=str(r.randint(100000,999999)); n=a.long//4; pos=r.randint(n//5,4*n//5); body=[]
    for j in range(n):
        if j==pos: body.append(f"IMPORTANT FACT: the secret code is {code}.")
        body.append(para(r,3)+".")
    P=" ".join(body)+"\n\nQuestion: what is the secret code? Answer with the 6-digit number only."
    L=stream(req(m,P,16)); rep["long_ttft"]=L["ttft"]; rep["long_prompt_tokens"]=(L["usage"] or {}).get("prompt_tokens"); rep["needle_ok"]= code in L["text"]
    R=stream(req(m,P,16)); rep["repeat_ttft"]=R["ttft"]; rep["repeat_cached"]=((R["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    m1=metrics(); rep["accept_rate"]= (m1[0]-m0[0])/(m1[1]-m0[1]) if m1[1]>m0[1] else None
    out["reps"].append(rep); print(json.dumps(rep),flush=True)
keys=["short_ttft","dec_tps","long_ttft","repeat_ttft","accept_rate"]
out["summary"]={k:spread([x[k] for x in out["reps"]]) for k in keys}; out["summary"]["needle_ok"]=f"{sum(x['needle_ok'] for x in out['reps'])}/{len(out['reps'])}"
print(json.dumps(out["summary"],indent=1))
if a.out: json.dump(out,open(a.out,"w"),indent=1)
