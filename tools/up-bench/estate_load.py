#!/usr/bin/env python3
"""estate_load.py [--n 12] [--max-tokens 256] -- replay recent flight-recorded real client bodies (Halo/pi/Hermes) concurrently against :8001.
Reports per-request TTFT, decode tok/s and aggregate output tok/s. Bodies are used as recorded (tools, system prompts) except stream/max_tokens/model."""
import argparse, glob, json, os, threading, time, http.client, statistics
ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=12); ap.add_argument("--max-tokens",type=int,default=256); ap.add_argument("--min-tok",type=int,default=8000); ap.add_argument("--max-tok",type=int,default=60000); ap.add_argument("--out"); a=ap.parse_args()
FR=os.path.expanduser("~/.local/share/vllm-qwen27b/flightrec")
c=http.client.HTTPConnection("127.0.0.1",8001,timeout=10); c.request("GET","/v1/models"); MODEL=json.loads(c.getresponse().read())["data"][0]["id"]
files=[]
for p in sorted(glob.glob(FR+"/*.json"),reverse=True):
    try: tok=int(os.path.basename(p).split("_")[1].replace("tok.json",""))
    except Exception: continue
    if a.min_tok<=tok<=a.max_tok: files.append((tok,p))
import random; random.Random(3).shuffle(files); files=files[:a.n]
res=[]; lock=threading.Lock()
def one(tok,p):
    body=json.load(open(p))
    for k in ("max_completion_tokens","stream_options","store","thinking_token_budget","reasoning_effort","n","logprobs","top_logprobs"): body.pop(k,None)
    body.update(model=MODEL,max_tokens=a.max_tokens,stream=True,stream_options={"include_usage":True},temperature=0)
    t0=time.time(); ttft=None; ct=None; pt=None; cached=None; err=None
    try:
        cn=http.client.HTTPConnection("127.0.0.1",8001,timeout=1800)
        cn.request("POST","/v1/chat/completions",json.dumps(body),{"Content-Type":"application/json","X-Client":"estate-load-up"})
        r=cn.getresponse(); buf=b""
        if r.status!=200: err=f"http {r.status}"
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
                        if d.get("usage"): ct=d["usage"].get("completion_tokens"); pt=d["usage"].get("prompt_tokens"); cached=(d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
                        for chn in d.get("choices",[]):
                            dl=chn.get("delta",{})
                            if ttft is None and (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning") or dl.get("tool_calls")): ttft=time.time()-t0; tf=time.time()
    except Exception as e: err=repr(e)
    t1=time.time()
    dec=((ct-1)/(t1-tf)) if (ct and ct>1 and ttft is not None and t1>tf) else None
    with lock: res.append(dict(tok=tok,prompt=pt,cached=cached,ct=ct,ttft=round(ttft,2) if ttft else None,dec_tps=round(dec,1) if dec else None,total=round(t1-t0,1),err=err))
W0=time.time(); th=[threading.Thread(target=one,args=f) for f in files]
[t.start() for t in th]; [t.join() for t in th]; wall=time.time()-W0
ok=[r for r in res if not r["err"] and r["ct"]]
summ=dict(n=len(res),ok=len(ok),wall_s=round(wall,1),agg_out_tok_s=round(sum(r["ct"] for r in ok)/wall,1),total_prompt_tok=sum(r["prompt"] or 0 for r in ok),
          med_dec_tps=round(statistics.median([r["dec_tps"] for r in ok if r["dec_tps"]]),1) if ok else None,
          med_ttft=round(statistics.median([r["ttft"] for r in ok if r["ttft"]]),1) if ok else None, errors=[r["err"] for r in res if r["err"]][:3])
print(json.dumps(summ)); 
if a.out: json.dump(dict(summary=summ,reqs=res),open(a.out,"w"),indent=1)
