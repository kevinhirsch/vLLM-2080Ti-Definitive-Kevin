#!/usr/bin/env python3
"""conc_short.py N [max_tokens] -- N concurrent short-context natural-text decodes; reports per-req tok/s and aggregate."""
import sys,json,http.client,threading,time
N=int(sys.argv[1]); MT=int(sys.argv[2]) if len(sys.argv)>2 else 300
c=http.client.HTTPConnection("127.0.0.1",8001,timeout=10); c.request("GET","/v1/models"); M=json.loads(c.getresponse().read())["data"][0]["id"]
out=[]
def one(i):
    t0=time.time(); c=http.client.HTTPConnection("127.0.0.1",8001,timeout=900)
    c.request("POST","/v1/chat/completions",json.dumps(dict(model=M,messages=[{"role":"user","content":f"Write a detailed essay about the history of topic number {i+int(t0)%1000}: the steam engine, with numbered sections."}],max_tokens=MT,min_tokens=MT,temperature=0,chat_template_kwargs={"enable_thinking":False})),{"Content-Type":"application/json"})
    r=json.loads(c.getresponse().read()); out.append((r["usage"]["completion_tokens"],time.time()-t0))
t0=time.time(); th=[threading.Thread(target=one,args=(i,)) for i in range(N)]; [t.start() for t in th]; [t.join() for t in th]; w=time.time()-t0
print(json.dumps(dict(N=N,wall=round(w,1),agg=round(sum(o[0] for o in out)/w,1),per_req=round(sum(o[0]/o[1] for o in out)/N,1))))
