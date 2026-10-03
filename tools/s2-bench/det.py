#!/usr/bin/env python3
"""det.py OUT.json  -- greedy single-stream replay of 6 frozen bodies (same selection as estate_load), saves outputs; det.py --cmp A B compares."""
import sys,json,glob,os,random,http.client
if sys.argv[1]=="--cmp":
    a=json.load(open(sys.argv[2])); b=json.load(open(sys.argv[3])); tot=0
    for k in a:
        x,y=a[k],b[k]; n=0
        for i,(c,d) in enumerate(zip(x,y)):
            if c!=d: break
            n=i+1
        else: n=min(len(x),len(y))
        print(k,"len",len(x),len(y),"common_prefix_chars",n, "IDENTICAL" if x==y else "DIVERGES"); tot+= (x==y)
    print("identical",tot,"of",len(a)); sys.exit()
FR=os.environ["ESTATE_FR"]; files=[]
for p in sorted(glob.glob(FR+"/*.json"),reverse=True):
    tok=int(os.path.basename(p).split("_")[1].replace("tok.json",""))
    if 18000<=tok<=36000: files.append(p)
random.Random(3).shuffle(files); files=files[:6]
c=http.client.HTTPConnection("127.0.0.1",8001,timeout=10); c.request("GET","/v1/models"); M=json.loads(c.getresponse().read())["data"][0]["id"]
out={}
for p in files:
    b=json.load(open(p))
    for k in ("max_completion_tokens","stream_options","store","thinking_token_budget","reasoning_effort","n","logprobs","top_logprobs"): b.pop(k,None)
    b.update(model=M,max_tokens=160,stream=False,temperature=0)
    cn=http.client.HTTPConnection("127.0.0.1",8001,timeout=1800); cn.request("POST","/v1/chat/completions",json.dumps(b),{"Content-Type":"application/json","X-Client":"s2-det"})
    r=json.loads(cn.getresponse().read()); m=r["choices"][0]["message"]
    out[os.path.basename(p)]=(m.get("content") or "")+"||"+(m.get("reasoning_content") or m.get("reasoning") or "")+"||"+json.dumps(m.get("tool_calls"))
json.dump(out,open(sys.argv[1],"w"))
print("saved",len(out))
