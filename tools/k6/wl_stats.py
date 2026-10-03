import json,sys,statistics as st,glob
rows=[]
for f in sys.argv[1:]:
    for l in open(f):
        try: r=json.loads(l)
        except: continue
        if r.get('route')!='local' or r.get('status')!=200: continue
        rows.append(r)
rows.sort(key=lambda r:r['t'])
t0,t1=rows[0]['t'],rows[-1]['t']; span=t1-t0
def q(xs,p): xs=sorted(xs); return xs[min(len(xs)-1,int(p*len(xs)))] if xs else None
n=len(rows)
comp=[r['computed_actual'] for r in rows if r.get('computed_actual') is not None]
ptok=[r['ptok'] for r in rows if r.get('ptok')]
out=[r['outtok'] for r in rows if r.get('outtok') is not None]
ttft=[r['ttft'] for r in rows if r.get('ttft') is not None]
tpot=[r['decode_time']/max(1,r['outtok']-1) for r in rows if r.get('decode_time') and (r.get('outtok') or 0)>20]
print(f"span {span/3600:.2f} h, local 200 requests {n}, rate {n/span*60:.2f}/min, with computed_actual {len(comp)}")
print("ptok mean %.0f p50 %d p90 %d"%(st.mean(ptok),q(ptok,.5),q(ptok,.9)))
if comp: print("computed mean %.0f p50 %d p90 %d  sum %.2fM -> %.0f tok/s avg (scaled to all reqs %.0f)"%(st.mean(comp),q(comp,.5),q(comp,.9),sum(comp)/1e6,sum(comp)/span, st.mean(comp)*n/span))
print("out mean %.0f p50 %d p90 %d -> %.1f tok/s avg"%(st.mean(out),q(out,.5),q(out,.9),sum(out)/span))
print("ttft p50 %.2f p90 %.2f p99 %.2f mean %.2f"%(q(ttft,.5),q(ttft,.9),q(ttft,.99),st.mean(ttft)))
print("eff TPOT (decode_time/out) p50 %.3f p90 %.3f mean %.3f -> per-stream tok/s p50 %.1f"%(q(tpot,.5),q(tpot,.9),st.mean(tpot),1/q(tpot,.5)))
# concurrency from intervals: start = t - duration
ev=[]
for r in rows:
    d=r.get('duration') or 0; s=r['t']-d
    ev.append((s,1)); ev.append((r['t'],-1))
ev.sort(); c=0; last=ev[0][0]; area=0; hist={}
for t,dlt in ev:
    hist[c]=hist.get(c,0)+(t-last); area+=c*(t-last); c+=dlt; last=t
tot=sum(hist.values())
print("mean in-flight %.2f; time share by in-flight:"%(area/tot), {k:round(v/tot,3) for k,v in sorted(hist.items()) if v/tot>0.005})
by={}
for r in rows: by.setdefault(r.get('client'),[0,0,0]); b=by[r.get('client')]; b[0]+=1; b[1]+=r.get('computed_actual') or 0; b[2]+=r.get('outtok') or 0
print("by client (n, computed, out):", sorted(by.items(), key=lambda x:-x[1][0])[:8])
