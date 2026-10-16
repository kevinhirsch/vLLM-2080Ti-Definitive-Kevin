#!/usr/bin/env python3
"""Minimal GGUF v3 header/tensor-table reader (handles types gguf-py lacks, e.g. Q2_0=42). Header only, no tensor data read."""
import struct, sys, json, collections
S = {0:('B',1),1:('b',1),2:('H',2),3:('h',2),4:('I',4),5:('i',4),6:('f',4),7:('?',1),10:('Q',8),11:('q',8),12:('d',8)}
def rd(f, fmt): n = struct.calcsize(fmt); return struct.unpack('<'+fmt, f.read(n))[0]
def rstr(f): n = rd(f,'Q'); return f.read(n).decode('utf8','replace')
def rval(f, t, keep=True):
    if t in S: return rd(f, S[t][0])
    if t == 8: return rstr(f)
    if t == 9:
        et = rd(f,'I'); n = rd(f,'Q')
        if et in S and not keep or et==8 and n>64:
            if et in S: f.seek(S[et][1]*n, 1)
            else:
                for _ in range(n): rstr(f)
            return f'<array {n}>'
        return [rval(f, et) for _ in range(n)]
    raise ValueError(t)
# ggml type -> (name, block_elems, block_bytes)
TY = {0:('F32',1,4),1:('F16',1,2),2:('Q4_0',32,18),8:('Q8_0',32,34),10:('Q2_K',256,84),11:('Q3_K',256,110),12:('Q4_K',256,144),13:('Q5_K',256,176),14:('Q6_K',256,210),
      16:('IQ2_XXS',256,66),17:('IQ2_XS',256,74),18:('IQ3_XXS',256,98),19:('IQ1_S',256,50),20:('IQ4_NL',32,18),21:('IQ3_S',256,110),22:('IQ2_S',256,82),23:('IQ4_XS',256,136),
      29:('IQ1_M',256,56),30:('BF16',1,2),42:('Q2_0',64,18)}
def read(path):
    with open(path,'rb') as f:
        assert f.read(4)==b'GGUF'; ver=rd(f,'I'); nt=rd(f,'Q'); nkv=rd(f,'Q')
        kv={}
        for _ in range(nkv):
            k=rstr(f); t=rd(f,'I'); kv[k]=rval(f,t,keep=False)
        ts=[]
        for _ in range(nt):
            name=rstr(f); nd=rd(f,'I'); dims=[rd(f,'Q') for _ in range(nd)]; ty=rd(f,'I'); off=rd(f,'Q')
            nm,be,bb=TY.get(ty,(f'T{ty}',1,0)); n=1
            for d in dims: n*=d
            ts.append(dict(name=name,dims=dims,type=ty,tname=nm,offset=off,bytes=n//be*bb))
        pos=f.tell(); align=kv.get('general.alignment',32); data_start=(pos+align-1)//align*align
    return kv, ts, data_start
if __name__=='__main__':
    out={}
    for p in sys.argv[1:]:
        kv,ts,ds=read(p); out[p]=dict(kv={k:v for k,v in kv.items() if not isinstance(v,list) or len(v)<16},data_start=ds,tensors=ts)
    json.dump(out,open('gguf_index.json','w'))
    allt=[t for v in out.values() for t in v['tensors']]
    agg=collections.defaultdict(lambda:[0,0])
    for t in allt:
        base=t['name']; 
        import re; base=re.sub(r'\.\d+\.','.N.',base)
        a=agg[(base,t['tname'])]; a[0]+=1; a[1]+=t['bytes']
    for (b,tn),(c,by) in sorted(agg.items(), key=lambda x:-x[1][1])[:60]: print(f'{by/2**30:8.2f} GiB  x{c:<3} {tn:8} {b}')
