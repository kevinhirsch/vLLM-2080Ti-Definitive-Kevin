import sys,json
for l in sys.stdin:
    if not l.startswith('{'): print(l.rstrip()[:300]); continue
    d=json.loads(l)
    if 'skipped' in d: print(d); continue
    s=f"Tq{d['Tq']} Tkv{d['Tkv']} H{d['Hq']}/{d['Hk']}: "
    for k,v in d.items():
        if isinstance(v,dict): s+=f"{k.replace('k1_','').replace('_bn16','')} {v['tflops']}TF x{v['speedup']} | "
    print(s)
