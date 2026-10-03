import re,csv,collections,sys
lines=open('cub/dis.txt').read().split('\n')
cur=None; inK=False; off2line={}
for l in lines:
    if l.startswith('.text.') : inK = 'tq_gqa_stage1' in l
    if not inK: continue
    m=re.search(r'//## File ".*tq_gqa\.cu", line (\d+)',l)
    if m: cur=int(m.group(1)); continue
    m=re.match(r'\s+/\*([0-9a-f]{4})\*/\s+(.*?);',l)
    if m: off2line[int(m.group(1),16)]=(cur,m.group(2))
rows=list(csv.reader(open(sys.argv[1])))
hdr=rows[1]; ie=hdr.index("Instructions Executed"); ad=hdr.index("Address"); ss=hdr.index("Warp Stall Sampling (All Samples)")
base=int(rows[2][ad],16)
byline=collections.Counter(); st=collections.Counter()
for r in rows[2:]:
    try: off=int(r[ad],16)-base; n=int(r[ie]); s=int(r[ss])
    except: continue
    if off in off2line: byline[off2line[off][0]]+=n; st[off2line[off][0]]+=s
tot=sum(byline.values()); ts=sum(st.values()); print('total',tot)
src=open('/home/kevin/Desktop/wt-integrate/vllm/v1/attention/ops/tq_gqa_ext/tq_gqa.cu').read().split('\n')
for ln,n in byline.most_common(24): print(f"{100*n/tot:5.1f}% inst {100*st[ln]/ts:5.1f}% stall L{ln}: {src[ln-1].strip()[:80]}")
