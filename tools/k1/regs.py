import re,sys
cur=None
for l in open(sys.argv[1] if len(sys.argv)>1 else '/tmp/k1_build.log'):
    m=re.search(r"k1fa_kernelILi(\d+)E((?:Lb[01]E)+)",l)
    if 'Compiling entry' in l:
        cur=None
        if m: cur='BN'+m.group(1)+' '+' '.join(x for x in re.findall(r'Lb([01])',m.group(2)))
    elif cur and 'spill' in l: sp=re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads',l).groups()
    elif cur and 'registers' in l:
        print(cur,'(C P L Q)', 'regs', re.search(r'Used (\d+) registers',l).group(1), 'spill st/ld', sp)
