import sys,time,torch,math,os
sys.path.insert(0,'/home/kevin/Desktop/wt-integrate')
from vllm.v1.attention.ops.triton_turboquant_decode import triton_turboquant_decode_attention as old
import vllm.v1.attention.ops.tq_gqa_cuda as C
torch.manual_seed(0); dev='cuda'
D=256; Hk=2; Hq=12; KPS=98; VB=128; SLOT=KPS+VB+4; BS=int(sys.argv[1]); ctx=int(sys.argv[2]); S=int(sys.argv[3]); QL=int(sys.argv[4]); NSP=[int(x) for x in sys.argv[5].split(',')] if len(sys.argv)>5 else [128]
nb=(ctx+BS-1)//BS+1; NBT=S*nb+2
kv=torch.randint(0,256,(NBT,BS,Hk,SLOT),dtype=torch.uint8,device=dev)
def put16(off,vals):
    kv[...,off:off+2]=vals.to(torch.float16).view(torch.uint8).reshape(*vals.shape,2)
sh=kv.shape[:3]
put16(96,torch.rand(sh,device=dev)*2+0.5); put16(KPS+VB,torch.rand(sh,device=dev)*0.2+0.05); put16(KPS+VB+2,-torch.rand(sh,device=dev))
cent=torch.tensor([-2.15,-1.34,-0.756,-0.245,0.245,0.756,1.34,2.15],device=dev)/16
Pi=torch.eye(D,device=dev)
perm=torch.randperm(S*nb,device=dev).to(torch.int32)  # shuffled pages
bt=perm.reshape(S,nb).contiguous()
lens=[ctx-(i*37)%900 for i in range(S)]
sl=torch.tensor(lens,dtype=torch.int32,device=dev)
q=(torch.randn(S*QL,Hq,D,device=dev)*3).to(torch.float16)
scale=1/16
ref_out=torch.empty(S*QL,Hq,D,dtype=torch.float16,device=dev); ref_lse=torch.empty(S*QL,Hq,device=dev)
def run_old():
    return old(q,kv,bt.repeat_interleave(QL,0),sl.repeat_interleave(QL),Pi,cent,scale,3,KPS,4,False,True,output_buf=ref_out,lse_buf=ref_lse,max_num_kv_splits=128)
def bench(f,n=30):
    f(); torch.cuda.synchronize(); t=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time()-t)/n*1e3
run_old(); t_old=bench(run_old); print(f"old S={S} QL={QL} ctx~{ctx}: {t_old:.3f} ms (stage1+2 incl. repeat_interleave)",flush=True)
for ns in NSP:
    out=torch.empty_like(ref_out); lse=torch.empty_like(ref_lse)
    f=lambda: C.tq_gqa_decode_attention(q,kv,bt,sl,Pi,cent,scale,True,q_per_seq=QL,num_splits=ns,output_buf=out,lse_buf=lse)
    f(); torch.cuda.synchronize()
    eo=(out.float()-ref_out.float()).abs().max().item(); rel=eo/ref_out.float().abs().max().item(); el=(lse-ref_lse).abs().max().item()
    t=bench(f); print(f"cuda NS={ns}: {t:.3f} ms ({t_old/t:.1f}x)  out_rel_err={rel:.3g} lse_abs_err={el:.3g} nan={torch.isnan(out).any().item()}",flush=True)
