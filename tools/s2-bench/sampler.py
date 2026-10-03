#!/usr/bin/env python3
"""sampler.py OUT.csv -- 1 Hz: engine metrics (running/waiting/kv usage/prefix hits+queries/preemptions/spec) + per-GPU clocks/power/temp/throttle reasons."""
import sys,time,subprocess,http.client,csv,re
out=csv.writer(open(sys.argv[1],"w",buffering=1))
out.writerow(["t","running","waiting","kv_usage","pc_hits","pc_queries","preempt","spec_acc","spec_drafts","g0_sm","g0_mem","g0_w","g0_c","g0_thr","g1_sm","g1_mem","g1_w","g1_c","g1_thr"])
def eng():
    try:
        c=http.client.HTTPConnection("127.0.0.1",8001,timeout=3); c.request("GET","/metrics"); t=c.getresponse().read().decode()
    except Exception: return [""]*8
    def g(k):
        return sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(k) and "created" not in l and not l.startswith("#"))
    return [g("vllm:num_requests_running"),g("vllm:num_requests_waiting"),g("vllm:kv_cache_usage_perc"),g("vllm:prefix_cache_hits_total"),g("vllm:prefix_cache_queries_total"),g("vllm:num_preemptions_total"),g("vllm:spec_decode_num_accepted_tokens_total"),g("vllm:spec_decode_num_drafts_total")]
Q="clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active"
while True:
    try:
        r=subprocess.run(["nvidia-smi","--query-gpu="+Q,"--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=5).stdout.strip().split("\n")
        g=[x.split(", ") for x in r]
        row=[round(time.time(),1)]+eng()+g[0]+g[1]
        out.writerow(row)
    except Exception as e: pass
    time.sleep(1)
