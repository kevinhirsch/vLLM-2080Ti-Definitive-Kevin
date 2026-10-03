#!/usr/bin/env python3
"""K6 trace-driven model: TP2 time-sliced (today) vs P/D disaggregation (TP1+TP1)
vs TP2 concurrent-stream multiplexing.  Inputs = gateway request telemetry
(arrival = t - duration, computed_actual prefill tokens, outtok)."""
import json, sys, math, heapq, argparse, statistics as st

def load(paths, t_from=None, t_to=None):
    rows = []
    for f in paths:
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if r.get('route') != 'local' or r.get('status') != 200: continue
            d = r.get('duration') or 0.0
            a = r['t'] - d
            if t_from and a < t_from: continue
            if t_to and a > t_to: continue
            c = r.get('computed_actual')
            if c is None: c = r.get('est_computed') or r.get('ptok') or 0
            o = r.get('outtok') or 1
            rows.append(dict(a=a, c=max(1, int(c)), p=int(r.get('ptok') or c), o=max(1, int(o)),
                             ttft_m=r.get('ttft'), dec_m=r.get('decode_time'), cl=r.get('client')))
    rows.sort(key=lambda r: r['a'])
    return rows

def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs)-1, int(p*len(xs)))] if xs else float('nan')

def summarize(name, rows, key_ttft='ttft', key_dec='dec'):
    tt = [r[key_ttft] for r in rows if r.get(key_ttft) is not None]
    rate = [ (r['o']-1)/r[key_dec] for r in rows if r.get(key_dec) and r['o'] > 20 ]
    e2e = [ r[key_ttft] + (r[key_dec] or 0) for r in rows if r.get(key_ttft) is not None]
    return dict(name=name, n=len(tt), ttft_p50=q(tt,.5), ttft_p90=q(tt,.9), ttft_p99=q(tt,.99), ttft_mean=st.mean(tt),
                dec_tps_p10=q(rate,.1), dec_tps_p50=q(rate,.5), e2e_mean=st.mean(e2e), e2e_p90=q(e2e,.9))

ALPHA = 2.5   # accepted tokens per decode step per stream (MTP-3, measured 2.46-2.59)

def sim_tp2(rows, R2=1450., d0=0.027, ds=0.010, share=0.75, block=1856, budget=3632,
            maxseq=16, mux=None):
    """Single TP2 server. mux=None: today's time-sliced step loop (chunk steps carry decode rows,
    prefill-share cooldown). mux=(s_dec, s_pf): decode runs on a hi-pri stream concurrently with
    prefill; decode step = d(b)*s_dec, prefill rate = R2*(1-s_pf) while decodes are active."""
    t = 0.0; i = 0; n = len(rows)
    waiting = []   # prefill queue FCFS: [req, remaining]
    dec = []       # decoding: [req, steps_left, t_first]
    cool_until = 0.0
    out = []
    pf_left_time = 0.0
    while i < n or waiting or dec:
        if not waiting and not dec:
            t = max(t, rows[i]['a'])
        while i < n and rows[i]['a'] <= t:
            r = dict(rows[i]); waiting.append([r, r['c']]); i += 1
        b = len(dec)
        if mux is None:
            # build one step
            pf = 0; finished_pf = []
            can_chunk = not (dec and t < cool_until)
            if waiting and len(dec) < maxseq:
                # short-first: allow short whole prompts in budget; long chunk only if can_chunk
                left = budget
                for w in waiting:
                    if left <= 0 or len(dec) + len(finished_pf) >= maxseq: break
                    rem = w[1]
                    if rem <= left and (rem <= block or can_chunk):
                        pf += rem; left -= rem; w[1] = 0; finished_pf.append(w)
                    elif can_chunk:
                        take = (left // block) * block
                        if take <= 0: break
                        take = min(take, rem); w[1] -= take; pf += take; left -= take
                        break   # FCFS long prefill blocks behind it
                    else:
                        break
            dt = (d0 + ds * b if b else 0.0) + pf / R2
            if dt <= 0:
                # nothing schedulable (cooldown with no decoders impossible) -> advance to next arrival
                t = rows[i]['a'] if i < n else t + 0.01; continue
            t += dt
            if pf > 0 and dec and any(True for _ in [0]):
                # long chunk step with decoders -> cooldown
                if any(w[1] > 0 for w in waiting) or pf > block:
                    cool_until = t + dt * (1 - share) / share if share < 1 else t
            for w in finished_pf:
                waiting.remove(w); r = w[0]; r['ttft'] = t - r['a']
                dec.append([r, math.ceil(max(0, r['o'] - 1) / ALPHA), t])
            nd = []
            for d in dec:
                if d[0] in [w[0] for w in finished_pf]: nd.append(d); continue
                d[1] -= 1
                if d[1] <= 0: d[0]['dec'] = t - d[2]; out.append(d[0])
                else: nd.append(d)
            # requests with zero decode steps
            dec = []
            for d in nd:
                if d[1] <= 0: d[0]['dec'] = max(0.0, t - d[2]); out.append(d[0])
                else: dec.append(d)
        else:
            s_dec, s_pf = mux
            # event step: advance by one decode step (if decoders) or by prefill-to-completion/next arrival
            nxt = rows[i]['a'] if i < n else float('inf')
            if dec:
                dt = (d0 + ds * b) * (s_dec if waiting else 1.0)
            elif waiting:
                dt = min(waiting[0][1] / R2, max(1e-4, nxt - t))
            else:
                t = nxt; continue
            rate = R2 * ((1 - s_pf) if dec else 1.0)
            budget_tok = rate * dt
            t += dt
            # prefill FCFS progress (one at a time, like a single chunk stream)
            while waiting and budget_tok > 0 and len(dec) < maxseq:
                w = waiting[0]; take = min(w[1], budget_tok); w[1] -= take; budget_tok -= take
                if w[1] <= 1e-9:
                    waiting.pop(0); r = w[0]; r['ttft'] = t - r['a']
                    dec.append([r, math.ceil(max(0, r['o'] - 1) / ALPHA) + 1, t])
            nd = []
            for d in dec:
                d[1] -= 1
                if d[1] <= 0: d[0]['dec'] = t - d[2]; out.append(d[0])
                else: nd.append(d)
            dec = nd
    return out

def sim_pd(rows, R1=800., d0=0.045, ds=0.020, d_pool=200_000, xfer_bw=40e9, kv_b=14_720, state_b=80e6,
           maxseq=16, p_pool_scale=1.0):
    """1 prefill card (TP1) + 1 decode card (TP1). FCFS prefill at R1. KV shipped over NVLink.
    D admits while resident tokens <= d_pool."""
    t_pf = 0.0
    ready = []   # (t_ready, idx, req)
    for k, r0 in enumerate(rows):
        r = dict(r0)
        start = max(t_pf, r['a'])
        t_pf = start + r['c'] / R1
        xfer = (r['p'] * kv_b + state_b) / xfer_bw
        r['_pf_done'] = t_pf; r['ttft'] = t_pf + xfer - r['a']
        ready.append((t_pf + xfer, k, r))
    ready.sort(key=lambda x: x[0])
    # decode server
    t = 0.0; j = 0; dq = []; dec = []; out = []; resident = 0
    while j < len(ready) or dq or dec:
        if not dq and not dec: t = max(t, ready[j][0])
        while j < len(ready) and ready[j][0] <= t: dq.append(ready[j][2]); j += 1
        while dq and len(dec) < maxseq and (resident + dq[0]['p'] + dq[0]['o'] <= d_pool or not dec):
            r = dq.pop(0); resident += r['p'] + r['o']
            r['ttft'] = max(r['ttft'], t - r['a'])   # waiting for D slot delays first visible token? (P emits 1st token; keep P time)
            dec.append([r, math.ceil(max(0, r['o'] - 1) / ALPHA), t])
        if not dec:
            t = ready[j][0] if j < len(ready) else t; continue
        dt = d0 + ds * len(dec); t += dt
        nd = []
        for d in dec:
            d[1] -= 1
            if d[1] <= 0:
                d[0]['dec'] = t - d[2] + (d[2] - (d[0]['a'] + d[0]['ttft'])); out.append(d[0]); resident -= d[0]['p'] + d[0]['o']
            else: nd.append(d)
        dec = nd
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    rows = load(a.files)
    res = []
    meas = [dict(r, ttft=r['ttft_m'], dec=r['dec_m']) for r in rows if r['ttft_m'] is not None]
    res.append(summarize('MEASURED (production telemetry)', meas))
    for R2 in (1279., 1450., 1600.):
        res.append(summarize(f'A TP2 today share0.75 R2={R2:.0f}', sim_tp2(rows, R2=R2)))
    res.append(summarize('A2 TP2 share0.50 R2=1450', sim_tp2(rows, R2=1450., share=0.5)))
    res.append(summarize('A3 TP2 no-slicing share1.0 R2=1450', sim_tp2(rows, R2=1450., share=1.0)))
    for k in (0.5, 0.55, 0.62):
        for pool in (150_000, 300_000, 10**9):
            res.append(summarize(f'B PD TP1 R1={k:.2f}*1450 Dpool={pool if pool<10**9 else "inf"} (cache-optimistic)',
                                 sim_pd(rows, R1=1450.*k, d_pool=pool)))
    for sd, sp in ((1.3, 0.10), (1.6, 0.20), (2.0, 0.30)):
        res.append(summarize(f'M TP2 hi-pri decode stream || prefill (dec x{sd}, pf -{int(sp*100)}%)',
                             sim_tp2(rows, R2=1450., mux=(sd, sp))))
    w = max(len(r['name']) for r in res)
    print(f"{'config':<{w}}  {'n':>5} {'TTFTp50':>8} {'p90':>7} {'p99':>7} {'mean':>6} | {'dec tok/s p10':>13} {'p50':>6} | {'e2e mean':>8} {'p90':>7}")
    for r in res:
        print(f"{r['name']:<{w}}  {r['n']:>5} {r['ttft_p50']:>8.2f} {r['ttft_p90']:>7.1f} {r['ttft_p99']:>7.1f} {r['ttft_mean']:>6.1f} | {r['dec_tps_p10']:>13.1f} {r['dec_tps_p50']:>6.1f} | {r['e2e_mean']:>8.1f} {r['e2e_p90']:>7.1f}")
    if a.json: json.dump(res, open(a.json, 'w'), indent=1)
