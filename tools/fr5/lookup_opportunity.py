"""FR5 offline opportunity sim (CPU only). For (prompt, response) token pairs, simulate greedy decode where at each
step a suffix-match (prompt-lookup style, longest match, most recent tie) draft of up to D tokens is proposed if the
match length >= L, vs MTP-only at a fixed mean tokens/step. Reports coverage and tokens/step.
Usage: lookup_opportunity.py corpus.jsonl  (each line: {"src":..,"prompt":[ids],"resp":[ids]})"""
import collections, json, sys

MTP = 3.28          # measured tokens/step (1 + accepted_per_draft 2.28) on summarisation C1 (prod-up window base_mb.json)
NMIN = 3            # smallest suffix indexed
MAXBACK = 32        # max backward match length measured
CAND = 48           # most recent candidates examined per lookup


class Idx:
    def __init__(self, ctx):
        self.ctx = list(ctx); self.d = collections.defaultdict(list)
        for i in range(NMIN, len(self.ctx) + 1):
            self.d[tuple(self.ctx[i - NMIN:i])].append(i)   # i = index just past the n-gram

    def push(self, tok):
        self.ctx.append(tok); i = len(self.ctx)
        if i >= NMIN:
            self.d[tuple(self.ctx[i - NMIN:i])].append(i)

    def match(self):
        """longest suffix match end position (index of the token that would follow) and its length"""
        c = self.ctx; n = len(c)
        if n < NMIN: return 0, 0
        best = (0, 0)
        for p in reversed(self.d.get(tuple(c[n - NMIN:n]), [])[-CAND - 1:]):
            if p == n: continue
            m = NMIN
            while m < MAXBACK and p - m - 1 >= 0 and c[p - m - 1] == c[n - m - 1]: m += 1
            if m > best[1]: best = (p, m)
        return best


def run(prompt, resp, L, D, m_copy, hybrid):
    """Return (steps, lookup_steps, accepted_sum, covered_tokens). Baseline: MTP everywhere (m_copy tok/step when a
    >=L match exists, else MTP). Hybrid: when a >=L match exists propose up to D lookup tokens and advance 1+accepted."""
    ix = Idx(prompt); T = len(resp); t = 0; steps = 0; ls = 0; acc = 0; cov = 0; carry = 0.0
    while t < T:
        p, m = ix.match(); ctx = ix.ctx; steps += 1
        if hybrid and m >= L:
            a = 0
            while a < D and p + a < len(ctx) and t + a < T and ctx[p + a] == resp[t + a]: a += 1
            ls += 1; acc += a; adv = 1 + a; cov += min(adv, T - t)
        else:
            mean = m_copy if m >= L else MTP
            carry += mean; adv = int(carry); carry -= adv; adv = max(adv, 1)
        for k in range(min(adv, T - t)): ix.push(resp[t + k])
        t += adv
    return steps, ls, acc, cov


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    by = collections.defaultdict(list)
    for r in rows: by[r["src"]].append(r)
    for src, rs in sorted(by.items()):
        T = sum(len(r["resp"]) for r in rs)
        print(f"== {src}: n={len(rs)} resp_tokens={T} mean_prompt={sum(len(r['prompt']) for r in rs)//len(rs)}")
        # coverage: fraction of response tokens inside a >=L suffix match that predicts next token correctly
        for L in (4, 6, 8, 12):
            for D in (8, 16):
                for m_copy in (MTP, 3.8):
                    sb = sh = ls = acc = cov = 0
                    for r in rs:
                        sb += run(r["prompt"], r["resp"], L, D, m_copy, False)[0]
                        h, l, a, c = run(r["prompt"], r["resp"], L, D, m_copy, True)
                        sh += h; ls += l; acc += a; cov += c
                    print(f"  L={L:2d} D={D:2d} mtp_in_copy={m_copy:.2f}: lookup_steps={ls} ({ls/max(sh,1):.1%}) "
                          f"mean_acc/lookup={acc/max(ls,1):.2f} cov_tokens={cov/T:.1%} | tok/step base={T/sb:.2f} "
                          f"hybrid={T/sh:.2f} gain={sb/sh-1:+.1%}")

if __name__ == "__main__": main()
