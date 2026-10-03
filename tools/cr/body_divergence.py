#!/usr/bin/env python3
"""Lane CR: message-level prefix stability of captured request bodies (flight-recorder copies).

For every body, find the earlier body (same tools/template root) sharing the longest message
prefix = its most likely session predecessor, then report whether the predecessor is a PURE
prefix (append-only) or where/how the history was rewritten.  Read-only.
usage: body_divergence.py DIR [--verbose]
"""
import glob, hashlib, json, os, sys, difflib

def root(j):
    ctk = j.get("chat_template_kwargs") or {}
    return hashlib.sha256(json.dumps([j.get("tools"), ctk.get("enable_thinking"), ctk.get("preserve_thinking"),
                                      j.get("reasoning_effort")], sort_keys=True, default=str).encode()).hexdigest()[:12]

def ser(m):
    return json.dumps(m, sort_keys=True, default=str)

d = sys.argv[1]; verbose = "--verbose" in sys.argv
items = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    try:
        j = json.load(open(f))
    except Exception:
        continue
    ms = [ser(m) for m in j.get("messages") or []]
    items.append((os.path.basename(f), root(j), ms, j))
res = {"pure_append": 0, "diverged": 0, "no_pred": 0, "root_changed_only": 0}
where = {}
lost_chars = 0; tot_chars = 0
for i, (fn, rt, ms, j) in enumerate(items):
    best = None
    for fn2, rt2, ms2, j2 in items[max(0, i - 400):i]:
        if not ms2 or ms2[0] != ms[0]:
            continue
        k = 0
        while k < min(len(ms), len(ms2)) and ms[k] == ms2[k]:
            k += 1
        score = sum(len(x) for x in ms[:k])
        if best is None or score > best[0]:
            best = (score, k, fn2, rt2, ms2, j2)
    if best is None:
        res["no_pred"] += 1; continue
    score, k, fn2, rt2, ms2, j2 = best
    predlen = sum(len(x) for x in ms2)
    tot_chars += predlen
    if k == len(ms2) and k < len(ms):
        if rt2 == rt:
            res["pure_append"] += 1
        else:
            res["root_changed_only"] += 1; lost_chars += predlen
        continue
    res["diverged"] += 1; lost_chars += predlen - score
    a = json.loads(ms2[k]) if k < len(ms2) else None
    b = json.loads(ms[k]) if k < len(ms) else None
    role = (a or {}).get("role", "?")
    keydiff = sorted(set((a or {}).keys()) ^ set((b or {}).keys()))
    tag = f"msg{k}/{len(ms2)} role={role} keydiff={keydiff}"
    where[tag] = where.get(tag, 0) + 1
    if verbose:
        sa, sb = str((a or {}).get("content"))[:4000], str((b or {}).get("content"))[:4000]
        sm = difflib.SequenceMatcher(None, sa, sb, autojunk=False)
        ops = [o for o in sm.get_opcodes() if o[0] != "equal"][:2]
        print(f"{fn} <- {fn2}: diverge at {tag}; matched {score}/{predlen} chars; pred msgs {len(ms2)} cur {len(ms)}")
        for o in ops:
            print("    ", o[0], repr(sa[o[1]:o[2]][:120]), "->", repr(sb[o[3]:o[4]][:120]))
print(json.dumps({"bodies": len(items), **res, "lost_char_share_of_pred": round(lost_chars / max(1, tot_chars), 3),
                  "divergence_points": dict(sorted(where.items(), key=lambda kv: -kv[1])[:15])}, indent=1))
