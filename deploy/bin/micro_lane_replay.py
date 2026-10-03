#!/usr/bin/env python3
"""Replay gateway request telemetry through the micro fast lane v2 (lane K4, 2026-10-03).

Answers, from MEASURED traffic and with the gateway's own predictor (keepalive-shim.py micro_observe/micro_predict and
probe_word -- nothing re-implemented here): how many requests would the learned tiny lane have admitted, how often was
it wrong (the call then produced a long answer), how many full-lane slot-seconds and queue-wait seconds move off the
main lanes, and how much work the liveness-probe shortcut removes.

    python3 micro_lane_replay.py                      # last 7 UTC day files, human readable
    python3 micro_lane_replay.py --days 3 --json
    python3 micro_lane_replay.py --min-samples 5 --max-out 128
"""
import argparse
import collections
import glob
import importlib.util
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
TELEMETRY = os.environ.get("SHIM_TELEMETRY_DIR", os.path.expanduser("~/.local/share/vllm-qwen27b/telemetry"))


def load_shim():
    os.environ.setdefault("SHIM_EXACT_TOKENS", "0")
    spec = importlib.util.spec_from_file_location("keepalive_shim_replay", HERE / "keepalive-shim.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_rows(days, tdir=TELEMETRY):
    rows = []
    for f in sorted(glob.glob(os.path.join(tdir, "requests-*.jsonl")))[-days:]:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and isinstance(rec.get("t"), (int, float)):
                    rows.append(rec)
    rows.sort(key=lambda r: r["t"])
    return rows


def replay(rows, shim, *, long_out=None):
    """Online replay: predict BEFORE observing, exactly like the live gateway. Returns a metrics dict."""
    shim._MICRO_HIST.clear()
    long_out = long_out or 2 * shim.MICRO_MAX_OUT
    done = [r for r in rows if r.get("route") in ("local", "held") and (r.get("status") or 200) < 400
            and r.get("outtok") is not None and r.get("preview")]
    m = collections.Counter()
    by_client = collections.Counter()
    total_slot_s = sum(r.get("duration") or 0 for r in done)
    for r in done:
        pred = (not r.get("tiny")) and shim.micro_predict(r["client"], r["preview"], r.get("ptok") or 0)
        if pred:
            m["admitted"] += 1
            m["freed_slot_s"] += r.get("duration") or 0
            m["freed_wait_s"] += r.get("admission_wait") or 0
            by_client[r["client"]] += 1
            if r["outtok"] > long_out:
                m["wrong_long"] += 1
        shim.micro_observe(r["client"], r["preview"], r["outtok"])
    m["local_completions"] = len(done)
    m["local_slot_s"] = total_slot_s
    m["already_tiny"] = sum(1 for r in done if r.get("tiny"))
    # probe shortcut: every probe request in the log that the gateway would have answered (fresh engine assumed when
    # the log shows local output within PROBE_FRESH_S before it; every PROBE_REAL_EVERY-th still goes through).
    last_ok, seen = 0.0, 0
    for r in rows:
        word = shim.probe_word(json.dumps({"messages": [{"role": "user", "content": r.get("preview") or ""}]}).encode())
        # the log keeps only a 70-char preview: a probe's injected-context tail is cut, which is still a probe
        if word and r.get("alias_kind") in (None, "", "builtin-local") and r.get("route") != "rejected":
            m["probe_requests"] += 1
            if r["t"] - last_ok <= shim.PROBE_FRESH_S:
                seen += 1
                if not (shim.PROBE_REAL_EVERY > 0 and seen % shim.PROBE_REAL_EVERY == 0):
                    m["probe_synth"] += 1
                    m["probe_prefill_tokens_saved"] += (r.get("est_computed") or 0) if r.get("route") in ("local", "held") else 0
                    m["probe_slot_s_saved"] += (r.get("duration") or 0) if r.get("route") in ("local", "held") else 0
                    m["probe_remote_saved"] += 1 if r.get("route") == "remote" else 0
                    m["probe_503_saved"] += 1 if (r.get("status") or 200) >= 400 else 0
        if r.get("route") in ("local", "held") and (r.get("status") or 200) < 400 and (r.get("outtok") or r.get("outtok_lb")):
            last_ok = r["t"]
    out = dict(m)
    out["precision"] = round(1 - m["wrong_long"] / m["admitted"], 4) if m["admitted"] else None
    out["freed_slot_share"] = round(m["freed_slot_s"] / total_slot_s, 4) if total_slot_s else None
    out["top_clients"] = by_client.most_common(8)
    out["signatures"] = len(shim._MICRO_HIST)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--telemetry-dir", default=TELEMETRY)
    ap.add_argument("--min-samples", type=int)
    ap.add_argument("--max-out", type=int)
    ap.add_argument("--history", type=int)
    ap.add_argument("--sig-chars", type=int)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    shim = load_shim()
    for attr, val in (("MICRO_MIN_SAMPLES", a.min_samples), ("MICRO_MAX_OUT", a.max_out),
                      ("MICRO_HISTORY", a.history), ("MICRO_SIG_CHARS", a.sig_chars)):
        if val:
            setattr(shim, attr, val)
    res = replay(read_rows(a.days, a.telemetry_dir), shim)
    if a.json:
        print(json.dumps(res, indent=2, default=str))
        return 0
    print("micro fast lane v2 replay (%d local completions, %.0f slot-seconds)" % (res["local_completions"], res["local_slot_s"]))
    print("  learned-tiny admitted : %d (already static-tiny: %d)" % (res.get("admitted", 0), res["already_tiny"]))
    print("  wrong (answer > %dx)  : %d  -> precision %s" % (2, res.get("wrong_long", 0), res["precision"]))
    print("  main-lane slot-seconds moved to the tiny lane: %.0f (%.2f%% of all)" % (res.get("freed_slot_s", 0), 100 * (res["freed_slot_share"] or 0)))
    print("  queue-wait seconds those requests paid today  : %.0f" % res.get("freed_wait_s", 0))
    print("  probe requests %d -> answered by the gateway %d (local slot-s %.0f, uncached prefill tokens %d, paid-remote trips %d, 503s %d)" % (
        res.get("probe_requests", 0), res.get("probe_synth", 0), res.get("probe_slot_s_saved", 0),
        res.get("probe_prefill_tokens_saved", 0), res.get("probe_remote_saved", 0), res.get("probe_503_saved", 0)))
    print("  top clients:", res["top_clients"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
