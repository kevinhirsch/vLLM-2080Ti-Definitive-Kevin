# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the E16/P0 tree-drafting oracle: the env-gated logger (vllm/v1/spec_decode/tree_oracle.py), its
proposer/runner hooks, and the analysis math (tools/tree_oracle/analyze.py)."""

import importlib.util
import json
import os
import random
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.spec_decode import tree_oracle as to

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_spec = importlib.util.spec_from_file_location("tree_oracle_analyze", os.path.join(ROOT, "tools/tree_oracle/analyze.py"))
an = importlib.util.module_from_spec(_spec)
sys.modules["tree_oracle_analyze"] = an
_spec.loader.exec_module(an)


def _oracle(tmp_path, **env):
    e = {"VLLM_TREE_ORACLE": "1", "VLLM_TREE_ORACLE_PATH": str(tmp_path / "o.jsonl"), "VLLM_TREE_ORACLE_TOPK": "4"}
    e.update(env)
    return to.TreeOracle(env=e, rank=0)


def _rows(o):
    o._writer.close()
    return [json.loads(x) for x in open(o.path)]


# ------------------------------------------------------------------------------------------------ logger

def test_disabled_by_default_is_noop(tmp_path):
    o = to.TreeOracle(env={"VLLM_TREE_ORACLE_PATH": str(tmp_path / "x.jsonl")}, rank=0)
    assert not o.enabled and not o.value and not o.timing
    o.begin_propose()
    o.capture_logits(torch.randn(2, 10))
    o.finish_propose(["a", "b"])
    o.begin_step()
    o.on_verified(["a"], [[1]], {"a": [1]}, {})
    o.on_step(1, 4, True, 10)
    assert o._writer is None and not os.path.exists(tmp_path / "x.jsonl")


def test_topk_with_probs_matches_softmax():
    logits = torch.randn(3, 50)
    ids, p = to.topk_with_probs(logits, 5)
    ref = torch.softmax(logits, -1)
    assert torch.equal(ids[:, 0], logits.argmax(-1))
    assert torch.allclose(p, ref.gather(1, ids), atol=1e-6)
    assert (p[:, :-1] >= p[:, 1:]).all()


def _logits_with_order(B, V, orders):
    """[B, V] logits whose descending order starts with orders[b]."""
    lg = torch.full((B, V), -10.0)
    for b, order in enumerate(orders):
        for r, tok in enumerate(order):
            lg[b, tok] = 10.0 - r
    return lg


def test_join_one_step_later_and_schema(tmp_path):
    o = _oracle(tmp_path)
    # step 0: propose 3 depths for reqs r0, r1 (row order of the batch)
    o.begin_step()
    o.begin_propose()
    o.capture_logits(_logits_with_order(2, 20, [[5, 6, 7, 8], [9, 1, 2, 3]]))
    o.capture_logits(_logits_with_order(2, 20, [[11, 12, 13, 14], [4, 3, 2, 1]]))
    o.capture_logits(_logits_with_order(2, 20, [[15, 16, 17, 18], [0, 19, 18, 17]]))
    o.finish_propose(["r0", "r1"])
    # nothing verified yet in step 0 (no previous propose)
    o.on_verified(["r0", "r1"], [[100], [101]], {}, {})
    # step 1: the batch order changed (r1 first); r0 rejects at depth 2 and emits rank-2 token 12; r1 accepts all
    o.begin_step()
    meta = {"r0": (500, 0.0, -1, 1.0, False), "r1": (900, 0.6, 20, 0.95, True)}
    o.on_verified(["r1", "r0"], [[9, 4, 0, 77], [5, 12]], {"r0": [5, 11, 15], "r1": [9, 4, 0]}, meta)
    rows = [r for r in _rows(o) if r["t"] == "v"]
    assert len(rows) == 2
    r1, r0 = rows
    assert r0["req"] == "r0" and r0["drafts"] == [5, 11, 15] and r0["out"] == [5, 12] and r0["acc"] == 1
    assert r0["ids"] == [[5, 6, 7, 8], [11, 12, 13, 14], [15, 16, 17, 18]]
    assert len(r0["p"]) == 3 and len(r0["p"][0]) == 4 and abs(sum(r0["p"][0]) - 1) < 0.01
    assert r0["ctx"] == 500 and r0["temp"] == 0.0 and r0["so"] is False and r0["seq"] == 0
    assert r1["ids"][0] == [9, 1, 2, 3] and r1["acc"] == 3 and r1["temp"] == 0.6 and r1["so"] is True
    # the pending batch was consumed: a second verify without a new propose logs nothing
    o.begin_step()
    o.on_verified(["r0"], [[1]], {"r0": [1, 2, 3]}, meta)
    assert len([r for r in _rows(o) if r["t"] == "v"]) == 2


def test_unscheduled_and_empty_rows_skipped(tmp_path):
    o = _oracle(tmp_path)
    o.begin_step()
    o.begin_propose()
    for _ in range(2):
        o.capture_logits(_logits_with_order(2, 10, [[1, 2, 3, 4], [5, 6, 7, 8]]))
    o.finish_propose(["a", "b"])
    o.begin_step()
    # a: drafts not scheduled (e.g. discarded partial prefill); b: no sampled tokens (invalid row)
    o.on_verified(["a", "b"], [[3], []], {"b": [5, 6]}, {})
    assert [r for r in _rows(o) if r["t"] == "v"] == []


def test_pause_file_and_max_rows(tmp_path):
    pause = tmp_path / "pause"
    o = _oracle(tmp_path, VLLM_TREE_ORACLE_PAUSE_FILE=str(pause), VLLM_TREE_ORACLE_MAX_ROWS="1")

    def one_step(reqs):
        o.begin_step()
        o.on_verified(reqs, [[1, 2]] * len(reqs), {r: [1, 3] for r in reqs}, {})
        o.begin_propose()
        o.capture_logits(_logits_with_order(len(reqs), 10, [[1, 2, 3, 4]] * len(reqs)))
        o.capture_logits(_logits_with_order(len(reqs), 10, [[3, 2, 1, 4]] * len(reqs)))
        o.finish_propose(reqs)

    pause.write_text("")
    o._pause_checked = -1e9
    one_step(["a"])
    assert o._paused and o._next is None          # nothing captured while paused
    pause.unlink()
    o._pause_checked = -1e9
    one_step(["a"])                                # resumes: captures
    o._pause_checked = -1e9
    one_step(["a", "b"])                           # verifies a (1 row), then max_rows reached
    o._pause_checked = -1e9
    one_step(["a", "b"])
    rows = _rows(o)
    assert [r["paused"] for r in rows if r["t"] == "pause"] == [True, False]
    assert len([r for r in rows if r["t"] == "v"]) == 1


def test_non_zero_rank_logs_nothing(tmp_path):
    o = to.TreeOracle(env={"VLLM_TREE_ORACLE": "1", "VLLM_TREE_ORACLE_TIMING": "1",
                           "VLLM_TREE_ORACLE_PATH": str(tmp_path / "r1.jsonl")}, rank=1)
    o.begin_step()
    o.begin_propose()
    o.capture_logits(torch.randn(1, 10))
    o.finish_propose(["a"])
    o.on_step(1, 4, True, 1)
    assert o._writer is None


def test_timing_rows(tmp_path):
    o = _oracle(tmp_path, VLLM_TREE_ORACLE="0", VLLM_TREE_ORACLE_TIMING="1")
    assert o.timing and not o.value
    o.draft_timer_stop(o.draft_timer_start())   # CPU-only box: no CUDA -> None, harmless
    o.on_step(2, 8, True, 1000)
    o.on_step(2, 8, True, 1001)
    rows = [r for r in _rows(o) if r["t"] == "s"]
    assert rows[0]["dt_ms"] is None and rows[1]["dt_ms"] >= 0 and rows[1]["B"] == 2 and rows[1]["uni"] is True


def test_proposer_greedy_sample_hook_same_argmax(tmp_path, monkeypatch):
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    logits = torch.randn(3, 40)
    fake = SimpleNamespace(use_local_argmax_reduction=False, use_heterogeneous_vocab=False,
                           model=SimpleNamespace(compute_logits=lambda h: logits))
    monkeypatch.setattr(to, "_ORACLE", to.TreeOracle(env={"VLLM_TREE_ORACLE_PATH": str(tmp_path / "n.jsonl")},
                                                     rank=0))
    off = SpecDecodeBaseProposer._greedy_sample(fake, torch.zeros(3, 4))
    o = _oracle(tmp_path)
    monkeypatch.setattr(to, "_ORACLE", o)
    o.begin_propose()
    on = SpecDecodeBaseProposer._greedy_sample(fake, torch.zeros(3, 4))
    assert torch.equal(off, on) and torch.equal(on, logits.argmax(-1))
    assert len(o._depth) == 1 and torch.equal(o._depth[0][0][:, 0], on)


def test_runner_hook_builds_meta_and_uniform_flag(tmp_path):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    import numpy as np

    o = _oracle(tmp_path, VLLM_TREE_ORACLE_TIMING="1")
    o.begin_step()
    o.begin_propose()
    for _ in range(3):
        o.capture_logits(_logits_with_order(2, 10, [[1, 2, 3, 4], [5, 6, 7, 8]]))
    o.finish_propose(["a", "b"])
    o.begin_step()
    sp = SimpleNamespace(temperature=0.6, top_k=20, top_p=0.95, structured_outputs=None)
    fake = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=2, num_computed_tokens_cpu=np.array([100, 2000])),
        requests={"a": SimpleNamespace(sampling_params=sp, num_computed_tokens=100),
                  "b": SimpleNamespace(sampling_params=sp, num_computed_tokens=2000)})
    so = SimpleNamespace(scheduled_spec_decode_tokens={"a": [1, 2, 3], "b": [5, 6, 7]}, scheduled_new_reqs=[],
                         total_num_scheduled_tokens=8)
    GPUModelRunner._tree_oracle_after_bookkeeping(fake, o, so, ["a", "b"], [[1, 2, 9], [5, 6, 7, 1]])
    rows = _rows(o)
    v = [r for r in rows if r["t"] == "v"]
    s = [r for r in rows if r["t"] == "s"]
    assert [r["acc"] for r in v] == [2, 3] and v[0]["top_k"] == 20 and v[1]["ctx"] == 2000
    assert s[0]["uni"] is True and s[0]["B"] == 2 and s[0]["ctx_max"] == 2000


# ------------------------------------------------------------------------------------------------ analysis math

def _step(drafts, ids, out, p=None, req="chatcmpl-tro-code-1-abc", seq=0, temp=0.0, ctx=1000):
    return an.Step({"req": req, "seq": seq, "ctx": ctx, "temp": temp, "drafts": drafts, "ids": ids,
                    "p": p or [[0.5, 0.2, 0.1, 0.05]] * len(ids), "out": out, "acc": len(out) - 1})


IDS = [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43], [50, 51, 52, 53]]
DR = [10, 20, 30, 40, 50]


def test_category_parsing():
    assert an.category("chatcmpl-tro-code_think-17-1a2b3c4d") == "code_think"
    assert an.category("chatcmpl-tro-vault-3-ffff") == "vault"
    assert an.category("chatcmpl-abc") == "other"
    assert an.category("x", [{"cat": "longctx", "t0": 1, "t1": 2}], 1.5) == "longctx"


def test_chain_tokens_exact():
    s = _step(DR, IDS, [10, 20, 99])  # accepted 2, rejected at depth 3
    for K, want in [(1, 2), (2, 3), (3, 3), (4, 3), (5, 3)]:
        assert an.Tree("c", K).evaluate(s)[0] == want
    s5 = _step(DR, IDS, [10, 20, 30, 40, 50, 7])  # all 5 accepted + bonus
    assert [an.Tree("c", K).evaluate(s5)[0] for K in range(1, 6)] == [2, 3, 4, 5, 6]


def test_leaf_rescue_exact():
    t7 = an.Tree("M7", 3, (1, 1, 1))
    assert t7.rows == 7
    assert t7.evaluate(_step(DR, IDS, [11]))[0] == 2             # rejected at d1, rank-2 leaf hit: leaf + bonus
    assert t7.evaluate(_step(DR, IDS, [12]))[0] == 1             # rank-3 not in a 1-leaf tree
    assert an.Tree("w2", 3, (2,)).evaluate(_step(DR, IDS, [12]))[0] == 2
    assert t7.evaluate(_step(DR, IDS, [10, 20, 31]))[0] == 4     # rescue at d3
    assert t7.evaluate(_step(DR, IDS, [10, 20, 30, 41]))[0] == 4  # chain fully accepted at D=3: no change
    assert an.Tree("d1only", 3, (1,)).evaluate(_step(DR, IDS, [10, 21]))[0] == 2  # no leaf at d2


def test_leaf_skips_chain_token_when_draft_not_top1():
    ids = [[11, 10, 12, 13]] + IDS[1:]   # chain draft 10 is the drafter's rank 2 (tie / mismatch)
    s = _step(DR, ids, [11])
    assert not s.top1_match
    assert s.leaf_candidates(0, 1) == [11]
    assert s.rescue_rank() == 1
    assert an.Tree("M5", 3, (1,)).evaluate(s)[0] == 2


def test_dynamic_tau_and_rows():
    p = [[0.6, 0.03, 0.01, 0.0]] + [[0.6, 0.3, 0.05, 0.0]] * 4
    s = _step(DR, IDS, [11], p=p)
    t = an.Tree("dyn", 3, (1, 1, 1), tau=0.05)
    tok, used = t.evaluate(s)
    assert tok == 1 and used == 2       # the d1 leaf (p=0.03) is gated off; d2 and d3 leaves are used


def test_truncated_steps_excluded():
    s = _step([10, 20], IDS[:3], [10, 20, 5])  # scheduler trimmed to 2 drafts
    assert an.Tree("c3", 3).evaluate(s) is None
    assert an.Tree("c2", 2).evaluate(s)[0] == 3


def test_branch_approx_uses_next_step():
    a = _step(DR, IDS, [11], req="chatcmpl-tro-x-1-a", seq=0)
    b = _step(DR, IDS, [10, 20, 99], req="chatcmpl-tro-x-1-a", seq=1)
    an.link_next([a, b])
    t = an.Tree("br", 3, (1,), branch={1: 2})
    assert t.approx and t.rows == 7
    assert t.evaluate(a)[0] == 0 + 2 + 2   # s1 + 2 children (next step accepted 2) + bonus
    assert t.evaluate(b)[0] == 3            # rejected at d3: no leaf there
    lone = _step(DR, IDS, [11], req="chatcmpl-tro-x-2-a", seq=5)
    an.link_next([lone])
    assert t.evaluate(lone) is None


def test_rescue_table_and_survival():
    steps = [_step(DR, IDS, [11]), _step(DR, IDS, [12]), _step(DR, IDS, [99]), _step(DR, IDS, [10, 21]),
             _step(DR, IDS, [10, 20, 30, 40, 50, 1])]
    rt = an.rescue_table(steps)
    assert rt["d1"]["rejections"] == 3 and rt["d1"]["r_top2"] == round(1 / 3, 4)
    assert rt["d1"]["r_top3"] == round(2 / 3, 4) and rt["d1"]["outside_topk"] == round(1 / 3, 4)
    assert rt["d2"]["r_top2"] == 1.0
    sv = an.survival(steps, 5)
    assert sv["P(acc>=1)"] == 0.4 and sv["P(acc>=5)"] == 0.2


def test_simulated_rescue_rate_recovered():
    """Monte-Carlo: per-depth accept prob q, rank-2 rescue prob r. The analysis must recover E[tokens] analytically."""
    rng = random.Random(1)
    q, r, n = 0.7, 0.35, 20000
    steps = []
    for i in range(n):
        out = []
        for d in range(5):
            if rng.random() < q:
                out.append(DR[d])
            else:
                out.append(IDS[d][1] if rng.random() < r else 999)
                break
        else:
            out.append(7)
        steps.append(_step(DR, IDS, out, req=f"chatcmpl-tro-sim-{i % 50}-x", seq=i))
    res = {t["tree"]: t for t in an.evaluate_trees(steps, [an.Tree("chain K=3", 3), an.Tree("M7", 3, (1, 1, 1))])}
    chain = 1 + q + q ** 2 + q ** 3
    gain = r * (1 - q) * (1 + q + q ** 2)  # rejection mass at depths 1..3 times r
    assert abs(res["chain K=3"]["tok_per_step"] - chain) < 0.03
    assert abs(res["M7"]["gain_tok"] - gain) < 0.02
    lo, hi = res["M7"]["gain_ci95"]
    assert lo < gain < hi


def test_cost_curve_and_prediction():
    def arm(K, step_ms, draft_ms):
        rows = []
        for i in range(60):
            rows.append({"t": "s", "B": 1, "q": K + 1, "uni": True, "dt_ms": step_ms, "draft_ms": draft_ms,
                         "ctx_max": 2000})
        rows.insert(30, {"t": "s", "B": 1, "q": 300, "uni": False, "dt_ms": 500.0, "draft_ms": 1.0, "ctx_max": 2000})
        return an.cost_curve(rows)
    costs = {3: arm(3, 29.0, 3.0), 5: arm(5, 32.0, 5.0), 7: arm(7, 35.0, 7.0)}
    c3 = costs[3]["B1|<4K"]
    assert c3["n"] == 58 and c3["step_ms_p50"] == 29.0 and c3["verify_ms_p50"] == 26.0
    trees = [{"tree": "M7", "D": 3, "rows": 7, "n": 10, "tok_per_step": 3.0, "base_tok_per_step": 2.7,
              "approx": False}]
    pred = an.predict(trees, costs)
    row = pred[0]
    assert row["base_step_ms"] == 29.0
    t = row["trees"][0]
    # verify(M=7) interpolated between M=6 (27) and M=8 (28) = 27.5; drafter(D=3) = 3 -> 30.5 ms
    assert t["step_ms"] == 30.5
    assert abs(t["net_pct"] - 100 * ((3.0 / 30.5) / (2.7 / 29.0) - 1)) < 0.01
    g = an.go_no_go(pred)
    assert g["verdict"] == "GO" and g["best_B1"]["tree"] == "M7"


def test_end_to_end_logger_to_report(tmp_path):
    """Logger JSONL -> analyze(): the schema the engine writes is the schema the analysis reads."""
    pause = tmp_path / "pause"
    o = _oracle(tmp_path, VLLM_TREE_ORACLE_TOPK="8", VLLM_TREE_ORACLE_TIMING="1",
                VLLM_TREE_ORACLE_PAUSE_FILE=str(pause), VLLM_TREE_ORACLE_PATH=str(tmp_path / "oracle_k5.jsonl"))
    rng = random.Random(0)
    reqs = [f"chatcmpl-tro-{c}-{i}-abcd" for i, c in enumerate(["vault", "code", "tool", "longctx"])]
    meta = {r: (1000, 0.0, -1, 1.0, False) for r in reqs}
    orders = [list(range(d * 10, d * 10 + 8)) for d in range(5)]
    for step in range(400):
        if step == 300:
            pause.write_text("")
        o._pause_checked = -1e9
        o.begin_step()
        if o._cur is not None:
            outs, spec = [], {}
            for r in reqs:
                out = []
                for d in range(5):
                    if rng.random() < 0.7:
                        out.append(orders[d][0])
                    else:
                        out.append(orders[d][1] if rng.random() < 0.3 else 999)
                        break
                else:
                    out.append(5)
                outs.append(out)
                spec[r] = [orders[d][0] for d in range(5)]
            o.on_verified(reqs, outs, spec, meta)
        o.on_step(len(reqs), 24, True, 1000)
        o.begin_propose()
        for d in range(5):
            o.capture_logits(_logits_with_order(len(reqs), 64, [orders[d]] * len(reqs)))
        o.finish_propose(reqs)
    o._writer.close()
    rep = an.analyze([o.path])
    assert rep["D_run"] == 5 and rep["steps"] > 1000 and rep["drafts_not_top1"] == 0
    names = {t["tree"]: t for t in rep["overall"]["trees"]}
    assert names["chain K=3"]["gain_tok"] == 0
    assert names["K3 + r2 leaf @each depth (M=7) [design proto]"]["gain_tok"] > 0.1
    assert {"cat=vault", "cat=code", "cat=tool", "cat=longctx"} <= set(rep["slices"])
    assert "5" in rep["cost"]          # paused-phase step rows become the clean K=5 cost arm
    md = an.to_markdown(rep)
    assert "Rescue rate" in md and "Go/no-go" in md
