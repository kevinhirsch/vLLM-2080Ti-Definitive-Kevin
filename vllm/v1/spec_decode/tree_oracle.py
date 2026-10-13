# SPDX-License-Identifier: Apache-2.0
"""E16/P0 tree-drafting value + cost oracle (env-gated, default OFF, zero effect when off).

Oracle A (VLLM_TREE_ORACLE=1): at every drafter sampling site, keep the MTP drafter's top-K token ids and
probabilities (softmax of the raw draft logits, temperature 1). One step later, when the target has verified those
drafts, join them per request with the verified drafts and the emitted tokens and write one JSONL row:

  {"t":"v","ts","req","seq","ctx","temp","top_k","top_p","so","drafts":[D],"ids":[[k]*D],"p":[[k]*D],
   "out":[a+1 tokens],"acc":a}

Because sample-and-match verification is lossless, out[a] (the token emitted after the first rejected draft) is a
draw from the target conditioned on rejecting drafts[a]: exactly the distribution a sibling leaf at that depth would
be verified against. That makes the rescue value of any "chain + leaves" tree exact from these rows.

Cost oracle (VLLM_TREE_ORACLE_TIMING=1): one row per engine step
  {"t":"s","ts","B","q","uni","dt_ms","draft_ms","ctx_max"}
dt_ms = host wall time since the previous step's same point (the engine step period: --no-async-scheduling makes
steps strictly sequential); draft_ms = GPU time of this step's drafter (CUDA events, read after the step's own sync).

Overhead: one topk + logsumexp on [B, vocab] per draft depth (eager, outside graphs), one non-blocking D2H copy per
step, JSON encoding and file I/O on a background thread. Only TP rank 0 logs.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from typing import Any

import torch

_TRUE = ("1", "true", "yes", "on")


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in _TRUE


class _Writer:
    """Background JSONL writer: callers enqueue dicts; a daemon thread encodes and appends them."""

    def __init__(self, path: str, flush_s: float = 1.0):
        self.path = path
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self._q: queue.Queue = queue.Queue()
        self._flush_s = flush_s
        self._closed = False
        self._t = threading.Thread(target=self._run, name="tree-oracle-writer", daemon=True)
        self._t.start()
        atexit.register(self.close)

    def put(self, row: dict) -> None:
        if not self._closed:
            self._q.put(row)

    def _drain(self, block: bool) -> bool:
        """Write everything queued (waiting up to flush_s for the first row when block). True if anything was written."""
        try:
            row = self._q.get(timeout=self._flush_s) if block else self._q.get_nowait()
        except queue.Empty:
            return False
        while True:
            self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            try:
                row = self._q.get_nowait()
            except queue.Empty:
                return True

    def _run(self) -> None:
        while True:
            wrote = self._drain(block=True)
            self._fh.flush()
            if self._closed and not wrote:
                break

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._t.join(timeout=10)
            if not self._t.is_alive():
                while self._drain(block=False):
                    pass
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass


def topk_with_probs(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k ids and their softmax probabilities (temperature 1) of [B, V] logits. Rank 0 == argmax."""
    lf = logits.float()
    vals, ids = torch.topk(lf, k, dim=-1)
    probs = torch.exp(vals - torch.logsumexp(lf, dim=-1, keepdim=True))
    return ids, probs


class TreeOracle:
    """Process-wide oracle state. All hooks are no-ops unless enabled; callers gate on `.value` / `.timing`."""

    def __init__(self, env: dict | None = None, rank: int | None = None):
        env = os.environ if env is None else env
        on = lambda n: str(env.get(n, "0")).strip().lower() in _TRUE  # noqa: E731
        self.value = on("VLLM_TREE_ORACLE")
        self.timing = on("VLLM_TREE_ORACLE_TIMING")
        self.k = max(1, int(env.get("VLLM_TREE_ORACLE_TOPK", "8")))
        self.max_rows = int(env.get("VLLM_TREE_ORACLE_MAX_ROWS", "0"))  # 0 = unlimited
        self.path = env.get("VLLM_TREE_ORACLE_PATH", "/tmp/vllm_tree_oracle.jsonl")
        # while this file exists the value logging is paused (timing continues): one boot serves a cost bench
        # (no top-k overhead in the measured step) and the value traffic
        self.pause_file = env.get("VLLM_TREE_ORACLE_PAUSE_FILE", "")
        self._paused = False
        self._pause_checked = -1e9
        self.rank = rank
        self._writer: _Writer | None = None
        self._rows = 0
        # value oracle: per-depth GPU tensors of the propose() in flight, then one pending batch per step
        self._depth: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._next: dict | None = None   # written by the propose of the current step
        self._cur: dict | None = None    # drafts being verified in the current step
        self._seq: dict[str, int] = {}
        # cost oracle
        self._t_prev: float | None = None
        self._ev: tuple[Any, Any] | None = None

    @property
    def enabled(self) -> bool:
        return self.value or self.timing

    def _active_rank(self) -> bool:
        if self.rank is None:
            try:
                from vllm.distributed import get_tp_group

                self.rank = get_tp_group().rank_in_group
            except Exception:
                self.rank = 0
        return self.rank == 0

    def writer(self) -> _Writer:
        if self._writer is None:
            self._writer = _Writer(self.path)
            self._writer.put({"t": "meta", "ts": time.time(), "pid": os.getpid(), "topk": self.k,
                              "value": self.value, "timing": self.timing,
                              "env": {k: v for k, v in os.environ.items()
                                      if k.startswith(("VLLM_TREE_ORACLE", "V02_SPEC"))}})
        return self._writer

    # ---------------------------------------------------------------- drafter side (llm_base_proposer.py)
    def begin_propose(self) -> None:
        self._depth = []

    def capture_logits(self, logits: torch.Tensor) -> None:
        """Called at every draft sampling site with this depth's [B, V] draft logits."""
        if not self.live or not self._active_rank():
            return
        if logits.is_cuda and torch.cuda.is_current_stream_capturing():
            return
        ids, probs = topk_with_probs(logits, self.k)
        self._depth.append((ids, probs))

    # ---------------------------------------------------------------- runner side (gpu_model_runner.py)
    def begin_step(self) -> None:
        """Start of sample_tokens: the drafts proposed last step are the ones verified now."""
        self._cur, self._next = self._next, None
        if self.pause_file:
            now = time.monotonic()
            if now - self._pause_checked >= 1.0:
                self._pause_checked = now
                paused = os.path.exists(self.pause_file)
                if paused != self._paused and self.value and self._active_rank():
                    self.writer().put({"t": "pause", "ts": time.time(), "paused": paused})
                self._paused = paused

    @property
    def live(self) -> bool:
        """Value logging active right now (enabled and not paused)."""
        return self.value and not self._paused

    def finish_propose(self, req_ids: list[str]) -> None:
        """After drafter.propose(): bind the captured depths to the batch's request ids (row order)."""
        depth, self._depth = self._depth, []
        if not self.live or not depth or not self._active_rank():
            return
        b = depth[0][0].shape[0]
        if any(d[0].shape[0] != b for d in depth):
            return
        ids = torch.stack([d[0] for d in depth], dim=1)            # [B, D, k]
        probs = torch.stack([d[1] for d in depth], dim=1)
        ids_cpu = ids.to("cpu", non_blocking=True)
        probs_cpu = probs.to("cpu", non_blocking=True)
        ev = torch.cuda.Event() if ids.is_cuda else None
        if ev is not None:
            ev.record()
        self._next = {"req_ids": list(req_ids[:b]), "ids": ids_cpu, "p": probs_cpu, "ev": ev}

    def on_verified(self, req_ids: list[str], valid_sampled: list[list[int]], scheduled_spec: dict,
                    meta: dict[str, tuple]) -> None:
        """After bookkeeping: join the verified step with the drafts' top-k. meta[req] = (ctx, temp, top_k, top_p, so)."""
        cur, self._cur = self._cur, None
        if not self.live or cur is None or not self._active_rank():
            return
        if self.max_rows and self._rows >= self.max_rows:
            return
        if cur["ev"] is not None:
            cur["ev"].synchronize()
        row_of = {r: i for i, r in enumerate(cur["req_ids"])}
        ids_l = cur["ids"].tolist()
        p_l = cur["p"].tolist()
        w = self.writer()
        ts = time.time()
        for i, req in enumerate(req_ids):
            drafts = scheduled_spec.get(req) if scheduled_spec else None
            j = row_of.get(req)
            if not drafts or j is None or i >= len(valid_sampled) or not valid_sampled[i]:
                continue
            out = list(valid_sampled[i])
            seq = self._seq.get(req, 0)
            self._seq[req] = seq + 1
            ctx, temp, top_k, top_p, so = meta.get(req, (None, None, None, None, None))
            w.put({"t": "v", "ts": round(ts, 4), "req": req, "seq": seq, "ctx": ctx, "temp": temp, "top_k": top_k,
                   "top_p": top_p, "so": so, "drafts": list(drafts), "ids": ids_l[j],
                   "p": [[round(x, 5) for x in r] for r in p_l[j]], "out": out, "acc": len(out) - 1})
            self._rows += 1
        if len(self._seq) > 4096:  # finished requests: keep the map bounded
            live = set(req_ids)
            self._seq = {r: s for r, s in self._seq.items() if r in live}

    # ---------------------------------------------------------------- cost oracle
    def draft_timer_start(self):
        if not self.timing or not self._active_rank() or not torch.cuda.is_available():
            return None
        e0 = torch.cuda.Event(enable_timing=True)
        e0.record()
        return e0

    def draft_timer_stop(self, e0) -> None:
        if e0 is None:
            return
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record()
        self._ev = (e0, e1)

    def on_step(self, num_reqs: int, total_tokens: int, uniform: bool, ctx_max: int | None) -> None:
        """After bookkeeping (the step's GPU work, drafter included, is complete)."""
        if not self.timing or not self._active_rank():
            return
        now = time.perf_counter()
        dt = None if self._t_prev is None else (now - self._t_prev) * 1e3
        self._t_prev = now
        draft_ms = None
        if self._ev is not None:
            try:
                draft_ms = self._ev[0].elapsed_time(self._ev[1])
            except Exception:
                draft_ms = None
            self._ev = None
        self.writer().put({"t": "s", "ts": round(time.time(), 4), "B": num_reqs, "q": total_tokens, "uni": uniform,
                           "dt_ms": None if dt is None else round(dt, 3),
                           "draft_ms": None if draft_ms is None else round(draft_ms, 3), "ctx_max": ctx_max})


_ORACLE: TreeOracle | None = None


def get_tree_oracle() -> TreeOracle:
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = TreeOracle()
    return _ORACLE
