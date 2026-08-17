# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXP-039 (S4): FSM-gated *scoped* drafter for verbatim re-emission.

Layers on top of the MTP/EAGLE proposer (does NOT replace it). Only proposes a
draft when the request is *provably* inside a verbatim re-emission span -- i.e.
the last ``G`` generated tokens exactly match a unique (or near-unique) G-gram in
the request's **prompt** tokens, so the continuation is determined by the context
being copied from (file rewrites, quoted blocks, diffs). Otherwise it returns the
MTP draft untouched, so MTP-K2 behaviour on generation-shaped work is unchanged.

Losslessness is unconditional: the output is only a speculative *draft*, verified
by vLLM's standard rejection sampler. A mis-fired gate can only waste a step's
speculative budget, never corrupt output -- so the gate is tuned purely for
acceptance probability.

Design: ``docs/exp039-scoped-drafter-design.md``.
Gated by ``VLLM_S4_SCOPED_DRAFTER=1`` (default off = strict no-op at the call site).
"""

import os

import numpy as np

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# Polynomial rolling-hash base for the fixed-window prompt index. Correctness
# never depends on this value (hash collisions are resolved by an exact G-token
# comparison), so any odd constant works; uint64 wraparound = mod 2**64.
_HASH_BASE = np.uint64(1_000_003)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


class _PromptIndex:
    """Fixed-window (length-G) hashed index over one request's prompt tokens.

    Stored as a sorted array of G-gram hashes plus the parallel array of match
    *end* positions (the index into ``prompt`` at which the continuation begins).
    Lookup is ``np.searchsorted`` -> O(log n); the prompt is immutable for the
    life of the request, so this is built once and never rebuilt.
    """

    __slots__ = ("prompt", "g", "n", "_sorted_h", "_sorted_ends", "_empty")

    def __init__(self, prompt: np.ndarray, g: int):
        # Work in int64 token space; hash in uint64 (natural mod 2**64).
        self.prompt = np.ascontiguousarray(prompt, dtype=np.int64)
        self.g = g
        self.n = int(self.prompt.shape[0])
        self._empty = self.n < g
        if self._empty:
            self._sorted_h = np.empty(0, dtype=np.uint64)
            self._sorted_ends = np.empty(0, dtype=np.int64)
            return
        num_windows = self.n - g + 1
        a = self.prompt.astype(np.uint64)
        # h[s] = sum_{t=0..g-1} a[s+t] * base**(g-1-t)  (mod 2**64), vectorised
        # over all windows s in [0, num_windows). g <= 16 iterations. uint64
        # overflow == the intended mod-2**64 wraparound, so silence the warning.
        h = np.zeros(num_windows, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for t in range(g):
                h = h * _HASH_BASE + a[t : t + num_windows]
        ends = np.arange(g, self.n + 1, dtype=np.int64)  # continuation start
        order = np.argsort(h, kind="stable")
        self._sorted_h = h[order]
        self._sorted_ends = ends[order]

    def candidates(self, needle_hash: np.uint64) -> np.ndarray:
        """End positions of prompt G-grams whose hash == needle_hash."""
        if self._empty:
            return self._sorted_ends  # empty
        lo = int(np.searchsorted(self._sorted_h, needle_hash, side="left"))
        hi = int(np.searchsorted(self._sorted_h, needle_hash, side="right"))
        return self._sorted_ends[lo:hi]


class _ReqState:
    """Per-request scoped-drafter state: index + FSM cursor + accounting hook."""

    __slots__ = (
        "index",
        "cursor",          # prompt pos currently being copied, or -1 (SEARCHING)
        "run_len",         # consecutive COPYING steps (accounting only)
        "last_draft",      # draft emitted last step (for acceptance reconcile)
        "last_source",     # "scoped" | "mtp"
    )

    def __init__(self, index: _PromptIndex):
        self.index = index
        self.cursor = -1
        self.run_len = 0
        self.last_draft: list[int] | None = None
        self.last_source = ""


class _SourceStats:
    __slots__ = ("drafts", "draft_tokens", "accepted", "per_pos")

    def __init__(self, k: int):
        self.drafts = 0
        self.draft_tokens = 0
        self.accepted = 0
        self.per_pos = [0] * k

    def observe(self, n_draft: int, n_accepted: int):
        self.drafts += 1
        self.draft_tokens += n_draft
        self.accepted += n_accepted
        for i in range(min(n_accepted, len(self.per_pos))):
            self.per_pos[i] += 1

    def summary(self) -> str:
        mal = 1 + (self.accepted / self.drafts) if self.drafts else float("nan")
        rate = (self.accepted / self.draft_tokens) if self.draft_tokens else float("nan")
        pos = ", ".join(
            f"{(p / self.drafts):.3f}" if self.drafts else "nan" for p in self.per_pos
        )
        return (
            f"drafts={self.drafts} draft_toks={self.draft_tokens} "
            f"accepted={self.accepted} mean_accept_len={mal:.2f} "
            f"acc_rate={rate:.3f} per_pos=[{pos}]"
        )


class ScopedReemissionDrafter:
    """FSM-gated scoped drafter. See module docstring.

    ``merge(mtp_drafts, input_batch, sampled_token_ids)`` takes the MTP proposer's
    per-request drafts and returns the merged ``list[list[int]]``: gate-open rows
    are replaced by the scoped continuation, gate-closed rows are MTP's draft
    verbatim.
    """

    def __init__(self, vllm_config: VllmConfig):
        spec = vllm_config.speculative_config
        assert spec is not None, "Speculative config must be set"
        # num_speculative_tokens sizes the KV lookahead / cudagraph / stats
        # buffers; a scoped draft longer than this would overflow the lookahead
        # reservation, so it is the hard ceiling on K_scoped (see design doc S5).
        self.num_speculative_tokens = int(spec.num_speculative_tokens)
        self.max_model_len = int(vllm_config.model_config.max_model_len)

        self.g = max(1, _env_int("VLLM_S4_G", 12))
        self.k_scoped = max(
            1, min(_env_int("VLLM_S4_K_SCOPED", 16), self.num_speculative_tokens)
        )
        # Max prompt occurrences of the G-gram for the gate to fire (1 = unique).
        self.min_uniq = max(1, _env_int("VLLM_S4_MIN_UNIQ", 1))
        self.log_every = _env_int("VLLM_S4_LOG_EVERY", 2000)

        self._states: dict[str, _ReqState] = {}
        # Accounting (design doc S6): scoped vs mtp, per-position, measured apart.
        self._stat_scoped = _SourceStats(self.num_speculative_tokens)
        self._stat_mtp = _SourceStats(self.num_speculative_tokens)
        self._gate_fires = 0
        self._copy_spans = 0
        self._span_len_total = 0
        self._since_log = 0

        logger.info(
            "S4 scoped-reemission drafter ENABLED: G=%d K_scoped=%d "
            "min_uniq=%d (num_spec_tokens=%d, mtp_draft_cap=%s)",
            self.g,
            self.k_scoped,
            self.min_uniq,
            self.num_speculative_tokens,
            os.environ.get("VLLM_MTP_DRAFT_CAP", "0"),
        )

    # -- lifecycle ---------------------------------------------------------

    def _get_state(self, req_id: str, input_batch, index_row: int) -> _ReqState:
        st = self._states.get(req_id)
        if st is None:
            num_prompt = int(input_batch.num_prompt_tokens[index_row])
            prompt = input_batch.token_ids_cpu[index_row, :num_prompt]
            st = _ReqState(_PromptIndex(prompt, self.g))
            self._states[req_id] = st
        return st

    def _evict_stale(self, active_req_ids) -> None:
        stale = self._states.keys() - set(active_req_ids)
        for req_id in stale:
            self._states.pop(req_id, None)

    # -- gate --------------------------------------------------------------

    def _hash_tail(self, tail: np.ndarray) -> np.uint64:
        # hash of a length-g token window with the same polynomial as
        # _PromptIndex. uint64 overflow == the intended mod-2**64 wraparound.
        base = _HASH_BASE
        with np.errstate(over="ignore"):
            h = np.uint64(0)
            for t in range(self.g):
                h = h * base + np.uint64(int(tail[t]))
        return h

    def _effective_tail(
        self, seq: np.ndarray, n: int, sampled_ids: list[int]
    ) -> np.ndarray:
        """Last G tokens of the sequence INCLUDING this step's sampled tokens.

        CRITICAL (root cause of EXP-039's acceptance collapse): ``merge`` runs in
        ``gpu_model_runner.propose_draft_token_ids``, which for the MTP/EAGLE path
        executes BEFORE ``_bookkeeping_sync`` writes this step's freshly-sampled
        tokens into ``token_ids_cpu`` and advances ``num_tokens_no_spec``. So the
        committed tail ``seq[:n]`` lags by ``len(sampled_ids)`` tokens. Splicing
        the just-sampled tokens on realigns the needle -- and therefore the
        continuation index -- with what MTP proposes from (it is seeded from these
        same tokens via ``next_token_ids``). Without this, the drafted position 0
        is the token JUST sampled (already emitted -> always rejected)."""
        g = self.g
        s = len(sampled_ids)
        if s >= g:
            return np.asarray(sampled_ids[s - g :], dtype=np.int64)
        head = np.asarray(seq[n - (g - s) : n], dtype=np.int64)
        tail = np.asarray(sampled_ids, dtype=np.int64)
        return np.concatenate([head, tail])

    def _scoped_draft(
        self, st: _ReqState, seq: np.ndarray, n: int, sampled_ids: list[int]
    ) -> list[int] | None:
        """Return the scoped continuation if the gate is OPEN, else None.

        Memoryless two-state FSM: COPYING iff the last G tokens (the committed
        tail spliced with THIS step's sampled tokens -- see ``_effective_tail``)
        uniquely (<= min_uniq) match a prompt G-gram; then draft the next K prompt
        tokens, i.e. the continuation that FOLLOWS this step's sampled tokens.
        """
        idx = st.index
        n_eff = n + len(sampled_ids)  # true length once this step is committed
        if idx.n < self.g or n_eff < self.g:
            st.cursor = -1
            return None
        if n_eff >= self.max_model_len:
            st.cursor = -1
            return None

        needle = self._effective_tail(seq, n, sampled_ids)
        needle_hash = self._hash_tail(needle)
        cands = idx.candidates(needle_hash)  # candidate end positions in prompt
        if cands.shape[0] == 0:
            st.cursor = -1
            return None

        prompt = idx.prompt
        # Verify exact match (guards hash collisions); collect valid end-positions
        # that also have at least one continuation token left.
        ends: list[int] = []
        for e in cands.tolist():
            s = e - self.g
            if e < idx.n and np.array_equal(prompt[s:e], needle):
                ends.append(e)
        if not ends:
            st.cursor = -1
            return None
        if len(ends) > self.min_uniq:
            # Ambiguous G-gram: fire only on the continuation prefix on which
            # ALL candidate positions agree (design doc S4). If they disagree at
            # the first token, the gate is closed.
            return self._agreed_continuation(st, prompt, ends, idx.n)

        # Unique (or within min_uniq with identical continuation start): copy.
        e = ends[0]
        k = min(self.k_scoped, idx.n - e)
        if k <= 0:
            st.cursor = -1
            return None
        st.cursor = e
        return prompt[e : e + k].tolist()

    def _agreed_continuation(
        self, st: _ReqState, prompt: np.ndarray, ends: list[int], n_prompt: int
    ) -> list[int] | None:
        max_k = min(self.k_scoped, min(n_prompt - e for e in ends))
        if max_k <= 0:
            st.cursor = -1
            return None
        first = ends[0]
        draft: list[int] = []
        for j in range(max_k):
            tok = int(prompt[first + j])
            if all(int(prompt[e + j]) == tok for e in ends[1:]):
                draft.append(tok)
            else:
                break
        if not draft:
            st.cursor = -1
            return None
        st.cursor = first
        return draft

    # -- accounting --------------------------------------------------------

    def _reconcile(self, st: _ReqState, sampled_ids: list[int]) -> None:
        """Score the draft emitted last step against what was actually accepted.

        The draft emitted at step T-1 is verified by the target model at step T;
        its accepted prefix arrives as THIS step's ``sampled_ids`` (the rejection
        sampler emits [accepted drafts..., bonus], so accepted == the matching
        prefix). This runs before the draft is overwritten. Tagged by source so
        scoped and MTP acceptance are measured separately (design doc S6).

        Note: this internal per-source accounting is a diagnostic (surfaced only
        in the periodic log); the AUTHORITATIVE per-position acceptance is vLLM's
        own rejection-sampler Prometheus metric, which the A/B bench scrapes.
        """
        draft = st.last_draft
        if draft is None:
            return
        confirmed = 0
        for i in range(min(len(draft), len(sampled_ids))):
            if int(sampled_ids[i]) == draft[i]:
                confirmed += 1
            else:
                break
        stat = self._stat_scoped if st.last_source == "scoped" else self._stat_mtp
        stat.observe(len(draft), confirmed)

    def _maybe_log(self) -> None:
        if self.log_every <= 0:
            return
        self._since_log += 1
        if self._since_log < self.log_every:
            return
        self._since_log = 0
        mean_span = (
            self._span_len_total / self._copy_spans if self._copy_spans else 0.0
        )
        logger.info(
            "S4 scoped-drafter: gate_fires=%d copy_spans=%d mean_span_len=%.1f\n"
            "  scoped: %s\n  mtp   : %s",
            self._gate_fires,
            self._copy_spans,
            mean_span,
            self._stat_scoped.summary(),
            self._stat_mtp.summary(),
        )

    # -- main entry point --------------------------------------------------

    def merge(
        self,
        mtp_drafts,
        input_batch,
        sampled_token_ids,
    ) -> list[list[int]]:
        """Merge MTP drafts with scoped drafts per request.

        ``mtp_drafts`` may be a ``torch.Tensor`` (padded batch) or ``list``. The
        returned value is always a ``list[list[int]]`` (list draft path).
        """
        if hasattr(mtp_drafts, "tolist"):
            base = mtp_drafts.tolist()
        else:
            base = mtp_drafts

        # Normalise sampled_token_ids to per-row lists of the tokens sampled THIS
        # step (bonus + accepted drafts), padding (-1) stripped. Partial prefills
        # produce empty rows -> no draft. These rows are (a) the seed the scoped
        # draft must continue FROM (they are not yet in token_ids_cpu, see
        # _effective_tail) and (b) the verification result used to reconcile the
        # previous step's draft.
        if hasattr(sampled_token_ids, "tolist"):
            sampled_rows = [
                [t for t in row if t != -1] for row in sampled_token_ids.tolist()
            ]
        else:
            sampled_rows = [
                [t for t in row if t != -1] for row in sampled_token_ids
            ]

        req_ids = input_batch.req_ids
        num_reqs = len(req_ids)
        merged: list[list[int]] = []

        for i in range(num_reqs):
            mtp_i = list(base[i]) if i < len(base) else []
            # Filter MTP padding (-1) that the padded tensor path leaves in rows.
            mtp_i = [t for t in mtp_i if t != -1]

            sampled_ids = sampled_rows[i] if i < len(sampled_rows) else []
            has_sampled = bool(sampled_ids)
            req_id = req_ids[i]
            index_row = input_batch.req_id_to_index[req_id]
            num_tokens = int(input_batch.num_tokens_no_spec[i])

            st = self._get_state(req_id, input_batch, index_row)
            seq = input_batch.token_ids_cpu[index_row]

            # Reconcile the previous step's draft against this step's accepted
            # tokens (which arrive in sampled_ids) for acceptance accounting.
            self._reconcile(st, sampled_ids)

            scoped = None
            if has_sampled:
                prev_cursor = st.cursor
                scoped = self._scoped_draft(st, seq, num_tokens, sampled_ids)
                if scoped is not None:
                    self._gate_fires += 1
                    if prev_cursor < 0:
                        # SEARCHING -> COPYING: a new verbatim span begins.
                        self._copy_spans += 1
                        st.run_len = 1
                    else:
                        st.run_len += 1
                    self._span_len_total += 1
                else:
                    st.run_len = 0

            if scoped is not None:
                chosen, source = scoped, "scoped"
            else:
                chosen, source = mtp_i, "mtp"

            # Stash for next-step reconciliation.
            st.last_draft = list(chosen) if chosen else None
            st.last_source = source

            merged.append(chosen)

        self._evict_stale(req_ids)
        self._maybe_log()
        return merged
