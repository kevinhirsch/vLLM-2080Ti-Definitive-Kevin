# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for the vllm-project/vllm#52805 fork port
("[Bugfix][Structured Output] Stop XGrammar token batches at termination"),
in ``vllm/v1/structured_output/backend_xgrammar.py``, ``XgrammarGrammar``.

Upstream bug: when a terminating token (e.g. EOS landing in an MTP draft
slot) appeared mid-batch -- not as the last token -- ``accept_tokens``
kept feeding the matcher tokens *after* termination (the ``FSM``'s own
post-termination "trying to accept new token" error), and
``validate_tokens`` didn't short-circuit once the matcher had already
terminated inside its own loop. ``reset()`` also never cleared
``_is_terminated``, so a request reused for a fresh sequence (e.g. a KV
cache slot recycled with prefix reuse) could stay stuck "terminated" from
its previous life. The PR author's own live-model test command matches
this fork's deployment shape: Qwen3.8-27B, TP=2, tool/reasoning parsers,
MTP with num_speculative_tokens=3.

The fix (mirrored here from the upstream diff):
  * ``accept_tokens``: on prior termination, return True (not False) --
    trailing tokens after EOS are a normal MTP artifact, not a real FSM
    failure. Move the termination check *inside* the per-token loop and
    break as soon as it fires, instead of feeding every token in the
    batch to an already-terminated matcher.
  * ``validate_tokens``: early-return ``[]`` if already terminated; break
    out of the validation loop as soon as the matcher terminates
    mid-batch (instead of validating tokens the matcher would silently
    have refused after termination).
  * ``reset()``: also reset ``_is_terminated`` back to False.

This exercises the real ``XgrammarGrammar`` dataclass directly (no CUDA
dependency in these methods) using a minimal duck-typed stand-in for
``xgr.GrammarMatcher`` instead of a real compiled grammar/tokenizer --
avoids depending on the exact installed xgrammar version's compile API
while still driving the actual production state-machine code in
accept_tokens/validate_tokens/reset.
"""

from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

EOS = 2
TRAILING = 5


class _FakeXgrMatcher:
    """Minimal stand-in for xgr.GrammarMatcher's mutation surface used by
    XgrammarGrammar: accept_token, is_terminated, rollback, reset. Accepts
    every token (a real matcher would also reject grammar-violating
    tokens, but the bug this covers is about tokens *after* a legal
    termination, not rejected ones) and terminates whenever a token from
    ``terminating_tokens`` is accepted -- standing in for a real grammar
    reaching its stop state on EOS.
    """

    def __init__(self, terminating_tokens: set[int]):
        self._terminating_tokens = terminating_tokens
        self._accepted: list[int] = []
        self._terminated = False
        self.accept_token_calls = 0

    def accept_token(self, token: int) -> bool:
        self.accept_token_calls += 1
        self._accepted.append(token)
        if token in self._terminating_tokens:
            self._terminated = True
        return True

    def is_terminated(self) -> bool:
        return self._terminated

    def rollback(self, num_tokens: int) -> None:
        del self._accepted[-num_tokens:]
        self._terminated = any(t in self._terminating_tokens for t in self._accepted)

    def reset(self) -> None:
        self._accepted.clear()
        self._terminated = False


def _make_grammar() -> XgrammarGrammar:
    return XgrammarGrammar(
        vocab_size=100,
        matcher=_FakeXgrMatcher({EOS}),
        ctx=None,  # type: ignore[arg-type]  # unused by the methods under test
    )


def test_xgrammar_accept_tokens_stops_at_termination() -> None:
    """Tokens after a terminating EOS do not reach the matcher, and
    accept_tokens keeps returning True (not False) once terminated."""
    grammar = _make_grammar()

    # EOS lands mid-batch (an MTP draft slot), followed by a trailing token.
    assert grammar.accept_tokens("req", [EOS, TRAILING]) is True
    assert grammar.is_terminated()
    # Only EOS was actually processed -- the loop broke before TRAILING.
    assert grammar.num_processed_tokens == 1
    assert grammar.matcher.accept_token_calls == 1  # type: ignore[attr-defined]

    # A later call after termination (e.g. more spec-decode tokens for an
    # already-finished request) must not touch the matcher again and must
    # still report success, not failure.
    calls_before = grammar.matcher.accept_token_calls  # type: ignore[attr-defined]
    assert grammar.accept_tokens("req", [TRAILING]) is True
    assert grammar.num_processed_tokens == 1
    assert grammar.matcher.accept_token_calls == calls_before  # type: ignore[attr-defined]

    grammar.reset()
    assert not grammar.is_terminated()
    assert grammar.num_processed_tokens == 0


def test_xgrammar_validate_tokens_stops_at_termination() -> None:
    """Validation rolls back after reaching a terminating EOS, and stops
    validating (and short-circuits entirely) once the grammar is already
    terminated."""
    grammar = _make_grammar()

    # Validating past a terminating token: only EOS is returned as
    # accepted, and the matcher is rolled back to its pre-EOS state (since
    # validate_tokens must not commit/advance real state).
    assert grammar.validate_tokens([EOS, TRAILING]) == [EOS]
    assert not grammar.matcher.is_terminated()  # type: ignore[attr-defined]
    # validate_tokens never touches grammar-level termination.
    assert not grammar.is_terminated()

    # Now actually commit EOS via accept_tokens.
    assert grammar.accept_tokens("req", [EOS]) is True
    assert grammar.is_terminated()

    # Once the grammar itself is terminated, validate_tokens short-circuits
    # without touching the matcher at all.
    calls_before = grammar.matcher.accept_token_calls  # type: ignore[attr-defined]
    assert grammar.validate_tokens([TRAILING]) == []
    assert grammar.matcher.accept_token_calls == calls_before  # type: ignore[attr-defined]
