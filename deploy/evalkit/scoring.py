"""
Deterministic scorers, one function per category. Pure functions of
(item, ...raw data...) -> result dict with at least {"passed": bool}.

Both run_eval.py (live, right after each response) and score_only.py
(offline, from saved raw files) call into this module, so scoring logic
lives in exactly one place and re-scoring is a real recomputation, not a
replay of cached booleans.
"""
import ast
import calendar
import json
import os
import re
import subprocess
import tempfile

from predicates import check_predicate, call_matches, expand_value_from_previous

# ---------------------------------------------------------------------------
# tool_call
# ---------------------------------------------------------------------------


def parse_tool_calls(message):
    """message = the OpenAI 'assistant' message dict. Returns
    [(name, args_dict_or_None, raw_arguments_str), ...]"""
    out = []
    for tc in (message or {}).get("tool_calls") or []:
        fn = tc.get("function", {})
        name = fn.get("name")
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = None
        out.append((name, args, raw_args))
    return out


def score_tool_call(item, message):
    check = item["check"]
    expected = check["expected_calls"]
    mode = check.get("mode", "any_order")

    calls = parse_tool_calls(message)
    if len(calls) != len(expected):
        return {
            "passed": False,
            "reason": f"expected {len(expected)} tool call(s), got {len(calls)}",
            "actual_calls": [{"name": n, "args": a, "raw_arguments": r} for n, a, r in calls],
        }

    if mode == "in_order":
        ok = all(call_matches(n, a, e) for (n, a, _r), e in zip(calls, expected))
        reason = "" if ok else "one or more calls didn't match the expected name/args at their position"
    else:  # any_order
        remaining = list(expected)
        ok = True
        for n, a, _r in calls:
            idx = next((i for i, e in enumerate(remaining) if call_matches(n, a, e)), None)
            if idx is None:
                ok = False
                break
            remaining.pop(idx)
        ok = ok and not remaining
        reason = "" if ok else "no matching expected-call predicate set found for one or more actual calls"

    return {
        "passed": ok,
        "reason": reason,
        "actual_calls": [{"name": n, "args": a, "raw_arguments": r} for n, a, r in calls],
    }


# ---------------------------------------------------------------------------
# agentic_chain
# ---------------------------------------------------------------------------


def evaluate_tool_step(step_spec, assistant_message, prev_inject_result):
    """Evaluate one non-final agentic_chain step against the model's
    assistant message. Returns a result dict; also returns which tool_call
    (if any) to use for injecting the canned result, so the chain can
    proceed even when the check fails (useful diagnostic signal)."""
    calls = parse_tool_calls(assistant_message)
    if not calls:
        return {"passed": False, "reason": "model made no tool_call", "used_call": None}, None
    if len(calls) != 1:
        # HOLE FIX: previously only calls[0] was ever inspected, so a model
        # that made the right call *plus* an extra spurious/parallel call in
        # the same turn silently passed. These chain steps are specified as
        # single-tool-per-turn, so more than one call is itself a failure.
        return {
            "passed": False,
            "reason": f"expected exactly 1 tool_call this step, got {len(calls)}",
            "used_call": None,
        }, None

    name, args, _raw = calls[0]
    expected = {
        "name": step_spec["expect_tool"],
        "args": [expand_value_from_previous(p, prev_inject_result or {}) for p in step_spec.get("arg_checks", [])],
    }
    ok = call_matches(name, args, expected)
    result = {
        "passed": ok,
        "reason": "" if ok else "tool name/args didn't match expectation",
        "used_call": {"name": name, "args": args},
    }
    tool_call_id = assistant_message["tool_calls"][0].get("id")
    return result, tool_call_id



# HOLE FIX: item specs (generate_items.py) build answer_check patterns for
# date-bearing facts straight from the literal ISO string in the injected
# tool result -- e.g. step_final(r"2026-08-20") for a track_package result
# of {"eta": "2026-08-20", ...}. Nothing about the prompt asks the model to
# echo that exact digit string back, and a correct, fully-grounded answer
# routinely paraphrases it into prose ("The estimated delivery date (ETA)
# is August 20, 2026"). The old literal-regex match then scored a
# factually-correct multi-step answer as a failure purely because of date
# formatting (agentic_chain_002 in baseline-v0-20260816: right tools, right
# args, right date, "content didn't match /2026-08-20/i"). Any ISO
# YYYY-MM-DD date embedded in an answer_check pattern is now expanded, in
# place, into an alternation that also accepts the common human-readable
# renderings of that same calendar date. The literal ISO string is always
# kept as one of the alternatives, so this only ever *widens* what counts
# as a pass -- it can't turn a previously-failing wrong-date answer into a
# pass, only stop penalizing correct answers for formatting.
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def _date_variants_pattern(year_s, month_s, day_s):
    """Build a regex alternation matching common human-readable renderings
    of the ISO date year_s-month_s-day_s, plus the literal ISO form itself.
    Falls back to just the literal ISO form if the digits don't form a
    valid calendar date (defensive -- an answer_check pattern could in
    principle contain a YYYY-MM-DD-shaped non-date, e.g. an ID)."""
    literal = re.escape(f"{year_s}-{month_s}-{day_s}")
    try:
        year, month, day = int(year_s), int(month_s), int(day_s)
        month_name = calendar.month_name[month]
        month_abbr = calendar.month_abbr[month]
        if not month_name or not (1 <= day <= 31):
            raise ValueError("not a valid calendar date")
    except (ValueError, IndexError):
        return literal

    day_noz = str(day)
    ordinal = r"(?:st|nd|rd|th)?"
    variants = [
        literal,                                                 # 2026-08-20
        rf"{month_name}\s+{day_noz}{ordinal},?\s+{year}",         # August 20, 2026 / August 20th 2026
        rf"{month_abbr}\.?\s+{day_noz}{ordinal},?\s+{year}",      # Aug 20, 2026
        rf"{day_noz}{ordinal}\s+{month_name}\s+{year}",           # 20 August 2026
        rf"{day_noz}{ordinal}\s+{month_abbr}\.?\s+{year}",        # 20 Aug 2026
        rf"{month}/{day}/{year}",                                 # 8/20/2026
        rf"{month_s}/{day_s}/{year}",                             # 08/20/2026
    ]
    return "(?:" + "|".join(variants) + ")"


def _tolerant_date_pattern(pattern):
    return _ISO_DATE_RE.sub(lambda m: _date_variants_pattern(*m.groups()), pattern)


def evaluate_final_step(step_spec, assistant_message):
    content = (assistant_message or {}).get("content") or ""
    calls = parse_tool_calls(assistant_message)
    if calls:
        return {"passed": False, "reason": "expected a final text answer, model called a tool instead", "content": content}
    check = step_spec["answer_check"]
    pattern = check["pattern"]
    ok = re.search(_tolerant_date_pattern(pattern), content, re.IGNORECASE) is not None
    if not ok:
        return {"passed": False, "reason": f"content didn't match /{pattern}/i", "content": content}
    # HOLE FIX: a bare keyword/fact regex can't tell success from failure --
    # "I was NOT able to confirm the email was sent" contains both "confirm"
    # and "sent" and used to pass every final-step check that used those
    # generic completion words. must_not_pattern is a cheap, still-fully-
    # deterministic negation guard: if the answer trips a common
    # negation/failure word near the claimed result, we don't trust the
    # positive keyword match. Every step_final() in generate_items.py sets
    # a sensible default; a specific item can override/disable it in
    # "answer_check" if a check's own pattern already can't be negated this
    # way (e.g. it's a positive-only fact like a confirmation code).
    must_not = check.get("must_not_pattern")
    if must_not:
        bad = re.search(must_not, content, re.IGNORECASE)
        if bad:
            return {
                "passed": False,
                "reason": f"content matched /{pattern}/i but also tripped negation guard /{must_not}/i "
                          f"(looks like a failure/negated statement, not a genuine completion)",
                "content": content,
            }
    return {"passed": True, "reason": "", "content": content}


def score_agentic_chain(item, raw_record):
    """Re-derive pass/fail for every step from the saved per-step
    request/response pairs (raw_record['steps']), independent of whatever
    was decided live at run time."""
    steps_spec = item["steps"]
    steps_raw = raw_record.get("steps", [])
    results = []
    prev_inject_result = None
    all_passed = True

    for i, spec in enumerate(steps_spec):
        if i >= len(steps_raw):
            results.append({"passed": False, "reason": "no raw data for this step (chain aborted earlier)"})
            all_passed = False
            continue
        raw_step = steps_raw[i]
        if raw_step.get("status") != "ok":
            results.append({"passed": False, "reason": f"step status={raw_step.get('status')}"})
            all_passed = False
            continue
        message = raw_step["response"]["choices"][0]["message"]

        if spec.get("final"):
            res = evaluate_final_step(spec, message)
        else:
            res, _tool_call_id = evaluate_tool_step(spec, message, prev_inject_result)
            prev_inject_result = spec.get("inject_result")

        results.append(res)
        all_passed = all_passed and res["passed"]

    return {"passed": all_passed, "steps": results}


# ---------------------------------------------------------------------------
# code_exec
# ---------------------------------------------------------------------------

# HOLE FIX: the language-tag group used to be followed by a literal '\n'
# with nothing allowed in between. Real model output sometimes has trailing
# whitespace after the tag (e.g. "```python \n" or a "```python\r\n" CRLF
# fence), which made the old pattern fail to recognize the fence at all and
# fall through to executing raw prose -- a correct answer would then fail
# with a SyntaxError purely because of fence-matching brittleness.
_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)


def extract_code(text):
    """Fence-tolerant code extraction: prefer the largest fenced block;
    fall back to the raw text if there are no fences at all."""
    blocks = _FENCE_RE.findall(text or "")
    if blocks:
        return max(blocks, key=len)
    return text or ""


def run_python(code, timeout_sec=10):
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                ["python3", path], capture_output=True, text=True, timeout=timeout_sec
            )
            return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode, "timed_out": False}
        except subprocess.TimeoutExpired as e:
            return {
                "stdout": e.stdout or "",
                "stderr": (e.stderr or "") + "\n[TIMEOUT]",
                "returncode": None,
                "timed_out": True,
            }
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _calls_itself(code, func_name):
    """AST check: does the function named func_name call itself somewhere
    in its own body? Used to gate style-constrained items (e.g. 'write this
    recursively') where a correct-stdout iterative or lookup-table solution
    would otherwise pass score_code_exec's stdout-only diff."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == func_name:
                    return True
    return False


def score_code_exec(item, content):
    check = item["check"]
    code = extract_code(content)
    # HOLE FIX: the harness (the concrete test-driver calls with literal
    # inputs) used to be embedded verbatim in the prompt and the model was
    # asked to reproduce it. That let a "solution" hardcode the expected
    # outputs for exactly the visible inputs (e.g.
    # `def is_prime(n): return {-5:False,...,7919:True}[n]`) and pass
    # perfectly without implementing anything general. The harness now
    # lives only in the item spec and is appended here, server-side, at
    # scoring time -- the model never sees the concrete test inputs, so it
    # has to implement the real function to pass.
    harness = check.get("harness", "")
    full_code = f"{code}\n\n{harness}" if harness else code
    run = run_python(full_code, check.get("timeout_sec", 10))
    expected = check["expected_stdout"].rstrip("\n")
    actual = run["stdout"].rstrip("\n")
    # HOLE FIX: stdout-diffing alone can't tell a genuine implementation of a
    # style-constrained item (e.g. "using recursion") from one that just
    # hardcodes/looks up the right answers for the harness's fixed inputs.
    # An item can opt into an AST-level check by setting
    # check.require_recursive to the name of the function that must call
    # itself; items that don't set it are scored exactly as before.
    require_recursive = check.get("require_recursive")
    recursive_ok = _calls_itself(code, require_recursive) if require_recursive else True
    passed = (not run["timed_out"]) and run["returncode"] == 0 and actual == expected and recursive_ok
    return {
        "passed": passed,
        "extracted_code": code,
        "executed_code": full_code,
        "expected_stdout": expected,
        "actual_stdout": run["stdout"],
        "stderr": run["stderr"][-4000:],
        "returncode": run["returncode"],
        "timed_out": run["timed_out"],
        "recursive_ok": recursive_ok,
    }


# ---------------------------------------------------------------------------
# json_schema
# ---------------------------------------------------------------------------

from json_validator import validate_schema  # noqa: E402

_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def _json_candidates(text):
    """HOLE FIX: score_json_schema used to call json.loads() on the raw
    response only. `response_format: json_object` is a request to the
    engine, not a guarantee -- across a config/quant change the engine may
    stop honoring it, or the model may wrap the object in a ```json fence
    or a lead-in sentence despite being told not to. A correct JSON answer
    wrapped that way used to fail outright. Try, in order: the raw text;
    the largest fenced code block; the outermost {...} span."""
    text = text or ""
    candidates = [text]
    blocks = _FENCE_RE.findall(text)
    if blocks:
        candidates.append(max(blocks, key=len))
    m = _BRACE_RE.search(text)
    if m:
        candidates.append(m.group(0))
    return candidates


def score_json_schema(item, content):
    check = item["check"]
    obj = None
    last_err = None
    for candidate in _json_candidates(content):
        try:
            obj = json.loads(candidate)
            break
        except json.JSONDecodeError as e:
            last_err = e
            continue
    if obj is None:
        return {
            "passed": False,
            "reason": f"invalid JSON even after fence/brace-extraction fallback: {last_err}",
            "errors": [str(last_err)],
        }
    errors = validate_schema(obj, check["schema"])
    return {"passed": len(errors) == 0, "errors": errors, "parsed": obj}


# ---------------------------------------------------------------------------
# instruction
# ---------------------------------------------------------------------------


def score_instruction(item, content):
    check = item["check"]
    t = check["type"]
    text = content or ""

    if t == "line_count":
        # HOLE FIX: "".strip("\n").split("\n") == [""], so n was 1 for an
        # empty (or newlines-only) response instead of the correct 0 lines.
        stripped = text.strip("\n")
        n = len(stripped.split("\n")) if stripped else 0
        return {"passed": n == check["value"], "actual": n}
    if t == "word_count":
        n = len(text.split())
        return {"passed": n == check["value"], "actual": n}
    if t == "no_vowels":
        found = re.findall(r"[aeiouAEIOU]", text)
        return {"passed": len(found) == 0, "found": found[:20]}
    if t == "starts_with":
        # HOLE FIX: bare startswith() also matched e.g. "INTRODUCINGLY..."
        # for a required prefix of "INTRODUCING". Require a word boundary
        # right after the prefix so only the exact leading token counts.
        pattern = re.escape(check["value"]) + r"(?!\w)"
        return {"passed": re.match(pattern, text.strip()) is not None}
    if t == "regex_fullmatch":
        # HOLE FIX: this used to hardcode re.DOTALL, which silently makes
        # every '.' in a pattern cross line boundaries. For a pattern like
        # the "exactly 5 hyphenated lines" check
        # (r"-\s+\S.*(?:\n-\s+\S.*){4}"), that let a response with extra
        # junk lines appended after the 5 real ones still fullmatch, because
        # the trailing '.*' could swallow the extra "\n-..." lines instead
        # of being stopped by end-of-line. Default is now no DOTALL (so '.'
        # behaves per-line as line-oriented patterns assume); a check can
        # opt back in with {"dotall": true} if it genuinely wants '.' to
        # span newlines.
        flags = re.DOTALL if check.get("dotall", False) else 0
        return {"passed": re.fullmatch(check["pattern"], text.strip(), flags) is not None}
    if t == "no_letter":
        letter = check["value"].lower()
        return {"passed": letter not in text.lower()}
    if t == "all_upper":
        letters = [c for c in text if c.isalpha()]
        passed = len(letters) > 0 and all(c.isupper() for c in letters)
        return {"passed": passed, "n_letters": len(letters)}
    if t == "char_count":
        n = len(text.strip())
        return {"passed": n == check["value"], "actual": n}
    if t == "exact":
        return {"passed": text.strip() == check["value"]}

    raise ValueError(f"unknown instruction check type: {t!r}")


# ---------------------------------------------------------------------------
# loop_probe
# ---------------------------------------------------------------------------

_LOOP_LINE_RE = re.compile(r"^\s*(\d{1,4})\s*[.):]\s*(.+?)\s*$", re.MULTILINE)


def score_loop_probe(item, content, finish_reason):
    check = item["check"]
    n_expected = check.get("count", 100)
    text = content or ""

    matches = _LOOP_LINE_RE.findall(text)
    numbers = [int(m[0]) for m in matches]
    items_text = [m[1] for m in matches]

    reasons = []

    order_ok = numbers == list(range(1, n_expected + 1))
    if not order_ok:
        reasons.append(
            f"numbered lines were not exactly 1..{n_expected} in order "
            f"(parsed {len(numbers)} numbered lines)"
        )

    norm = [t.strip().lower() for t in items_text]
    distinct_ok = len(norm) == n_expected and len(set(norm)) == n_expected
    if not distinct_ok:
        dupes = len(norm) - len(set(norm))
        reasons.append(f"{dupes} duplicate item line(s) detected (possible repeated-pattern degeneration)")

    # HOLE FIX: the prompt explicitly demands "no title, no intro, no
    # closing remarks, no blank lines" but nothing used to check that --
    # _LOOP_LINE_RE just extracts whatever numbered lines it can find and
    # silently ignores everything else, so a response with a chatty
    # preamble/postamble around a perfect 1..100 list still passed. Count
    # every non-blank line in the response and require it to equal exactly
    # the number of matched numbered lines (i.e. no stray lines at all).
    non_blank_lines = [ln for ln in text.strip("\n").split("\n") if ln.strip() != ""]
    no_extra_ok = len(non_blank_lines) == len(numbers)
    if not no_extra_ok:
        extra = len(non_blank_lines) - len(numbers)
        reasons.append(
            f"{extra} extra non-numbered line(s) detected (intro/outro/commentary not allowed)"
        )

    stop_ok = finish_reason == "stop"
    if not stop_ok:
        reasons.append(f"finish_reason={finish_reason!r} (required 'stop')")

    passed = order_ok and distinct_ok and no_extra_ok and stop_ok
    return {
        "passed": passed,
        "reasons": reasons,
        "n_lines_parsed": len(numbers),
        "finish_reason": finish_reason,
        "order_ok": order_ok,
        "distinct_ok": distinct_ok,
        "no_extra_ok": no_extra_ok,
    }


# ---------------------------------------------------------------------------
# long_ctx
# ---------------------------------------------------------------------------

_SIX_DIGIT_RE = re.compile(r"\b\d{6}\b")


def is_garbled(text):
    """Best-effort structural garble/degeneration detector: not about
    factual correctness, just 'did the model produce coherent-looking
    prose or did it fall over into repetition/mojibake'."""
    t = (text or "").strip()
    if len(t) < 8:
        return True
    if re.search(r"(.)\1{19,}", t):  # any single char hammered 20+ times in a row
        return True
    if re.search(r"(.{15,}?)\1{3,}", t, re.DOTALL):  # a >=15-char chunk repeated 4+ times
        return True
    # HOLE FIX: digits were excluded from the "coherent" character set, so a
    # genuinely correct, coherent one-sentence answer that happens to be
    # numerically detailed (e.g. "...a 27B-parameter model with a 256000
    # token context...", which is exactly the flavor of fact this codebase's
    # own corpus is full of) was penalized toward the garble threshold for
    # no good reason. isalnum() keeps digits without letting through the
    # mojibake/symbol soup this heuristic exists to catch.
    keep = sum(1 for c in t if c.isalnum() or c.isspace() or c in ".,;:'\"-()")
    if keep / len(t) < 0.6:
        return True
    if len(t.split()) > 120:  # asked for one sentence; generous cap catches runaway loops
        return True
    return False


def score_long_ctx(item, content):
    check = item["check"]
    t = check["type"]
    text = content or ""

    if t == "single_fact":
        expected = check["expected_number"]
        found = [int(x) for x in _SIX_DIGIT_RE.findall(text)]
        passed = expected in found
        return {"passed": passed, "expected_number": expected, "found_numbers": found}

    if t == "multi_fact":
        expected_list = check["expected_numbers"]
        found = [int(x) for x in _SIX_DIGIT_RE.findall(text)]
        # HOLE FIX: plain membership over `found` (every 6-digit number in
        # the response) accepted the right numbers in any order, plus extra
        # numbers, as a pass. The prompt demands exactly these three numbers,
        # in this order, and nothing else -- require an exact positional
        # match against the full extracted sequence.
        passed = found == expected_list
        return {"passed": passed, "expected_numbers": expected_list, "found_numbers": found}

    if t == "purpose":
        pattern = check.get("pattern", r"vllm|inference|LLM|serving")
        regex_ok = re.search(pattern, text, re.IGNORECASE) is not None
        garbled = is_garbled(text)
        return {"passed": regex_ok and not garbled, "regex_ok": regex_ok, "garbled": garbled}

    raise ValueError(f"unknown long_ctx check type: {t!r}")
