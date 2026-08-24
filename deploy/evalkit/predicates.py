"""
Deterministic predicate evaluation for tool-call arguments.

Used by both the tool_call and agentic_chain checkers. No external deps: a
"path" is a small dotted/bracket mini-language (e.g. "user.id",
"items[0].name") resolved by hand against the parsed JSON arguments of a
tool call -- this is the "jsonpath" referenced in the brief, implemented
inline rather than pulling in a jsonpath library.
"""
import re

_TOKEN_RE = re.compile(r'[^.\[\]]+|\[\d+\]')


def resolve_path(obj, path):
    """Resolve a dotted/bracket path against a JSON-like object.
    "" or "." means "the whole object". Raises KeyError/IndexError/TypeError
    on a missing/invalid segment -- callers decide what that means.
    """
    if path in ("", "."):
        return obj
    cur = obj
    for tok in _TOKEN_RE.findall(path):
        if tok.startswith('['):
            idx = int(tok[1:-1])
            cur = cur[idx]
        else:
            if not isinstance(cur, dict) or tok not in cur:
                raise KeyError(f"path segment '{tok}' not found")
            cur = cur[tok]
    return cur


_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def check_predicate(args_dict, pred):
    """Evaluate one predicate against parsed tool-call arguments.

    pred = {"path": "<dotted path>", "match": "<kind>", ...kind-specific...}
    match kinds: exact, regex, contains, numeric, type, in, exists, absent.
    """
    path = pred.get("path", "")
    match = pred["match"]

    if match == "absent":
        try:
            resolve_path(args_dict, path)
            return False
        except (KeyError, IndexError, TypeError):
            return True

    try:
        value = resolve_path(args_dict, path)
    except (KeyError, IndexError, TypeError):
        return False

    if match == "exists":
        return True
    if match == "exact":
        return value == pred["value"]
    if match == "regex":
        flags = re.DOTALL | re.MULTILINE if pred.get("dotall", True) else re.MULTILINE
        return re.search(pred["pattern"], str(value), flags) is not None
    if match == "contains":
        if isinstance(value, (list, str)):
            return pred["value"] in value
        return pred["value"] in str(value)
    if match == "numeric":
        try:
            return abs(float(value) - float(pred["value"])) <= pred.get("tol", 1e-6)
        except (TypeError, ValueError):
            return False
    if match == "type":
        expected = _TYPE_MAP.get(pred["value"])
        if expected is None:
            return False
        if pred["value"] in ("integer", "number") and isinstance(value, bool):
            return False
        return isinstance(value, expected)
    if match == "in":
        return value in pred["values"]

    raise ValueError(f"unknown predicate match type: {match!r}")


def expand_value_from_previous(pred, prev_result):
    """If a predicate references 'value_from_previous_result' (a path into
    the previous step's injected tool result), resolve it into a concrete
    'value'/'values' key so check_predicate can be used unmodified. Returns
    a new dict; leaves pred untouched if there's nothing to expand.
    """
    if "value_from_previous_result" in pred:
        pred = dict(pred)
        src_path = pred.pop("value_from_previous_result")
        resolved = resolve_path(prev_result, src_path)
        if pred.get("match") == "in":
            pred["values"] = resolved
        else:
            pred["value"] = resolved
    return pred


def call_matches(name, args, expected):
    """expected = {"name": ..., "args": [predicate, ...]}"""
    if args is None or name != expected["name"]:
        return False
    for pred in expected.get("args", []):
        if not check_predicate(args, pred):
            return False
    return True
