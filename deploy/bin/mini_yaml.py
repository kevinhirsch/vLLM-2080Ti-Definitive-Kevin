#!/usr/bin/env python3
"""mini_yaml.py -- a dependency-free parser for the YAML subset window specs use (lane RL, 2026-10-03).

The test gate's venv has no PyYAML, and window specs must parse identically everywhere, so windowctl ALWAYS uses this
(test_mini_yaml.py checks it against PyYAML on every deploy/windows/*.yaml wherever PyYAML exists). Supported:
  block mappings and sequences (indentation), `- key: v` sequence items, flow [a, b] and {k: v} (nested),
  literal block scalars `|` / `|-` / `|+`, folded `>` / `>-`, "double" (JSON-style escapes) and 'single' ('' escape)
  quotes, plain scalars (true/false/yes/no/on/off, null/~, ints, floats, strings), # comments.
Anything else (anchors, tags, multi-document, complex keys) raises MiniYAMLError instead of guessing.
"""
from __future__ import annotations

import re

_INT = re.compile(r"^[-+]?(0|[1-9][0-9_]*)$")
_FLOAT = re.compile(r"^[-+]?(\.[0-9]+|[0-9][0-9_]*(\.[0-9]*)?)([eE][-+]?[0-9]+)?$")
_BOOL = {"true": True, "True": True, "TRUE": True, "false": False, "False": False, "FALSE": False,
         "yes": True, "Yes": True, "no": False, "No": False, "on": True, "On": True, "off": False, "Off": False}
_NULL = {"null", "Null", "NULL", "~", ""}
_ESC = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", '"': '"', "\\": "\\", "/": "/", " ": " ", "b": "\b", "f": "\f",
        "e": "\x1b", "a": "\a", "v": "\v"}


class MiniYAMLError(ValueError):
    pass


def _scalar(text: str):
    t = text.strip()
    if t in _NULL:
        return None
    if t in _BOOL:
        return _BOOL[t]
    if _INT.match(t):
        return int(t.replace("_", ""))
    if _FLOAT.match(t) and any(c.isdigit() for c in t):
        return float(t.replace("_", ""))
    if t[:1] in ("&", "*", "!", "%", "@", "`"):
        raise MiniYAMLError(f"unsupported YAML construct: {t[:40]!r}")
    return t


def _dq(s: str, i: int):
    """Parse a double-quoted string starting at s[i] == '"'. Returns (value, index after the closing quote)."""
    out, i = [], i + 1
    while i < len(s):
        c = s[i]
        if c == "\\":
            n = s[i + 1] if i + 1 < len(s) else ""
            if n in _ESC:
                out.append(_ESC[n])
                i += 2
                continue
            if n == "x":
                out.append(chr(int(s[i + 2:i + 4], 16)))
                i += 4
                continue
            if n == "u":
                out.append(chr(int(s[i + 2:i + 6], 16)))
                i += 6
                continue
            raise MiniYAMLError(f"bad escape \\{n}")
        if c == '"':
            return "".join(out), i + 1
        out.append(c)
        i += 1
    raise MiniYAMLError("unterminated double-quoted string")


def _sq(s: str, i: int):
    out, i = [], i + 1
    while i < len(s):
        c = s[i]
        if c == "'":
            if s[i + 1:i + 2] == "'":
                out.append("'")
                i += 2
                continue
            return "".join(out), i + 1
        out.append(c)
        i += 1
    raise MiniYAMLError("unterminated single-quoted string")


def _strip_comment(s: str) -> str:
    """Drop a trailing ` # comment` that is outside quotes."""
    i, q = 0, None
    while i < len(s):
        c = s[i]
        if q:
            if q == '"' and c == "\\":
                i += 2
                continue
            if c == q:
                if q == "'" and s[i + 1:i + 2] == "'":
                    i += 2
                    continue
                q = None
        elif c in "\"'" and (i == 0 or s[i - 1] in " \t[{,:"):
            q = c
        elif c == "#" and (i == 0 or s[i - 1] in " \t"):
            return s[:i].rstrip()
        i += 1
    return s.rstrip()


def _split_key(s: str):
    """`key: rest` -> (key, rest) when s is a mapping entry (colon followed by space/end, outside quotes/brackets)."""
    if s[:1] in "\"'":
        val, j = (_dq if s[0] == '"' else _sq)(s, 0)
        rest = s[j:]
        if rest.startswith(":") and (len(rest) == 1 or rest[1] in " \t"):
            return val, rest[1:].strip()
        return None
    depth = 0
    for i, c in enumerate(s):
        if c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
        elif c == ":" and depth == 0 and (i + 1 == len(s) or s[i + 1] in " \t"):
            key = s[:i].strip()
            if not key or key[0] in "[{":
                return None
            return key, s[i + 1:].strip()
        elif c in " \t" and depth == 0 and s[:i].strip() and s[i + 1:i + 2] == "#":
            return None
    return None


class _Flow:
    def __init__(self, s):
        self.s, self.i = s, 0

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \t":
            self.i += 1

    def value(self, stop):
        self.ws()
        c = self.s[self.i:self.i + 1]
        if c == "[":
            self.i += 1
            out = []
            while True:
                self.ws()
                if self.s[self.i:self.i + 1] == "]":
                    self.i += 1
                    return out
                out.append(self.value(",]"))
                self.ws()
                if self.s[self.i:self.i + 1] == ",":
                    self.i += 1
                elif self.s[self.i:self.i + 1] != "]":
                    raise MiniYAMLError(f"expected , or ] in {self.s!r}")
        if c == "{":
            self.i += 1
            out = {}
            while True:
                self.ws()
                if self.s[self.i:self.i + 1] == "}":
                    self.i += 1
                    return out
                k = self.value(":,}")
                self.ws()
                if self.s[self.i:self.i + 1] != ":":
                    raise MiniYAMLError(f"expected : in flow mapping {self.s!r}")
                self.i += 1
                out[k] = self.value(",}")
                self.ws()
                if self.s[self.i:self.i + 1] == ",":
                    self.i += 1
                elif self.s[self.i:self.i + 1] != "}":
                    raise MiniYAMLError(f"expected , or }} in {self.s!r}")
        if c == '"':
            v, self.i = _dq(self.s, self.i)
            return v
        if c == "'":
            v, self.i = _sq(self.s, self.i)
            return v
        j = self.i
        while j < len(self.s) and self.s[j] not in stop:
            if self.s[j] == ":" and ":" in stop and not (j + 1 < len(self.s) and self.s[j + 1] not in " \t,}]"):
                break
            j += 1
        tok, self.i = self.s[self.i:j], j
        return _scalar(tok)


def _inline(rest: str):
    rest = rest.strip()
    if rest[:1] in "[{":
        f = _Flow(rest)
        v = f.value("")
        f.ws()
        if f.i != len(rest):
            raise MiniYAMLError(f"trailing text after flow value: {rest!r}")
        return v
    if rest[:1] == '"':
        v, j = _dq(rest, 0)
        if rest[j:].strip():
            raise MiniYAMLError(f"trailing text after string: {rest!r}")
        return v
    if rest[:1] == "'":
        v, j = _sq(rest, 0)
        if rest[j:].strip():
            raise MiniYAMLError(f"trailing text after string: {rest!r}")
        return v
    return _scalar(rest)


class _Parser:
    def __init__(self, text):
        if "\t" in "".join(l[:len(l) - len(l.lstrip())] for l in text.splitlines()):
            raise MiniYAMLError("tabs in indentation")
        self.lines = text.splitlines()
        self.i = 0

    @staticmethod
    def indent(line):
        return len(line) - len(line.lstrip(" "))

    def peek(self):
        """Index of the next significant line (not blank, not a comment), or None."""
        j = self.i
        while j < len(self.lines):
            s = self.lines[j].strip()
            if s and not s.startswith("#"):
                if s in ("---", "...") or s.startswith("--- "):
                    if s == "---" and not any(l.strip() and not l.strip().startswith("#") for l in self.lines[:j]):
                        j += 1          # a leading document marker is fine
                        continue
                    raise MiniYAMLError("multi-document YAML is not supported")
                return j
            j += 1
        return None

    def node(self, min_indent):
        j = self.peek()
        if j is None:
            return None
        ind = self.indent(self.lines[j])
        if ind < min_indent:
            return None
        s = self.lines[j].strip()
        if s == "-" or s.startswith("- "):
            return self.seq(ind)
        if _split_key(_strip_comment(s)) is not None:
            return self.mapping(ind)
        self.i = j + 1
        return _inline(_strip_comment(s))

    def block_scalar(self, header, parent_indent):
        style, chomp = header[0], header[1:2]
        raw = []
        j = self.i
        while j < len(self.lines):
            line = self.lines[j]
            if line.strip() and self.indent(line) <= parent_indent:
                break
            raw.append(line)
            j += 1
        self.i = j
        trailing_blank = False
        while raw and not raw[-1].strip():
            trailing_blank = True
            raw.pop()
        body = [l for l in raw if l.strip()]
        if not body:
            return ""
        ind = min(self.indent(l) for l in body)
        lines = [l[ind:] if l.strip() else "" for l in raw]
        if style == "|":
            text = "\n".join(lines)
        else:
            paras, cur = [], []
            for l in lines:
                if l == "":
                    paras.append(" ".join(cur))
                    cur = []
                else:
                    cur.append(l)
            paras.append(" ".join(cur))
            text = "\n".join(paras)
        if chomp == "-":
            return text
        if chomp == "+":
            return text + "\n" + ("\n" if trailing_blank else "")
        return text + "\n"

    def value_after_key(self, rest, ind):
        if rest[:1] in ("|", ">") and re.fullmatch(r"[|>][-+]?", rest):
            return self.block_scalar(rest, ind)
        if rest == "":
            j = self.peek()
            if j is not None:
                nind = self.indent(self.lines[j])
                s = self.lines[j].strip()
                if nind > ind or (nind == ind and (s == "-" or s.startswith("- "))):
                    return self.node(nind)
            return None
        return _inline(rest)

    def mapping(self, ind):
        out = {}
        while True:
            j = self.peek()
            if j is None or self.indent(self.lines[j]) != ind:
                if j is not None and self.indent(self.lines[j]) > ind:
                    raise MiniYAMLError(f"line {j + 1}: unexpected indentation")
                return out
            s = _strip_comment(self.lines[j].strip())
            if s == "-" or s.startswith("- "):
                return out
            kv = _split_key(s)
            if kv is None:
                raise MiniYAMLError(f"line {j + 1}: expected 'key: value', got {s[:60]!r}")
            k, rest = kv
            if k in out:
                raise MiniYAMLError(f"line {j + 1}: duplicate key {k!r}")
            self.i = j + 1
            out[k] = self.value_after_key(rest, ind)

    def seq(self, ind):
        out = []
        while True:
            j = self.peek()
            if j is None or self.indent(self.lines[j]) != ind:
                return out
            s = self.lines[j].strip()
            if not (s == "-" or s.startswith("- ")):
                return out
            content = s[1:].lstrip(" ")
            if content == "" or content.startswith("#"):
                self.i = j + 1
                out.append(self.node(ind + 1))
                continue
            sub = ind + (len(s) - len(content))
            if _split_key(_strip_comment(content)) is not None:
                # "- key: v" starts a mapping whose lines sit at the column of `key`
                self.lines[j] = " " * sub + content
                self.i = j
                out.append(self.mapping(sub))
                continue
            self.i = j + 1
            out.append(_inline(_strip_comment(content)))


def load(text: str):
    p = _Parser(text)
    v = p.node(0)
    if p.peek() is not None:
        raise MiniYAMLError(f"line {p.peek() + 1}: unexpected content")
    return v
