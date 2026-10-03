#!/usr/bin/env python3
"""Lane SH: the gateway shim's "include parts" -- how keepalive-shim.py is split into files without changing behaviour.

keepalive-shim.py runs as one namespace: 39 module-level containers and 22 `global` statements are shared state, and
the tests patch 200+ of its attributes. Moving code into real modules would change what a patch or a `global` touches.
So the shim is cut at its own section banners into gateway_part_*.py files, and at the exact place each block used to
be, the shim runs

    _include_gateway_part("gateway_part_<name>.py")

which executes that file's bytes inside the shim's OWN globals (see _include_gateway_part in keepalive-shim.py).
Same namespace, same definition order: behaviour is identical by construction. A part file starts with header lines
beginning with "# gateway-part:"; everything after them is the original code, byte for byte.

    python3 gateway_parts.py expand [SHIM]            print the monolith (includes replaced by part bodies)
    python3 gateway_parts.py check  [SHIM] --against REV
                                                      exit 0 iff expand(SHIM) == expand(SHIM at git REV): the proof
                                                      that a split commit is a pure move
    python3 gateway_parts.py split  SHIM FIRST LAST NAME "TITLE"
                                                      move lines FIRST..LAST (1-based, inclusive) into gateway_part_NAME.py

Stdlib only; never imports the shim. Tests that grep the gateway's source call expanded_source().
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHIM = os.path.join(HERE, "keepalive-shim.py")
HEADER_PREFIX = "# gateway-part:"
INCLUDE_RE = re.compile(r'^_include_gateway_part\("(gateway_part_[a-z0-9_]+\.py)"\)[ \t]*$', re.M)


def part_names(shim_text):
    """Part filenames in include order."""
    return INCLUDE_RE.findall(shim_text)


def part_body(part_text):
    """The original code inside a part file: its leading '# gateway-part:' header lines removed."""
    lines = part_text.split("\n")
    i = 0
    while i < len(lines) and lines[i].startswith(HEADER_PREFIX):
        i += 1
    return "\n".join(lines[i:])


def expand_text(shim_text, read_part):
    """read_part(filename) -> text. Each include line becomes the part's body (which ends with the newline the
    replaced block ended with), so expansion of a pure split is byte-identical to the file before the split."""
    def sub(m):
        body = part_body(read_part(m.group(1)))
        return body[:-1] if body.endswith("\n") else body     # split() writes the block + exactly one newline
    return INCLUDE_RE.sub(sub, shim_text)


def expanded_source(shim_path=SHIM):
    d = os.path.dirname(os.path.abspath(shim_path))
    with open(shim_path, encoding="utf-8") as fh:
        text = fh.read()

    def read_part(name):
        with open(os.path.join(d, name), encoding="utf-8") as fh:
            return fh.read()
    return expand_text(text, read_part)


def expanded_at(rev, repo=None, rel="deploy/bin"):
    repo = repo or os.path.dirname(os.path.dirname(HERE))

    def show(name):
        return subprocess.check_output(["git", "show", "%s:%s/%s" % (rev, rel, name)], cwd=repo).decode("utf-8")
    return expand_text(show("keepalive-shim.py"), show)


def split(shim_path, first, last, name, title):
    """Move lines first..last (1-based, inclusive) of the shim into gateway_part_<name>.py, verbatim."""
    fname = "gateway_part_%s.py" % name
    with open(shim_path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    block = lines[first - 1:last]
    header = [HEADER_PREFIX + " " + title,
              HEADER_PREFIX + " executed inside keepalive-shim.py's own namespace by _include_gateway_part() -- not an",
              HEADER_PREFIX + " importable module. Names here are the shim's globals. See gateway_parts.py."]
    part_path = os.path.join(os.path.dirname(os.path.abspath(shim_path)), fname)
    if os.path.exists(part_path):
        raise SystemExit("%s exists" % part_path)
    with open(part_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header + block) + "\n")
    lines[first - 1:last] = ['_include_gateway_part("%s")' % fname]
    with open(shim_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return fname


def main(argv):
    if len(argv) >= 2 and argv[1] == "expand":
        sys.stdout.write(expanded_source(argv[2] if len(argv) > 2 else SHIM))
        return 0
    if len(argv) >= 2 and argv[1] == "check":
        rest = argv[2:]
        rev = rest[rest.index("--against") + 1]
        shim = rest[0] if rest and rest[0] != "--against" else SHIM
        same = expanded_source(shim) == expanded_at(rev)
        print("pure move: expanded source is byte-identical to %s" % rev if same else "DIFFERS from %s" % rev)
        return 0 if same else 1
    if len(argv) == 7 and argv[1] == "split":
        print(split(argv[2], int(argv[3]), int(argv[4]), argv[5], argv[6]))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
