#!/usr/bin/env bash
# make_variant.sh TUNED_MTP_SAFETENSORS OUT_DIR
# Builds a NEW model-variant dir: symlinks to every original file (target weights, tokenizer, configs) except model-mtp.safetensors, which is the tuned copy.
# The original model dir is never modified.
set -euo pipefail
SRC=/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven
T=$1; OUT=$2
[ -f "$T" ] || { echo "missing $T"; exit 2; }
[ "$OUT" != "$SRC" ] || { echo "refusing to write into the original dir"; exit 2; }
mkdir -p "$OUT"
for f in "$SRC"/* ; do b=$(basename "$f"); [ "$b" = model-mtp.safetensors ] && continue; ln -sfn "$f" "$OUT/$b"; done
cp -f "$T" "$OUT/model-mtp.safetensors"
python3 - "$SRC/model-mtp.safetensors" "$OUT/model-mtp.safetensors" <<'P'
import json, struct, sys
def hdr(p):
    f = open(p, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); h.pop('__metadata__', None); return h
a, b = hdr(sys.argv[1]), hdr(sys.argv[2])
assert a.keys() == b.keys() and all(a[k]['shape'] == b[k]['shape'] and a[k]['dtype'] == b[k]['dtype'] for k in a), "tuned mtp layout differs from original"
print("variant dir ok:", sys.argv[2], len(a), "tensors, layout identical to the original")
P
