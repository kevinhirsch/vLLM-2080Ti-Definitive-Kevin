"""Turn copy_probe pairs (text) into token pairs for lookup_opportunity.py. Usage: tokenize_pairs.py in.jsonl out.jsonl"""
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven")
with open(sys.argv[2], "w") as o:
    for l in open(sys.argv[1]):
        d = json.loads(l)
        o.write(json.dumps({"src": d["src"], "prompt": tok.encode(d["prompt_text"], add_special_tokens=False),
                            "resp": tok.encode(d["resp_text"], add_special_tokens=False)}) + "\n")
