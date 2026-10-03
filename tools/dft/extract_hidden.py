#!/usr/bin/env python3
"""Lane DFT (GPU, inside the offline window, engine STOPPED): run the TARGET model (same flags as production, no speculative decoding,
eager) over the corpus sequences and dump its post-norm hidden states for the chosen windows.
usage: extract_hidden.py MANIFEST.json OUTDIR [--deadline EPOCH_S] [--order val,train]"""
import argparse, json, os, sys, time, random
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("manifest"); ap.add_argument("outdir")
ap.add_argument("--deadline", type=float, default=0); ap.add_argument("--util", type=float, default=0.80)
ap.add_argument("--max-len", type=int, default=32768); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.makedirs(a.outdir, exist_ok=True)
man = json.load(open(a.manifest))
rng = random.Random(1); rng.shuffle(man)
man = sorted(man, key=lambda m: 0 if m["split"] == "val" else 1)   # val first so a deadline never starves the held-out set
if a.limit: man = man[: a.limit]
def main():
    global done
    from vllm import LLM, SamplingParams
    M = "/home/kevin/Desktop/models/Qwen3.8-27B-HauhauCS-Aggressive-W4A16-twolven"
    llm = LLM(model=M, dtype="half", tensor_parallel_size=2, generation_config=os.path.expanduser("~/.local/share/vllm-qwen27b/gencfg"),
              gpu_memory_utilization=a.util, max_model_len=a.max_len,
              hf_overrides={"rope_parameters": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 262144, "mrope_interleaved": True,
                                               "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000}},
              enable_chunked_prefill=True, max_num_seqs=2, max_num_batched_tokens=3584, kv_cache_dtype="turboquant_k3v4_nc",
              enable_prefix_caching=False, enforce_eager=True, worker_extension_cls="dft_ext.DumpExt",
              additional_config={"gdn_prefill_backend": "flashqla_legacy"}, limit_mm_per_prompt={"image": 0, "video": 0},
              language_model_only=True, seed=0)
    print("hook on", llm.collective_rpc("dft_arm")[0], flush=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    done = tot = 0; t0 = time.time(); stats = []
    for m in man:
        outp = f"{a.outdir}/{m['id']}.npy"
        if os.path.exists(outp): continue
        if a.deadline and time.time() > a.deadline:
            print("deadline reached", flush=True); break
        z = np.load(f"{os.path.dirname(a.manifest)}/seqs/{m['id']}.npz")
        ids = z["ids"].tolist()
        llm.collective_rpc("dft_arm")
        ts = time.time()
        llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
        r = llm.collective_rpc("dft_dump", args=(outp, [tuple(w) for w in m["windows"]], len(ids)))
        if not r[0]["ok"]:
            print("MISMATCH", m["id"], r, flush=True); continue
        done += 1; tot += len(ids)
        stats.append((len(ids), time.time() - ts))
        if done % 10 == 0 or done < 4:
            print(f"[{done}/{len(man)}] {tot} tok, {tot / (time.time() - t0):.0f} tok/s overall, last {len(ids) / stats[-1][1]:.0f} tok/s", flush=True)
    print(f"EXTRACT DONE {done} seqs {tot} tokens in {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
