"""Lane E04 CPU tests: the C3 shared-memory layout of tq_gqa_stage1_c3 (no GPU).

1. Python mirrors of c3_qk / c3_v / c3_s are parsed out of the .cu source (so the test follows the code) and checked
   for: in-bounds, bijective, accesses never crossing a 16-byte chunk, S_s alias disjoint from the live K_s bytes,
   total footprint fits 3 CTAs in 64 KiB (256 B granules), bank conflicts no worse than the shipped layout for every
   shared-memory access pattern of the kernel (per warp / per half-warp for 64/128-bit accesses).
2. Source contracts: shipped stage1/stage2 device code byte-identical to d2b14aa5e7; the C3 kernel equals the
   shipped kernel modulo the listed addressing substitutions; wrapper gate VLLM_TQ_GQA_C3 selects decode_c3.
"""
from __future__ import annotations

import importlib
import pathlib
import re
import subprocess
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CU = ROOT / "vllm/v1/attention/ops/tq_gqa_ext/tq_gqa.cu"
SRC = CU.read_text()
BASE_SHA = "d2b14aa5e7"

HD, CH, LDP = 256, 16, 24
# shipped layout constants
LDQ, LDK, LDV, LDS = 264, 264, 26, 20


def _c_expr_fn(name):
    m = re.search(r"__device__ __forceinline__ int " + name + r"\(([^)]*)\) \{ return (.*?); \}", SRC)
    assert m, name
    args = [a.split()[-1] for a in m.group(1).split(",")]
    expr = m.group(2)
    return eval(f"lambda {', '.join(args)}: {expr}")  # C int ops on non-negative ints == Python semantics


c3_qk = _c_expr_fn("c3_qk")
c3_v = _c_expr_fn("c3_v")
c3_s = _c_expr_fn("c3_s")


def _define(name):
    m = re.search(r"#define " + name + r" (.*?)(?:\s*/\*.*\*/)?\n", SRC)
    assert m, name
    expr = m.group(1)
    for k, v in DEFS.items():
        expr = re.sub(r"\b" + k + r"\b", str(v), expr)
    return eval(expr)


DEFS = {"HD": HD, "CH": CH, "LDP": LDP}
for _n in ("C3_Q_OFF", "C3_KV_OFF", "C3_KV_BYTES", "C3_S_OFF", "C3_P_OFF", "C3_SMEM_BYTES"):
    DEFS[_n] = _define(_n)


def ways(words):
    """max bank-conflict degree for one shared-memory wavefront: distinct 32-bit words per bank."""
    banks = {}
    for w in set(words):
        banks.setdefault(w % 32, set()).add(w)
    return max(len(s) for s in banks.values())


# ---------------------------------------------------------------- layout geometry

def test_footprint_fits_three_ctas():
    used = DEFS["C3_SMEM_BYTES"]
    assert used == 21600
    granule = -(-used // 256) * 256
    assert 3 * granule <= 64 * 1024
    shipped = 16 * LDQ * 2 + max(HD * LDV, CH * LDK) * 2 + 4 * 16 * LDS * 4 + 16 * LDP * 2 + CH * 8 + CH * 4 * 2 + 16 * 4 * 3 + 8 * 4 + 64
    assert shipped == 28192 and 3 * (-(-shipped // 256) * 256) > 64 * 1024  # why the shipped layout is capped at 2


def test_offsets_ordered_and_aligned():
    assert DEFS["C3_Q_OFF"] == 0 and DEFS["C3_KV_OFF"] == 8192
    assert DEFS["C3_KV_OFF"] % 16 == 0 and DEFS["C3_S_OFF"] % 16 == 0 and DEFS["C3_P_OFF"] % 16 == 0
    assert DEFS["C3_P_OFF"] == DEFS["C3_KV_OFF"] + DEFS["C3_KV_BYTES"]


def test_qk_bijective_in_bounds_chunk_local():
    seen = set()
    for row in range(16):
        for col in range(HD):
            i = c3_qk(row, col)
            assert 0 <= i < 16 * HD
            seen.add(i)
            # same 8-half chunk for col and its chunk-mates; vector accesses never straddle
            assert c3_qk(row, col) - c3_qk(row, col & ~7) == (col & 7)
    assert len(seen) == 16 * HD


def test_v_bijective_in_bounds_pair_aligned():
    seen = set()
    for d in range(HD):
        for tk in range(0, CH, 2):
            i = c3_v(d, tk)
            assert i % 2 == 0 and 0 <= i and i + 1 < HD * 24
            seen.add(i)
    assert len(seen) == HD * CH // 2
    assert HD * 24 * 2 == DEFS["C3_KV_BYTES"]


def test_s_bijective_and_alias_disjoint_from_live_k():
    seen = {c3_s(kq, r, c) for kq in range(4) for r in range(16) for c in range(16)}
    assert seen == set(range(4 * 16 * 16))
    # K_s (live during the score phase) occupies bytes [KV_OFF, KV_OFF + max qk*2]; S_s starts after it
    k_end = DEFS["C3_KV_OFF"] + (max(c3_qk(r, c) for r in range(CH) for c in range(HD)) + 1) * 2
    assert k_end <= DEFS["C3_S_OFF"]
    assert DEFS["C3_S_OFF"] + 4 * 16 * 16 * 4 <= DEFS["C3_P_OFF"]
    # float2 store pairs (col even, col+1) stay adjacent
    for kq in range(4):
        for r in range(16):
            for c in range(0, 16, 2):
                assert c3_s(kq, r, c + 1) == c3_s(kq, r, c) + 1 and c3_s(kq, r, c) % 2 == 0


# ---------------------------------------------------------------- bank conflicts (C3 <= shipped, per access)

def _lanes():
    return [(lane, lane >> 2, lane & 3) for lane in range(32)]


def test_bank_q_k_fragment_loads():
    for kq in range(4):
        for ks in range(8):
            k0 = kq * 64 + ks * 8
            for rows in (lambda g: g, lambda g: g + 8):
                new = [(DEFS["C3_Q_OFF"] // 2 + c3_qk(rows(g), k0 + 2 * t4)) // 2 for _, g, t4 in _lanes()]
                old = [(rows(g) * LDQ + k0 + 2 * t4) // 2 for _, g, t4 in _lanes()]
                assert ways(new) <= ways(old) == 1
            for nt in range(2):
                new = [c3_qk(nt * 8 + g, k0 + 2 * t4) // 2 for _, g, t4 in _lanes()]
                assert ways(new) == 1


def test_bank_k_store_128bit():
    # warp w writes row tk (w, w+8), lane -> 16 bytes at col 8*lane; 128-bit stores = 4 quarter-warp wavefronts
    for tk in range(16):
        for qw in range(4):
            words = []
            for lane in range(qw * 8, qw * 8 + 8):
                base = c3_qk(tk, 8 * lane) // 2
                words += [base + i for i in range(4)]
            assert ways(words) == 1


def test_bank_q_staging_store():
    # tid -> c4 = tid & 63, m = (tid >> 6) + 4 i; two half2 stores (64-bit pattern) -> per half-warp wavefront
    for warp in range(8):
        for i in range(4):
            for half in range(2):
                words = []
                for lane in range(half * 16, half * 16 + 16):
                    tid = warp * 32 + lane
                    c4, m = tid & 63, (tid >> 6) + 4 * i
                    base = c3_qk(m, c4 * 4) // 2
                    words += [base, base + 1]
                assert ways(words) == 1


def test_bank_v_transpose_store_and_pv_load():
    for th in range(2):
        for q in range(CH // 4):
            tk = th * (CH // 2) + 2 * q
            for warp in range(4):
                for hi in (0, 1):
                    js = [(warp * 32 + lane) & 127 for lane in range(32)]
                    new = [c3_v(2 * j + hi, tk) // 2 for j in js]
                    old = [((2 * j + hi) * LDV + tk) // 2 for j in js]
                    assert ways(new) == 1 <= ways(old)
    for warp in range(8):
        for dt in range(4):
            for ks in range(2):
                new = [c3_v((warp * 4 + dt) * 8 + g, ks * 8 + 2 * t4) // 2 for _, g, t4 in _lanes()]
                old = [(((warp * 4 + dt) * 8 + g) * LDV + ks * 8 + 2 * t4) // 2 for _, g, t4 in _lanes()]
                assert ways(new) == 1 <= ways(old)


def test_bank_scores_store_and_softmax_load():
    for warp in range(8):
        nt, kq = warp & 1, warp >> 1
        for rowoff in (0, 8):
            for half in range(2):  # float2 stores: half-warp wavefronts
                words = []
                for lane in range(half * 16, half * 16 + 16):
                    g, t4 = lane >> 2, lane & 3
                    w = c3_s(kq, g + rowoff, nt * 8 + 2 * t4)
                    words += [w, w + 1]
                assert ways(words) == 1
        for kq2 in range(4):
            srow_scol = [((warp * 32 + lane) >> 4, (warp * 32 + lane) & 15) for lane in range(32)]
            assert ways([c3_s(kq2, r, c) for r, c in srow_scol]) == 1


# ---------------------------------------------------------------- source contracts

def _git_show(rev, rel):
    return subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=ROOT, check=True, capture_output=True, text=True).stdout


BASE_SRC = _git_show(BASE_SHA, "vllm/v1/attention/ops/tq_gqa_ext/tq_gqa.cu")


def _kernel(src, head, end_marker):
    i = src.index(head)
    return src[i:src.index(end_marker, i)]


def test_shipped_device_code_unchanged():
    base_dev = BASE_SRC.split("void tq_gqa_decode(", 1)[0]
    assert SRC.startswith(base_dev.split("// Reduce over KV splits.", 1)[0])
    stage2 = _kernel(BASE_SRC, "// Reduce over KV splits.", "void tq_gqa_decode(")
    assert stage2 in SRC
    base_host = _kernel(BASE_SRC, "void tq_gqa_decode(", "PYBIND11_MODULE")
    assert base_host in SRC  # shipped host launcher verbatim


def test_c3_kernel_is_shipped_kernel_modulo_addressing():
    old = _kernel(BASE_SRC, "__global__ void __launch_bounds__(NTHR, 2) tq_gqa_stage1(", "// Reduce over KV splits.")
    new = _kernel(SRC, "__global__ void __launch_bounds__(NTHR, 3) tq_gqa_stage1_c3(", "void tq_gqa_decode(")
    new = new.split("// Reduce over KV splits.")[0] if "// Reduce over KV splits." in new else new
    # strip everything that is allowed to differ, then demand equality
    def norm(t):
        t = re.sub(r"//[^\n]*", "", t)
        t = re.sub(r"\s+", " ", t)
        return t
    # every memory index expression replaced by a token
    subs_old = [
        (r"__launch_bounds__\(NTHR, 2\) tq_gqa_stage1\(", "LB("),
        (r"__half\* Q_s = .*?float\* vs_s", "LAYOUT float* vs_s"),
        (r"Q_s \+ m \* LDQ \+ c4 \* 4", "QW"), (r"KV_s \+ tk \* LDK \+ 8 \* lane", "KW"),
        (r"&Q_s\[g \* LDQ \+ k0 \+ 2 \* t4\]", "QA0"), (r"&Q_s\[\(g \+ 8\) \* LDQ \+ k0 \+ 2 \* t4\]", "QA1"),
        (r"&KV_s\[\(nt \* 8 \+ g\) \* LDK \+ k0 \+ 2 \* t4\]", "KB"),
        (r"float\* sp = S_s.*?= cc\[3\];", "SSTORE"),
        (r"float s = S_s\[.*?\];", "SLOAD"),
        (r"Vt \+ \(2 \* j\) \* LDV \+ tk", "VW0"), (r"Vt \+ \(2 \* j \+ 1\) \* LDV \+ tk", "VW1"),
        (r"&KV_s\[d \* LDV \+ ks \* 8 \+ 2 \* t4\]", "VB"),
    ]
    subs_new = [
        (r"__launch_bounds__\(NTHR, 3\) tq_gqa_stage1_c3\(", "LB("),
        (r"__half\* Q_s = .*?float\* vs_s", "LAYOUT float* vs_s"),
        (r"Q_s \+ c3_qk\(m, c4 \* 4\)", "QW"), (r"KV_s \+ c3_qk\(tk, 8 \* lane\)", "KW"),
        (r"&Q_s\[c3_qk\(g, k0 \+ 2 \* t4\)\]", "QA0"), (r"&Q_s\[c3_qk\(g \+ 8, k0 \+ 2 \* t4\)\]", "QA1"),
        (r"&KV_s\[c3_qk\(nt \* 8 \+ g, k0 \+ 2 \* t4\)\]", "KB"),
        (r"\*reinterpret_cast<float2\*>\(S_s \+ c3_s\(kq, g, .*?make_float2\(cc\[2\], cc\[3\]\);", "SSTORE"),
        (r"float s = S_s\[.*?\];", "SLOAD"),
        (r"Vt \+ c3_v\(2 \* j, tk\)", "VW0"), (r"Vt \+ c3_v\(2 \* j \+ 1, tk\)", "VW1"),
        (r"&KV_s\[c3_v\(d, ks \* 8 \+ 2 \* t4\)\]", "VB"),
    ]
    o, n = norm(old), norm(new)
    for pat, tok in subs_old:
        o, k = re.subn(pat, tok, o, count=1)
        assert k == 1, pat
    for pat, tok in subs_new:
        n, k = re.subn(pat, tok, n, count=1)
        assert k == 1, pat
    assert o == n.strip() or o.strip() == n.strip()
    # the softmax sum order is unchanged: kq 0+1+2+3
    assert re.search(r"c3_s\(0, srow, scol\)\] \+ S_s\[c3_s\(1, srow, scol\)\] \+ S_s\[c3_s\(2, srow, scol\)\] \+\s+S_s\[c3_s\(3, srow, scol\)\]", SRC)


def test_c3_host_launch():
    host = _kernel(SRC, "void tq_gqa_decode_c3(", "PYBIND11_MODULE")
    assert "tq_gqa_stage1_c3<<<grid, NTHR, smem, stream>>>(" in host
    assert "const size_t smem = C3_SMEM_BYTES;" in host
    assert "tq_gqa_stage2<<<" in host  # stage 2 unchanged
    assert 'm.def("decode_c3", &tq_gqa_decode_c3' in SRC and 'm.def("decode", &tq_gqa_decode' in SRC


@pytest.mark.parametrize("env,expect", [(None, "decode"), ("0", "decode"), ("1", "decode_c3")])
def test_wrapper_gate(monkeypatch, env, expect):
    import torch

    if env is None:
        monkeypatch.delenv("VLLM_TQ_GQA_C3", raising=False)
    else:
        monkeypatch.setenv("VLLM_TQ_GQA_C3", env)
    sys.modules.pop("vllm.v1.attention.ops.tq_gqa_cuda", None)
    C = importlib.import_module("vllm.v1.attention.ops.tq_gqa_cuda")
    called = []
    fake = types.SimpleNamespace(decode=lambda *a: called.append("decode"), decode_c3=lambda *a: called.append("decode_c3"))
    monkeypatch.setattr(C, "_EXT", fake)
    S, QL, Hq, Hk, D = 1, 4, 12, 2, 256
    q = torch.zeros(S * QL, Hq, D, dtype=torch.float16)
    kv = torch.zeros(2, 16, Hk, 230, dtype=torch.uint8)
    C.tq_gqa_decode_attention(q, kv, torch.zeros(S, 2, dtype=torch.int32), torch.ones(S, dtype=torch.int32),
                              torch.eye(D), torch.zeros(8), 0.0625, True, q_per_seq=QL, num_splits=8)
    assert called == [expect]
    assert C.c3_enabled() == (expect == "decode_c3")
    sys.modules.pop("vllm.v1.attention.ops.tq_gqa_cuda", None)
