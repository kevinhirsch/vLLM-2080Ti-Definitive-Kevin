"""CPU test: the shared memfd copy of a PLE table is one copy for every rank of an engine.

On HNET00 CUDA refuses a read-only pin of the table file's page-cache pages (cudaError 801; Linux forbids
long-term pins of file-backed pages), so HostRegisteredTable copies the table once into a memfd that the
ranks (children of one EngineCore) share. Two child processes here play the ranks: both must see the file's
bytes, and a byte one writes must be visible to the other (one physical copy, not two).
"""
import multiprocessing as mp
import os

os.environ.setdefault("TRITON_INTERPRET", "1")  # same import order as test_ple_uva (triton reads it once)

from vllm.models.qwen4_exp.nvidia import model as _model  # noqa: F401,E402  (imports ple_host without the cycle)
from vllm.models.qwen4_exp.nvidia import ple_host  # noqa: E402


def _rank(path, offset, nbytes, rank, q, barrier):
    mm, _fd = ple_host._shared_table_copy(path, offset, nbytes)
    first = bytes(mm[:16])
    barrier.wait()
    if rank == 0:
        mm[0] = 0xAB
    barrier.wait()
    q.put((rank, first, mm[0]))


def test_two_ranks_share_one_copy(tmp_path):
    payload = os.urandom(3 * 4096 + 123)
    path = tmp_path / "table.bin"
    path.write_bytes(b"H" * 100 + payload)
    ctx = mp.get_context("fork")
    q, barrier = ctx.Queue(), ctx.Barrier(2)
    procs = [ctx.Process(target=_rank, args=(str(path), 100, len(payload), r, q, barrier)) for r in (0, 1)]
    for p in procs:
        p.start()
    got = dict((r, (first, b0)) for r, first, b0 in (q.get(timeout=60) for _ in procs))
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    assert got[0][0] == got[1][0] == payload[:16]  # both mapped the file's table bytes
    assert got[1][1] == 0xAB  # rank 0's write is visible to rank 1: one shared copy
