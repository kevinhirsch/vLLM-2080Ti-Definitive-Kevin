# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduction test for ring buffer deadlock caused by lost ZMQ notifications.

When the reader is in the SpinCondition hot-path (within busy_loop_s of the
last read), it only calls sched_yield() and never polls the ZMQ notification
socket.  If the writer calls notify() during this window the zmq.CONFLATE
setting can discard the notification.  The reader then enters the cold-path
and blocks on the poller waiting for a notification that never comes; on the
writer side acquire_write() spins waiting for the reader's ACK.

The fix adds a non-blocking ZMQ poll at the start of every hot-path wait()
call so that notifications are never lost.
"""

import threading
import time
from unittest import mock

import zmq

from vllm.distributed.device_communicators.shm_broadcast import (
    MessageQueue,
    SpinCondition,
)


def test_spincondition_hot_path_notify_not_lost():
    """Verify that notify() sent while the reader is in hot-path is still
    visible via a non-blocking poll.  Without the fix the notify is lost
    because wait() only calls sched_yield() during the busy-loop window."""
    ctx = zmq.Context()
    addr = "inproc://test_spincondition_hot_path"

    # Writer binds first (PUB), then reader connects (SUB)
    writer_cond = SpinCondition(
        is_reader=False, context=ctx, notify_address=addr
    )
    reader_cond = SpinCondition(
        is_reader=True, context=ctx, notify_address=addr
    )

    try:
        # Force the reader into persistent hot-path mode
        reader_cond.busy_loop_s = 9999
        reader_cond.record_read()

        # Writer sends a notification
        writer_cond.notify()

        # Give ZMQ a moment to deliver the inproc message
        time.sleep(0.05)

        # Reader calls wait() — without the fix this returns immediately
        # without ever seeing the notify.  With the fix it drains the
        # notification via _poll_notifications().
        reader_cond.wait(timeout_ms=0)

        # Verify the notify was consumed: a second poll on the socket
        # should return nothing.
        events = dict(reader_cond.poller.poll(timeout=0))
        notify_key = reader_cond.local_notify_socket
        assert notify_key not in events, (
            "Notify was not consumed by wait() in hot-path"
        )
    finally:
        for sock in (
            writer_cond.local_notify_socket,
            reader_cond.local_notify_socket,
            reader_cond.read_cancel_socket,
            reader_cond.write_cancel_socket,
        ):
            sock.close(linger=0)
        ctx.term()


def test_message_queue_notify_loss_no_deadlock():
    """Verify that a notify() sent while the reader is in hot-path is
    drained by _poll_notifications() inside wait().

    The scenario:
      1. Reader is in persistent hot-path (busy_loop_s >> 0) with no data
         in the ring buffer, so acquire_read()'s check() returns False
         and the reader spins in wait().
      2. Writer calls notify() (the SUB socket receives the message).
      3. Without the fix, wait() only calls sched_yield() and never polls
         the ZMQ socket, so the notify stays in the SUB socket.
      4. With the fix, _poll_notifications() drains the notify on the
         first wait() call after it arrives.

    After letting the reader spin for 0.5 s we poll the reader's SUB
    socket non-blocking.  Without the fix the notify is still pending
    (events non-empty); with the fix it was drained (events empty).
    """
    writer = MessageQueue(
        n_reader=1,
        n_local_reader=1,
        max_chunk_bytes=1024,
        max_chunks=2,
    )
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    writer.wait_until_ready()
    reader.wait_until_ready()

    stop = threading.Event()
    errors: list = []

    def reader_thread():
        try:
            while not stop.is_set():
                reader._spin_condition.wait(timeout_ms=0)
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=reader_thread, daemon=True)
    try:
        # Force reader into persistent hot-path.
        reader._spin_condition.busy_loop_s = 9999
        reader._spin_condition.record_read()

        t.start()

        # Give the reader a moment to enter the hot-path spin loop.
        time.sleep(0.05)

        # Writer sends a notification (no data written — the ring is empty,
        # so the reader's check() would return False and it relies solely on
        # wait() to observe the notify).
        writer._spin_condition.notify()

        # Let the reader spin long enough to call wait() many times.  With
        # the fix the notify is drained on the first wait() after arrival.
        time.sleep(0.5)

        # Verify the notify was consumed from the SUB socket.  A
        # non-blocking poll should return no events for the notify socket.
        events = dict(
            reader._spin_condition.poller.poll(timeout=0)
        )
        notify_key = reader._spin_condition.local_notify_socket
        assert notify_key not in events, (
            "Notify was not drained during hot-path wait(); it is still "
            "pending in the SUB socket.  Without the fix, wait() only "
            "calls sched_yield() and never polls the ZMQ socket, so a "
            "notify sent while the reader is in hot-path is never "
            "consumed — leading to a lost-wakeup deadlock."
        )
    finally:
        stop.set()
        t.join(timeout=5)
        writer.shutdown()
        reader.shutdown()
        for sock in (
            writer.local_socket,
            writer._spin_condition.local_notify_socket,
            reader.local_socket,
            reader._spin_condition.local_notify_socket,
            reader._spin_condition.read_cancel_socket,
            reader._spin_condition.write_cancel_socket,
        ):
            sock.close(linger=0)


def test_writer_ring_buffer_deadlock_recovered():
    """Verify that the writer calls notify() inside the acquire_write()
    spin loop.

    Deadlock cycle (observed in production via GDB traces):
      1. GPU stall or long computation on a worker delays it from reading
         the shared-memory ring buffer.
      2. Writer (EngineCore) fills all max_chunks slots.
      3. Worker finishes GPU work, returns to dequeue() — but more than
         busy_loop_s has elapsed, so SpinCondition.wait() enters the
         **cold path** and blocks on zmq.Poller.poll().
      4. Writer wraps the ring: acquire_write() finds the next slot still
         written (not yet ACKed by the reader).  Writer enters the spin
         loop.
      5. Without the fix the spin loop only calls sched_yield(), so the
         reader stays blocked on the ZMQ poller forever — the writer
         cannot exit acquire_write() to call the post-enqueue notify(),
         and the reader never wakes to ACK the slot.  **Complete deadlock.**

    The fix adds self._spin_condition.notify() inside the acquire_write()
    spin loop so that any reader parked in the cold path is woken to
    re-check and ACK the ring-buffer slot.

    Test strategy: fill the ring buffer so the next enqueue() must spin
    in acquire_write() (slot is written but never ACKed).  Spy on
    SpinCondition.notify() to verify it is called *during* the spin loop.
    Without the fix, notify() is never called while spinning — the only
    notify() is the post-enqueue one at line 907, which cannot fire until
    acquire_write() returns (which requires the reader to ACK — a
    deadlock).  After verifying the spy, drain the reader so the writer
    can proceed and the test exits cleanly.
    """
    writer = MessageQueue(
        n_reader=1,
        n_local_reader=1,
        max_chunk_bytes=1024,
        max_chunks=2,
    )
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    writer.wait_until_ready()
    reader.wait_until_ready()

    t = None
    original_notify = writer._spin_condition.notify
    try:
        # Fill the ring buffer (max_chunks = 2).  Reader never dequeues.
        writer.enqueue({"seq": 0})
        writer.enqueue({"seq": 1})
        # Writer's current_idx is now 0 (wrapped).  Slot 0 is
        # written_flag=1, read_count=0 (never ACKed by the reader).

        # Spy on notify() to count calls from the spin loop.  Any notify()
        # calls observed while the writer is spinning in acquire_write()
        # must originate from the spin loop itself — the post-enqueue
        # notify() (line 907) cannot fire until acquire_write() returns.
        notify_calls: list = []
        def counting_notify():
            notify_calls.append(time.monotonic())
            original_notify()
        writer._spin_condition.notify = counting_notify

        # Start a writer thread that will spin in acquire_write().
        # The timeout prevents an indefinite hang if the test breaks.
        errors: list = []
        def writer_thread():
            try:
                writer.enqueue({"seq": 2}, timeout=30)
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=writer_thread, daemon=True)
        t.start()

        # Let the writer spin for 0.5 s.  During this window any
        # notify() calls can only come from the acquire_write() spin
        # loop — the writer cannot exit acquire_write() until the
        # reader ACKs, which hasn't happened yet.
        time.sleep(0.5)
        spin_notify_count = len(notify_calls)
        assert spin_notify_count > 0, (
            "Writer did not call notify() during the acquire_write() spin "
            f"loop (0 calls in 0.5 s).  Without the fix, only sched_yield() "
            "is called in the spin loop, so a reader blocked in the "
            "SpinCondition cold-path would never be woken — deadlock."
        )

        # Drain the ring so the writer can proceed and exit the spin.
        for i in range(3):
            msg = reader.dequeue(timeout=5)
            assert msg["seq"] == i

        t.join(timeout=5)
        assert not errors, f"Writer thread errored: {errors}"
        assert not t.is_alive(), "Writer thread did not finish"
    finally:
        writer._spin_condition.notify = original_notify
        # Drain any unread data so the writer thread (if still alive)
        # can exit acquire_write() instead of spinning until timeout.
        if t is not None and t.is_alive():
            try:
                for _ in range(3):
                    reader.dequeue(timeout=2)
            except Exception:
                pass
            t.join(timeout=5)
        writer.shutdown()
        reader.shutdown()
        for sock in (
            writer.local_socket,
            writer._spin_condition.local_notify_socket,
            reader.local_socket,
            reader._spin_condition.local_notify_socket,
            reader._spin_condition.read_cancel_socket,
            reader._spin_condition.write_cancel_socket,
        ):
            sock.close(linger=0)