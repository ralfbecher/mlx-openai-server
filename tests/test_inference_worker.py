"""Tests for queued cancellation behavior in the inference worker."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from app.core.inference_worker import InferenceWorker


@pytest.mark.anyio
async def test_submit_timeout_skips_work_that_has_not_started() -> None:
    """Timed-out queued work should be skipped when it reaches the worker."""

    worker = InferenceWorker(queue_size=2, timeout=1.0)
    worker.start()

    blocker_started = threading.Event()
    unblock = threading.Event()
    executed_late_work = threading.Event()

    def blocking_job() -> str:
        blocker_started.set()
        unblock.wait(timeout=1.0)
        return "first"

    def queued_job() -> str:
        executed_late_work.set()
        return "second"

    first_task = asyncio.create_task(worker.submit(blocking_job))
    await asyncio.to_thread(blocker_started.wait, 1.0)

    worker._timeout = 0.01
    with pytest.raises(TimeoutError, match="Inference timed out"):
        await worker.submit(queued_job)

    unblock.set()
    assert await first_task == "first"

    await asyncio.sleep(0.05)
    assert not executed_late_work.is_set()

    stats = worker.get_stats()
    assert stats["abandoned_requests"] == 1

    worker.stop()


@pytest.mark.anyio
async def test_submit_cancellation_skips_work_that_has_not_started() -> None:
    """Caller cancellation should prevent queued work from executing later."""

    worker = InferenceWorker(queue_size=2, timeout=1.0)
    worker.start()

    blocker_started = threading.Event()
    unblock = threading.Event()
    executed_cancelled_work = threading.Event()

    def blocking_job() -> str:
        blocker_started.set()
        unblock.wait(timeout=1.0)
        return "first"

    def queued_job() -> str:
        executed_cancelled_work.set()
        return "second"

    first_task = asyncio.create_task(worker.submit(blocking_job))
    await asyncio.to_thread(blocker_started.wait, 1.0)

    second_task = asyncio.create_task(worker.submit(queued_job))
    await asyncio.sleep(0)
    second_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await second_task

    unblock.set()
    assert await first_task == "first"

    await asyncio.sleep(0.05)
    assert not executed_cancelled_work.is_set()

    stats = worker.get_stats()
    assert stats["abandoned_requests"] == 1

    worker.stop()


@pytest.mark.anyio
async def test_submit_timeout_does_not_hide_started_work_completion() -> None:
    """Active work may keep running after timeout, but it should not be double-counted."""

    worker = InferenceWorker(queue_size=1, timeout=0.01)
    worker.start()

    started = threading.Event()
    finished = threading.Event()

    def slow_job() -> str:
        started.set()
        time.sleep(0.05)
        finished.set()
        return "done"

    with pytest.raises(TimeoutError, match="Inference timed out"):
        await worker.submit(slow_job)

    await asyncio.to_thread(started.wait, 1.0)
    await asyncio.to_thread(finished.wait, 1.0)

    stats = worker.get_stats()
    assert stats["completed_requests"] == 1
    assert stats["abandoned_requests"] == 0

    worker.stop()
