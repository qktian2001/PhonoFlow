from __future__ import annotations

from pathlib import Path

import pytest

from phonoflow_scheduler.cpu_queue_config import CpuQueueConfig
from phonoflow_scheduler.cpu_queue_filelock import FileCpuQueue


def _queue(tmp_path: Path, *, max_running_jobs: int = 2, timeout: float | None = 0.01) -> FileCpuQueue:
    return FileCpuQueue(
        CpuQueueConfig.from_values(
            enabled=True,
            total_cpu_slots=24,
            max_running_jobs=max_running_jobs,
            default_job_cpu_slots=12,
            acquire_timeout_s=timeout,
            poll_interval_s=0.001,
            state_dir=tmp_path,
        )
    )


def test_file_queue_acquires_and_releases_fifo(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    for index in range(3):
        queue.submit_request(f"job_{index}", "user-a", 12, "test")

    first = queue.acquire_for_job("job_0")
    second = queue.acquire_for_job("job_1")
    snapshot = queue.status()

    assert first.allocated_slots == 12
    assert second.allocated_slots == 12
    assert snapshot.used_cpu_slots == 24
    assert [request.job_id for request in snapshot.pending_requests] == ["job_2"]

    queue.release(first)
    third = queue.acquire_for_job("job_2")

    assert third.job_id == "job_2"
    assert queue.status().used_cpu_slots == 24


def test_file_queue_timeout_marks_request(tmp_path: Path) -> None:
    queue = _queue(tmp_path, max_running_jobs=1, timeout=0.001)
    queue.submit_request("job_0", "user-a", 12, "test")
    queue.submit_request("job_1", "user-a", 12, "test")
    queue.acquire_for_job("job_0")

    with pytest.raises(TimeoutError):
        queue.acquire_for_job("job_1", timeout_s=0.001)

    state = queue.status()
    assert "job_1" not in [request.job_id for request in state.pending_requests]
    assert any(request.job_id == "job_1" and request.status == "timeout" for request in state.completed_recent)


def test_file_queue_cancelled_request_unblocks_acquire(tmp_path: Path) -> None:
    queue = _queue(tmp_path, max_running_jobs=1, timeout=None)
    queue.submit_request("job_0", "user-a", 12, "test")
    queue.submit_request("job_1", "user-a", 12, "test")
    queue.acquire_for_job("job_0")
    queue.cancel("job_1")

    with pytest.raises(RuntimeError, match="cancelled"):
        queue.acquire_for_job("job_1", timeout_s=0.05)

    state = queue.status()
    assert "job_1" not in [request.job_id for request in state.pending_requests]
    assert any(request.job_id == "job_1" and request.status == "cancelled" for request in state.completed_recent)


def test_file_queue_stale_lease_reclaims_slots(tmp_path: Path) -> None:
    queue = FileCpuQueue(
        CpuQueueConfig.from_values(
            enabled=True,
            total_cpu_slots=12,
            max_running_jobs=1,
            default_job_cpu_slots=12,
            acquire_timeout_s=0.05,
            poll_interval_s=0.001,
            stale_lease_s=0.001,
            state_dir=tmp_path,
        )
    )
    queue.submit_request("job_0", "user-a", 12, "test")
    queue.acquire_for_job("job_0")
    queue.submit_request("job_1", "user-a", 12, "test")

    lease = queue.acquire_for_job("job_1", timeout_s=0.05)

    assert lease.job_id == "job_1"
    assert queue.status().used_cpu_slots == 12
