from __future__ import annotations

from phonoflow_scheduler.cpu_queue import NoopCpuQueue, cpu_job_guard, get_cpu_queue
from phonoflow_scheduler.cpu_queue_config import CpuQueueConfig


def test_get_cpu_queue_defaults_to_noop() -> None:
    queue = get_cpu_queue(CpuQueueConfig.from_values())

    assert isinstance(queue, NoopCpuQueue)
    lease = queue.acquire_for_job("job_1")
    assert lease.job_id == "job_1"
    assert lease.allocated_slots == 1


def test_cpu_job_guard_noop_releases_without_error() -> None:
    queue = NoopCpuQueue()

    with cpu_job_guard(queue, "job_1") as lease:
        assert lease.job_id == "job_1"
        assert lease.allocated_slots == 1

    assert queue.status().running_job_count == 0
