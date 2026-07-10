from __future__ import annotations

from datetime import datetime, timedelta, timezone

from phonoflow_scheduler.cpu_queue_config import CpuQueueConfig
from phonoflow_scheduler.cpu_queue_state import (
    CpuQueueJobRequest,
    CpuQueueState,
    cancel_pending_request,
    enqueue_request,
    expire_stale_leases,
    mark_job_finished,
    normalize_job_slots,
    release_lease,
    snapshot_for_admin,
    snapshot_for_user,
    try_start_next_jobs,
)


def _config(max_running_jobs: int = 2) -> CpuQueueConfig:
    return CpuQueueConfig.from_values(
        enabled=True,
        total_cpu_slots=24,
        max_running_jobs=max_running_jobs,
        default_job_cpu_slots=12,
    )


def _request(index: int, *, user_id: str = "user-a", slots: int = 12) -> CpuQueueJobRequest:
    return CpuQueueJobRequest.new(
        job_id=f"job_{index:03d}",
        user_id=user_id,
        owner=f"{user_id}@example.com",
        requested_slots=slots,
    )


def test_fifo_starts_two_of_ten_jobs_with_24_slots() -> None:
    config = _config(max_running_jobs=2)
    state = CpuQueueState.empty(config)
    for index in range(10):
        state = enqueue_request(state, _request(index))

    state = try_start_next_jobs(state, config)

    assert state.used_cpu_slots == 24
    assert state.available_cpu_slots == 0
    assert state.running_job_count == 2
    assert [request.job_id for request in state.pending_requests] == [f"job_{index:03d}" for index in range(2, 10)]
    assert [lease.job_id for lease in state.active_leases] == ["job_000", "job_001"]


def test_release_starts_next_fifo_job() -> None:
    config = _config(max_running_jobs=2)
    state = CpuQueueState.empty(config)
    for index in range(3):
        state = enqueue_request(state, _request(index))
    state = try_start_next_jobs(state, config)

    state = release_lease(state, state.active_leases[0].lease_id)
    state = mark_job_finished(state, "job_000")
    state = try_start_next_jobs(state, config)

    assert [lease.job_id for lease in state.active_leases] == ["job_001", "job_002"]
    assert state.used_cpu_slots == 24


def test_max_running_jobs_limits_even_when_slots_available() -> None:
    config = CpuQueueConfig.from_values(
        enabled=True,
        total_cpu_slots=24,
        max_running_jobs=1,
        default_job_cpu_slots=6,
    )
    state = CpuQueueState.empty(config)
    for index in range(3):
        state = enqueue_request(state, _request(index, slots=6))

    state = try_start_next_jobs(state, config)

    assert state.running_job_count == 1
    assert state.used_cpu_slots == 6
    assert [lease.job_id for lease in state.active_leases] == ["job_000"]
    assert [request.job_id for request in state.pending_requests] == ["job_001", "job_002"]


def test_fifo_does_not_backfill_small_later_job() -> None:
    config = CpuQueueConfig.from_values(
        enabled=True,
        total_cpu_slots=12,
        max_running_jobs=2,
        default_job_cpu_slots=12,
    )
    state = CpuQueueState.empty(config)
    state = enqueue_request(state, _request(0, slots=8))
    state = enqueue_request(state, _request(1, slots=8))
    state = enqueue_request(state, _request(2, slots=4))

    state = try_start_next_jobs(state, config)

    assert [lease.job_id for lease in state.active_leases] == ["job_000"]
    assert [request.job_id for request in state.pending_requests] == ["job_001", "job_002"]


def test_stale_lease_returns_slots_and_marks_failed() -> None:
    config = _config(max_running_jobs=2)
    state = CpuQueueState.empty(config)
    state = enqueue_request(state, _request(0))
    state = try_start_next_jobs(state, config)
    old = state.active_leases[0]
    state = state.copy_with(
        active_leases=[
            old.copy_with(acquired_at="2026-07-06T00:00:00+00:00", expires_at="2026-07-06T00:01:00+00:00")
        ]
    )

    state = expire_stale_leases(state, datetime(2026, 7, 7, tzinfo=timezone.utc), stale_lease_s=60.0)

    assert state.active_leases == []
    assert state.used_cpu_slots == 0
    assert state.completed_recent[0].status == "failed"


def test_cancel_pending_and_user_snapshot_redacts_other_jobs() -> None:
    config = _config()
    state = CpuQueueState.empty(config)
    state = enqueue_request(state, _request(0, user_id="alice"))
    state = enqueue_request(state, _request(1, user_id="bob"))
    state = cancel_pending_request(state, "job_001")

    user_snapshot = snapshot_for_user(state, "alice")
    admin_snapshot = snapshot_for_admin(state)

    assert user_snapshot["pending_requests"][0]["job_id"] == "job_000"
    assert "admin_managed" not in user_snapshot
    assert all(item.get("user_id") != "bob" for item in user_snapshot["current_user_jobs"])
    assert admin_snapshot["config"]["total_cpu_slots"] == 24
    assert state.completed_recent[0].status == "cancelled"


def test_normalize_job_slots_clamps_to_admin_limits() -> None:
    config = CpuQueueConfig.from_values(
        enabled=True,
        total_cpu_slots=24,
        default_job_cpu_slots=12,
        min_job_cpu_slots=2,
        max_job_cpu_slots=16,
    )

    assert normalize_job_slots(None, config) == 12
    assert normalize_job_slots(1, config) == 2
    assert normalize_job_slots(99, config) == 16
