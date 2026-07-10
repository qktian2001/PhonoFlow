"""Unified API for optional CPU queue implementations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Protocol

from phonoflow_scheduler.cpu_queue_config import CpuQueueConfig
from phonoflow_scheduler.cpu_queue_filelock import FileCpuQueue
from phonoflow_scheduler.cpu_queue_state import (
    CpuQueueJobRequest,
    CpuQueueLease,
    CpuQueueState,
    enqueue_request,
    mark_job_finished,
    release_lease,
    snapshot_for_admin,
    snapshot_for_user,
    try_start_next_jobs,
)


class CpuQueueProtocol(Protocol):
    def submit_request(
        self,
        job_id: str,
        user_id: str | None,
        requested_slots: int | None,
        reason: str | None = None,
        owner: str | None = None,
    ) -> CpuQueueJobRequest:
        ...

    def acquire_for_job(self, job_id: str, timeout_s: float | None = None) -> CpuQueueLease:
        ...

    def release(self, lease_or_lease_id: CpuQueueLease | str) -> None:
        ...

    def cancel(self, job_id: str) -> None:
        ...

    def status(self) -> CpuQueueState:
        ...

    def user_status(self, user_id: str | None = None) -> dict[str, object]:
        ...

    def admin_status(self) -> dict[str, object]:
        ...


class NoopCpuQueue:
    """Disabled CPU queue that preserves existing behavior."""

    def __init__(self) -> None:
        self._state = CpuQueueState.empty(CpuQueueConfig.from_values(enabled=True, total_cpu_slots=1))

    def submit_request(
        self,
        job_id: str,
        user_id: str | None = None,
        requested_slots: int | None = None,
        reason: str | None = None,
        owner: str | None = None,
    ) -> CpuQueueJobRequest:
        return CpuQueueJobRequest.new(
            job_id=job_id,
            user_id=user_id,
            owner=owner,
            requested_slots=requested_slots or 1,
            reason=reason,
        )

    def acquire_for_job(self, job_id: str, timeout_s: float | None = None) -> CpuQueueLease:
        request = CpuQueueJobRequest.new(job_id=job_id, requested_slots=1, reason="CPU queue disabled.")
        return CpuQueueLease.new(request=request, allocated_slots=1, stale_lease_s=86400.0)

    def release(self, lease_or_lease_id: CpuQueueLease | str) -> None:
        return None

    def cancel(self, job_id: str) -> None:
        return None

    def status(self) -> CpuQueueState:
        return self._state

    def user_status(self, user_id: str | None = None) -> dict[str, object]:
        payload = snapshot_for_user(self._state, user_id)
        payload["enabled"] = False
        return payload

    def admin_status(self) -> dict[str, object]:
        payload = snapshot_for_admin(self._state, CpuQueueConfig.from_values())
        payload["enabled"] = False
        return payload


@dataclass
class InMemoryCpuQueue:
    """Test-only in-memory queue implementation."""

    config: CpuQueueConfig
    state: CpuQueueState = field(init=False)

    def __post_init__(self) -> None:
        self.state = CpuQueueState.empty(self.config)

    def submit_request(
        self,
        job_id: str,
        user_id: str | None = None,
        requested_slots: int | None = None,
        reason: str | None = None,
        owner: str | None = None,
    ) -> CpuQueueJobRequest:
        request = CpuQueueJobRequest.new(
            job_id=job_id,
            user_id=user_id,
            owner=owner,
            requested_slots=requested_slots or self.config.default_job_cpu_slots,
            reason=reason,
        )
        self.state = enqueue_request(self.state, request)
        return request

    def acquire_for_job(self, job_id: str, timeout_s: float | None = None) -> CpuQueueLease:
        self.state = try_start_next_jobs(self.state, self.config)
        lease = next((item for item in self.state.active_leases if item.job_id == job_id), None)
        if lease is None:
            raise TimeoutError(f"Timed out waiting for CPU queue lease for job {job_id}.")
        return lease

    def release(self, lease_or_lease_id: CpuQueueLease | str) -> None:
        lease_id = lease_or_lease_id.lease_id if isinstance(lease_or_lease_id, CpuQueueLease) else str(lease_or_lease_id)
        lease = next((item for item in self.state.active_leases if item.lease_id == lease_id), None)
        self.state = release_lease(self.state, lease_id)
        if lease is not None:
            self.state = mark_job_finished(self.state, lease.job_id)
        self.state = try_start_next_jobs(self.state, self.config)

    def cancel(self, job_id: str) -> None:
        from phonoflow_scheduler.cpu_queue_state import cancel_pending_request

        self.state = cancel_pending_request(self.state, job_id)

    def status(self) -> CpuQueueState:
        return self.state

    def user_status(self, user_id: str | None = None) -> dict[str, object]:
        return snapshot_for_user(self.state, user_id)

    def admin_status(self) -> dict[str, object]:
        return snapshot_for_admin(self.state, self.config)


def get_cpu_queue(config: CpuQueueConfig | None = None) -> CpuQueueProtocol:
    resolved = config or CpuQueueConfig.from_values()
    if not resolved.enabled:
        return NoopCpuQueue()
    return FileCpuQueue(resolved)


def submit_cpu_job(
    queue: CpuQueueProtocol,
    *,
    job_id: str,
    user_id: str | None,
    requested_slots: int | None,
    reason: str | None = None,
    owner: str | None = None,
) -> CpuQueueJobRequest:
    return queue.submit_request(job_id, user_id, requested_slots, reason=reason, owner=owner)


def wait_for_cpu_lease(queue: CpuQueueProtocol, job_id: str, timeout_s: float | None = None) -> CpuQueueLease:
    return queue.acquire_for_job(job_id, timeout_s=timeout_s)


@contextmanager
def cpu_job_guard(queue: CpuQueueProtocol, job_id: str, timeout_s: float | None = None) -> Iterator[CpuQueueLease]:
    lease = queue.acquire_for_job(job_id, timeout_s=timeout_s)
    try:
        yield lease
    finally:
        queue.release(lease)


def get_cpu_queue_snapshot(queue: CpuQueueProtocol, user_id: str | None = None, *, admin: bool = False) -> dict[str, object]:
    return queue.admin_status() if admin else queue.user_status(user_id)
