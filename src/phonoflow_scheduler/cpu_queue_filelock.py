"""File-backed CPU queue using a JSON state file and ``fcntl`` locking."""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from phonoflow_scheduler.cpu_queue_config import CpuQueueConfig
from phonoflow_scheduler.cpu_queue_state import (
    CpuQueueJobRequest,
    CpuQueueLease,
    CpuQueueState,
    cancel_pending_request,
    enqueue_request,
    expire_stale_leases,
    mark_job_failed,
    mark_job_finished,
    release_lease,
    snapshot_for_admin,
    snapshot_for_user,
    try_start_next_jobs,
)


CPU_QUEUE_DIR_ENV = "PHONOFLOW_CPU_QUEUE_DIR"


class FileCpuQueue:
    """Small PBS-style queue stored as a locked JSON file."""

    def __init__(self, config: CpuQueueConfig) -> None:
        if not config.enabled:
            raise ValueError("FileCpuQueue requires enabled=True.")
        if sys.platform.startswith("win"):
            raise RuntimeError("FileCpuQueue requires fcntl file locking and is not supported on native Windows.")
        self.config = config
        self.state_dir = _state_dir(config)
        self.state_file = self.state_dir / "cpu_queue_state.json"
        self.lock_file = self.state_dir / "cpu_queue_state.lock"

    def submit_request(
        self,
        job_id: str,
        user_id: str | None,
        requested_slots: int | None,
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
        with self._locked_state() as state:
            existing = _request_or_completed(state, job_id)
            if existing is not None and existing.status in {"queued", "running"}:
                yield_state = state
                request = existing
            else:
                yield_state = enqueue_request(state, request)
            self._write_state(yield_state)
        return request

    def acquire_for_job(self, job_id: str, timeout_s: float | None = None) -> CpuQueueLease:
        timeout = self.config.acquire_timeout_s if timeout_s is None else timeout_s
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            with self._locked_state() as state:
                state = expire_stale_leases(state, _now_datetime(), self.config.stale_lease_s)
                existing = _request_or_completed(state, job_id)
                lease = _lease_for_job(state, job_id)
                if existing is not None and lease is None and existing.status in {"cancelled", "failed", "timeout", "finished"}:
                    self._write_state(state)
                    raise RuntimeError(f"CPU queue request for job {job_id} was {existing.status}.")
                if existing is None and lease is None:
                    state = enqueue_request(
                        state,
                        CpuQueueJobRequest.new(
                            job_id=job_id,
                            requested_slots=self.config.default_job_cpu_slots,
                            reason="Implicit CPU queue request.",
                        ),
                    )
                state = try_start_next_jobs(state, self.config)
                lease = _lease_for_job(state, job_id)
                self._write_state(state)
                if lease is not None:
                    return lease
            if deadline is not None and time.monotonic() >= deadline:
                with self._locked_state() as state:
                    state = _mark_timeout(state, job_id)
                    self._write_state(state)
                raise TimeoutError(f"Timed out waiting for CPU queue lease for job {job_id}.")
            time.sleep(self.config.poll_interval_s)

    def release(self, lease_or_lease_id: CpuQueueLease | str) -> None:
        lease_id = lease_or_lease_id.lease_id if isinstance(lease_or_lease_id, CpuQueueLease) else str(lease_or_lease_id)
        with self._locked_state() as state:
            lease = next((item for item in state.active_leases if item.lease_id == lease_id), None)
            state = release_lease(state, lease_id)
            if lease is not None:
                state = mark_job_finished(state, lease.job_id)
            state = try_start_next_jobs(state, self.config)
            self._write_state(state)

    def cancel(self, job_id: str) -> None:
        with self._locked_state() as state:
            lease = _lease_for_job(state, job_id)
            if lease is not None:
                state = release_lease(state, lease.lease_id)
                state = mark_job_failed(state, job_id)
            else:
                state = cancel_pending_request(state, job_id)
            state = try_start_next_jobs(state, self.config)
            self._write_state(state)

    def status(self) -> CpuQueueState:
        with self._locked_state() as state:
            state = expire_stale_leases(state, _now_datetime(), self.config.stale_lease_s)
            state = try_start_next_jobs(state, self.config)
            self._write_state(state)
            return state

    def user_status(self, user_id: str | None = None) -> dict[str, object]:
        return snapshot_for_user(self.status(), user_id)

    def admin_status(self) -> dict[str, object]:
        return snapshot_for_admin(self.status(), self.config)

    @contextmanager
    def _locked_state(self) -> Iterator[CpuQueueState]:
        import fcntl

        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_file.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield self._read_state()
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _read_state(self) -> CpuQueueState:
        if not self.state_file.exists():
            return CpuQueueState.empty(self.config)
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return CpuQueueState.empty(self.config)
        if not isinstance(data, dict):
            return CpuQueueState.empty(self.config)
        return CpuQueueState.from_dict(data, self.config)

    def _write_state(self, state: CpuQueueState) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_file)


def _state_dir(config: CpuQueueConfig) -> Path:
    if config.state_dir is not None:
        return config.state_dir
    env_dir = os.environ.get(CPU_QUEUE_DIR_ENV)
    if env_dir:
        return Path(env_dir)
    return Path("work") / "cpu_queue"


def _lease_for_job(state: CpuQueueState, job_id: str) -> CpuQueueLease | None:
    return next((lease for lease in state.active_leases if lease.job_id == job_id), None)


def _request_or_completed(state: CpuQueueState, job_id: str) -> CpuQueueJobRequest | None:
    for request in [*state.pending_requests, *state.completed_recent]:
        if request.job_id == job_id:
            return request
    return None


def _mark_timeout(state: CpuQueueState, job_id: str) -> CpuQueueState:
    from phonoflow_scheduler.cpu_queue_state import CpuQueueJobRequest, utc_now_iso

    pending = []
    completed = list(state.completed_recent)
    found = False
    for request in state.pending_requests:
        if request.job_id == job_id:
            completed.append(request.copy_with(status="timeout", finished_at=utc_now_iso(), reason="Timed out waiting for CPU slots."))
            found = True
        else:
            pending.append(request)
    if not found:
        completed.append(
            CpuQueueJobRequest.new(job_id=job_id, requested_slots=1, reason="Timed out waiting for CPU slots.").copy_with(
                status="timeout",
                finished_at=utc_now_iso(),
            )
        )
    return state.copy_with(pending_requests=pending, completed_recent=completed[-100:])


def _now_datetime():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
