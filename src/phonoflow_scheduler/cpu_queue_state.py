"""Pure state transitions for the optional CPU resource queue."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from uuid import uuid4

from phonoflow_scheduler.config import clean_positive_int
from phonoflow_scheduler.cpu_queue_config import CpuQueueConfig


CPU_QUEUE_PENDING_STATUSES = {"queued"}
CPU_QUEUE_RUNNING_STATUSES = {"running"}
CPU_QUEUE_DONE_STATUSES = {"finished", "cancelled", "failed", "timeout"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CpuQueueJobRequest:
    request_id: str
    job_id: str
    user_id: str | None
    owner: str | None
    requested_slots: int
    allocated_slots: int | None
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    reason: str | None = None

    @classmethod
    def new(
        cls,
        *,
        job_id: str,
        user_id: str | None = None,
        owner: str | None = None,
        requested_slots: int = 1,
        reason: str | None = None,
    ) -> "CpuQueueJobRequest":
        return cls(
            request_id=f"req_{uuid4().hex}",
            job_id=str(job_id),
            user_id=None if user_id is None else str(user_id),
            owner=None if owner is None else str(owner),
            requested_slots=clean_positive_int(requested_slots),
            allocated_slots=None,
            status="queued",
            created_at=utc_now_iso(),
            reason=reason,
        )

    def copy_with(self, **changes: object) -> "CpuQueueJobRequest":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CpuQueueJobRequest":
        return cls(
            request_id=str(data["request_id"]),
            job_id=str(data["job_id"]),
            user_id=None if data.get("user_id") is None else str(data.get("user_id")),
            owner=None if data.get("owner") is None else str(data.get("owner")),
            requested_slots=clean_positive_int(data.get("requested_slots") or 1),
            allocated_slots=None if data.get("allocated_slots") is None else clean_positive_int(data.get("allocated_slots")),
            status=str(data.get("status") or "queued"),
            created_at=str(data.get("created_at") or utc_now_iso()),
            started_at=None if data.get("started_at") is None else str(data.get("started_at")),
            finished_at=None if data.get("finished_at") is None else str(data.get("finished_at")),
            reason=None if data.get("reason") is None else str(data.get("reason")),
        )


@dataclass(frozen=True)
class CpuQueueLease:
    lease_id: str
    request_id: str
    job_id: str
    user_id: str | None
    allocated_slots: int
    acquired_at: str
    expires_at: str | None = None

    @classmethod
    def new(
        cls,
        *,
        request: CpuQueueJobRequest,
        allocated_slots: int,
        stale_lease_s: float,
        now: datetime | None = None,
    ) -> "CpuQueueLease":
        acquired = now or datetime.now(timezone.utc)
        expires = acquired.timestamp() + float(stale_lease_s)
        return cls(
            lease_id=f"lease_{uuid4().hex}",
            request_id=request.request_id,
            job_id=request.job_id,
            user_id=request.user_id,
            allocated_slots=clean_positive_int(allocated_slots),
            acquired_at=acquired.isoformat(),
            expires_at=datetime.fromtimestamp(expires, timezone.utc).isoformat(),
        )

    def copy_with(self, **changes: object) -> "CpuQueueLease":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CpuQueueLease":
        return cls(
            lease_id=str(data["lease_id"]),
            request_id=str(data["request_id"]),
            job_id=str(data["job_id"]),
            user_id=None if data.get("user_id") is None else str(data.get("user_id")),
            allocated_slots=clean_positive_int(data.get("allocated_slots") or 1),
            acquired_at=str(data.get("acquired_at") or utc_now_iso()),
            expires_at=None if data.get("expires_at") is None else str(data.get("expires_at")),
        )


@dataclass(frozen=True)
class CpuQueueState:
    total_cpu_slots: int
    used_cpu_slots: int
    available_cpu_slots: int
    max_running_jobs: int
    running_job_count: int
    pending_requests: list[CpuQueueJobRequest]
    active_leases: list[CpuQueueLease]
    completed_recent: list[CpuQueueJobRequest]
    updated_at: str

    @classmethod
    def empty(cls, config: CpuQueueConfig) -> "CpuQueueState":
        total = normalize_total_slots(config.total_cpu_slots)
        return cls(
            total_cpu_slots=total,
            used_cpu_slots=0,
            available_cpu_slots=total,
            max_running_jobs=clean_positive_int(config.max_running_jobs),
            running_job_count=0,
            pending_requests=[],
            active_leases=[],
            completed_recent=[],
            updated_at=utc_now_iso(),
        )

    def copy_with(self, **changes: object) -> "CpuQueueState":
        next_state = replace(self, **changes)
        return _recount(next_state)

    def to_dict(self) -> dict[str, object]:
        return {
            "total_cpu_slots": self.total_cpu_slots,
            "used_cpu_slots": self.used_cpu_slots,
            "available_cpu_slots": self.available_cpu_slots,
            "max_running_jobs": self.max_running_jobs,
            "running_job_count": self.running_job_count,
            "pending_requests": [request.to_dict() for request in self.pending_requests],
            "active_leases": [lease.to_dict() for lease in self.active_leases],
            "completed_recent": [request.to_dict() for request in self.completed_recent],
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object], config: CpuQueueConfig) -> "CpuQueueState":
        total = normalize_total_slots(data.get("total_cpu_slots") or config.total_cpu_slots)
        state = cls(
            total_cpu_slots=total,
            used_cpu_slots=int(data.get("used_cpu_slots") or 0),
            available_cpu_slots=int(data.get("available_cpu_slots") or total),
            max_running_jobs=clean_positive_int(data.get("max_running_jobs") or config.max_running_jobs),
            running_job_count=int(data.get("running_job_count") or 0),
            pending_requests=[
                CpuQueueJobRequest.from_dict(item)
                for item in data.get("pending_requests", [])  # type: ignore[arg-type]
                if isinstance(item, dict)
            ],
            active_leases=[
                CpuQueueLease.from_dict(item)
                for item in data.get("active_leases", [])  # type: ignore[arg-type]
                if isinstance(item, dict)
            ],
            completed_recent=[
                CpuQueueJobRequest.from_dict(item)
                for item in data.get("completed_recent", [])  # type: ignore[arg-type]
                if isinstance(item, dict)
            ],
            updated_at=str(data.get("updated_at") or utc_now_iso()),
        )
        return _recount(state)


def normalize_total_slots(value: object | None) -> int:
    return clean_positive_int(value or 1)


def normalize_job_slots(value: object | None, config: CpuQueueConfig) -> int:
    slots = config.default_job_cpu_slots if value is None else clean_positive_int(value)
    slots = max(config.min_job_cpu_slots, slots)
    if config.max_job_cpu_slots is not None:
        slots = min(slots, config.max_job_cpu_slots)
    if config.total_cpu_slots is not None:
        slots = min(slots, config.total_cpu_slots)
    return clean_positive_int(slots)


def can_start_job(state: CpuQueueState, request: CpuQueueJobRequest, config: CpuQueueConfig) -> bool:
    requested = normalize_job_slots(request.requested_slots, config)
    return (
        state.running_job_count < config.max_running_jobs
        and state.used_cpu_slots + requested <= state.total_cpu_slots
    )


def enqueue_request(state: CpuQueueState, request: CpuQueueJobRequest) -> CpuQueueState:
    if _find_request(state, request.job_id) is not None or _find_lease(state, request.job_id) is not None:
        return state
    return state.copy_with(pending_requests=[*state.pending_requests, request.copy_with(status="queued")])


def try_start_next_jobs(state: CpuQueueState, config: CpuQueueConfig) -> CpuQueueState:
    current = _recount(state.copy_with(max_running_jobs=config.max_running_jobs, total_cpu_slots=normalize_total_slots(config.total_cpu_slots)))
    pending = list(current.pending_requests)
    while pending:
        request = pending[0]
        if not can_start_job(current, request, config):
            break
        current = acquire_lease_for_request(current.copy_with(pending_requests=pending), request, config)
        pending = list(current.pending_requests)
    return current


def acquire_lease_for_request(
    state: CpuQueueState,
    request: CpuQueueJobRequest,
    config: CpuQueueConfig,
) -> CpuQueueState:
    slots = normalize_job_slots(request.requested_slots, config)
    now = datetime.now(timezone.utc)
    lease = CpuQueueLease.new(request=request, allocated_slots=slots, stale_lease_s=config.stale_lease_s, now=now)
    pending = [item for item in state.pending_requests if item.request_id != request.request_id]
    running_request = request.copy_with(
        allocated_slots=slots,
        status="running",
        started_at=now.isoformat(),
        reason=request.reason,
    )
    completed = [item for item in state.completed_recent if item.job_id != request.job_id]
    completed.append(running_request)
    return state.copy_with(
        pending_requests=pending,
        active_leases=[*state.active_leases, lease],
        completed_recent=_trim_recent(completed),
    )


def release_lease(state: CpuQueueState, lease_id: str) -> CpuQueueState:
    return state.copy_with(active_leases=[lease for lease in state.active_leases if lease.lease_id != lease_id])


def mark_job_finished(state: CpuQueueState, job_id: str) -> CpuQueueState:
    return _mark_done(state, job_id, "finished", "Job finished.")


def mark_job_failed(state: CpuQueueState, job_id: str) -> CpuQueueState:
    return _mark_done(state, job_id, "failed", "Job failed.")


def cancel_pending_request(state: CpuQueueState, job_id: str) -> CpuQueueState:
    now = utc_now_iso()
    cancelled: list[CpuQueueJobRequest] = []
    pending: list[CpuQueueJobRequest] = []
    for request in state.pending_requests:
        if request.job_id == job_id:
            cancelled.append(request.copy_with(status="cancelled", finished_at=now, reason="Cancelled before CPU lease."))
        else:
            pending.append(request)
    return state.copy_with(pending_requests=pending, completed_recent=_trim_recent([*state.completed_recent, *cancelled]))


def expire_stale_leases(state: CpuQueueState, now: datetime, stale_lease_s: float) -> CpuQueueState:
    keep: list[CpuQueueLease] = []
    expired_jobs: list[str] = []
    for lease in state.active_leases:
        expires_at = _parse_iso(lease.expires_at)
        acquired_at = _parse_iso(lease.acquired_at)
        stale_by_age = acquired_at is not None and (now - acquired_at).total_seconds() > float(stale_lease_s)
        if (expires_at is not None and now >= expires_at) or stale_by_age:
            expired_jobs.append(lease.job_id)
        else:
            keep.append(lease)
    current = state.copy_with(active_leases=keep)
    for job_id in expired_jobs:
        current = _mark_done(current, job_id, "failed", "CPU queue lease expired.")
    return current


def snapshot_for_admin(state: CpuQueueState, config: CpuQueueConfig | None = None) -> dict[str, object]:
    payload = state.to_dict()
    payload["config"] = (config or CpuQueueConfig.from_values(enabled=True, total_cpu_slots=state.total_cpu_slots)).to_dict()
    payload["enabled"] = bool(config.enabled) if config is not None else True
    payload["total_jobs"] = state.running_job_count + len(state.pending_requests) + len(state.completed_recent)
    payload["queued_jobs"] = len(state.pending_requests)
    payload["running_jobs"] = state.running_job_count
    return payload


def snapshot_for_user(state: CpuQueueState, user_id: str | None) -> dict[str, object]:
    user = None if user_id is None else str(user_id)
    current_user_jobs = [
        _public_request_dict(request)
        for request in [*state.pending_requests, *state.completed_recent]
        if user is None or request.user_id == user
    ]
    return {
        "enabled": True,
        "total_jobs": state.running_job_count + len(state.pending_requests),
        "queued_jobs": len(state.pending_requests),
        "running_jobs": state.running_job_count,
        "total_cpu_slots": state.total_cpu_slots,
        "used_cpu_slots": state.used_cpu_slots,
        "available_cpu_slots": state.available_cpu_slots,
        "max_running_jobs": state.max_running_jobs,
        "running_job_count": state.running_job_count,
        "pending_requests": [
            _public_request_dict(request)
            for request in state.pending_requests
            if user is None or request.user_id == user
        ],
        "active_leases": [
            _public_lease_dict(lease)
            for lease in state.active_leases
            if user is None or lease.user_id == user
        ],
        "current_user_jobs": current_user_jobs,
        "updated_at": state.updated_at,
    }


def _mark_done(state: CpuQueueState, job_id: str, status: str, reason: str) -> CpuQueueState:
    now = utc_now_iso()
    leases = [lease for lease in state.active_leases if lease.job_id != job_id]
    existing = _find_completed(state, job_id) or _find_request(state, job_id)
    if existing is None:
        existing = CpuQueueJobRequest.new(job_id=job_id, requested_slots=1, reason=reason)
    done = existing.copy_with(status=status, finished_at=now, reason=reason)
    completed = [item for item in state.completed_recent if item.job_id != job_id]
    completed.append(done)
    pending = [item for item in state.pending_requests if item.job_id != job_id]
    return state.copy_with(active_leases=leases, pending_requests=pending, completed_recent=_trim_recent(completed))


def _recount(state: CpuQueueState) -> CpuQueueState:
    used = sum(clean_positive_int(lease.allocated_slots) for lease in state.active_leases)
    total = clean_positive_int(state.total_cpu_slots)
    if used > total:
        used = total
    return replace(
        state,
        used_cpu_slots=used,
        available_cpu_slots=max(0, total - used),
        running_job_count=len(state.active_leases),
        updated_at=utc_now_iso(),
    )


def _trim_recent(items: list[CpuQueueJobRequest], limit: int = 100) -> list[CpuQueueJobRequest]:
    return items[-limit:]


def _find_request(state: CpuQueueState, job_id: str) -> CpuQueueJobRequest | None:
    return next((request for request in state.pending_requests if request.job_id == job_id), None)


def _find_completed(state: CpuQueueState, job_id: str) -> CpuQueueJobRequest | None:
    return next((request for request in reversed(state.completed_recent) if request.job_id == job_id), None)


def _find_lease(state: CpuQueueState, job_id: str) -> CpuQueueLease | None:
    return next((lease for lease in state.active_leases if lease.job_id == job_id), None)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _public_request_dict(request: CpuQueueJobRequest) -> dict[str, object]:
    return {
        "job_id": request.job_id,
        "requested_slots": request.requested_slots,
        "allocated_slots": request.allocated_slots,
        "status": request.status,
        "created_at": request.created_at,
        "started_at": request.started_at,
        "finished_at": request.finished_at,
        "reason": request.reason,
    }


def _public_lease_dict(lease: CpuQueueLease) -> dict[str, object]:
    return {
        "job_id": lease.job_id,
        "allocated_slots": lease.allocated_slots,
        "acquired_at": lease.acquired_at,
        "expires_at": lease.expires_at,
    }
