"""Configuration for the optional PBS-style CPU resource queue."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from phonoflow_scheduler.config import clean_positive_int


CpuQueueSchedulingPolicy = Literal["fifo"]


@dataclass(frozen=True)
class CpuQueueConfig:
    """Admin-managed CPU slot policy for Web/CLI job admission.

    ``total_cpu_slots`` is the total CPU resource pool PhonoFlow may use. It is
    not the same concept as ``force_workers``; an allocated job slot count may be
    mapped to per-job workers by the caller as a resource policy choice.
    """

    enabled: bool = False
    total_cpu_slots: int | None = None
    max_running_jobs: int = 1
    default_job_cpu_slots: int = 1
    min_job_cpu_slots: int = 1
    max_job_cpu_slots: int | None = None
    scheduling_policy: CpuQueueSchedulingPolicy = "fifo"
    allow_backfill: bool = False
    acquire_timeout_s: float | None = None
    poll_interval_s: float = 0.5
    state_dir: Path | None = None
    stale_lease_s: float = 86400.0
    admin_managed: bool = True

    @classmethod
    def from_values(
        cls,
        *,
        enabled: object | None = False,
        total_cpu_slots: object | None = None,
        max_running_jobs: object = 1,
        default_job_cpu_slots: object = 1,
        min_job_cpu_slots: object = 1,
        max_job_cpu_slots: object | None = None,
        scheduling_policy: object = "fifo",
        allow_backfill: object = False,
        acquire_timeout_s: object | None = None,
        poll_interval_s: object = 0.5,
        state_dir: str | Path | None = None,
        stale_lease_s: object = 86400.0,
        admin_managed: object = True,
    ) -> "CpuQueueConfig":
        policy = str(scheduling_policy or "fifo").strip().lower()
        if policy != "fifo":
            raise ValueError("scheduling_policy must be 'fifo'.")
        is_enabled = _as_bool(enabled)
        total = None if total_cpu_slots is None or str(total_cpu_slots).strip() == "" else clean_positive_int(total_cpu_slots)
        if is_enabled and total is None:
            raise ValueError("total_cpu_slots is required when CPU queue is enabled.")
        min_slots = clean_positive_int(min_job_cpu_slots)
        default_slots = max(min_slots, clean_positive_int(default_job_cpu_slots))
        max_slots = None if max_job_cpu_slots is None or str(max_job_cpu_slots).strip() == "" else clean_positive_int(max_job_cpu_slots)
        if max_slots is not None:
            max_slots = max(min_slots, max_slots)
            default_slots = min(default_slots, max_slots)
        if total is not None:
            default_slots = min(default_slots, total)
            if max_slots is not None:
                max_slots = min(max_slots, total)
        poll = float(poll_interval_s)
        if poll <= 0:
            raise ValueError("poll_interval_s must be positive.")
        stale = float(stale_lease_s)
        if stale <= 0:
            raise ValueError("stale_lease_s must be positive.")
        timeout = None if acquire_timeout_s is None or str(acquire_timeout_s).strip() == "" else float(acquire_timeout_s)
        if timeout is not None and timeout < 0:
            raise ValueError("acquire_timeout_s must be non-negative.")
        return cls(
            enabled=is_enabled,
            total_cpu_slots=total,
            max_running_jobs=clean_positive_int(max_running_jobs),
            default_job_cpu_slots=default_slots,
            min_job_cpu_slots=min_slots,
            max_job_cpu_slots=max_slots,
            scheduling_policy="fifo",
            allow_backfill=_as_bool(allow_backfill),
            acquire_timeout_s=timeout,
            poll_interval_s=poll,
            state_dir=Path(state_dir) if state_dir is not None and str(state_dir).strip() else None,
            stale_lease_s=stale,
            admin_managed=_as_bool(admin_managed),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "total_cpu_slots": self.total_cpu_slots,
            "max_running_jobs": self.max_running_jobs,
            "default_job_cpu_slots": self.default_job_cpu_slots,
            "min_job_cpu_slots": self.min_job_cpu_slots,
            "max_job_cpu_slots": self.max_job_cpu_slots,
            "scheduling_policy": self.scheduling_policy,
            "allow_backfill": self.allow_backfill,
            "acquire_timeout_s": self.acquire_timeout_s,
            "poll_interval_s": self.poll_interval_s,
            "state_dir": str(self.state_dir) if self.state_dir is not None else None,
            "stale_lease_s": self.stale_lease_s,
            "admin_managed": self.admin_managed,
        }


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on", "enabled"}
