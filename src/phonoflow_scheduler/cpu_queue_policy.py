"""Policy helpers for admin-managed CPU queue settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from phonoflow_scheduler.cpu_queue_config import CpuQueueConfig
from phonoflow_scheduler.cpu_queue_state import normalize_job_slots


CPU_QUEUE_ENABLED_ENV = "PHONOFLOW_CPU_QUEUE_ENABLED"
CPU_QUEUE_TOTAL_SLOTS_ENV = "PHONOFLOW_CPU_QUEUE_TOTAL_SLOTS"
CPU_QUEUE_MAX_RUNNING_JOBS_ENV = "PHONOFLOW_CPU_QUEUE_MAX_RUNNING_JOBS"
CPU_QUEUE_JOB_SLOTS_ENV = "PHONOFLOW_CPU_QUEUE_JOB_SLOTS"
CPU_QUEUE_DIR_ENV = "PHONOFLOW_CPU_QUEUE_DIR"
CPU_QUEUE_TIMEOUT_ENV = "PHONOFLOW_CPU_QUEUE_TIMEOUT"


def build_cpu_queue_config_from_env() -> CpuQueueConfig:
    enabled = os.environ.get(CPU_QUEUE_ENABLED_ENV, "0")
    total = os.environ.get(CPU_QUEUE_TOTAL_SLOTS_ENV)
    return CpuQueueConfig.from_values(
        enabled=enabled,
        total_cpu_slots=total,
        max_running_jobs=os.environ.get(CPU_QUEUE_MAX_RUNNING_JOBS_ENV, 1),
        default_job_cpu_slots=os.environ.get(CPU_QUEUE_JOB_SLOTS_ENV, 1),
        max_job_cpu_slots=os.environ.get(CPU_QUEUE_JOB_SLOTS_ENV),
        acquire_timeout_s=os.environ.get(CPU_QUEUE_TIMEOUT_ENV),
        state_dir=os.environ.get(CPU_QUEUE_DIR_ENV),
    )


def build_cpu_queue_config_from_admin_settings(
    *,
    enabled: object = False,
    total_cpu_slots: object | None = None,
    max_running_jobs: object = 1,
    default_job_cpu_slots: object = 1,
    min_job_cpu_slots: object = 1,
    max_job_cpu_slots: object | None = None,
    state_dir: str | Path | None = None,
    acquire_timeout_s: object | None = None,
) -> CpuQueueConfig:
    return CpuQueueConfig.from_values(
        enabled=enabled,
        total_cpu_slots=total_cpu_slots,
        max_running_jobs=max_running_jobs,
        default_job_cpu_slots=default_job_cpu_slots,
        min_job_cpu_slots=min_job_cpu_slots,
        max_job_cpu_slots=max_job_cpu_slots,
        acquire_timeout_s=acquire_timeout_s,
        state_dir=state_dir,
        admin_managed=True,
    )


def effective_job_cpu_slots(policy: CpuQueueConfig, requested_slots: int | None = None) -> int:
    return normalize_job_slots(requested_slots, policy)


def validate_admin_policy(policy: CpuQueueConfig) -> dict[str, Any]:
    if not policy.enabled:
        return {"enabled": False, "estimated_reserved_slots": 0, "warning": None}
    if policy.total_cpu_slots is None:
        raise ValueError("total_cpu_slots is required when CPU queue is enabled.")
    estimated = int(policy.max_running_jobs) * int(policy.default_job_cpu_slots)
    if estimated > int(policy.total_cpu_slots):
        raise ValueError(
            "CPU queue policy exceeds total_cpu_slots: "
            f"max_running_jobs={policy.max_running_jobs} x default_job_cpu_slots={policy.default_job_cpu_slots} "
            f"> total_cpu_slots={policy.total_cpu_slots}."
        )
    return {
        "enabled": True,
        "estimated_reserved_slots": estimated,
        "total_cpu_slots": policy.total_cpu_slots,
        "max_running_jobs": policy.max_running_jobs,
        "default_job_cpu_slots": policy.default_job_cpu_slots,
        "warning": None,
    }
