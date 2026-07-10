"""Shared CPU resource scheduling helpers.

This module keeps the historical PhonoFlow API shape while delegating the
actual CPU-budget calculation to :mod:`phonoflow_scheduler`.
"""

from __future__ import annotations

from typing import Any

from phonoflow_scheduler.config import clean_positive_int
from phonoflow_scheduler.resources import validate_resource_budget


def estimate_cpu_budget(
    *,
    max_concurrent_jobs: int = 1,
    batch_workers: int = 1,
    force_workers: int = 1,
    deepmd_torch_threads: int | None = 1,
    os_cpu_count: int | None = None,
) -> dict[str, Any]:
    """Estimate requested CPU parallelism and return a non-fatal warning if oversubscribed."""

    budget = validate_resource_budget(
        max_concurrent_jobs=max_concurrent_jobs,
        batch_workers=batch_workers,
        force_workers=force_workers,
        deepmd_torch_threads=deepmd_torch_threads,
        os_cpu_count=os_cpu_count,
    )
    return {
        "os_cpu_count": budget.os_cpu_count,
        "max_concurrent_jobs": budget.max_concurrent_jobs,
        "batch_workers": budget.batch_workers,
        "force_workers": budget.force_workers,
        "deepmd_torch_threads": budget.deepmd_torch_threads,
        "web_requested_cpu_parallelism": budget.web_requested_cpu_parallelism,
        "batch_requested_cpu_parallelism": budget.batch_requested_cpu_parallelism,
        "estimated_cpu_parallelism": budget.estimated_cpu_workers,
        "estimated_cpu_workers": budget.estimated_cpu_workers,
        "oversubscribed": budget.oversubscribed,
        "warning": budget.warning,
    }
