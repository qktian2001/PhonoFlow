"""Pure CPU resource-budget helpers."""

from __future__ import annotations

import os

from phonoflow_scheduler.config import ResourceBudget, clean_positive_int


def estimate_cpu_workers(
    *,
    max_concurrent_jobs: int = 1,
    force_workers: int = 1,
    deepmd_torch_threads: int | None = 1,
) -> ResourceBudget:
    """Estimate Web-level requested CPU workers."""

    return validate_resource_budget(
        max_concurrent_jobs=max_concurrent_jobs,
        batch_workers=1,
        force_workers=force_workers,
        deepmd_torch_threads=deepmd_torch_threads,
        os_cpu_count=None,
    )


def recommend_force_workers(
    *,
    cpu_count: int | None = None,
    max_concurrent_jobs: int = 1,
    deepmd_torch_threads: int | None = 1,
) -> int:
    """Return a conservative per-job worker recommendation."""

    cpus = clean_positive_int(cpu_count if cpu_count is not None else os.cpu_count() or 1)
    jobs = clean_positive_int(max_concurrent_jobs)
    torch_threads = clean_positive_int(deepmd_torch_threads or 1)
    return max(1, cpus // max(1, jobs * torch_threads))


def validate_resource_budget(
    *,
    max_concurrent_jobs: int = 1,
    batch_workers: int = 1,
    force_workers: int = 1,
    deepmd_torch_threads: int | None = 1,
    os_cpu_count: int | None = None,
) -> ResourceBudget:
    """Estimate requested CPU parallelism and mark oversubscription."""

    max_jobs = clean_positive_int(max_concurrent_jobs)
    batch = clean_positive_int(batch_workers)
    force = clean_positive_int(force_workers)
    torch_threads = clean_positive_int(deepmd_torch_threads or 1)
    web_requested = max_jobs * force * torch_threads
    batch_requested = batch * force * torch_threads
    estimated = max(web_requested, batch_requested)
    cpu_count = os_cpu_count if os_cpu_count is not None else os.cpu_count()
    warning = None
    if cpu_count is not None and estimated > int(cpu_count):
        warning = (
            "Requested CPU parallelism may oversubscribe the machine: "
            f"max_concurrent_jobs={max_jobs} x force_workers={force} x "
            f"deepmd_torch_threads={torch_threads} = {web_requested}; "
            f"batch_workers={batch} x force_workers={force} x "
            f"deepmd_torch_threads={torch_threads} = {batch_requested}; "
            f"os.cpu_count()={cpu_count}."
        )
    return ResourceBudget(
        max_concurrent_jobs=max_jobs,
        batch_workers=batch,
        force_workers=force,
        deepmd_torch_threads=torch_threads,
        web_requested_cpu_parallelism=web_requested,
        batch_requested_cpu_parallelism=batch_requested,
        estimated_cpu_workers=estimated,
        os_cpu_count=cpu_count,
        oversubscribed=warning is not None,
        warning=warning,
    )
