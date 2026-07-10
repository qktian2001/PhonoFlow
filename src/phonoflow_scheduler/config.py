"""Scheduler configuration data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ForceParallelBackend = Literal["origin", "direct", "serial", "process", "worker_queue"]


def clean_positive_int(value: object, default: int = 1) -> int:
    """Return a positive integer, falling back to default for invalid values."""

    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        parsed = default
    return max(1, parsed)


@dataclass(frozen=True)
class ResourceBudget:
    """Estimated CPU scheduling budget."""

    max_concurrent_jobs: int = 1
    batch_workers: int = 1
    force_workers: int = 1
    deepmd_torch_threads: int = 1
    web_requested_cpu_parallelism: int = 1
    batch_requested_cpu_parallelism: int = 1
    estimated_cpu_workers: int = 1
    os_cpu_count: int | None = None
    oversubscribed: bool = False
    warning: str | None = None


@dataclass(frozen=True)
class ForceScheduleConfig:
    """Process/serial force-loop scheduling parameters."""

    force_workers: int = 1
    force_parallel_backend: ForceParallelBackend = "serial"
    deepmd_torch_threads: int | None = 1
    max_concurrent_jobs: int = 1
    batch_workers: int = 1
    chunk_size: int | None = None
    max_pending_tasks: int | None = None
    calculator_initializer: str = "worker"
    backend_name: str | None = None
    base_payload: Any | None = None

    @classmethod
    def from_values(
        cls,
        *,
        force_workers: object = 1,
        force_parallel_backend: object = "serial",
        deepmd_torch_threads: object | None = 1,
        max_concurrent_jobs: object = 1,
        batch_workers: object = 1,
        chunk_size: object | None = None,
        max_pending_tasks: object | None = None,
        calculator_initializer: str = "worker",
        backend_name: object | None = None,
        base_payload: Any | None = None,
    ) -> "ForceScheduleConfig":
        """Build a cleaned config from user-facing values."""

        backend = str(force_parallel_backend or "serial").lower()
        if backend not in {"origin", "direct", "serial", "process", "worker_queue"}:
            backend = "serial"
        torch_threads = None if deepmd_torch_threads is None else clean_positive_int(deepmd_torch_threads)
        return cls(
            force_workers=clean_positive_int(force_workers),
            force_parallel_backend=backend,  # type: ignore[arg-type]
            deepmd_torch_threads=torch_threads,
            max_concurrent_jobs=clean_positive_int(max_concurrent_jobs),
            batch_workers=clean_positive_int(batch_workers),
            chunk_size=None if chunk_size is None else clean_positive_int(chunk_size),
            max_pending_tasks=None if max_pending_tasks is None else clean_positive_int(max_pending_tasks),
            calculator_initializer=calculator_initializer,
            backend_name=None if backend_name is None else str(backend_name).lower(),
            base_payload=base_payload,
        )
