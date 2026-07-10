from __future__ import annotations

from phonoflow_scheduler.resources import (
    estimate_cpu_workers,
    recommend_force_workers,
    validate_resource_budget,
)


def test_estimate_cpu_workers_uses_web_budget_formula() -> None:
    budget = estimate_cpu_workers(
        max_concurrent_jobs=2,
        force_workers=12,
        deepmd_torch_threads=1,
    )

    assert budget.estimated_cpu_workers == 24
    assert budget.max_concurrent_jobs == 2
    assert budget.force_workers == 12
    assert budget.deepmd_torch_threads == 1


def test_validate_resource_budget_reports_oversubscription() -> None:
    budget = validate_resource_budget(
        max_concurrent_jobs=2,
        batch_workers=1,
        force_workers=16,
        deepmd_torch_threads=2,
        os_cpu_count=24,
    )

    assert budget.oversubscribed is True
    assert budget.estimated_cpu_workers == 64
    assert "oversubscribe" in (budget.warning or "")


def test_recommend_force_workers_divides_available_cpu_by_concurrency() -> None:
    assert recommend_force_workers(cpu_count=24, max_concurrent_jobs=1, deepmd_torch_threads=1) == 24
    assert recommend_force_workers(cpu_count=24, max_concurrent_jobs=2, deepmd_torch_threads=1) == 12
    assert recommend_force_workers(cpu_count=24, max_concurrent_jobs=4, deepmd_torch_threads=1) == 6
