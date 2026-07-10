from __future__ import annotations

from pathlib import Path

import pytest

from phonoflow_scheduler.cpu_queue_policy import (
    build_cpu_queue_config_from_admin_settings,
    build_cpu_queue_config_from_env,
    effective_job_cpu_slots,
    validate_admin_policy,
)


def test_build_cpu_queue_config_from_env_defaults_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [
        "PHONOFLOW_CPU_QUEUE_ENABLED",
        "PHONOFLOW_CPU_QUEUE_TOTAL_SLOTS",
        "PHONOFLOW_CPU_QUEUE_MAX_RUNNING_JOBS",
        "PHONOFLOW_CPU_QUEUE_JOB_SLOTS",
    ]:
        monkeypatch.delenv(name, raising=False)

    config = build_cpu_queue_config_from_env()

    assert config.enabled is False


def test_build_cpu_queue_config_from_env_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHONOFLOW_CPU_QUEUE_ENABLED", "1")
    monkeypatch.setenv("PHONOFLOW_CPU_QUEUE_TOTAL_SLOTS", "24")
    monkeypatch.setenv("PHONOFLOW_CPU_QUEUE_MAX_RUNNING_JOBS", "2")
    monkeypatch.setenv("PHONOFLOW_CPU_QUEUE_JOB_SLOTS", "12")
    monkeypatch.setenv("PHONOFLOW_CPU_QUEUE_DIR", str(tmp_path))

    config = build_cpu_queue_config_from_env()

    assert config.enabled is True
    assert config.total_cpu_slots == 24
    assert config.max_running_jobs == 2
    assert config.default_job_cpu_slots == 12
    assert config.state_dir == tmp_path


def test_admin_settings_validate_slot_budget() -> None:
    config = build_cpu_queue_config_from_admin_settings(
        enabled=True,
        total_cpu_slots=24,
        max_running_jobs=2,
        default_job_cpu_slots=12,
    )

    assert validate_admin_policy(config)["estimated_reserved_slots"] == 24


def test_admin_settings_reject_oversubscribed_policy() -> None:
    config = build_cpu_queue_config_from_admin_settings(
        enabled=True,
        total_cpu_slots=24,
        max_running_jobs=3,
        default_job_cpu_slots=12,
    )

    with pytest.raises(ValueError, match="exceeds total_cpu_slots"):
        validate_admin_policy(config)


def test_effective_job_cpu_slots_clamps_requested_slots() -> None:
    config = build_cpu_queue_config_from_admin_settings(
        enabled=True,
        total_cpu_slots=24,
        max_running_jobs=2,
        default_job_cpu_slots=12,
        max_job_cpu_slots=16,
    )

    assert effective_job_cpu_slots(config, None) == 12
    assert effective_job_cpu_slots(config, 99) == 16
