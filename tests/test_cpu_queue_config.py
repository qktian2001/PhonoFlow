from __future__ import annotations

from pathlib import Path

import pytest

from phonoflow_scheduler.cpu_queue_config import CpuQueueConfig


def test_cpu_queue_config_defaults_disabled() -> None:
    config = CpuQueueConfig.from_values()

    assert config.enabled is False
    assert config.total_cpu_slots is None
    assert config.max_running_jobs == 1
    assert config.default_job_cpu_slots == 1
    assert config.scheduling_policy == "fifo"
    assert config.allow_backfill is False


def test_cpu_queue_config_normalizes_slots_and_state_dir(tmp_path: Path) -> None:
    config = CpuQueueConfig.from_values(
        enabled=True,
        total_cpu_slots="24",
        max_running_jobs="2",
        default_job_cpu_slots="12",
        max_job_cpu_slots="16",
        state_dir=str(tmp_path),
    )

    assert config.enabled is True
    assert config.total_cpu_slots == 24
    assert config.max_running_jobs == 2
    assert config.default_job_cpu_slots == 12
    assert config.max_job_cpu_slots == 16
    assert config.state_dir == tmp_path


def test_cpu_queue_config_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="scheduling_policy"):
        CpuQueueConfig.from_values(enabled=True, total_cpu_slots=24, scheduling_policy="priority")


def test_cpu_queue_config_requires_total_slots_when_enabled() -> None:
    with pytest.raises(ValueError, match="total_cpu_slots"):
        CpuQueueConfig.from_values(enabled=True)
