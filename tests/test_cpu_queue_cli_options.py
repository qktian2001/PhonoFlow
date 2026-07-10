from __future__ import annotations

from phonoflow.config import WorkflowConfig


def test_workflow_config_accepts_cpu_queue_fields(tmp_path):
    config = WorkflowConfig(
        cpu_queue_enabled=True,
        cpu_queue_total_slots=24,
        cpu_queue_max_running_jobs=2,
        cpu_queue_job_slots=12,
        cpu_queue_state_dir=tmp_path,
        cpu_queue_timeout=30.0,
    )

    assert config.cpu_queue_enabled is True
    assert config.cpu_queue_total_slots == 24
    assert config.cpu_queue_max_running_jobs == 2
    assert config.cpu_queue_job_slots == 12
    assert config.cpu_queue_state_dir == tmp_path
    assert config.cpu_queue_timeout == 30.0


def test_cpu_queue_positive_validation():
    try:
        WorkflowConfig(cpu_queue_enabled=True, cpu_queue_total_slots=0)
    except Exception as exc:
        assert "must be at least 1" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("Expected invalid CPU queue total slots to fail validation.")
