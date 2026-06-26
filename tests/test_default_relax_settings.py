from phonoflow.config import WorkflowConfig


def test_default_relax_settings():
    config = WorkflowConfig()
    assert config.relax is True
    assert config.relax_cell is True
    assert config.fmax == 1e-5
    assert config.max_steps == 2000


def test_no_relax_cell_mode_values():
    config = WorkflowConfig(relax=True, relax_cell=False)
    assert config.relax is True
    assert config.relax_cell is False


def test_no_relax_mode_values():
    config = WorkflowConfig(relax=False)
    assert config.relax is False
