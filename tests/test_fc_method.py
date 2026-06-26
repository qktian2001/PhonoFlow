from pathlib import Path

import pytest

from phonoflow.config import WorkflowConfig
from phonoflow.exceptions import ConfigError
from phonoflow.workflow.pipeline import run_single_workflow


def test_fc_method_finite_displacement_is_default_and_valid():
    config = WorkflowConfig()
    assert config.fc_method == "finite-displacement"


def test_fc_method_hiphive_is_reserved_for_future_release(tmp_path):
    config = WorkflowConfig(
        input_path=Path("examples/Si.vasp"),
        outdir=tmp_path,
        backend="dummy",
        fc_method="hiphive",
    )

    with pytest.raises(ConfigError, match="planned for a future release"):
        run_single_workflow(config)
