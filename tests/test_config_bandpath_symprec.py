from __future__ import annotations

from phonoflow.analysis.bandpath import DEFAULT_SEEKPATH_SYMPREC, DEFAULT_SEEKPATH_WITH_TIME_REVERSAL
from phonoflow.config import WorkflowConfig


def test_workflow_config_has_independent_bandpath_seekpath_settings() -> None:
    config = WorkflowConfig()

    assert config.phonopy_symprec == 1e-5
    assert config.symprec == 1e-5
    assert config.bandpath_symprec == DEFAULT_SEEKPATH_SYMPREC
    assert config.bandpath_with_time_reversal is DEFAULT_SEEKPATH_WITH_TIME_REVERSAL


def test_workflow_config_bandpath_settings_are_overridable() -> None:
    config = WorkflowConfig(bandpath_symprec=1e-6, bandpath_with_time_reversal=True)

    assert config.bandpath_symprec == 1e-6
    assert config.bandpath_with_time_reversal is True


def test_workflow_config_kpath_mode_defaults_to_auto() -> None:
    config = WorkflowConfig()

    assert config.kpath_mode == "auto"


def test_workflow_config_kpath_mode_is_overridable() -> None:
    config = WorkflowConfig(kpath_mode="2d_ase")

    assert config.kpath_mode == "2d_ase"


def test_legacy_symprec_alias_maps_to_phonopy_symprec() -> None:
    config = WorkflowConfig(symprec=1e-3)

    assert config.phonopy_symprec == 1e-3
    assert "symprec" not in config.model_dump()
