from __future__ import annotations

from phonoflow.config import WorkflowConfig
from phonoflow.thermal.fc3_finite_displacement import _resolve_phono3py_symprec


def test_phono3py_symprec_default_is_official_api_default_not_global_symprec() -> None:
    config = WorkflowConfig(symprec=1e-3)
    assert config.phono3py_symprec == 1e-5
    assert _resolve_phono3py_symprec(config) == 1e-5


def test_phono3py_cutoff_frequency_default_is_recorded_official_default() -> None:
    config = WorkflowConfig()
    assert config.phono3py_cutoff_frequency == 1e-4


def test_traditional_finite_displacement_symmetrizes_fc2_and_fc3_by_default() -> None:
    config = WorkflowConfig(compute_kappa=True, fc3_method="finite-displacement")
    assert config.phono3py_symmetrize_fc2 is True
    assert config.phono3py_symmetrize_fc3 is True


def test_legacy_fc2_asr_alias_does_not_override_explicit_new_field() -> None:
    config = WorkflowConfig(phono3py_fc2_asr=True, phono3py_symmetrize_fc2=False)
    assert config.phono3py_symmetrize_fc2 is False
