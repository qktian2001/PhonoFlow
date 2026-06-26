from pathlib import Path

import pytest
from ase import Atoms

from phonoflow.calculators.calorine_backend import CalorineBackend
from phonoflow.config import WorkflowConfig
from phonoflow.io.structure_io import write_structure
from phonoflow.workflow.pipeline import run_single_workflow


def test_calorine_backend_import_guard_returns_bool():
    backend = CalorineBackend()
    assert isinstance(backend.check_available(), bool)


def test_calorine_missing_error_is_clear(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(CalorineBackend, "check_available", lambda self: False)
    input_path = tmp_path / "Si.vasp"
    atoms = Atoms(
        "Si2",
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=[5.43, 5.43, 5.43],
        pbc=True,
    )
    write_structure(atoms, input_path)

    config = WorkflowConfig(
        input_path=input_path,
        model_path=tmp_path / "nep.txt",
        outdir=tmp_path / "out",
        backend="calorine",
    )
    with pytest.raises(Exception, match="Calorine is required for real NEP/NEP89 calculations"):
        run_single_workflow(config)


def test_calorine_cpunep_import_if_available():
    backend = CalorineBackend()
    if not backend.check_available():
        pytest.skip("Calorine CPUNEP is not installed in this environment.")

    from calorine.calculators import CPUNEP

    assert CPUNEP is not None
