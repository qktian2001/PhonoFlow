from pathlib import Path

import pytest
from ase.build import bulk

from phonoflow.config import WorkflowConfig
from phonoflow.defaults import infer_default_config, infer_supercell_dim, resolve_backend_name
from phonoflow.exceptions import BackendUnavailableError, ConfigError


def test_infer_supercell_dim_for_si_bulk():
    atoms = bulk("Si", "diamond", a=5.43)
    dims = infer_supercell_dim(atoms, target_supercell_length=20.0)
    assert len(dims) == 3
    assert all(dim >= 1 for dim in dims)
    assert all(dim <= 6 for dim in dims)


def test_infer_default_config_outdir_and_defaults(monkeypatch):
    monkeypatch.setattr("phonoflow.defaults.resolve_backend_name", lambda requested: "calorine")
    atoms = bulk("Si", "diamond", a=5.43)
    config = infer_default_config(
        atoms=atoms,
        input_path=Path("examples/Si.vasp"),
        model_path=Path("nep.txt"),
        user_config=WorkflowConfig(input_path=Path("examples/Si.vasp"), model_path=Path("nep.txt")),
    )
    assert config.backend == "calorine"
    assert config.outdir == Path("results") / "Si_calorine"
    assert isinstance(config.supercell_dim, list)
    assert config.target_supercell_length == 15.0
    assert config.max_supercell_atoms == 1000
    assert config.min_supercell_dim == 1
    assert config.max_supercell_dim == 6
    assert config.supercell_info["n_atoms_supercell"] == 128
    assert config.supercell_info["supercell_lengths_resolved"]
    assert config.displacement == 0.01
    assert config.relax is True
    assert config.relax_cell is True
    assert config.fmax == 1e-5
    assert config.max_steps == 2000


def test_auto_backend_infers_registered_dpa_from_model_path():
    atoms = bulk("Si", "diamond", a=5.43)
    model = Path("models/DPA-3.2-5M.pt")

    config = infer_default_config(
        atoms=atoms,
        input_path=Path("examples/Si.vasp"),
        model_path=model,
        user_config=WorkflowConfig(input_path=Path("examples/Si.vasp"), model_path=model),
    )

    assert config.backend == "deepmd"
    assert config.backend_alias == "dpa32"
    assert config.dpa_model_name == "DPA-3.2-5M.pt"
    assert config.deepmd_model_head == "OMat24"
    assert config.model_path == model
    assert config.relax is False
    assert config.outdir == Path("results") / "Si_dpa32"


def test_auto_backend_infers_generic_deepmd_from_model_path():
    atoms = bulk("Si", "diamond", a=5.43)
    model = Path("models/custom-dpa-ft.pth")

    config = infer_default_config(
        atoms=atoms,
        input_path=Path("examples/Si.vasp"),
        model_path=model,
        user_config=WorkflowConfig(input_path=Path("examples/Si.vasp"), model_path=model),
    )

    assert config.backend == "deepmd"
    assert config.backend_alias == "dpa"
    assert config.dpa_model_name == "custom-dpa-ft.pth"
    assert config.deepmd_model_head is None
    assert config.model_path == model
    assert config.relax is False
    assert config.outdir == Path("results") / "Si_dpa"


def test_backend_auto_selection_prefers_calorine(monkeypatch):
    class AvailableCalorine:
        def check_available(self):
            return True

    monkeypatch.setattr("phonoflow.defaults.CalorineBackend", AvailableCalorine)
    assert resolve_backend_name("auto") == "calorine"


def test_backend_auto_selection_requires_calorine(monkeypatch):
    class MissingCalorine:
        def check_available(self):
            return False

    monkeypatch.setattr("phonoflow.defaults.CalorineBackend", MissingCalorine)
    with pytest.raises(BackendUnavailableError, match="Calorine is required"):
        resolve_backend_name("auto")


def test_backend_pynep_is_removed():
    with pytest.raises(ConfigError, match="PyNEP backend has been removed"):
        resolve_backend_name("pynep")
