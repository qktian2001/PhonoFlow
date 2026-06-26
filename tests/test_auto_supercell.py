from pathlib import Path

from ase import Atoms
from ase.build import bulk

from phonoflow.config import WorkflowConfig
from phonoflow.defaults import infer_default_config, infer_supercell_dim, infer_supercell_info
from phonoflow.io.structure_io import write_structure
from phonoflow.workflow.pipeline import run_single_workflow


def test_auto_supercell_target_15_defaults(monkeypatch):
    monkeypatch.setattr("phonoflow.defaults.resolve_backend_name", lambda requested: "calorine")
    atoms = bulk("Si", "diamond", a=5.43)
    config = infer_default_config(
        atoms,
        Path("examples/Si.vasp"),
        Path("nep.txt"),
        WorkflowConfig(input_path=Path("examples/Si.vasp"), model_path=Path("nep.txt")),
    )
    assert config.target_supercell_length == 15.0
    assert config.supercell_dim == [4, 4, 4]
    assert config.supercell_info["n_atoms_supercell"] == 128
    assert len(config.supercell_info["supercell_lengths_resolved"]) == 3


def test_auto_supercell_user_target_override():
    atoms = bulk("Si", "diamond", a=5.43)
    dims = infer_supercell_dim(atoms, target_supercell_length=12.0)
    assert dims == [4, 4, 4]


def test_auto_supercell_vacuum_direction_kept_at_one():
    atoms = Atoms(
        "C2",
        positions=[[0.0, 0.0, 10.0], [1.42, 0.0, 10.0]],
        cell=[2.46, 2.46, 20.0],
        pbc=True,
    )
    info = infer_supercell_info(atoms, vacuum_like_directions=["c"])
    assert info["supercell_dim"][2] == 1
    assert info["auto_supercell_notes"]


def test_dry_run_records_auto_supercell_fields(tmp_path: Path):
    input_path = tmp_path / "Si.vasp"
    model_path = tmp_path / "nep.txt"
    outdir = tmp_path / "dry"
    atoms = bulk("Si", "diamond", a=5.43)
    write_structure(atoms, input_path)
    model_path.write_text("dummy potential placeholder\n", encoding="utf-8")

    result = run_single_workflow(
        WorkflowConfig(
            input_path=input_path,
            model_path=model_path,
            outdir=outdir,
            backend="dummy",
            dry_run=True,
        )
    )["report"]

    assert result["target_supercell_length"] == 15.0
    assert result["supercell_dim_resolved"] == [4, 4, 4]
    assert result["n_atoms_supercell"] == 128
    assert result["supercell_lengths_resolved"]
