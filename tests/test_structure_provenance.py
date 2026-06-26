from __future__ import annotations

from pathlib import Path

from ase import Atoms

from phonoflow.workflow.structure_provenance import build_structure_provenance, structure_hash


def test_structure_hash_is_stable_for_copied_atoms() -> None:
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[3, 3, 3], pbc=True)

    assert structure_hash(atoms) == structure_hash(atoms.copy())


def test_structure_provenance_records_same_fc2_fc3_structure(tmp_path: Path) -> None:
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[3, 3, 3], pbc=True)

    data = build_structure_provenance(
        input_atoms=atoms,
        relaxed_atoms=atoms.copy(),
        fc2_atoms=atoms.copy(),
        fc3_atoms=atoms.copy(),
        input_structure_path=Path("examples/Si.vasp"),
        relaxed_structure_path=tmp_path / "relaxed.vasp",
        fc2_source_structure_path=tmp_path / "fc_source_structure.vasp",
        fc3_source_structure_path=tmp_path / "fc_source_structure.vasp",
        relax_backend="deepmd",
        force_constants_backend="deepmd",
        structure_stage_mode="single_stage",
    )

    assert data["same_structure_for_fc2_fc3"] is True
    assert data["input_structure_hash"] == data["relaxed_structure_hash"]
    assert data["fc2_source_structure_hash"] == data["fc3_source_structure_hash"]
    assert data["relax_backend"] == "deepmd"
    assert data["force_constants_backend"] == "deepmd"
