from ase import Atoms
from ase.build import bulk

from phonoflow.analysis.structure_type import classify_structure_type


def test_structure_type_bulk_has_no_vacuum_direction():
    atoms = bulk("Si", "diamond", a=5.43)
    result = classify_structure_type(atoms)
    assert result["structure_type"] == "bulk"
    assert result["vacuum_like_directions"] == []


def test_structure_type_2d_detects_vacuum_direction():
    atoms = Atoms(
        "C2",
        positions=[[0.0, 0.0, 10.0], [1.42, 0.0, 10.0]],
        cell=[2.46, 2.46, 20.0],
        pbc=True,
    )
    result = classify_structure_type(atoms)
    assert "c" in result["vacuum_like_directions"]
    assert result["structure_type"] in {"2d", "interface_or_slab"}
    assert result["warnings"]
