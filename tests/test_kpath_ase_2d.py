from __future__ import annotations

from ase import Atoms

from phonoflow.analysis.bandpath import generate_kpath, get_band_path
from phonoflow.kpath.dimensionality import infer_dimensionality_by_vacuum, standardize_2d_for_ase_bandpath
from phonoflow.kpath.kpath_ase_2d import generate_ase_2d_kpath


def _hex2d_atoms(vacuum_axis: int = 2) -> Atoms:
    base_cell = [
        [2.46, 0.0, 0.0],
        [-1.23, 2.130422493, 0.0],
        [0.0, 0.0, 20.0],
    ]
    base_scaled = [
        [0.0, 0.0, 0.5],
        [1.0 / 3.0, 2.0 / 3.0, 0.5],
    ]
    periodic_axes = [axis for axis in range(3) if axis != 2]
    permutation = periodic_axes.copy()
    permutation.insert(vacuum_axis, 2)
    remapped_cell = [base_cell[index] for index in permutation]
    remapped_scaled = [[coords[index] for index in permutation] for coords in base_scaled]
    return Atoms("C2", cell=remapped_cell, scaled_positions=remapped_scaled, pbc=[True, True, True])


def _square2d_atoms() -> Atoms:
    return Atoms("Si", cell=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 18.0]], positions=[[0.0, 0.0, 9.0]], pbc=[True, True, True])


def _rect2d_atoms() -> Atoms:
    return Atoms("Si", cell=[[3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 18.0]], positions=[[0.0, 0.0, 9.0]], pbc=[True, True, True])


def _bulk_si_atoms() -> Atoms:
    return Atoms(
        "Si2",
        cell=[[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        pbc=[True, True, True],
    )


def test_hex2d_auto_kpath_uses_ase_hex2d_and_zero_z_coordinates() -> None:
    result = generate_kpath(_hex2d_atoms(), kpath_mode="auto", npoints=9)

    assert result.mode == "2d_ase"
    assert result.dimensionality == "2D"
    assert result.bravais == "HEX2D"
    assert result.display_path == "Γ-M-K-Γ"
    assert all(abs(coords[2]) < 1e-12 for coords in result.special_points.values())

    band_path = get_band_path(_hex2d_atoms(), mode="auto", npoints=9)
    assert band_path.source == "ase-2d"
    assert all(abs(float(point[2])) < 1e-12 for segment in band_path.qpoints for point in segment)


def test_square_and_rectangular_2d_lattices_use_ase_bravais_detection() -> None:
    square = generate_ase_2d_kpath(_square2d_atoms(), npoints=8)
    rect = generate_ase_2d_kpath(_rect2d_atoms(), npoints=8)

    assert square.bravais == "SQR"
    assert rect.bravais == "RECT"
    assert all(abs(coords[2]) < 1e-12 for coords in square.special_points.values())
    assert all(abs(coords[2]) < 1e-12 for coords in rect.special_points.values())


def test_non_c_vacuum_axis_is_detected_and_standardized_for_ase() -> None:
    atoms = _hex2d_atoms(vacuum_axis=0)

    dimensionality = infer_dimensionality_by_vacuum(atoms)
    standardized, permutation = standardize_2d_for_ase_bandpath(atoms, vacuum_axis=0)
    result = generate_kpath(atoms, kpath_mode="auto", npoints=9)

    assert dimensionality["dimension"] == "2D"
    assert dimensionality["vacuum_axis"] == 0
    assert dimensionality["vacuum_axis_name"] == "a"
    assert permutation == [1, 2, 0]
    assert list(standardized.pbc) == [True, True, False]
    assert result.vacuum_axis == 0
    assert result.vacuum_axis_name == "a"
    assert all(abs(coords[2]) < 1e-12 for coords in result.special_points.values())


def test_bulk_si_auto_mode_keeps_seekpath() -> None:
    result = generate_kpath(_bulk_si_atoms(), kpath_mode="auto", npoints=7)

    assert result.mode == "3d_seekpath"
    assert result.dimensionality == "3D"
    assert result.source == "seekpath"


def test_forced_2d_ase_mode_works_without_auto_resolution() -> None:
    result = generate_kpath(_hex2d_atoms(), kpath_mode="2d_ase", npoints=9)

    assert result.mode == "2d_ase"
    assert result.source == "ase_cell_bandpath"
    assert result.path_labels == ["G", "M", "K", "G"]
