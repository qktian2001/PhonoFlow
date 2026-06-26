from __future__ import annotations

import re
from typing import Any

import numpy as np

from phonoflow.kpath.dimensionality import infer_dimensionality_by_vacuum, standardize_2d_for_ase_bandpath
from phonoflow.kpath.schema import KPathResult

ALLOWED_2D_BRAVAIS = {"OBL", "RECT", "CRECT", "HEX2D", "SQR"}


def identify_ase_2d_bravais(atoms_2d: Any, eps: float = 2e-4) -> dict[str, Any]:
    from ase.lattice import identify_lattice

    atoms_tmp = atoms_2d.copy()
    atoms_tmp.pbc = [True, True, False]
    lattice, operation = identify_lattice(atoms_tmp.cell, eps=float(eps), pbc=[True, True, False])
    if lattice.name not in ALLOWED_2D_BRAVAIS:
        raise RuntimeError(f"ASE did not identify a supported 2D Bravais lattice: {lattice.name}")
    try:
        operation_payload = np.asarray(operation, dtype=float).tolist()
    except Exception:
        operation_payload = str(operation)
    return {
        "bravais": lattice.name,
        "bravais_longname": getattr(lattice, "longname", lattice.name),
        "operation": operation_payload,
    }


def parse_ase_path_string(path: str) -> list[str]:
    labels: list[str] = []
    for chunk in str(path).split(","):
        labels.extend(re.findall(r"[A-Z][0-9']*", chunk))
    return labels


def to_phonopy_band(path_labels: list[str], special_points: dict[str, list[float]]) -> list[list[list[float]]]:
    bands: list[list[list[float]]] = []
    for start, end in zip(path_labels[:-1], path_labels[1:], strict=False):
        p1 = list(map(float, special_points[start]))
        p2 = list(map(float, special_points[end]))
        if len(p1) != 3 or len(p2) != 3:
            raise ValueError("Phonopy band points must be 3D coordinates.")
        p1[2] = 0.0
        p2[2] = 0.0
        bands.append([p1, p2])
    return bands


def display_label(label: str) -> str:
    return {"G": "Γ", "GAMMA": "Γ"}.get(str(label), str(label))


def display_path_string(labels: list[str]) -> str:
    return "-".join(display_label(label) for label in labels)


def generate_ase_2d_kpath(atoms: Any, npoints: int = 101, eps: float = 2e-4) -> KPathResult:
    dim_info = infer_dimensionality_by_vacuum(atoms)
    if dim_info["dimension"] != "2D":
        raise RuntimeError("generate_ase_2d_kpath was called for a non-2D structure.")

    atoms_2d, permutation = standardize_2d_for_ase_bandpath(atoms, vacuum_axis=int(dim_info["vacuum_axis"]))
    bravais_info = identify_ase_2d_bravais(atoms_2d, eps=eps)
    bandpath = atoms_2d.cell.bandpath(npoints=int(npoints), eps=float(eps), pbc=[True, True, False])
    path_labels = parse_ase_path_string(bandpath.path)

    special_points: dict[str, list[float]] = {}
    for label, coord in bandpath.special_points.items():
        values = list(map(float, coord))
        if len(values) == 2:
            values = [values[0], values[1], 0.0]
        elif len(values) == 3:
            values = [values[0], values[1], 0.0]
        else:
            raise ValueError(f"Invalid special point coordinate: {label} = {coord}")
        values[2] = 0.0
        special_points[str(label)] = values

    missing = [label for label in path_labels if label not in special_points]
    if missing:
        raise RuntimeError(
            f"Missing special point coordinates for labels: {missing}. "
            f"path={bandpath.path}, available={list(special_points)}"
        )

    used_points: dict[str, list[float]] = {}
    for label in path_labels:
        if label in special_points and label not in used_points:
            used_points[label] = list(special_points[label])

    path_segments = list(zip(path_labels[:-1], path_labels[1:], strict=False))
    phonopy_band = to_phonopy_band(path_labels, special_points)

    return KPathResult(
        mode="2d_ase",
        dimensionality="2D",
        source="ase_cell_bandpath",
        bravais=bravais_info["bravais"],
        path_string="-".join(path_labels),
        display_path=display_path_string(path_labels),
        path_labels=path_labels,
        special_points=used_points,
        path_segments=path_segments,
        phonopy_band=phonopy_band,
        vacuum_axis=int(dim_info["vacuum_axis"]),
        vacuum_axis_name=str(dim_info["vacuum_axis_name"]),
        warning=None,
        metadata={
            "dimensionality": dim_info,
            "ase_bravais": bravais_info,
            "axis_permutation": permutation,
            "ase_raw_path": str(bandpath.path),
            "primitive_lattice": np.asarray(atoms_2d.cell.array, dtype=float).tolist(),
            "reciprocal_lattice": np.asarray(atoms_2d.cell.reciprocal().array, dtype=float).tolist(),
        },
    )
