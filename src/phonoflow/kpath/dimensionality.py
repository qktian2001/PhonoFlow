from __future__ import annotations

from typing import Any

import numpy as np


def infer_dimensionality_by_vacuum(
    atoms: Any,
    vacuum_threshold: float = 8.0,
    min_cell_length_for_vacuum: float = 12.0,
    max_slab_fraction: float = 0.70,
) -> dict[str, Any]:
    """Infer whether a structure behaves like a 2D slab from its vacuum direction."""

    cell = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.get_positions(), dtype=float)
    lengths = np.linalg.norm(cell, axis=1)

    candidates: list[dict[str, Any]] = []
    for axis in range(3):
        length = float(lengths[axis])
        if length <= 1e-8:
            continue
        direction = cell[axis] / length
        proj = positions @ direction if len(positions) else np.asarray([0.0], dtype=float)
        atomic_span = float(proj.max() - proj.min()) if proj.size else 0.0
        vacuum_length = float(length - atomic_span)
        slab_fraction = float(atomic_span / length) if length > 0 else 1.0
        if (
            length >= float(min_cell_length_for_vacuum)
            and vacuum_length >= float(vacuum_threshold)
            and slab_fraction <= float(max_slab_fraction)
        ):
            candidates.append(
                {
                    "axis": axis,
                    "cell_length": length,
                    "atomic_span": atomic_span,
                    "vacuum_length": vacuum_length,
                    "slab_fraction": slab_fraction,
                }
            )

    if not candidates:
        return {
            "dimension": "3D",
            "vacuum_axis": None,
            "vacuum_axis_name": None,
            "reason": "No large vacuum direction detected.",
        }

    best = max(candidates, key=lambda item: item["vacuum_length"])
    axis_names = ["a", "b", "c"]
    return {
        "dimension": "2D",
        "vacuum_axis": int(best["axis"]),
        "vacuum_axis_name": axis_names[int(best["axis"])],
        "reason": f"Detected slab vacuum along {axis_names[int(best['axis'])]} axis.",
        **best,
    }


def standardize_2d_for_ase_bandpath(atoms: Any, vacuum_axis: int):
    """Temporarily move the vacuum axis to ``c`` for ASE 2D bandpath generation."""

    atoms2 = atoms.copy()
    old_cell = np.asarray(atoms2.cell.array, dtype=float)
    old_scaled = np.asarray(atoms2.get_scaled_positions(wrap=False), dtype=float)

    periodic_axes = [axis for axis in range(3) if axis != int(vacuum_axis)]
    permutation = periodic_axes + [int(vacuum_axis)]

    new_cell = old_cell[permutation]
    new_scaled = old_scaled[:, permutation]

    atoms2.set_cell(new_cell, scale_atoms=False)
    atoms2.set_scaled_positions(new_scaled)
    atoms2.pbc = [True, True, False]
    return atoms2, permutation
