"""Lightweight structure-type heuristics for relaxation guidance."""

from __future__ import annotations

from typing import Any

import numpy as np


VACUUM_WARNING = (
    "The input structure appears to contain a vacuum-like direction or a slab/interface "
    "geometry. The current default relaxation mode optimizes both atomic positions and "
    "the cell. For 2D, surface, slab, or interface systems, this may collapse the vacuum "
    "region or distort the intended geometry. Consider using --no-relax-cell to keep the "
    "cell fixed and relax atomic positions only."
)


def classify_structure_type(atoms: Any) -> dict[str, Any]:
    """Classify a structure with simple vacuum-direction heuristics.

    The classifier is intentionally conservative. It is meant to explain the default
    relaxation choice, not to impose automatic dimensional constraints.
    """

    lengths = np.asarray(atoms.cell.lengths(), dtype=float)
    scaled = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    atom_extents = [_periodic_extent(scaled[:, axis]) * float(lengths[axis]) for axis in range(3)]

    vacuum_like_directions: list[str] = []
    warnings: list[str] = []
    axis_names = ["a", "b", "c"]
    for axis, (length, extent) in enumerate(zip(lengths, atom_extents)):
        if length <= 0:
            continue
        vacuum_ratio = 1.0 - (extent / length)
        if length >= 15.0 and vacuum_ratio >= 0.35:
            vacuum_like_directions.append(axis_names[axis])

    if len(vacuum_like_directions) == 0:
        structure_type = "bulk"
    elif len(vacuum_like_directions) == 1:
        structure_type = "2d"
        warnings.append(VACUUM_WARNING)
    else:
        structure_type = "interface_or_slab"
        warnings.append(VACUUM_WARNING)

    return {
        "structure_type": structure_type,
        "classification_confidence": "heuristic",
        "classification_method": "cell-length and periodic atom-extent heuristic",
        "cell_lengths": [float(value) for value in lengths],
        "atom_extents": [float(value) for value in atom_extents],
        "vacuum_like_directions": vacuum_like_directions,
        "warnings": warnings,
    }


def _periodic_extent(values: np.ndarray) -> float:
    """Return the smallest fractional span occupied on a periodic interval."""

    if values.size <= 1:
        return 0.0
    wrapped = np.mod(values, 1.0)
    wrapped.sort()
    gaps = np.diff(wrapped)
    wrap_gap = (wrapped[0] + 1.0) - wrapped[-1]
    largest_gap = float(max(np.max(gaps), wrap_gap))
    return float(max(0.0, min(1.0, 1.0 - largest_gap)))
