"""Canonical structure hashing and provenance helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def structure_hash(atoms: Any) -> str:
    return _sha256_json(_canonical_structure_payload(atoms))


def cell_hash(atoms: Any) -> str:
    return _sha256_json({"cell": _rounded(np.asarray(atoms.cell.array, dtype=float))})


def positions_hash(atoms: Any) -> str:
    return _sha256_json(
        {
            "symbols": list(atoms.get_chemical_symbols()),
            "scaled_positions": _rounded(_folded_scaled_positions(atoms)),
        }
    )


def build_structure_provenance(
    *,
    input_atoms: Any,
    relaxed_atoms: Any,
    fc2_atoms: Any,
    fc3_atoms: Any,
    input_structure_path: Path | str | None,
    relaxed_structure_path: Path | str | None,
    fc2_source_structure_path: Path | str | None,
    fc3_source_structure_path: Path | str | None,
    relax_backend: str,
    force_constants_backend: str,
    structure_stage_mode: str,
) -> dict[str, Any]:
    """Build the structure provenance block embedded in workflow outputs."""

    fc2_hash = structure_hash(fc2_atoms)
    fc3_hash = structure_hash(fc3_atoms)
    return {
        "input_structure_path": str(input_structure_path) if input_structure_path is not None else None,
        "input_structure_hash": structure_hash(input_atoms),
        "relaxed_structure_path": str(relaxed_structure_path) if relaxed_structure_path is not None else None,
        "relaxed_structure_hash": structure_hash(relaxed_atoms),
        "fc2_source_structure_path": str(fc2_source_structure_path) if fc2_source_structure_path is not None else None,
        "fc2_source_structure_hash": fc2_hash,
        "fc3_source_structure_path": str(fc3_source_structure_path) if fc3_source_structure_path is not None else None,
        "fc3_source_structure_hash": fc3_hash,
        "same_structure_for_fc2_fc3": fc2_hash == fc3_hash,
        "relax_backend": relax_backend,
        "force_constants_backend": force_constants_backend,
        "structure_stage_mode": structure_stage_mode,
    }


def write_structure_provenance(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _canonical_structure_payload(atoms: Any) -> dict[str, Any]:
    return {
        "symbols": list(atoms.get_chemical_symbols()),
        "natoms": int(len(atoms)),
        "pbc": [bool(value) for value in atoms.get_pbc()],
        "cell": _rounded(np.asarray(atoms.cell.array, dtype=float)),
        "scaled_positions": _rounded(_folded_scaled_positions(atoms)),
    }


def _folded_scaled_positions(atoms: Any) -> np.ndarray:
    scaled = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    return np.mod(scaled, 1.0)


def _rounded(array: np.ndarray) -> list[Any]:
    return np.round(np.asarray(array, dtype=float), decimals=10).tolist()


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
