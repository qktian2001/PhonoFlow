"""Crystal structure readers and writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read, write

from phonoflow.exceptions import StructureReadError, StructureWriteError


def read_structure(path: Path) -> Any:
    """Read a structure with ASE."""

    path = Path(path)
    if not path.exists():
        raise StructureReadError(f"Structure file not found: {path}")

    try:
        return read(path)
    except Exception as exc:
        raise StructureReadError(f"Could not read structure '{path}' with ASE: {exc}") from exc


def write_structure(atoms: Any, path: Path) -> None:
    """Write a structure in VASP format."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write(path, atoms, format="vasp", direct=True, sort=False)
    except Exception as exc:
        raise StructureWriteError(f"Could not write VASP structure '{path}': {exc}") from exc
