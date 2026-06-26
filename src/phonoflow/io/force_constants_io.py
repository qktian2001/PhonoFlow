"""Force-constant export helpers."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile
from typing import Any

import numpy as np


def write_force_constants_text(
    force_constants: Any,
    outdir: Path,
    n_supercell_atoms: int,
    write_phonopy: bool = True,
    write_shengbte: bool = True,
    phonopy_filename: str = "FORCE_CONSTANTS",
    shengbte_filename: str = "FORCE_CONSTANTS_2ND",
) -> dict[str, Any]:
    """Write full FC2 in Phonopy text format and ShengBTE-style FC2 name."""

    array = _validate_full_fc2(force_constants, n_supercell_atoms)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    output: dict[str, Any] = {
        "force_constants_text_exported": False,
        "force_constants_text_format": "phonopy-text-fc2",
        "force_constants_text_shape": [int(value) for value in array.shape],
        "phonopy_force_constants_file": None,
        "shengbte_fc2_file": None,
    }

    phonopy_path = outdir / phonopy_filename
    shengbte_path = outdir / shengbte_filename

    if write_phonopy:
        _write_phonopy_force_constants_text(array, phonopy_path)
        _ensure_nonempty(phonopy_path)
        output["phonopy_force_constants_file"] = phonopy_path.name
        output["force_constants_text_exported"] = True

    if write_shengbte:
        if write_phonopy and phonopy_path.exists():
            copyfile(phonopy_path, shengbte_path)
        else:
            _write_phonopy_force_constants_text(array, shengbte_path)
        _ensure_nonempty(shengbte_path)
        output["shengbte_fc2_file"] = shengbte_path.name
        output["force_constants_text_exported"] = True

    return output


def _write_phonopy_force_constants_text(force_constants: np.ndarray, path: Path) -> None:
    try:
        from phonopy.file_IO import write_FORCE_CONSTANTS
    except Exception as exc:
        raise RuntimeError("Phonopy write_FORCE_CONSTANTS API is not available.") from exc

    write_FORCE_CONSTANTS(force_constants, filename=str(path))


def _validate_full_fc2(force_constants: Any, n_supercell_atoms: int) -> np.ndarray:
    array = np.asarray(force_constants, dtype=float)
    expected_shape = (int(n_supercell_atoms), int(n_supercell_atoms), 3, 3)
    if array.shape != expected_shape:
        raise ValueError(
            "Cannot export FORCE_CONSTANTS text because force constants are not full "
            f"supercell FC2. Got shape {array.shape}; expected {expected_shape}. "
            "Try --primitive-matrix identity, check Phonopy compact force-constant settings, "
            "or use force_constants.hdf5 only."
        )
    return array


def _ensure_nonempty(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Force-constants text export produced an empty file: {path}")
