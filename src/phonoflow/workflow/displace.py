"""Finite-displacement helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


def ase_atoms_to_phonopy_atoms(atoms: Atoms) -> PhonopyAtoms:
    """Convert ASE Atoms to PhonopyAtoms."""

    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )


def phonopy_atoms_to_ase_atoms(phonopy_atoms: PhonopyAtoms) -> Atoms:
    """Convert PhonopyAtoms to ASE Atoms."""

    return Atoms(
        symbols=list(phonopy_atoms.symbols),
        cell=phonopy_atoms.cell,
        scaled_positions=phonopy_atoms.scaled_positions,
        pbc=True,
    )


def create_phonopy(
    atoms: Atoms,
    supercell_dim: list[int],
    primitive_matrix: str = "P",
    symprec: float = 1e-5,
) -> Phonopy:
    """Create a Phonopy object from an ASE structure."""

    unitcell = ase_atoms_to_phonopy_atoms(atoms)
    primitive = _resolve_primitive_matrix_argument(primitive_matrix)
    return Phonopy(unitcell, supercell_matrix=supercell_dim, primitive_matrix=primitive, symprec=symprec)


def _resolve_primitive_matrix_argument(primitive_matrix: str) -> Any:
    normalized = primitive_matrix.lower()
    if normalized == "p":
        return "P"
    if normalized == "auto":
        return "auto"
    if normalized == "identity":
        return np.eye(3)
    if normalized == "none":
        return None
    raise ValueError("primitive_matrix must be one of: auto, P, identity, none")


def generate_displacements(phonon: Phonopy, displacement: float) -> list[Any]:
    """Generate finite-displacement supercells with Phonopy."""

    phonon.generate_displacements(distance=displacement)
    supercells = list(phonon.supercells_with_displacements or [])
    if not supercells:
        raise RuntimeError("Phonopy did not generate any displaced supercells.")
    return supercells
