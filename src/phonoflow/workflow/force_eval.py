"""Force evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms

from phonoflow.calculators.base import CalculatorBackend
from phonoflow.workflow.force_audit import build_force_audit_record, write_force_audit_files
from phonoflow.workflow.displace import phonopy_atoms_to_ase_atoms


def evaluate_forces(
    displaced_supercells: list[PhonopyAtoms],
    backend: CalculatorBackend,
    model_path: Path | None,
    log: Callable[[str], None] | None = None,
    audit_outdir: Path | None = None,
    audit_label: str = "fc2",
) -> list[np.ndarray]:
    """Evaluate forces for displaced Phonopy supercells."""

    if hasattr(backend, "set_model_path"):
        backend.set_model_path(model_path)

    forces: list[np.ndarray] = []
    audit_records: list[dict[str, Any]] = []
    total = len(displaced_supercells)
    for index, supercell in enumerate(displaced_supercells, start=1):
        if log is not None:
            log(f"Evaluating forces for displaced supercell {index}/{total}")
        atoms = phonopy_atoms_to_ase_atoms(supercell)
        result: dict[str, Any] = backend.calculate_energy_forces(atoms)
        force_array = np.asarray(result["forces"], dtype=float)
        forces.append(force_array)
        if audit_outdir is not None:
            audit_records.append(
                build_force_audit_record(
                    index - 1,
                    atoms,
                    energy=result.get("energy"),
                    forces=force_array,
                )
            )
    if audit_outdir is not None:
        write_force_audit_files(audit_outdir, audit_label, audit_records, np.asarray(forces, dtype=float))
    return forces
