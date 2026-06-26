"""Dummy backend for testing workflow plumbing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from phonoflow.calculators.base import CalculatorBackend
from phonoflow.config import WorkflowConfig


class DummyBackend(CalculatorBackend):
    """Backend that returns deterministic non-physical values."""

    name = "dummy"

    def check_available(self) -> bool:
        return True

    def calculate_energy_forces(self, atoms: Any) -> dict[str, Any]:
        n_atoms = len(atoms)
        return {"energy": 0.0, "forces": np.zeros((n_atoms, 3), dtype=float)}

    def relax_structure(self, atoms: Any, outdir: Path, config: WorkflowConfig) -> tuple[Any, dict[str, Any]]:
        relax_info = {
            "relax_converged": True,
            "final_max_force_eV_per_A": 0.0,
            "n_steps": 0,
            "notes": "Dummy relaxation copied the input structure.",
        }
        return atoms.copy(), relax_info
