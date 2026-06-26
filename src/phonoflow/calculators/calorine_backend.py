"""Calorine CPUNEP backend for NEP/NEP89 calculations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from phonoflow.calculators.base import CalculatorBackend
from phonoflow.config import WorkflowConfig
from phonoflow.exceptions import BackendUnavailableError, ConfigError, WorkflowError


class CalorineBackend(CalculatorBackend):
    """Backend for Calorine CPUNEP energy and force calculations."""

    name = "calorine"

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else None

    def check_available(self) -> bool:
        try:
            from calorine.calculators import CPUNEP  # noqa: F401
        except Exception:
            return False
        return True

    def set_model_path(self, model_path: Path | None) -> None:
        """Set the NEP/NEP89 potential path used by subsequent calculations."""

        self.model_path = Path(model_path) if model_path is not None else None

    def create_calculator(self, model_path: Path | None = None) -> Any:
        """Create a Calorine CPUNEP calculator from a potential file."""

        potential_path = Path(model_path or self.model_path) if (model_path or self.model_path) else None
        if potential_path is None:
            raise ConfigError(
                "backend='calorine' requires --model-path pointing to a NEP/NEP89 potential file."
            )
        if not potential_path.exists():
            raise ConfigError(
                f"NEP model file not found: {potential_path}\n"
                "Please provide a valid --model-path, for example nep.txt or nep89.txt."
            )

        try:
            from calorine.calculators import CPUNEP
        except ImportError as exc:
            raise BackendUnavailableError(
                "Calorine is required for real NEP/NEP89 calculations.\n"
                "Install it with:\n"
                "python -m pip install calorine"
            ) from exc
        except Exception as exc:
            raise BackendUnavailableError(
                "Calorine is importable, but `from calorine.calculators import CPUNEP` failed. "
                f"Your Calorine installation may not expose the CPUNEP API. Details: {exc}"
            ) from exc

        try:
            return CPUNEP(str(potential_path))
        except Exception as exc:
            raise BackendUnavailableError(
                f"Failed to initialize Calorine CPUNEP with model file {potential_path}. "
                "Please check whether the NEP/NEP89 potential file is valid and compatible "
                "with Calorine. Possible causes include an unsupported NEP89 format, an "
                "older Calorine version, or a non-NEP potential file. "
                f"Details: {exc}"
            ) from exc

    def supports_stress(self) -> bool:
        """Calorine CPUNEP exposes stress through the ASE calculator interface."""

        return True

    def calculate_energy_forces(self, atoms: Any) -> dict[str, Any]:
        """Calculate energy and forces with Calorine CPUNEP."""

        atoms = atoms.copy()
        atoms.calc = self.create_calculator()
        try:
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
        except Exception as exc:
            raise WorkflowError(
                "Calorine CPUNEP failed during energy/force evaluation. Check the input "
                "structure, potential compatibility, and Calorine installation. "
                f"Details: {exc}"
            ) from exc

        if forces.shape != (len(atoms), 3):
            raise WorkflowError(
                f"Calorine CPUNEP returned forces with shape {forces.shape}; expected ({len(atoms)}, 3)."
            )
        return {"energy": energy, "forces": forces}

    def relax_structure(self, atoms: Any, outdir: Path, config: WorkflowConfig) -> tuple[Any, dict[str, Any]]:
        """Run ASE relaxation through the workflow helper."""

        from phonoflow.workflow.relax import run_ase_relaxation

        return run_ase_relaxation(atoms, self, outdir, config)
