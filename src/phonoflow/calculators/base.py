"""Abstract calculator backend API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from phonoflow.config import WorkflowConfig


class CalculatorBackend(ABC):
    """Common interface for force and relaxation backends."""

    name: str

    @abstractmethod
    def check_available(self) -> bool:
        """Return True if this backend can run in the current environment."""

    @abstractmethod
    def calculate_energy_forces(self, atoms: Any) -> dict[str, Any]:
        """Return an energy and force dictionary."""

    def supports_stress(self) -> bool:
        """Return True when the backend can provide stress for cell relaxation."""

        return False

    def apply_config(self, config: WorkflowConfig) -> None:
        """Apply workflow-level backend options before calculations start."""

        return None

    @abstractmethod
    def relax_structure(self, atoms: Any, outdir: Path, config: WorkflowConfig) -> tuple[Any, dict[str, Any]]:
        """Return relaxed atoms and relaxation metadata."""
