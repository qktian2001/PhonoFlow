"""Reserved GPUMD command-line backend interface."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from phonoflow.calculators.base import CalculatorBackend
from phonoflow.config import WorkflowConfig


class GPUMDBackend(CalculatorBackend):
    """Placeholder backend for future GPUMD-powered calculations."""

    name = "gpumd"

    def check_available(self) -> bool:
        return shutil.which("gpumd") is not None

    def calculate_energy_forces(self, atoms: Any) -> dict[str, Any]:
        # TODO(v0.3): implement GPUMD command-line force evaluation.
        raise NotImplementedError("GPUMD force evaluation is planned for a later release.")

    def relax_structure(self, atoms: Any, outdir: Path, config: WorkflowConfig) -> tuple[Any, dict[str, Any]]:
        # TODO(v0.3): implement GPUMD command-line structure relaxation.
        raise NotImplementedError("GPUMD relaxation is planned for a later release.")
