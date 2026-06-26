"""Core thermal-conductivity workflow dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from phonoflow.calculators.base import CalculatorBackend
from phonoflow.config import WorkflowConfig
from phonoflow.thermal.config import disabled_thermal_result, unavailable_thermal_result
from phonoflow.thermal.fc3_finite_displacement import run_finite_displacement_kappa_workflow
from phonoflow.thermal.fc3_hiphive import run_hiphive_kappa_workflow


def run_thermal_conductivity_workflow(
    atoms: Any,
    backend: CalculatorBackend,
    config: WorkflowConfig,
    outdir: Path,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the requested thermal-conductivity workflow without breaking phonons."""

    if not config.compute_kappa:
        return disabled_thermal_result()
    if config.fc3_method == "finite-displacement":
        return run_finite_displacement_kappa_workflow(atoms, backend, config, outdir, log=log)
    if config.fc3_method == "hiphive":
        result = run_hiphive_kappa_workflow(atoms, backend, config, outdir, log=log)
        result["enabled"] = True
        result["kappa_method"] = config.kappa_method
        return result
    return unavailable_thermal_result(
        reason=f"Unsupported fc3_method: {config.fc3_method}",
        fc3_method=str(config.fc3_method),
        kappa_method=config.kappa_method,
    )
