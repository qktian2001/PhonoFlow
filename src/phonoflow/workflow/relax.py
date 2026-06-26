"""Structure relaxation workflow helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE, LBFGS

from phonoflow.calculators.base import CalculatorBackend
from phonoflow.config import WorkflowConfig
from phonoflow.exceptions import ConfigError


OPTIMIZERS = {
    "FIRE": FIRE,
    "LBFGS": LBFGS,
}


def relax_structure(
    atoms: Any,
    backend: CalculatorBackend,
    outdir: Path,
    config: WorkflowConfig,
) -> tuple[Any, dict[str, Any]]:
    """Delegate relaxation to the selected backend."""

    if backend.name == "calorine":
        return run_ase_relaxation(atoms, backend, outdir, config)
    return backend.relax_structure(atoms, outdir, config)


def run_ase_relaxation(
    atoms: Any,
    backend: CalculatorBackend,
    outdir: Path,
    config: WorkflowConfig,
) -> tuple[Any, dict[str, Any]]:
    """Relax atomic positions, optionally together with the cell, using ASE."""

    if not hasattr(backend, "create_calculator"):
        raise ConfigError(f"Backend '{backend.name}' does not provide an ASE calculator.")
    if config.relax_cell and not backend.supports_stress():
        raise ConfigError(
            "Cell relaxation requires stress support from the selected backend. "
            "The current backend failed to provide stress. Use --no-relax-cell to relax "
            "atomic positions only."
        )

    outdir.mkdir(parents=True, exist_ok=True)
    relaxed_atoms = atoms.copy()
    relaxed_atoms.calc = backend.create_calculator()
    initial_cell = np.asarray(relaxed_atoms.cell.array, dtype=float)
    initial_cell_parameters = relaxed_atoms.cell.cellpar()
    initial_volume = float(relaxed_atoms.get_volume())

    if config.relax_cell:
        _assert_stress_available(relaxed_atoms)

    optimizer_cls = OPTIMIZERS.get(config.optimizer.upper())
    if optimizer_cls is None:
        raise ConfigError(f"Unsupported optimizer '{config.optimizer}'. Choose FIRE or LBFGS.")

    log_path = outdir / "relax.log"
    optimizable = FrechetCellFilter(relaxed_atoms) if config.relax_cell else relaxed_atoms
    optimizer = optimizer_cls(optimizable, logfile=str(log_path))
    optimizer.run(fmax=config.fmax, steps=config.max_steps)

    forces = np.asarray(relaxed_atoms.get_forces(), dtype=float)
    force_norms = np.linalg.norm(forces, axis=1) if len(forces) else np.array([0.0])
    final_max_force = float(np.max(force_norms))
    n_steps = int(getattr(optimizer, "nsteps", 0))
    relax_converged = final_max_force <= config.fmax
    final_cell = np.asarray(relaxed_atoms.cell.array, dtype=float)
    final_cell_parameters = relaxed_atoms.cell.cellpar()
    final_volume = float(relaxed_atoms.get_volume())
    stress_gpa = _stress_gpa(relaxed_atoms)

    return relaxed_atoms, {
        "relax_converged": relax_converged,
        "final_max_force_eV_per_A": final_max_force,
        "final_stress_GPa": stress_gpa,
        "n_steps": n_steps,
        "fmax": float(config.fmax),
        "max_steps": int(config.max_steps),
        "optimizer": optimizer_cls.__name__,
        "relax": True,
        "relax_cell": bool(config.relax_cell),
        "relax_mode": "cell" if config.relax_cell else "positions",
        "constant_cell": not config.relax_cell,
        "initial_cell": initial_cell.tolist(),
        "final_cell": final_cell.tolist(),
        "initial_cell_lengths": [float(value) for value in initial_cell_parameters[:3]],
        "final_cell_lengths": [float(value) for value in final_cell_parameters[:3]],
        "initial_cell_angles": [float(value) for value in initial_cell_parameters[3:]],
        "final_cell_angles": [float(value) for value in final_cell_parameters[3:]],
        "initial_volume": initial_volume,
        "final_volume": final_volume,
        "volume_change_percent": _volume_change_percent(initial_volume, final_volume),
        "warnings": [],
    }


def run_ase_position_relaxation(
    atoms: Any,
    backend: CalculatorBackend,
    outdir: Path,
    config: WorkflowConfig,
) -> tuple[Any, dict[str, Any]]:
    """Compatibility wrapper for older imports."""

    return run_ase_relaxation(atoms, backend, outdir, config)


def _assert_stress_available(atoms: Any) -> None:
    try:
        atoms.get_stress(voigt=True)
    except Exception as exc:
        raise ConfigError(
            "Cell relaxation requires stress support from the selected backend. "
            "The current backend failed to provide stress. Use --no-relax-cell to relax "
            "atomic positions only."
        ) from exc


def _stress_gpa(atoms: Any) -> list[float] | None:
    try:
        stress = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    except Exception:
        return None
    return [float(value) for value in stress * 160.21766208]


def _volume_change_percent(initial_volume: float, final_volume: float) -> float | None:
    if abs(initial_volume) < 1e-12:
        return None
    return float((final_volume - initial_volume) / initial_volume * 100.0)
