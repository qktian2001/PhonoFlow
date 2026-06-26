"""DeepMD-kit ASE backend for DPA/DeepMD model families."""

from __future__ import annotations

import os
import inspect
from pathlib import Path
from typing import Any

import numpy as np

from phonoflow.calculators.base import CalculatorBackend
from phonoflow.config import WorkflowConfig
from phonoflow.exceptions import BackendUnavailableError, ConfigError, WorkflowError


DETERMINISTIC_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "DP_INTRA_OP_PARALLELISM_THREADS": "1",
    "DP_INTER_OP_PARALLELISM_THREADS": "1",
}
DPA4_DEFAULT_INFER_BATCH_SIZE = "64"


class DeepMDBackend(CalculatorBackend):
    """Backend for DeepMD-kit ``deepmd.calculator.DP`` models."""

    name = "deepmd"

    def __init__(self, model_path: Path | None = None, backend_alias: str = "deepmd") -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.backend_alias = backend_alias.lower()
        self.reuse_calculator = True
        self.force_backend = "ase"
        self.device = "cpu"
        self.model_head: str | None = None
        self.deterministic = False
        self._calculator_cache: dict[str, Any] = {}
        self.deterministic_warnings: list[str] = []

    def check_available(self) -> bool:
        try:
            from deepmd.calculator import DP  # noqa: F401
        except Exception:
            return False
        return True

    def apply_config(self, config: WorkflowConfig) -> None:
        self.reuse_calculator = bool(config.deepmd_reuse_calculator)
        self.force_backend = str(config.deepmd_force_backend)
        self.device = str(config.deepmd_device)
        self.model_head = config.deepmd_model_head
        self.deterministic = bool(config.deepmd_deterministic)
        if self.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ["HIP_VISIBLE_DEVICES"] = "-1"
            os.environ["ROCR_VISIBLE_DEVICES"] = "-1"
        if self.backend_alias in {"dpa4", "dpa4neo"}:
            os.environ.setdefault("DP_INFER_BATCH_SIZE", DPA4_DEFAULT_INFER_BATCH_SIZE)
        if self.deterministic:
            self.deterministic_warnings = _apply_deterministic_settings()

    def set_model_path(self, model_path: Path | None) -> None:
        self.model_path = Path(model_path) if model_path is not None else None

    def create_calculator(self, model_path: Path | None = None) -> Any:
        potential_path = Path(model_path or self.model_path) if (model_path or self.model_path) else None
        if potential_path is None:
            raise ConfigError("backend='deepmd' requires --model-path pointing to a DeepMD/DPA model file.")
        if not potential_path.exists():
            raise ConfigError(f"DeepMD/DPA model file not found: {potential_path}")

        cache_key = f"{potential_path}|head={self.model_head or ''}"
        if self.reuse_calculator and cache_key in self._calculator_cache:
            return self._calculator_cache[cache_key]

        try:
            from deepmd.calculator import DP
        except ImportError as exc:
            raise BackendUnavailableError(
                "deepmd-kit is required for backend='deepmd'/'dpa4'. Install a DeepMD-kit "
                "build compatible with the model, torch, and platform."
            ) from exc
        except Exception as exc:
            raise BackendUnavailableError(
                "deepmd-kit is importable, but `from deepmd.calculator import DP` failed. "
                "For DPA4/SeZM, check deepmd-kit, torch, e3nn, MPI, and ABI compatibility. "
                f"Details: {exc}"
            ) from exc

        try:
            kwargs: dict[str, Any] = {}
            if self.device != "auto" and _dp_accepts_device(DP):
                kwargs["device"] = self.device
            if self.model_head is not None and _dp_accepts_parameter(DP, "head"):
                kwargs["head"] = self.model_head
            calculator = DP(model=str(potential_path), **kwargs)
        except Exception as exc:
            raise BackendUnavailableError(
                f"Failed to initialize deepmd.calculator.DP with model file {potential_path}. "
                "Possible causes include unsupported model type, missing SeZM/e3nn support, "
                "or torch/deepmd ABI mismatch. "
                f"Details: {exc}"
            ) from exc
        if self.reuse_calculator:
            self._calculator_cache[cache_key] = calculator
        return calculator

    def supports_stress(self) -> bool:
        return True

    def calculate_energy_forces(self, atoms: Any) -> dict[str, Any]:
        if self.force_backend == "deeppot":
            return self._calculate_via_deeppot(atoms)
        calculator = self.create_calculator()
        atoms_for_calc = atoms.copy()
        try:
            atoms_for_calc.calc = calculator
            energy = float(atoms_for_calc.get_potential_energy())
            forces = np.asarray(atoms_for_calc.get_forces(), dtype=float)
            stress = _safe_stress(atoms_for_calc)
        except Exception as exc:
            raise WorkflowError(
                "DeepMD/DPA force evaluation failed through ASE DP calculator. "
                "Check model compatibility, deepmd-kit runtime, and input structure. "
                f"Details: {exc}"
            ) from exc
        return _validated_result(atoms_for_calc, energy, forces, stress)

    def _calculate_via_deeppot(self, atoms: Any) -> dict[str, Any]:
        potential_path = Path(self.model_path) if self.model_path is not None else None
        if potential_path is None:
            raise ConfigError("backend='deepmd' requires --model-path for DeepPot direct evaluation.")
        try:
            from deepmd.infer import DeepPot
        except Exception as exc:
            raise BackendUnavailableError(
                "DeepPot direct evaluation requires `deepmd.infer.DeepPot`; use "
                "--deepmd-force-backend ase if this API is unavailable. "
                f"Details: {exc}"
            ) from exc
        try:
            kwargs: dict[str, Any] = {}
            if self.model_head is not None and _dp_accepts_parameter(DeepPot, "head"):
                kwargs["head"] = self.model_head
            deep_pot = DeepPot(str(potential_path), **kwargs)
            coord = np.asarray(atoms.get_positions(), dtype=float).reshape(1, -1)
            cell = np.asarray(atoms.cell.array, dtype=float).reshape(1, -1)
            atype = np.asarray(atoms.get_atomic_numbers(), dtype=int)
            energy, forces, virial = deep_pot.eval(coord, cell, atype)
            energy_value = float(np.asarray(energy).reshape(-1)[0])
            force_array = np.asarray(forces, dtype=float).reshape(len(atoms), 3)
            stress = _virial_to_voigt_stress(virial, atoms.get_volume())
        except Exception as exc:
            raise WorkflowError(f"DeepPot direct evaluation failed: {exc}") from exc
        return _validated_result(atoms, energy_value, force_array, stress)

    def relax_structure(self, atoms: Any, outdir: Path, config: WorkflowConfig) -> tuple[Any, dict[str, Any]]:
        from phonoflow.workflow.relax import run_ase_relaxation

        return run_ase_relaxation(atoms, self, outdir, config)


def _apply_deterministic_settings() -> list[str]:
    warnings: list[str] = []
    for name, value in DETERMINISTIC_ENV_VARS.items():
        os.environ.setdefault(name, value)
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except Exception as exc:
            warnings.append(f"Could not set torch interop threads: {exc}")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            warnings.append(f"Could not enable torch deterministic algorithms: {exc}")
    except Exception as exc:
        warnings.append(f"Could not apply torch deterministic settings: {exc}")
    return warnings


def _dp_accepts_device(dp_cls: Any) -> bool:
    return _dp_accepts_parameter(dp_cls, "device")


def _dp_accepts_parameter(dp_cls: Any, name: str) -> bool:
    try:
        signature = inspect.signature(dp_cls)
    except Exception:
        return True
    return name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _safe_stress(atoms: Any) -> list[float] | None:
    try:
        return [float(value) for value in np.asarray(atoms.get_stress(voigt=True), dtype=float)]
    except Exception:
        return None


def _validated_result(atoms: Any, energy: float, forces: np.ndarray, stress: list[float] | None) -> dict[str, Any]:
    if forces.shape != (len(atoms), 3):
        raise WorkflowError(
            f"DeepMD/DPA returned forces with shape {forces.shape}; expected ({len(atoms)}, 3)."
        )
    result: dict[str, Any] = {"energy": float(energy), "forces": forces}
    if stress is not None:
        result["stress"] = stress
    return result


def _virial_to_voigt_stress(virial: Any, volume: float) -> list[float] | None:
    if virial is None or abs(float(volume)) < 1e-12:
        return None
    tensor = np.asarray(virial, dtype=float).reshape(-1, 3, 3)[0] / float(volume)
    return [
        float(tensor[0, 0]),
        float(tensor[1, 1]),
        float(tensor[2, 2]),
        float(tensor[1, 2]),
        float(tensor[0, 2]),
        float(tensor[0, 1]),
    ]
