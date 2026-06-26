"""Calculator backend registry."""

from __future__ import annotations

from pathlib import Path

from phonoflow.calculators.base import CalculatorBackend
from phonoflow.calculators.calorine_backend import CalorineBackend
from phonoflow.calculators.deepmd_backend import DeepMDBackend
from phonoflow.calculators.dummy import DummyBackend
from phonoflow.calculators.gpumd_backend import GPUMDBackend
from phonoflow.exceptions import ConfigError


PYNEP_REMOVED_MESSAGE = (
    "Unsupported backend: pynep. PyNEP backend has been removed. Please use "
    "backend=calorine for real NEP/NEP89 calculations or backend=dummy for tests."
)


def get_backend(name: str, model_path: Path | None = None) -> CalculatorBackend:
    """Create a calculator backend by name."""

    normalized = name.lower()
    if normalized == "dummy":
        return DummyBackend()
    if normalized == "calorine":
        return CalorineBackend(model_path=model_path)
    if normalized in {"deepmd", "dpa", "dpa3", "dpa4", "dpa31", "dpa32", "dpa33", "dpa4neo"}:
        return DeepMDBackend(model_path=model_path, backend_alias=normalized)
    if normalized == "pynep":
        raise ConfigError(PYNEP_REMOVED_MESSAGE)
    if normalized == "gpumd":
        return GPUMDBackend()
    raise ConfigError(
        f"Unknown backend '{name}'. Choose one of: auto, dummy, calorine, gpumd, deepmd, dpa, dpa31, dpa32, dpa33, dpa4neo."
    )


__all__ = [
    "CalculatorBackend",
    "DummyBackend",
    "CalorineBackend",
    "DeepMDBackend",
    "GPUMDBackend",
    "get_backend",
]
