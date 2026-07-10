from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from phonoflow.calculators import get_backend
from phonoflow.config import WorkflowConfig
from phonoflow.exceptions import BackendUnavailableError, ConfigError


def _install_fake_deepmd(monkeypatch: pytest.MonkeyPatch, calculator_cls: type) -> None:
    deepmd_module = types.ModuleType("deepmd")
    calculator_module = types.ModuleType("deepmd.calculator")
    calculator_module.DP = calculator_cls
    monkeypatch.setitem(sys.modules, "deepmd", deepmd_module)
    monkeypatch.setitem(sys.modules, "deepmd.calculator", calculator_module)


def test_dpa_aliases_resolve_to_deepmd_backend(tmp_path: Path) -> None:
    model = tmp_path / "dpa4.pt"
    model.write_bytes(b"fake-model")

    for alias in ("deepmd", "dpa", "dpa3", "dpa4"):
        backend = get_backend(alias, model_path=model)
        assert backend.name == "deepmd"
        assert getattr(backend, "backend_alias") == alias


def test_deepmd_missing_dependency_has_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "deepmd", None)
    monkeypatch.setitem(sys.modules, "deepmd.calculator", None)
    model = tmp_path / "dpa4.pt"
    model.write_bytes(b"fake-model")
    backend = get_backend("dpa4", model_path=model)

    assert backend.check_available() is False
    with pytest.raises(BackendUnavailableError, match="deepmd-kit"):
        backend.create_calculator()


def test_deepmd_backend_reuses_calculator_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[str] = []

    class FakeDP:
        implemented_properties = ["energy", "forces", "stress"]

        def __init__(self, model: str) -> None:
            created.append(model)

        def get_potential_energy(self, atoms: Atoms | None = None) -> float:
            return -1.25

        def get_forces(self, atoms: Atoms | None = None) -> np.ndarray:
            assert atoms is not None
            return np.zeros((len(atoms), 3))

        def get_stress(self, atoms: Atoms | None = None) -> np.ndarray:
            return np.zeros(6)

    _install_fake_deepmd(monkeypatch, FakeDP)
    model = tmp_path / "dpa4.pt"
    model.write_bytes(b"fake-model")
    backend = get_backend("dpa4", model_path=model)
    backend.apply_config(WorkflowConfig(deepmd_reuse_calculator=True))

    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[3, 3, 3], pbc=True)
    backend.calculate_energy_forces(atoms)
    backend.calculate_energy_forces(atoms)

    assert created == [str(model)]


def test_deepmd_backend_passes_multitask_model_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[dict[str, str | None]] = []

    class FakeDP:
        def __init__(self, model: str, head: str | None = None) -> None:
            created.append({"model": model, "head": head})

    _install_fake_deepmd(monkeypatch, FakeDP)
    model = tmp_path / "DPA-3.2-5M.pt"
    model.write_bytes(b"fake-model")
    backend = get_backend("dpa32", model_path=model)
    backend.apply_config(
        WorkflowConfig(
            backend="deepmd",
            model_path=model,
            deepmd_model_head="OMat24",
        )
    )

    backend.create_calculator()

    assert created == [{"model": str(model), "head": "OMat24"}]


def test_dpa4neo_backend_sets_default_inference_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DP_INFER_BATCH_SIZE", raising=False)
    model = tmp_path / "DPA4-Neo-OMat24-v20260528_rc.pt"
    model.write_bytes(b"fake-model")
    backend = get_backend("dpa4neo", model_path=model)

    backend.apply_config(WorkflowConfig(backend="deepmd", model_path=model))

    assert __import__("os").environ["DP_INFER_BATCH_SIZE"] == "64"


def test_deepmd_deterministic_forces_torch_threads_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    class FakeTorch:
        @staticmethod
        def set_num_threads(value: int) -> None:
            calls.append(("threads", value))

        @staticmethod
        def set_num_interop_threads(value: int) -> None:
            calls.append(("interop", value))

        @staticmethod
        def use_deterministic_algorithms(value: bool) -> None:
            calls.append(("deterministic", int(value)))

        @staticmethod
        def get_num_threads() -> int:
            return 1

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    backend = get_backend("deepmd")

    backend.apply_config(WorkflowConfig(backend="deepmd", deepmd_deterministic=True, deepmd_torch_threads=8))

    assert ("threads", 1) in calls
    assert any("deterministic mode is enabled" in warning for warning in backend.deterministic_warnings)
    assert backend.runtime_thread_info["actual_torch_num_threads"] == 1


def test_deepmd_non_deterministic_uses_explicit_torch_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    class FakeTorch:
        @staticmethod
        def set_num_threads(value: int) -> None:
            calls.append(("threads", value))

        @staticmethod
        def set_num_interop_threads(value: int) -> None:
            calls.append(("interop", value))

        @staticmethod
        def get_num_threads() -> int:
            thread_calls = [value for name, value in calls if name == "threads"]
            return thread_calls[-1] if thread_calls else 0

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    backend = get_backend("deepmd")

    backend.apply_config(WorkflowConfig(backend="deepmd", deepmd_deterministic=False, deepmd_torch_threads=6))

    assert ("threads", 6) in calls
    assert ("interop", 6) in calls
    assert backend.runtime_thread_info["actual_torch_num_threads"] == 6


def test_deepmd_backend_validates_missing_model_path() -> None:
    backend = get_backend("deepmd")
    with pytest.raises(ConfigError, match="requires --model-path"):
        backend.create_calculator()
