from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.build import bulk

from phonoflow.config import WorkflowConfig
from phonoflow.defaults import infer_default_config
from phonoflow.thermal.fc3_finite_displacement import _resolve_kappa_mesh as resolve_fd_kappa_mesh
from phonoflow.thermal.kappa_io import select_kappa_hdf5_path
from phonoflow.workflow.phonon import _run_total_dos


def test_infer_default_config_unifies_default_q_mesh() -> None:
    config = infer_default_config(
        atoms=bulk("Si", "diamond", a=5.43),
        input_path=Path("examples/Si.vasp"),
        model_path=Path("nep89_potential/nep89_20250409.txt"),
        user_config=WorkflowConfig(
            input_path=Path("examples/Si.vasp"),
            model_path=Path("nep89_potential/nep89_20250409.txt"),
            backend="calorine",
            compute_kappa=True,
        ),
    )

    assert config.mesh == [21, 21, 21]
    assert config.kappa_mesh == [21, 21, 21]


def test_infer_default_config_uses_2d_q_mesh_for_c_axis_vacuum() -> None:
    atoms = Atoms(
        "Si2",
        cell=[3.8, 3.8, 25.0],
        scaled_positions=[(0.0, 0.0, 0.5), (0.5, 0.5, 0.5)],
        pbc=True,
    )

    config = infer_default_config(
        atoms=atoms,
        input_path=Path("examples/Si.vasp"),
        model_path=Path("nep89_potential/nep89_20250409.txt"),
        user_config=WorkflowConfig(
            input_path=Path("examples/Si.vasp"),
            model_path=Path("nep89_potential/nep89_20250409.txt"),
            backend="calorine",
            compute_kappa=True,
        ),
    )

    assert config.mesh == [51, 51, 1]
    assert config.kappa_mesh == [51, 51, 1]


def test_explicit_q_mesh_is_not_overridden_by_2d_default() -> None:
    atoms = Atoms(
        "Si2",
        cell=[3.8, 3.8, 25.0],
        scaled_positions=[(0.0, 0.0, 0.5), (0.5, 0.5, 0.5)],
        pbc=True,
    )

    config = infer_default_config(
        atoms=atoms,
        input_path=Path("examples/Si.vasp"),
        model_path=Path("nep89_potential/nep89_20250409.txt"),
        user_config=WorkflowConfig(
            input_path=Path("examples/Si.vasp"),
            model_path=Path("nep89_potential/nep89_20250409.txt"),
            backend="calorine",
            mesh=[9, 9, 3],
            compute_kappa=True,
        ),
    )

    assert config.mesh == [9, 9, 3]
    assert config.kappa_mesh == [9, 9, 3]


def test_explicit_kappa_mesh_alias_populates_common_q_mesh() -> None:
    config = infer_default_config(
        atoms=bulk("Si", "diamond", a=5.43),
        input_path=Path("examples/Si.vasp"),
        model_path=Path("nep89_potential/nep89_20250409.txt"),
        user_config=WorkflowConfig(
            input_path=Path("examples/Si.vasp"),
            model_path=Path("nep89_potential/nep89_20250409.txt"),
            backend="calorine",
            mesh="auto",
            kappa_mesh=[7, 7, 7],
            compute_kappa=True,
        ),
    )

    assert config.mesh == [7, 7, 7]
    assert config.kappa_mesh == [7, 7, 7]
    assert resolve_fd_kappa_mesh(config) == [7, 7, 7]


def test_run_total_dos_uses_gamma_centered_q_mesh(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class FakePhonon:
        def run_mesh(self, mesh, **kwargs):
            calls["mesh"] = list(mesh)
            calls["kwargs"] = dict(kwargs)

        def run_total_dos(self):
            calls["ran_total_dos"] = True

        def get_total_dos_dict(self):
            return {
                "frequency_points": [0.0, 1.0],
                "total_dos": [0.5, 1.5],
            }

    monkeypatch.setattr("phonoflow.workflow.phonon.plot_phonon_dos", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "phonoflow.workflow.phonon.write_dos_diagnostics",
        lambda **kwargs: {"source_file": "phonon_dos.dat", "plot_file": "phonon_dos.png"},
    )

    result = _run_total_dos(
        FakePhonon(),
        WorkflowConfig(mesh=[4, 5, 6]),
        tmp_path,
    )

    assert calls["mesh"] == [4, 5, 6]
    assert calls["kwargs"] == {"is_gamma_center": True}
    assert calls["ran_total_dos"] is True
    assert result["dos_generated"] is True


def test_select_kappa_hdf5_path_prefers_current_q_mesh(tmp_path: Path) -> None:
    stale = tmp_path / "kappa-m111111.hdf5"
    target = tmp_path / "kappa-m212121.hdf5"
    stale.write_bytes(b"old")
    target.write_bytes(b"new")

    assert select_kappa_hdf5_path(tmp_path, [21, 21, 21]) == target
