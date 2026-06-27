from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from typer.testing import CliRunner

from phonoflow.cli import app
from phonoflow.config import WorkflowConfig
from phonoflow.defaults import infer_default_config, resolve_dpa_model_path
from phonoflow.exceptions import ConfigError
from phonoflow.workflow.relax import run_ase_relaxation


def test_dpa3_alias_auto_resolves_registered_model_and_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    dpa3_model = model_dir / "DPA-3.2-5M.pt"
    dpa3_model.write_bytes(b"dpa3")
    monkeypatch.setattr("phonoflow.defaults.DEFAULT_DPA_MODEL_DIR", model_dir)

    config = infer_default_config(
        atoms=bulk("Si", "diamond", a=5.43),
        input_path=Path("examples/Si.vasp"),
        model_path=None,
        user_config=WorkflowConfig(
            input_path=Path("examples/Si.vasp"),
            backend="dpa3",
            compute_kappa=True,
        ),
    )

    assert config.backend == "deepmd"
    assert config.backend_alias == "dpa3"
    assert config.dpa_model_name == "DPA-3.2-5M.pt"
    assert config.model_path == dpa3_model
    assert config.deepmd_model_head == "OMat24"
    assert config.relax is False
    assert config.supercell_info["source"] == "auto"
    assert config.supercell_dim == config.supercell_info["supercell_dim"]
    assert config.fc3_supercell_dim == "auto"
    assert config.displacement == 0.01
    assert config.fc3_displacement == 0.03
    assert config.phono3py_symprec == 1e-5
    assert config.phono3py_cutoff_frequency == 1e-4
    assert config.kappa_mesh == [21, 21, 21]
    assert config.deepmd_reuse_calculator is True
    assert config.deepmd_deterministic is True
    assert config.save_force_audit is True
    assert config.phono3py_symmetrize_fc2 is True


def test_dpa_default_geometry_policy_matches_nep89_auto_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    dpa3_model = model_dir / "DPA-3.2-5M.pt"
    dpa3_model.write_bytes(b"dpa3")
    monkeypatch.setattr("phonoflow.defaults.DEFAULT_DPA_MODEL_DIR", model_dir)
    monkeypatch.setattr("phonoflow.defaults.resolve_backend_name", lambda requested: "calorine" if requested == "calorine" else "deepmd")
    atoms = bulk("Si", "diamond", a=5.43)
    input_path = Path("examples/Si.vasp")

    nep89 = infer_default_config(
        atoms=atoms,
        input_path=input_path,
        model_path=Path("nep89_potential/nep89_20250409.txt"),
        user_config=WorkflowConfig(
            input_path=input_path,
            model_path=Path("nep89_potential/nep89_20250409.txt"),
            backend="calorine",
            compute_kappa=True,
        ),
    )
    dpa3 = infer_default_config(
        atoms=atoms,
        input_path=input_path,
        model_path=None,
        user_config=WorkflowConfig(input_path=input_path, backend="dpa3", compute_kappa=True),
    )

    assert nep89.supercell_info["source"] == "auto"
    assert dpa3.supercell_info["source"] == "auto"
    assert dpa3.supercell_dim == nep89.supercell_dim
    assert dpa3.fc3_supercell_dim == nep89.fc3_supercell_dim == "auto"
    assert dpa3.kappa_mesh == nep89.kappa_mesh == [21, 21, 21]
    assert dpa3.phono3py_symprec == nep89.phono3py_symprec == 1e-5
    assert dpa3.phono3py_cutoff_frequency == nep89.phono3py_cutoff_frequency == 1e-4
    assert dpa3.displacement == nep89.displacement == 0.01
    assert dpa3.phono3py_symmetrize_fc2 is True
    assert nep89.phono3py_symmetrize_fc2 is True


def test_dpa4_alias_auto_resolves_registered_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    dpa4_model = model_dir / "DPA4-Neo-OMat24-v20260528_rc.pt"
    dpa4_model.write_bytes(b"dpa4")
    monkeypatch.setattr("phonoflow.defaults.DEFAULT_DPA_MODEL_DIR", model_dir)

    resolved = resolve_dpa_model_path("dpa4", None)

    assert resolved.model_name == "DPA4-Neo-OMat24-v20260528_rc.pt"
    assert resolved.model_path == dpa4_model
    assert resolved.backend_alias == "dpa4"


def test_generic_dpa_requires_explicit_model_path() -> None:
    with pytest.raises(ConfigError, match="--model-path"):
        resolve_dpa_model_path("dpa", None)


def test_nep89_defaults_enable_phono3py_symmetrize_fc2_for_traditional_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("phonoflow.defaults.resolve_backend_name", lambda requested: "calorine")
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

    assert config.backend == "calorine"
    assert config.phono3py_symmetrize_fc2 is True
    assert config.displacement == 0.01


def test_dpa_dry_run_result_records_alias_model_name_and_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    dpa3_model = model_dir / "DPA-3.2-5M.pt"
    dpa3_model.write_bytes(b"dpa3")
    monkeypatch.setattr("phonoflow.defaults.DEFAULT_DPA_MODEL_DIR", model_dir)
    outdir = tmp_path / "dry"

    result = CliRunner().invoke(
        app,
        [
            "single",
            "--input-path",
            str(Path(__file__).resolve().parents[1] / "examples" / "Si.vasp"),
            "--backend",
            "dpa3",
            "--outdir",
            str(outdir),
            "--dry-run",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads((outdir / "result.json").read_text(encoding="utf-8"))
    assert data["backend_requested"] == "dpa3"
    assert data["backend_resolved"] == "deepmd"
    assert data["backend_alias"] == "dpa3"
    assert data["dpa_model_name"] == "DPA-3.2-5M.pt"
    assert data["deepmd_model_head"] == "OMat24"
    assert data["model_path"] == str(dpa3_model)
    assert data["model_file_hash"]
    assert data["supercell_dim_requested"] == "auto"
    assert data["supercell_info"]["source"] == "auto"
    assert data["phono3py_symmetrize_fc2"] is True
    assert data["wte_capability"]["phono3py_version"]
    assert "wte_module_found" in data["wte_capability"]
    assert "installation_hint" in data["wte_capability"]
    run_command = (outdir / "run_command.txt").read_text(encoding="utf-8")
    assert "--deepmd-model-head OMat24" in run_command


def test_run_cli_auto_detects_dpa_model_path_without_backend(tmp_path: Path) -> None:
    model = tmp_path / "DPA-3.2-5M.pt"
    model.write_bytes(b"dpa3")
    outdir = tmp_path / "run-dpa-auto"

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--input-path",
            str(Path(__file__).resolve().parents[1] / "examples" / "Si.vasp"),
            "--model-path",
            str(model),
            "--outdir",
            str(outdir),
            "--dry-run",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads((outdir / "result.json").read_text(encoding="utf-8"))
    assert data["backend_requested"] == "auto"
    assert data["backend_resolved"] == "deepmd"
    assert data["backend_alias"] == "dpa32"
    assert data["dpa_model_name"] == "DPA-3.2-5M.pt"
    assert data["deepmd_model_head"] == "OMat24"
    assert data["model_path"] == str(model)


def test_single_cli_accepts_explicit_deepmd_model_head(tmp_path: Path) -> None:
    model = tmp_path / "custom-multitask.pt"
    model.write_bytes(b"model")
    outdir = tmp_path / "dry-head"

    result = CliRunner().invoke(
        app,
        [
            "single",
            "--input-path",
            str(Path(__file__).resolve().parents[1] / "examples" / "Si.vasp"),
            "--backend",
            "deepmd",
            "--model-path",
            str(model),
            "--deepmd-model-head",
            "CustomHead",
            "--outdir",
            str(outdir),
            "--dry-run",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads((outdir / "result.json").read_text(encoding="utf-8"))
    assert data["deepmd_model_head"] == "CustomHead"


def test_deepmd_engineering_flags_are_described_as_phonoflow_policies() -> None:
    result = CliRunner().invoke(app, ["single", "--help"])
    help_text = " ".join(result.output.replace("│", " ").split())

    assert result.exit_code == 0
    assert "PhonoFlow reproducibility policy" in help_text
    assert "PhonoFlow performance policy" in help_text
    assert "PhonoFlow diagnostic artifact" in help_text


def test_single_cli_output_includes_traceable_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    dpa3_model = model_dir / "DPA-3.2-5M.pt"
    dpa3_model.write_bytes(b"dpa3")
    monkeypatch.setattr("phonoflow.defaults.DEFAULT_DPA_MODEL_DIR", model_dir)
    outdir = tmp_path / "dry"

    result = CliRunner().invoke(
        app,
        [
            "single",
            "--input-path",
            str(Path(__file__).resolve().parents[1] / "examples" / "Si.vasp"),
            "--backend",
            "dpa3",
            "--outdir",
            str(outdir),
            "--dry-run",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "[1/8] Reading input structure" in result.output
    assert "[2/8] Resolving default settings" in result.output
    assert result.output.count("Resolved PhonoFlow settings") == 1
    assert "backend: dpa3 -> deepmd" in result.output
    assert "supercell:" in result.output
    assert "phono3py_symmetrize_fc2: True" in result.output
    assert (outdir / "validation.log").exists()


def test_dpa_relax_uses_nep89_relax_model_instead_of_force_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    dpa3_model = model_dir / "DPA-3.2-5M.pt"
    dpa3_model.write_bytes(b"dpa3")
    nep89_model = tmp_path / "nep89.txt"
    nep89_model.write_text("nep89", encoding="utf-8")
    monkeypatch.setattr("phonoflow.defaults.DEFAULT_DPA_MODEL_DIR", model_dir)
    monkeypatch.setattr("phonoflow.defaults.DEFAULT_NEP89_MODEL_PATH", nep89_model)

    config = infer_default_config(
        atoms=bulk("Si", "diamond", a=5.43),
        input_path=Path("examples/Si.vasp"),
        model_path=None,
        user_config=WorkflowConfig(
            input_path=Path("examples/Si.vasp"),
            backend="dpa3",
            relax=True,
            option_sources={"relax": "user"},
        ),
    )

    assert config.model_path == dpa3_model
    assert config.relax_model_path == nep89_model
    assert config.relax_model_path != config.model_path


def test_ase_relaxation_uses_selected_backend_model_not_force_config_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ZeroCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": 0.0,
                "forces": np.zeros((len(atoms), 3), dtype=float),
                "stress": np.zeros(6, dtype=float),
            }

    seen_model_paths: list[Path | None] = []

    class RelaxBackend:
        name = "calorine"
        model_path = tmp_path / "nep89.txt"

        def supports_stress(self) -> bool:
            return True

        def create_calculator(self, model_path=None):
            seen_model_paths.append(model_path)
            return ZeroCalculator()

    class NoOpOptimizer:
        def __init__(self, atoms, logfile):
            self.nsteps = 0

        def run(self, fmax, steps):
            return None

    monkeypatch.setitem(__import__("phonoflow.workflow.relax", fromlist=["OPTIMIZERS"]).OPTIMIZERS, "FIRE", NoOpOptimizer)
    dpa_model = tmp_path / "DPA-3.2-5M.pt"
    config = WorkflowConfig(
        model_path=dpa_model,
        backend="deepmd",
        relax=True,
        relax_cell=True,
        relax_model_path=RelaxBackend.model_path,
    )

    run_ase_relaxation(
        bulk("Si", "diamond", a=5.43),
        RelaxBackend(),
        tmp_path / "relax",
        config,
    )

    assert seen_model_paths == [None]


def test_dpa_relax_dry_run_records_force_and_relax_models_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    dpa3_model = model_dir / "DPA-3.2-5M.pt"
    dpa3_model.write_bytes(b"dpa3")
    nep89_model = tmp_path / "nep89.txt"
    nep89_model.write_text("nep89", encoding="utf-8")
    monkeypatch.setattr("phonoflow.defaults.DEFAULT_DPA_MODEL_DIR", model_dir)
    monkeypatch.setattr("phonoflow.defaults.DEFAULT_NEP89_MODEL_PATH", nep89_model)
    outdir = tmp_path / "dry-relax"

    result = CliRunner().invoke(
        app,
        [
            "single",
            "--input-path",
            str(Path(__file__).resolve().parents[1] / "examples" / "Si.vasp"),
            "--backend",
            "dpa3",
            "--relax",
            "--outdir",
            str(outdir),
            "--dry-run",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads((outdir / "result.json").read_text(encoding="utf-8"))
    assert data["force_backend"] == "deepmd"
    assert data["force_model_path"] == str(dpa3_model)
    assert data["relax_backend"] == "calorine"
    assert data["relax_model_path"] == str(nep89_model)
    assert data["relax_enabled"] is True
    settings = json.loads((outdir / "resolved_settings.json").read_text(encoding="utf-8"))
    assert settings["relax_backend_requested"]["value"] == "auto"
    assert settings["relax_backend_resolved"]["value"] == "calorine"
    assert settings["relax_model_path"]["value"] == str(nep89_model)
