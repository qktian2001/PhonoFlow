import json
from pathlib import Path

from ase import Atoms

from phonoflow.config import WorkflowConfig
from phonoflow.io.structure_io import write_structure
from phonoflow.workflow.pipeline import run_single_workflow


def test_dry_run_writes_settings_without_running_phonons(tmp_path: Path):
    input_path = tmp_path / "Si.vasp"
    model_path = tmp_path / "nep.txt"
    outdir = tmp_path / "dry"
    atoms = Atoms(
        "Si2",
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=[5.43, 5.43, 5.43],
        pbc=True,
    )
    write_structure(atoms, input_path)
    model_path.write_text("dummy potential placeholder\n", encoding="utf-8")

    result = run_single_workflow(
        WorkflowConfig(
            input_path=input_path,
            model_path=model_path,
            outdir=outdir,
            backend="dummy",
            dry_run=True,
        )
    )

    assert result["status"] == "dry-run"
    assert (outdir / "resolved_settings.json").exists()
    assert (outdir / "resolved_settings.yaml").exists()
    assert (outdir / "run_command.txt").exists()
    assert not (outdir / "relaxed.vasp").exists()
    assert not (outdir / "force_constants.hdf5").exists()
    dry_result = json.loads((outdir / "result.json").read_text(encoding="utf-8"))
    assert dry_result["dry_run"] is True


def test_dry_run_writes_detailed_settings_and_timing_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "Si.vasp"
    model_path = tmp_path / "nep.txt"
    outdir = tmp_path / "dry_timing"
    atoms = Atoms(
        "Si2",
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=[5.43, 5.43, 5.43],
        pbc=True,
    )
    write_structure(atoms, input_path)
    model_path.write_text("dummy potential placeholder\n", encoding="utf-8")

    result = run_single_workflow(
        WorkflowConfig(
            input_path=input_path,
            model_path=model_path,
            outdir=outdir,
            backend="dummy",
            dry_run=True,
            compute_kappa=False,
        )
    )

    run_log = (outdir / "run.log").read_text(encoding="utf-8")
    assert "[1/8] Reading input structure" in run_log
    assert "[2/8] Resolving default settings" in run_log
    assert "Resolved PhonoFlow settings" in run_log
    assert "skipped because compute_kappa=false" in run_log
    assert "[8/8] Writing reports and artifacts" in run_log
    assert "result.json: written" in run_log
    assert (outdir / "resolved_settings_table.txt").exists()
    settings = json.loads((outdir / "resolved_settings.json").read_text(encoding="utf-8"))
    assert settings["group_velocity"]["value"] is True
    timing = json.loads((outdir / "timing_breakdown.json").read_text(encoding="utf-8"))
    assert timing["stages"]["fc2_phonon"]["seconds"] >= 0
    assert timing["stages"]["fc3_thermal"]["status"] == "skipped"
    assert timing["stages"]["reporting_packaging"]["status"] == "completed"
    assert result["report"]["timing_breakdown"] == timing
