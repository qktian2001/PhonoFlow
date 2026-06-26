from pathlib import Path

from ase import Atoms

from phonoflow.config import WorkflowConfig
from phonoflow.io.structure_io import write_structure
from phonoflow.workflow.pipeline import run_single_workflow


def test_existing_result_outdir_gets_timestamped_without_overwrite(tmp_path: Path):
    input_path = _write_input(tmp_path)
    outdir = tmp_path / "result"
    outdir.mkdir()
    (outdir / "result.json").write_text('{"success": true}\n', encoding="utf-8")

    result = run_single_workflow(WorkflowConfig(input_path=input_path, outdir=outdir, backend="dummy", dry_run=True))

    assert result["outdir"] != str(outdir)
    assert Path(result["outdir"]).name.startswith("result_")


def test_overwrite_keeps_existing_outdir(tmp_path: Path):
    input_path = _write_input(tmp_path)
    outdir = tmp_path / "result"
    outdir.mkdir()
    (outdir / "result.json").write_text('{"success": true}\n', encoding="utf-8")

    result = run_single_workflow(
        WorkflowConfig(input_path=input_path, outdir=outdir, backend="dummy", dry_run=True, overwrite=True)
    )

    assert result["outdir"] == str(outdir)


def _write_input(tmp_path: Path) -> Path:
    input_path = tmp_path / "Si.vasp"
    atoms = Atoms(
        "Si2",
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=[5.43, 5.43, 5.43],
        pbc=True,
    )
    write_structure(atoms, input_path)
    return input_path
