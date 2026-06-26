from pathlib import Path

from ase import Atoms

from phonoflow.config import WorkflowConfig
from phonoflow.io.structure_io import write_structure
from phonoflow.workflow.pipeline import run_single_workflow


def test_single_pipeline_creates_expected_outputs(tmp_path: Path):
    input_path = tmp_path / "Si.vasp"
    outdir = tmp_path / "result"
    atoms = Atoms(
        "Si2",
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=[5.43, 5.43, 5.43],
        pbc=True,
    )
    write_structure(atoms, input_path)

    config = WorkflowConfig(input_path=input_path, outdir=outdir, backend="dummy", relax=True)
    result = run_single_workflow(config)

    assert result["status"] == "success"
    assert (outdir / "resolved_config.yaml").exists()
    assert (outdir / "input_structure.vasp").exists()
    assert (outdir / "relaxed.vasp").exists()
    assert (outdir / "stability_report.json").exists()
    assert (outdir / "stability_report.txt").exists()
