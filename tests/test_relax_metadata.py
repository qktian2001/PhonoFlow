from pathlib import Path

from ase import Atoms

from phonoflow.config import WorkflowConfig
from phonoflow.io.structure_io import write_structure
from phonoflow.workflow.pipeline import run_single_workflow


def test_relax_warning_for_2d_dry_run(tmp_path: Path):
    input_path = tmp_path / "slab.vasp"
    model_path = tmp_path / "nep.txt"
    outdir = tmp_path / "dry"
    atoms = Atoms(
        "C2",
        positions=[[0.0, 0.0, 10.0], [1.42, 0.0, 10.0]],
        cell=[2.46, 2.46, 20.0],
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
    report = result["report"]
    assert report["structure_type"] in {"2d", "interface_or_slab"}
    assert report["relax"] is True
    assert report["relax_cell"] is True
    assert report["relax_mode"] == "cell"
    assert report["fmax"] == 1e-5
    assert report["max_steps"] == 2000
    assert report["relax_warnings"]


def test_no_relax_cell_dry_run_records_positions_mode(tmp_path: Path):
    input_path = tmp_path / "Si.vasp"
    model_path = tmp_path / "nep.txt"
    outdir = tmp_path / "dry"
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    write_structure(atoms, input_path)
    model_path.write_text("dummy potential placeholder\n", encoding="utf-8")

    result = run_single_workflow(
        WorkflowConfig(
            input_path=input_path,
            model_path=model_path,
            outdir=outdir,
            backend="dummy",
            dry_run=True,
            relax_cell=False,
        )
    )
    report = result["report"]
    assert report["relax"] is True
    assert report["relax_cell"] is False
    assert report["relax_mode"] == "positions"
    assert report["constant_cell"] is True


def test_no_relax_dry_run_records_none_mode(tmp_path: Path):
    input_path = tmp_path / "Si.vasp"
    model_path = tmp_path / "nep.txt"
    outdir = tmp_path / "dry"
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    write_structure(atoms, input_path)
    model_path.write_text("dummy potential placeholder\n", encoding="utf-8")

    result = run_single_workflow(
        WorkflowConfig(
            input_path=input_path,
            model_path=model_path,
            outdir=outdir,
            backend="dummy",
            dry_run=True,
            relax=False,
        )
    )
    report = result["report"]
    assert report["relax"] is False
    assert report["relax_mode"] == "none"
    assert report["constant_cell"] is False
