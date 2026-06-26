from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py


def _load_validator():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_output.py"
    spec = importlib.util.spec_from_file_location("validate_output", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validate_output_passes_for_minimal_fake_real_output(tmp_path: Path):
    validator = _load_validator()
    outdir = tmp_path / "out"
    outdir.mkdir()
    for filename in validator.REQUIRED_FILES:
        (outdir / filename).write_text("placeholder\n", encoding="utf-8")

    result = _fake_result(outdir, dos=True)
    (outdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with h5py.File(outdir / "force_constants.hdf5", "w") as handle:
        handle.create_dataset("force_constants", data=[0.0])
    (outdir / "band.yaml").write_text("phonon: []\n", encoding="utf-8")
    (outdir / "phonon_band.png").write_bytes(b"x" * 2048)
    _write_fake_band_exports(outdir)
    _write_fake_settings_exports(outdir)
    _write_fake_spacegroup_report(outdir)
    (outdir / "phonon_dos.dat").write_text("0.0 0.0\n", encoding="utf-8")
    (outdir / "phonon_dos.png").write_bytes(b"x" * 2048)

    report = validator.validate_output(outdir)
    assert report["passed"] is True


def test_validate_output_does_not_require_dos_when_disabled(tmp_path: Path):
    validator = _load_validator()
    outdir = tmp_path / "out"
    outdir.mkdir()
    for filename in validator.REQUIRED_FILES:
        (outdir / filename).write_text("placeholder\n", encoding="utf-8")
    result = _fake_result(outdir, dos=False)
    (outdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with h5py.File(outdir / "force_constants.hdf5", "w") as handle:
        handle.create_dataset("force_constants", data=[0.0])
    (outdir / "band.yaml").write_text("phonon: []\n", encoding="utf-8")
    (outdir / "phonon_band.png").write_bytes(b"x" * 2048)
    _write_fake_band_exports(outdir)
    _write_fake_settings_exports(outdir)
    _write_fake_spacegroup_report(outdir)
    report = validator.validate_output(outdir)
    assert report["passed"] is True


def test_validate_output_fails_for_missing_file(tmp_path: Path):
    validator = _load_validator()
    outdir = tmp_path / "out"
    outdir.mkdir()

    report = validator.validate_output(outdir)
    assert report["passed"] is False


def test_validate_output_checks_available_thermal_files(tmp_path: Path):
    validator = _load_validator()
    outdir = tmp_path / "out"
    outdir.mkdir()
    for filename in validator.REQUIRED_FILES:
        (outdir / filename).write_text("placeholder\n", encoding="utf-8")

    result = _fake_result(outdir, dos=True)
    result["thermal_conductivity"] = {
        "enabled": True,
        "available": True,
        "files": {
            "fc3_hdf5": "fc3.hdf5",
            "kappa_hdf5": "kappa-m111.hdf5",
            "thermal_conductivity_csv": "thermal_conductivity.csv",
            "thermal_conductivity_png": "thermal_conductivity.png",
        },
        "lifetime": {
            "available": True,
            "data_file": "phonon_lifetime.csv",
            "plot_file": "phonon_lifetime.png",
            "unit": "ps",
        },
    }
    (outdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with h5py.File(outdir / "force_constants.hdf5", "w") as handle:
        handle.create_dataset("force_constants", data=[0.0])
    _write_fake_band_exports(outdir)
    _write_fake_settings_exports(outdir)
    _write_fake_spacegroup_report(outdir)
    for filename in [
        "band.yaml",
        "phonon_dos.dat",
        "phonon_dos.png",
        "phonon_band.png",
        "fc3.hdf5",
        "kappa-m111.hdf5",
        "thermal_conductivity.csv",
        "thermal_conductivity.png",
        "phonon_lifetime.csv",
        "phonon_lifetime.png",
    ]:
        (outdir / filename).write_bytes(b"x" * 2048)

    report = validator.validate_output(outdir)

    assert report["passed"] is True


def _fake_result(outdir: Path, dos: bool) -> dict:
    output_files = {
        "band_plot": "phonon_band.png",
        "band_data": "phonon_band.dat",
        "band_csv": "phonon_band.csv",
        "band_long_csv": "phonon_band_long.csv",
        "band_segments": "phonon_band_segments.json",
        "band_metadata": "phonon_band_metadata.json",
        "band_path": "band_path.json",
        "summary": "summary.txt",
        "resolved_settings_json": "resolved_settings.json",
        "resolved_settings_yaml": "resolved_settings.yaml",
        "run_command": "run_command.txt",
        "spacegroup_report_json": "spacegroup_report.json",
        "spacegroup_report_txt": "spacegroup_report.txt",
    }
    if dos:
        output_files.update({"dos_data": "phonon_dos.dat", "dos_plot": "phonon_dos.png"})
    return {
        "project": "PhonoFlow",
        "version": "0.2.0a9",
        "backend": "auto",
        "backend_resolved": "calorine",
        "success": True,
        "output_directory": str(outdir),
        "structure_formula": "Si2",
        "n_atoms_unitcell": 2,
        "structure_type": "bulk",
        "structure_classification": {
            "structure_type": "bulk",
            "classification_confidence": "heuristic",
            "vacuum_like_directions": [],
            "atom_extents": [1.0, 1.0, 1.0],
            "cell_lengths": [5.0, 5.0, 5.0],
        },
        "vacuum_like_directions": [],
        "atom_extents": [1.0, 1.0, 1.0],
        "cell_lengths": [5.0, 5.0, 5.0],
        "supercell_dim_resolved": [2, 2, 2],
        "supercell_dim_requested": "auto",
        "target_supercell_length": 20.0,
        "min_supercell_dim": 1,
        "max_supercell_dim": 6,
        "max_supercell_atoms": 1000,
        "n_atoms_supercell": 16,
        "supercell_lengths_resolved": [10.0, 10.0, 10.0],
        "auto_supercell_warnings": [],
        "relax": True,
        "relax_cell": True,
        "relax_mode": "cell",
        "constant_cell": False,
        "fmax": 1e-5,
        "max_steps": 2000,
        "optimizer": "FIRE",
        "initial_cell_lengths": [5.0, 5.0, 5.0],
        "final_cell_lengths": [5.0, 5.0, 5.0],
        "initial_cell_angles": [90.0, 90.0, 90.0],
        "final_cell_angles": [90.0, 90.0, 90.0],
        "initial_volume": 125.0,
        "final_volume": 125.0,
        "volume_change_percent": 0.0,
        "relax_converged": True,
        "final_max_force_eV_per_A": 0.001,
        "final_stress_GPa": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "relax_warnings": [],
        "initial_spacegroup": _fake_spacegroup(),
        "final_spacegroup": _fake_spacegroup(),
        "spacegroup_changed": False,
        "spacegroup_change_summary": "Space group preserved: Fm-3m (No. 225).",
        "spacegroup_report_json": "spacegroup_report.json",
        "spacegroup_report_txt": "spacegroup_report.txt",
        "symprec": 1e-3,
        "angle_tolerance": -1.0,
        "n_displaced_supercells": 1,
        "minimum_frequency_THz": 0.0,
        "maximum_frequency_THz": 15.0,
        "has_imaginary_frequency": False,
        "settings_summary": {"backend_resolved": {"value": "calorine", "source": "auto", "note": ""}},
        "input_file_hash": "a" * 64,
        "model_file_hash": "b" * 64,
        "software_versions": {"PhonoFlow": "0.2.0a9"},
        "group_velocity": {
            "available": False,
            "reason": "not generated in fake output",
            "data_file": None,
            "plot_file": None,
        },
        "thermal_conductivity": {
            "enabled": False,
            "available": False,
            "reason": "Thermal conductivity calculation was not requested.",
        },
        "dos": dos,
        "output_files": output_files,
    }


def _write_fake_band_exports(outdir: Path) -> None:
    (outdir / "phonon_band.dat").write_text(
        "# segment 0: Gamma -> X\n0.0 0.0 1.0\n\n", encoding="utf-8"
    )
    (outdir / "phonon_band.csv").write_text(
        "segment_index,q_index_global,q_index_in_segment,distance,qx,qy,qz,branch_1_THz\n"
        "0,0,0,0.0,0,0,0,0.0\n",
        encoding="utf-8",
    )
    (outdir / "phonon_band_long.csv").write_text(
        "segment_index,q_index_global,q_index_in_segment,distance,qx,qy,qz,branch_index,frequency_THz\n"
        "0,0,0,0.0,0,0,0,1,0.0\n",
        encoding="utf-8",
    )
    (outdir / "phonon_band_segments.json").write_text(
        json.dumps({"segments": [{"segment_index": 0, "start_label": "Gamma", "end_label": "X"}]}),
        encoding="utf-8",
    )
    (outdir / "phonon_band_metadata.json").write_text(
        json.dumps({"tick_positions": [0.0, 1.0], "tick_labels": ["Γ", "X"]}),
        encoding="utf-8",
    )
    (outdir / "band_path.json").write_text(
        json.dumps({"path_source": "fallback", "segments": [["Gamma", "X"]]}),
        encoding="utf-8",
    )


def _write_fake_settings_exports(outdir: Path) -> None:
    settings = {"backend_resolved": {"value": "calorine", "source": "auto", "note": ""}}
    (outdir / "resolved_settings.json").write_text(json.dumps(settings), encoding="utf-8")
    (outdir / "resolved_settings.yaml").write_text("backend_resolved:\n  value: calorine\n", encoding="utf-8")
    (outdir / "run_command.txt").write_text("python -m phonoflow run\n", encoding="utf-8")


def _fake_spacegroup() -> dict:
    return {
        "spacegroup_number": 225,
        "international_symbol": "Fm-3m",
        "hall_symbol": "-F 4 2 3",
        "pointgroup": "m-3m",
        "crystal_system": "cubic",
        "symprec": 1e-3,
        "angle_tolerance": -1.0,
        "dataset_available": True,
        "error": None,
    }


def _write_fake_spacegroup_report(outdir: Path) -> None:
    report = {
        "symprec": 1e-3,
        "angle_tolerance": -1.0,
        "initial": _fake_spacegroup(),
        "final": _fake_spacegroup(),
        "changed": False,
        "change_summary": "Space group preserved: Fm-3m (No. 225).",
        "warnings": [],
    }
    (outdir / "spacegroup_report.json").write_text(json.dumps(report), encoding="utf-8")
    (outdir / "spacegroup_report.txt").write_text("Space Group Report\n", encoding="utf-8")
