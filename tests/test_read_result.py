import json
from pathlib import Path

from typer.testing import CliRunner

from phonoflow.cli import app


def _write_result(path: Path) -> Path:
    result_path = path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "project": "PhonoFlow",
                "version": "0.2.0a9",
                "backend_resolved": "calorine",
                "backend_requested": "auto",
                "structure_formula": "Si2",
                "n_atoms_unitcell": 2,
                "target_supercell_length": 20.0,
                "supercell_dim_resolved": [2, 2, 2],
                "supercell_lengths_resolved": [10.0, 10.0, 10.0],
                "n_atoms_supercell": 16,
                "primitive_matrix_resolved": "P",
                "mesh_resolved": [20, 20, 20],
                "n_displaced_supercells": 1,
                "relax_converged": True,
                "final_max_force_eV_per_A": 0.001,
                "initial_spacegroup": {"spacegroup_number": 225, "international_symbol": "Fm-3m"},
                "final_spacegroup": {"spacegroup_number": 225, "international_symbol": "Fm-3m"},
                "spacegroup_changed": False,
                "spacegroup_change_summary": "Space group preserved.",
                "symprec": 1e-3,
                "angle_tolerance": -1.0,
                "minimum_frequency_THz": 0.0,
                "maximum_frequency_THz": 15.0,
                "has_imaginary_frequency": False,
                "export_fc2_text": True,
                "force_constants_text_exported": True,
                "phonopy_force_constants_file": "FORCE_CONSTANTS",
                "shengbte_fc2_file": "FORCE_CONSTANTS_2ND",
                "input_file_hash": "a" * 64,
                "model_file_hash": "b" * 64,
                "elapsed_time_seconds": 1.23,
                "warnings": [],
                "dos": True,
                "thermal_conductivity": {
                    "enabled": True,
                    "available": True,
                    "fc3_method": "finite-displacement",
                    "kappa_method": "rta",
                    "files": {
                        "thermal_conductivity_csv": "thermal_conductivity.csv",
                        "thermal_conductivity_png": "thermal_conductivity.png",
                    },
                    "summary": {"300": {"kappa_trace_over_3": 1.0}},
                    "kappa_unit": "W/m-K",
                    "lifetime": {
                        "available": True,
                        "data_file": "phonon_lifetime.csv",
                        "plot_file": "phonon_lifetime.png",
                        "unit": "ps",
                        "mean_lifetime_ps": 1.5,
                        "max_lifetime_ps": 3.0,
                    },
                },
                "output_directory": str(path),
            }
        ),
        encoding="utf-8",
    )
    return result_path


def test_read_result_accepts_directory(tmp_path: Path):
    runner = CliRunner()
    result_path = _write_result(tmp_path)
    result = runner.invoke(app, ["read-result", str(tmp_path)])
    assert result.exit_code == 0
    assert "calorine" in result.stdout
    assert result_path.name in result.stdout


def test_read_result_accepts_file(tmp_path: Path):
    runner = CliRunner()
    result_path = _write_result(tmp_path)
    result = runner.invoke(app, ["read-result", str(result_path)])
    assert result.exit_code == 0
    assert "Si2" in result.stdout


def test_read_result_displays_spacegroup_fields(tmp_path: Path):
    runner = CliRunner()
    result_path = _write_result(tmp_path)
    result = runner.invoke(app, ["read-result", str(result_path)])
    assert result.exit_code == 0
    assert "Fm-3m" in result.stdout
    assert "target_supercell_length" in result.stdout


def test_read_result_json_output(tmp_path: Path):
    runner = CliRunner()
    result_path = _write_result(tmp_path)
    result = runner.invoke(app, ["read-result", str(result_path), "--json"])
    assert result.exit_code == 0
    assert '"backend_resolved": "calorine"' in result.stdout


def test_read_result_displays_thermal_conductivity(tmp_path: Path):
    runner = CliRunner()
    result_path = _write_result(tmp_path)
    result = runner.invoke(app, ["read-result", str(result_path)])
    assert result.exit_code == 0
    assert "thermal_conductivity" in result.stdout
    assert "thermal_conductivity.csv" in result.stdout
    assert "phonon_lifetime.csv" in result.stdout
