"""Validate PhonoFlow single-workflow output files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import h5py
import yaml


REQUIRED_FILES = [
    "resolved_config.yaml",
    "resolved_settings.json",
    "resolved_settings.yaml",
    "run_command.txt",
    "input_structure.vasp",
    "relaxed.vasp",
    "relax.log",
    "force_constants.hdf5",
    "phonopy.yaml",
    "band.yaml",
    "phonon_band.png",
    "phonon_band.dat",
    "phonon_band.csv",
    "phonon_band_long.csv",
    "phonon_band_segments.json",
    "phonon_band_metadata.json",
    "band_path.json",
    "result.json",
    "stability_report.json",
    "stability_report.txt",
    "spacegroup_report.json",
    "spacegroup_report.txt",
    "summary.txt",
    "run.log",
]

REQUIRED_RESULT_FIELDS = [
    "project",
    "version",
    "backend",
    "backend_resolved",
    "success",
    "output_directory",
    "structure_formula",
    "n_atoms_unitcell",
    "supercell_dim_resolved",
    "relax_converged",
    "final_max_force_eV_per_A",
    "n_displaced_supercells",
    "minimum_frequency_THz",
    "maximum_frequency_THz",
    "has_imaginary_frequency",
    "output_files",
    "settings_summary",
    "input_file_hash",
    "model_file_hash",
    "software_versions",
    "structure_type",
    "structure_classification",
    "vacuum_like_directions",
    "atom_extents",
    "cell_lengths",
    "relax",
    "relax_cell",
    "relax_mode",
    "constant_cell",
    "fmax",
    "max_steps",
    "optimizer",
    "initial_cell_lengths",
    "final_cell_lengths",
    "initial_cell_angles",
    "final_cell_angles",
    "initial_volume",
    "final_volume",
    "volume_change_percent",
    "final_stress_GPa",
    "relax_warnings",
    "target_supercell_length",
    "min_supercell_dim",
    "max_supercell_dim",
    "max_supercell_atoms",
    "n_atoms_supercell",
    "supercell_lengths_resolved",
    "auto_supercell_warnings",
    "initial_spacegroup",
    "final_spacegroup",
    "spacegroup_changed",
    "spacegroup_change_summary",
    "spacegroup_report_json",
    "spacegroup_report_txt",
    "symprec",
    "angle_tolerance",
    "group_velocity",
    "thermal_conductivity",
]


def validate_output(outdir: Path) -> dict[str, Any]:
    """Validate a PhonoFlow real single-workflow output directory."""

    outdir = Path(outdir)
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, message: str) -> None:
        checks.append({"name": name, "passed": passed, "message": message})

    if not outdir.is_dir():
        add("outdir", False, f"Output directory does not exist: {outdir}")
        return {"passed": False, "checks": checks}

    for filename in REQUIRED_FILES:
        path = outdir / filename
        exists = path.exists()
        nonzero = exists and path.stat().st_size > 0
        add(f"file:{filename}", nonzero, "exists and non-empty" if nonzero else "missing or empty")

    result_path = outdir / "result.json"
    result: dict[str, Any] = {}
    if result_path.exists() and result_path.stat().st_size > 0:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            add("result.json:parse", True, "valid JSON")
        except Exception as exc:
            add("result.json:parse", False, f"invalid JSON: {exc}")
    else:
        add("result.json:parse", False, "result.json missing or empty")

    if result:
        add("result.json:success", result.get("success") is True, f"success={result.get('success')!r}")
        for field in REQUIRED_RESULT_FIELDS:
            add(
                f"result.json:{field}",
                field in result,
                "present" if field in result else "missing",
            )
        output_files = result.get("output_files", {})
        if isinstance(output_files, dict):
            for label, filename in output_files.items():
                path = outdir / str(filename)
                add(
                    f"output_files:{label}",
                    path.exists() and path.stat().st_size > 0,
                    str(filename) if path.exists() and path.stat().st_size > 0 else f"missing or empty: {filename}",
                )
        _validate_settings_exports(outdir, result, add)
        _validate_supercell_fields(result, add)
        _validate_spacegroup_report(outdir, result, add)
        _validate_fc2_text_exports(outdir, result, add)
        _validate_group_velocity_exports(outdir, result, add)
        _validate_thermal_conductivity_exports(outdir, result, add)

    hdf5_path = outdir / "force_constants.hdf5"
    if hdf5_path.exists() and hdf5_path.stat().st_size > 0:
        try:
            with h5py.File(hdf5_path, "r") as handle:
                add("force_constants.hdf5:open", True, f"datasets={list(handle.keys())}")
        except Exception as exc:
            add("force_constants.hdf5:open", False, f"could not open: {exc}")

    band_yaml_path = outdir / "band.yaml"
    if band_yaml_path.exists() and band_yaml_path.stat().st_size > 0:
        try:
            with band_yaml_path.open("r", encoding="utf-8") as handle:
                band_data = yaml.safe_load(handle)
            add("band.yaml:read", band_data is not None, "readable YAML")
        except Exception as exc:
            add("band.yaml:read", False, f"could not read: {exc}")

    png_path = outdir / "phonon_band.png"
    add(
        "phonon_band.png:size",
        png_path.exists() and png_path.stat().st_size > 1024,
        ">1 KB" if png_path.exists() and png_path.stat().st_size > 1024 else "missing, empty, or <=1 KB",
    )

    _validate_band_exports(outdir, add)

    if result.get("dos") is True:
        for filename in ["phonon_dos.dat", "phonon_dos.png"]:
            path = outdir / filename
            add(
                f"dos:{filename}",
                path.exists() and path.stat().st_size > 0,
                "exists and non-empty" if path.exists() and path.stat().st_size > 0 else "missing or empty",
            )

    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _validate_settings_exports(outdir: Path, result: dict[str, Any], add: Any) -> None:
    add("result.json:settings_summary", isinstance(result.get("settings_summary"), dict), "settings_summary present")
    for field in ["input_file_hash", "model_file_hash"]:
        value = result.get(field)
        add(f"result.json:{field}", isinstance(value, str) and len(value) == 64, "SHA256 present")
    versions = result.get("software_versions", {})
    add("result.json:software_versions", isinstance(versions, dict) and bool(versions), "software_versions present")
    for filename in ["resolved_settings.json", "resolved_settings.yaml", "run_command.txt"]:
        path = outdir / filename
        add(
            f"settings:{filename}",
            path.exists() and path.stat().st_size > 0,
            "exists and non-empty" if path.exists() and path.stat().st_size > 0 else "missing or empty",
        )
    settings_json = outdir / "resolved_settings.json"
    if settings_json.exists() and settings_json.stat().st_size > 0:
        try:
            data = json.loads(settings_json.read_text(encoding="utf-8"))
            add("resolved_settings.json:parse", isinstance(data, dict), "valid JSON mapping")
        except Exception as exc:
            add("resolved_settings.json:parse", False, f"invalid JSON: {exc}")
    settings_yaml = outdir / "resolved_settings.yaml"
    if settings_yaml.exists() and settings_yaml.stat().st_size > 0:
        try:
            data = yaml.safe_load(settings_yaml.read_text(encoding="utf-8"))
            add("resolved_settings.yaml:parse", isinstance(data, dict), "valid YAML mapping")
        except Exception as exc:
            add("resolved_settings.yaml:parse", False, f"invalid YAML: {exc}")


def _validate_supercell_fields(result: dict[str, Any], add: Any) -> None:
    if result.get("supercell_dim_requested") != "auto":
        return
    for field in ["target_supercell_length", "supercell_dim_resolved", "n_atoms_supercell"]:
        add(f"supercell:{field}", field in result and result.get(field) is not None, "present")
    target = result.get("target_supercell_length")
    add(
        "supercell:target_supercell_length",
        isinstance(target, (int, float)) and target > 0,
        f"target={target}",
    )


def _validate_spacegroup_report(outdir: Path, result: dict[str, Any], add: Any) -> None:
    for filename in ["spacegroup_report.json", "spacegroup_report.txt"]:
        path = outdir / filename
        add(
            f"spacegroup:{filename}",
            path.exists() and path.stat().st_size > 0,
            "exists and non-empty" if path.exists() and path.stat().st_size > 0 else "missing or empty",
        )
    report_path = outdir / "spacegroup_report.json"
    if report_path.exists() and report_path.stat().st_size > 0:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            add("spacegroup_report.json:parse", isinstance(report, dict), "valid JSON")
            for field in ["initial", "final", "changed"]:
                add(
                    f"spacegroup_report.json:{field}",
                    field in report,
                    "present" if field in report else "missing",
                )
        except Exception as exc:
            add("spacegroup_report.json:parse", False, f"invalid JSON: {exc}")
    for field in ["initial_spacegroup", "final_spacegroup", "spacegroup_changed"]:
        add(
            f"result.json:{field}",
            field in result,
            "present" if field in result else "missing",
        )


def _validate_band_exports(outdir: Path, add: Any) -> None:
    csv_path = outdir / "phonon_band.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        add(
            "phonon_band.csv:columns",
            "distance" in header and any(column.startswith("branch_") for column in header),
            "contains distance and branch columns",
        )

    long_csv_path = outdir / "phonon_band_long.csv"
    if long_csv_path.exists() and long_csv_path.stat().st_size > 0:
        header = long_csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        add(
            "phonon_band_long.csv:columns",
            "distance" in header and "frequency_THz" in header,
            "contains distance and frequency_THz columns",
        )


def _validate_group_velocity_exports(outdir: Path, result: dict[str, Any], add: Any) -> None:
    group_velocity = result.get("group_velocity")
    add("group_velocity:schema", isinstance(group_velocity, dict), "present" if isinstance(group_velocity, dict) else "missing")
    if not isinstance(group_velocity, dict) or group_velocity.get("available") is not True:
        return
    for field in ["data_file", "plot_file"]:
        filename = group_velocity.get(field)
        path = outdir / str(filename)
        add(
            f"group_velocity:{field}",
            bool(filename) and path.exists() and path.stat().st_size > 0,
            str(filename) if bool(filename) and path.exists() and path.stat().st_size > 0 else f"missing or empty: {filename}",
        )

    for filename in ["phonon_band_segments.json", "band_path.json"]:
        path = outdir / filename
        if path.exists() and path.stat().st_size > 0:
            try:
                json.loads(path.read_text(encoding="utf-8"))
                add(f"{filename}:parse", True, "valid JSON")
            except Exception as exc:
                add(f"{filename}:parse", False, f"invalid JSON: {exc}")

    metadata_path = outdir / "phonon_band_metadata.json"
    if metadata_path.exists() and metadata_path.stat().st_size > 0:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            tick_labels = metadata.get("tick_labels", [])
            has_gamma = any("Gamma" in str(label) or "Γ" in str(label) for label in tick_labels)
            add("phonon_band_metadata.json:parse", True, "valid JSON")
            add("phonon_band_metadata.json:tick_labels", bool(tick_labels), "tick_labels present")
            add("phonon_band_metadata.json:gamma", has_gamma, "tick_labels include Gamma")
        except Exception as exc:
            add("phonon_band_metadata.json:parse", False, f"invalid JSON: {exc}")


def _validate_fc2_text_exports(outdir: Path, result: dict[str, Any], add: Any) -> None:
    should_check = result.get("export_fc2_text") is True or result.get("force_constants_text_exported") is True
    if not should_check:
        return

    output_files = result.get("output_files", {})
    expected = {
        "shengbte_force_constants_2nd": output_files.get(
            "shengbte_force_constants_2nd", "FORCE_CONSTANTS_2ND"
        ),
    }
    duplicate = outdir / "FORCE_CONSTANTS"
    add(
        "fc2_text:duplicate_FORCE_CONSTANTS_absent",
        not duplicate.exists(),
        "duplicate FORCE_CONSTANTS absent" if not duplicate.exists() else "duplicate FORCE_CONSTANTS present",
    )
    for label, filename in expected.items():
        path = outdir / str(filename)
        exists = path.exists() and path.stat().st_size > 0
        add(f"fc2_text:{label}", exists, "exists and non-empty" if exists else f"missing or empty: {filename}")
        if exists:
            _validate_force_constants_first_line(path, result, add)


def _validate_thermal_conductivity_exports(outdir: Path, result: dict[str, Any], add: Any) -> None:
    thermal = result.get("thermal_conductivity")
    add(
        "thermal_conductivity:schema",
        isinstance(thermal, dict),
        "present" if isinstance(thermal, dict) else "missing",
    )
    if not isinstance(thermal, dict):
        return
    if thermal.get("enabled") is not True:
        return
    if thermal.get("available") is not True:
        add("thermal_conductivity:available", True, f"not available: {thermal.get('reason')}")
        return

    files = thermal.get("files") or {}
    for field in ["fc3_hdf5", "kappa_hdf5", "thermal_conductivity_csv", "thermal_conductivity_png"]:
        filename = files.get(field)
        path = outdir / str(filename)
        add(
            f"thermal_conductivity:{field}",
            bool(filename) and path.exists() and path.stat().st_size > 0,
            str(filename) if bool(filename) and path.exists() and path.stat().st_size > 0 else f"missing or empty: {filename}",
        )

    lifetime = thermal.get("lifetime") or {}
    if isinstance(lifetime, dict) and lifetime.get("available") is True:
        for field in ["data_file", "plot_file"]:
            filename = lifetime.get(field)
            path = outdir / str(filename)
            add(
                f"phonon_lifetime:{field}",
                bool(filename) and path.exists() and path.stat().st_size > 0,
                str(filename) if bool(filename) and path.exists() and path.stat().st_size > 0 else f"missing or empty: {filename}",
            )


def _validate_force_constants_first_line(path: Path, result: dict[str, Any], add: Any) -> None:
    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0].strip()
        n_atoms = int(first_line.split()[0])
    except Exception as exc:
        add(f"{path.name}:first_line", False, f"first line is not an integer: {exc}")
        return

    expected_atoms = result.get("n_atoms_supercell")
    if expected_atoms is None:
        add(f"{path.name}:first_line", True, f"integer atom count={n_atoms}")
    else:
        add(
            f"{path.name}:first_line",
            n_atoms == int(expected_atoms),
            f"atom count={n_atoms}, expected={expected_atoms}",
        )


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""

    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("Usage: python scripts/validate_output.py <outdir>")
        return 2

    report = validate_output(Path(args[0]))
    status = "PASS" if report["passed"] else "FAIL"
    print(f"PhonoFlow output validation: {status}")
    print(f"{'Status':<8} {'Check':<40} Details")
    print("-" * 80)
    for check in report["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        print(f"{marker:<8} {check['name']:<40} {check['message']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
