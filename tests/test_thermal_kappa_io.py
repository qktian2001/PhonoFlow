from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np

from phonoflow.thermal.config import disabled_thermal_result, unavailable_thermal_result
from phonoflow.thermal.kappa_io import (
    extract_lifetime_from_hdf5,
    inspect_kappa_hdf5,
    parse_kappa_hdf5,
    summarize_kappa,
    write_thermal_conductivity_csv,
)
from phonoflow.thermal.plots import plot_thermal_conductivity


def test_disabled_thermal_result_schema() -> None:
    result = disabled_thermal_result()

    assert result["enabled"] is False
    assert result["available"] is False
    assert "not requested" in result["reason"]


def test_unavailable_thermal_result_schema() -> None:
    result = unavailable_thermal_result(
        enabled=True,
        reason="phono3py failed",
        warnings=["experimental"],
        fc3_method="finite-displacement",
        kappa_method="rta",
    )

    assert result["enabled"] is True
    assert result["available"] is False
    assert result["fc3_method"] == "finite-displacement"
    assert result["kappa_method"] == "rta"
    assert result["warnings"] == ["experimental"]


def test_parse_fake_kappa_hdf5_and_write_outputs(tmp_path: Path) -> None:
    kappa_path = tmp_path / "kappa-m111.hdf5"
    with h5py.File(kappa_path, "w") as handle:
        handle.create_dataset("temperature", data=np.array([300.0, 600.0]))
        handle.create_dataset(
            "kappa",
            data=np.array(
                [
                    [10.0, 11.0, 12.0, 1.0, 2.0, 3.0],
                    [5.0, 6.0, 7.0, 0.5, 0.6, 0.7],
                ]
            ),
        )

    parsed = parse_kappa_hdf5(kappa_path)
    rows = parsed["rows"]
    summary = summarize_kappa(rows)
    csv_path = write_thermal_conductivity_csv(rows, tmp_path / "thermal_conductivity.csv")
    png_path = plot_thermal_conductivity(rows, tmp_path / "thermal_conductivity.png", dpi=80)

    assert rows[0]["temperature_K"] == 300.0
    assert rows[0]["kxx"] == 10.0
    assert rows[0]["kyz"] == 1.0
    assert rows[0]["kxz"] == 2.0
    assert rows[0]["kxy"] == 3.0
    assert rows[0]["kappa_trace_over_3"] == 11.0
    assert summary["300"]["kappa_trace_over_3"] == 11.0
    assert csv_path.stat().st_size > 0
    assert png_path.stat().st_size > 0
    with csv_path.open("r", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))[0]["temperature_K"] == "300.0000000000"


def test_lifetime_missing_dataset_is_unavailable(tmp_path: Path) -> None:
    kappa_path = tmp_path / "kappa-m111.hdf5"
    with h5py.File(kappa_path, "w") as handle:
        handle.create_dataset("temperature", data=np.array([300.0]))

    result = extract_lifetime_from_hdf5(kappa_path, tmp_path, dpi=80)

    assert result["available"] is False
    assert "No lifetime or gamma dataset" in result["reason"]


def test_lifetime_dataset_writes_csv_and_plot(tmp_path: Path) -> None:
    kappa_path = tmp_path / "kappa-m111.hdf5"
    with h5py.File(kappa_path, "w") as handle:
        handle.create_dataset("temperature", data=np.array([300.0]))
        handle.create_dataset("frequency", data=np.array([[0.0, 1.0], [2.0, 3.0]]))
        handle.create_dataset("lifetime", data=np.array([[[1.0, 2.0], [3.0, 4.0]]]))

    result = extract_lifetime_from_hdf5(kappa_path, tmp_path, dpi=80)

    assert result["available"] is True
    assert result["data_file"] == "phonon_lifetime.csv"
    assert result["plot_file"] == "phonon_lifetime.png"
    assert result["diagnostics_file"] == "phonon_lifetime_diagnostics.json"
    assert result["unit"] == "ps"
    assert result["mean_lifetime_ps"] == 2.5
    assert result["max_lifetime_ps"] == 4.0
    assert (tmp_path / "phonon_lifetime.csv").stat().st_size > 0
    assert (tmp_path / "phonon_lifetime.png").stat().st_size > 0
    assert (tmp_path / "phonon_lifetime_diagnostics.json").stat().st_size > 0
    diagnostics = json.loads((tmp_path / "phonon_lifetime_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["source_file"] == "kappa-m111.hdf5"
    assert diagnostics["unit"] == "ps"
    assert diagnostics["n_nan"] == 0
    json.dumps(result)


def test_gamma_dataset_converts_to_lifetime_ps(tmp_path: Path) -> None:
    kappa_path = tmp_path / "kappa-m111.hdf5"
    gamma = np.array([[[0.25, 0.5], [1.0, 2.0]]])
    with h5py.File(kappa_path, "w") as handle:
        handle.create_dataset("temperature", data=np.array([300.0]))
        handle.create_dataset("frequency", data=np.array([[1.0, 2.0], [3.0, 4.0]]))
        handle.create_dataset("gamma", data=gamma)

    result = extract_lifetime_from_hdf5(kappa_path, tmp_path, dpi=80)

    assert result["available"] is True
    assert result["source"] == "gamma"
    assert result["unit"] == "ps"
    assert result["warnings"]
    with (tmp_path / "phonon_lifetime.csv").open("r", encoding="utf-8") as handle:
        row = list(csv.DictReader(handle))[0]
    assert abs(float(row["lifetime_ps"]) - 1.0 / (4.0 * np.pi * 0.25)) < 1e-12
    assert row["gamma_or_linewidth_raw"] == "0.25"


def test_lifetime_parsing_ignores_non_contiguous_grid_point_ids(tmp_path: Path) -> None:
    kappa_path = tmp_path / "kappa-m222.hdf5"
    gamma = np.array([[[0.25, 0.5], [1.0, 2.0]]])
    with h5py.File(kappa_path, "w") as handle:
        handle.create_dataset("temperature", data=np.array([300.0]))
        handle.create_dataset("grid_point", data=np.array([0, 68]))
        handle.create_dataset("frequency", data=np.array([[1.0, 2.0], [3.0, 4.0]]))
        handle.create_dataset("gamma", data=gamma)

    result = extract_lifetime_from_hdf5(kappa_path, tmp_path, dpi=80)

    assert result["available"] is True
    with (tmp_path / "phonon_lifetime.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["q_index"] for row in rows] == ["0", "0", "1", "1"]
    assert result["n_points"] == 4
    fields = inspect_kappa_hdf5(kappa_path)
    assert "grid_point" in fields["fields_found"]
    assert fields["fields"]["gamma"]["shape"] == [1, 2, 2]


def test_single_temperature_thermal_plot_is_bar(tmp_path: Path) -> None:
    rows = [
        {
            "temperature_K": 300.0,
            "kxx": 1.0,
            "kyy": 2.0,
            "kzz": 3.0,
            "kxy": 0.0,
            "kyz": 0.0,
            "kxz": 0.0,
            "kappa_trace_over_3": 2.0,
        }
    ]

    png_path = plot_thermal_conductivity(rows, tmp_path / "single_temperature.png", dpi=80)

    assert png_path.stat().st_size > 0
