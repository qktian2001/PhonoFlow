"""Tests for core phonon group velocity postprocessing."""

from __future__ import annotations

import csv
import json

import numpy as np

from phonoflow.analysis.group_velocity import (
    THZ_ANGSTROM_TO_KM_PER_S,
    compute_phonon_group_velocity,
    convert_thz_angstrom_to_km_s,
    write_group_velocity_outputs,
)


def test_group_velocity_missing_input_returns_unavailable(tmp_path):
    result = compute_phonon_group_velocity(output_dir=tmp_path, mesh=[2, 2, 2])

    assert result["available"] is False
    assert "Phonopy object" in result["reason"]


def test_group_velocity_unit_conversion():
    values = np.array([[[10.0, 0.0, -5.0]]])

    converted = convert_thz_angstrom_to_km_s(values)

    assert THZ_ANGSTROM_TO_KM_PER_S == 0.1
    assert converted.tolist() == [[[1.0, 0.0, -0.5]]]


def test_write_group_velocity_outputs_writes_csv_and_scatter(tmp_path):
    frequencies = np.array([[0.0, 1.5], [2.0, -0.2]], dtype=float)
    velocities = np.array(
        [
            [[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]],
            [[0.0, 0.0, 30.0], [4.0, 3.0, 0.0]],
        ],
        dtype=float,
    )

    result = write_group_velocity_outputs(frequencies, velocities, tmp_path, plot=True, dpi=80)

    assert result["available"] is True
    assert result["unit"] == "km/s"
    assert result["data_file"] == "phonon_group_velocity.csv"
    assert result["plot_file"] == "phonon_group_velocity.png"
    assert result["diagnostics_file"] == "phonon_group_velocity_diagnostics.json"
    assert result["n_points"] == 4
    assert result["negative_frequency_points"] == 1
    assert (tmp_path / "phonon_group_velocity.csv").stat().st_size > 0
    assert (tmp_path / "phonon_group_velocity.png").stat().st_size > 0
    assert (tmp_path / "phonon_group_velocity_diagnostics.json").stat().st_size > 0

    with (tmp_path / "phonon_group_velocity.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["frequency_THz"] == "0.0000000000"
    assert rows[0]["vg_abs_km_s"] == "1.0000000000"
    diagnostics = json.loads((tmp_path / "phonon_group_velocity_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["generated_by"] == "phonopy mesh group velocities"
    assert diagnostics["n_qpoints"] == 2
    assert diagnostics["n_branches"] == 2
    assert diagnostics["unit"] == "km/s"
