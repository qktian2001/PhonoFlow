from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phonoflow.analysis.postprocessing_diagnostics import (
    write_band_diagnostics,
    write_dos_diagnostics,
    write_lifetime_diagnostics,
)
from phonoflow.band.data import BandData, BandSegment


def test_band_diagnostics_fields(tmp_path: Path) -> None:
    band_yaml = tmp_path / "band.yaml"
    band_png = tmp_path / "phonon_band.png"
    band_yaml.write_text("phonon: []\n", encoding="utf-8")
    band_png.write_bytes(b"x" * 2048)
    band_data = BandData(
        segments=[
            BandSegment(
                index=0,
                start_label="Γ",
                end_label="X",
                distances=np.array([0.0, 1.0]),
                qpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
                frequencies=np.array([[0.0, 1.0], [2.0, 3.0]]),
            )
        ],
        tick_positions=[0.0, 1.0],
        tick_labels=["Γ", "X"],
    )

    diagnostics = write_band_diagnostics(
        band_data=band_data,
        source_file=band_yaml,
        plot_file=band_png,
        output_path=tmp_path / "phonon_band_diagnostics.json",
        high_symmetry_path={"display": "Γ — X"},
    )

    payload = json.loads((tmp_path / "phonon_band_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["diagnostics_file"] == "phonon_band_diagnostics.json"
    assert payload["source_file"] == "band.yaml"
    assert payload["generated_by"] == "phonopy band structure"
    assert payload["n_qpoints"] == 2
    assert payload["n_branches"] == 2
    assert payload["min_frequency_THz"] == 0.0
    assert payload["max_frequency_THz"] == 3.0
    assert payload["unit"] == "THz"
    assert payload["high_symmetry_path"]["display"] == "Γ — X"


def test_dos_diagnostics_fields(tmp_path: Path) -> None:
    dos_dat = tmp_path / "phonon_dos.dat"
    dos_png = tmp_path / "phonon_dos.png"
    dos_dat.write_text("0.0 0.0\n1.0 2.0\n", encoding="utf-8")
    dos_png.write_bytes(b"x" * 2048)

    diagnostics = write_dos_diagnostics(
        frequencies=np.array([0.0, 1.0, 2.0]),
        total_dos=np.array([0.0, 2.0, 1.0]),
        source_file=dos_dat,
        plot_file=dos_png,
        output_path=tmp_path / "phonon_dos_diagnostics.json",
    )

    payload = json.loads((tmp_path / "phonon_dos_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["diagnostics_file"] == "phonon_dos_diagnostics.json"
    assert payload["source_file"] == "phonon_dos.dat"
    assert payload["generated_by"] == "phonopy total DOS"
    assert payload["n_dos_points"] == 3
    assert payload["frequency_range_THz"] == [0.0, 2.0]
    assert payload["dos_unit"] == "arbitrary units from phonopy total DOS"


def test_lifetime_diagnostics_counts_nan_inf_zero(tmp_path: Path) -> None:
    data_file = tmp_path / "phonon_lifetime.csv"
    plot_file = tmp_path / "phonon_lifetime.png"
    source_file = tmp_path / "kappa-m111.hdf5"
    data_file.write_text("placeholder\n", encoding="utf-8")
    plot_file.write_bytes(b"x" * 2048)
    source_file.write_bytes(b"x" * 2048)
    rows = [
        {"temperature_K": 300.0, "q_index": 0, "branch_index": 0, "frequency_THz": 0.0, "lifetime_ps": 0.0},
        {"temperature_K": 300.0, "q_index": 0, "branch_index": 1, "frequency_THz": 1.0, "lifetime_ps": float("nan")},
        {"temperature_K": 300.0, "q_index": 1, "branch_index": 0, "frequency_THz": 2.0, "lifetime_ps": float("inf")},
        {"temperature_K": 300.0, "q_index": 1, "branch_index": 1, "frequency_THz": 3.0, "lifetime_ps": 5.0},
    ]

    diagnostics = write_lifetime_diagnostics(
        rows=rows,
        source_file=source_file,
        data_file=data_file,
        plot_file=plot_file,
        output_path=tmp_path / "phonon_lifetime_diagnostics.json",
        source="gamma",
        warnings=["converted"],
    )

    payload = json.loads((tmp_path / "phonon_lifetime_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["diagnostics_file"] == "phonon_lifetime_diagnostics.json"
    assert payload["source_file"] == "kappa-m111.hdf5"
    assert payload["n_qpoints"] == 2
    assert payload["n_branches"] == 2
    assert payload["n_nan"] == 1
    assert payload["n_inf"] == 1
    assert payload["n_zero"] == 1
    assert payload["mean_lifetime_ps"] == 5.0
