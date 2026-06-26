from pathlib import Path

import json

from phonoflow.plotting.plot_band import (
    band_data_from_phonopy_dict,
    export_phonon_band_data,
    load_band_yaml_segments,
    plot_phonon_band_from_band_yaml,
)
from phonoflow.analysis.bandpath import BandPath
import numpy as np


def test_plot_phonon_band_from_band_yaml_outputs_nonempty_png(tmp_path: Path):
    band_yaml = _write_test_band_yaml(tmp_path)
    output_png = tmp_path / "phonon_band.png"

    plot_phonon_band_from_band_yaml(band_yaml, output_png)

    assert output_png.exists()
    assert output_png.stat().st_size > 1024


def test_band_yaml_parsing_segments_and_discontinuous_label(tmp_path: Path):
    band_data = load_band_yaml_segments(_write_test_band_yaml(tmp_path))
    assert band_data.n_segments == 2
    assert band_data.n_qpoints == 4
    assert band_data.n_branches == 2
    assert band_data.tick_labels == ["Γ", "X|U", "Γ"]
    assert band_data.segments[0].distances.tolist() == [0.0, 1.0]
    assert band_data.segments[1].distances.tolist() == [1.0, 2.0]


def test_band_data_exports_csv_dat_and_metadata(tmp_path: Path):
    band_data = load_band_yaml_segments(_write_test_band_yaml(tmp_path))
    files = export_phonon_band_data(band_data, tmp_path)

    csv_text = (tmp_path / files["band_csv"]).read_text(encoding="utf-8")
    long_text = (tmp_path / files["band_long_csv"]).read_text(encoding="utf-8")
    dat_text = (tmp_path / files["band_data"]).read_text(encoding="utf-8")
    metadata = json.loads((tmp_path / files["band_metadata"]).read_text(encoding="utf-8"))
    segments = json.loads((tmp_path / files["band_segments"]).read_text(encoding="utf-8"))

    assert "distance" in csv_text.splitlines()[0]
    assert "branch_1_THz" in csv_text.splitlines()[0]
    assert "frequency_THz" in long_text.splitlines()[0]
    assert "\n\n# segment 1" in dat_text
    assert metadata["tick_positions"] == [0.0, 1.0, 2.0]
    assert metadata["tick_labels"] == ["Γ", "X|U", "Γ"]
    assert segments["segments"][1]["start_label"] == "U"


def test_band_data_from_explicit_path_uses_linearcoord_and_labels():
    band_path = BandPath(
        qpoints=[
            np.array([[0, 0, 0], [0.5, 0, 0]], dtype=float),
            np.array([[0.5, 0, 0], [0.625, 0.25, 0.625]], dtype=float),
            np.array([[0.375, 0.375, 0.75], [0, 0, 0]], dtype=float),
        ],
        labels=["G", "X", "X", "U", "K", "G"],
        source="seekpath",
        segments=[("G", "X"), ("X", "U"), ("K", "G")],
        explicit_kpoints_rel=np.zeros((5, 3)),
        explicit_kpoints_linearcoord=np.array([0.0, 1.0, 1.5, 1.5, 2.5]),
        explicit_kpoints_labels=["G", "X", "U", "K", "G"],
        segment_linearcoords=[
            np.array([0.0, 1.0]),
            np.array([1.0, 1.5]),
            np.array([1.5, 2.5]),
        ],
    )
    band_data = band_data_from_phonopy_dict(
        {"frequencies": [np.zeros((2, 2)), np.ones((2, 2)), np.full((2, 2), 2.0)]},
        band_path,
    )

    assert band_data.tick_positions == [0.0, 1.0, 1.5, 2.5]
    assert band_data.tick_labels == ["Γ", "X", "U|K", "Γ"]
    assert band_data.segments[1].distances.tolist() == [1.0, 1.5]


def _write_test_band_yaml(tmp_path: Path) -> Path:
    band_yaml = tmp_path / "band.yaml"
    band_yaml.write_text(
        """
segment_nqpoint:
- 2
- 2
labels:
- [G, X]
- [U, G]
phonon:
- distance: 0.0
  band:
  - frequency: 0.0
  - frequency: 1.0
- distance: 1.0
  band:
  - frequency: 2.0
  - frequency: 3.0
- distance: 1.0
  band:
  - frequency: -0.05
  - frequency: 2.5
- distance: 2.0
  band:
  - frequency: 1.5
  - frequency: 4.0
""".strip(),
        encoding="utf-8",
    )
    return band_yaml
