import json
from pathlib import Path

import numpy as np

from phonoflow.band.data import BandData, BandSegment
from phonoflow.band.export import export_phonon_band_data


def test_band_data_export_writes_expected_files(tmp_path: Path):
    band_data = BandData(
        segments=[
            BandSegment(
                index=0,
                start_label="G",
                end_label="X",
                distances=np.array([0.0, 1.0]),
                qpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
                frequencies=np.array([[0.0, 1.0], [2.0, 3.0]]),
            )
        ],
        tick_positions=[0.0, 1.0],
        tick_labels=["Γ", "X"],
    )

    files = export_phonon_band_data(band_data, tmp_path)

    for key in ["band_csv", "band_long_csv", "band_data", "band_segments", "band_metadata"]:
        path = tmp_path / files[key]
        assert path.exists()
        assert path.stat().st_size > 0

    metadata = json.loads((tmp_path / files["band_metadata"]).read_text(encoding="utf-8"))
    assert metadata["minimum_frequency_THz"] == 0.0
    assert metadata["maximum_frequency_THz"] == 3.0
    assert metadata["tick_labels"] == ["Γ", "X"]
