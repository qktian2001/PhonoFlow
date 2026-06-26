from pathlib import Path

import numpy as np

from phonoflow.band.data import BandData, BandSegment
from phonoflow.band.plot import plot_phonon_band


def test_plot_phonon_band_writes_nonempty_png(tmp_path: Path):
    band_data = BandData(
        segments=[
            BandSegment(
                index=0,
                start_label="Γ",
                end_label="X",
                distances=np.array([0.0, 1.0]),
                qpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
                frequencies=np.array([[-1e-8, 1.0], [2.0, 3.0]]),
            )
        ],
        tick_positions=[0.0, 1.0],
        tick_labels=["Γ", "X"],
    )

    output_png = tmp_path / "phonon_band.png"
    plot_phonon_band(band_data, output_png)

    assert output_png.exists()
    assert output_png.stat().st_size > 1024
