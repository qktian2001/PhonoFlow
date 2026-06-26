from __future__ import annotations

import yaml

from phonoflow.analysis.bandpath import (
    BandPath,
    high_symmetry_path_metadata,
    write_band_yaml_path_metadata,
)


def test_band_yaml_gets_phonoflow_path_metadata(tmp_path) -> None:
    band_yaml = tmp_path / "band.yaml"
    band_yaml.write_text(
        "nqpoint: 2\n"
        "segment_nqpoint:\n"
        "- 2\n"
        "phonon:\n"
        "- distance: 0.0\n"
        "  band:\n"
        "  - frequency: 0.0\n"
        "- distance: 1.0\n"
        "  band:\n"
        "  - frequency: 1.0\n",
        encoding="utf-8",
    )
    band_path = BandPath(
        qpoints=[],
        labels=["Γ", "X"],
        source="seekpath",
        segments=[("Γ", "X"), ("Y", "Γ")],
        explicit_kpoints_rel=[],
        explicit_kpoints_linearcoord=[],
        explicit_kpoints_labels=[],
        segment_linearcoords=[],
    )
    metadata = high_symmetry_path_metadata(band_path.segments, source=band_path.source)

    write_band_yaml_path_metadata(band_yaml, band_path, metadata)

    data = yaml.safe_load(band_yaml.read_text(encoding="utf-8"))
    assert data["labels"] == [["Γ", "X"], ["Y", "Γ"]]
    assert data["phonoflow_high_symmetry_path"]["display"] == "Γ — X | Y — Γ"
    assert data["phonoflow_seekpath"]["symprec"] == 1e-5
    assert data["phonoflow_seekpath"]["with_time_reversal"] is False
