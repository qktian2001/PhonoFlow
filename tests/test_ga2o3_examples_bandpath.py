from __future__ import annotations

from pathlib import Path

from ase.io import read

from phonoflow.analysis.bandpath import get_band_path, high_symmetry_path_metadata


def test_examples_ga2o3_bandpath_uses_official_seekpath_segments() -> None:
    atoms = read(Path("examples/Ga2O3.vasp"))

    band_path = get_band_path(atoms)
    metadata = high_symmetry_path_metadata(band_path.segments, source=band_path.source)
    display = metadata["display"]

    assert display.startswith("Γ — X | Y — Γ | Γ — Z | R — Γ")
    assert "Γ — X'" in display
    assert "Γ — V'" in display
    assert "Γ — T | T — H_2" not in display
    assert "H_0 — L" not in display
    assert "S_0" not in display
    assert metadata["symprec"] == 1e-5
    assert metadata["with_time_reversal"] is False
