from __future__ import annotations

from phonoflow.analysis.bandpath import (
    DEFAULT_SEEKPATH_SYMPREC,
    DEFAULT_SEEKPATH_WITH_TIME_REVERSAL,
    format_path_segments,
    high_symmetry_path_metadata,
)


def test_seekpath_defaults_are_strict_and_full_path() -> None:
    assert DEFAULT_SEEKPATH_SYMPREC == 1e-5
    assert DEFAULT_SEEKPATH_WITH_TIME_REVERSAL is False


def test_high_symmetry_path_metadata_comes_from_segments() -> None:
    segments = [("GAMMA", "X"), ("Y", "GAMMA"), ("GAMMA", "X'")]

    metadata = high_symmetry_path_metadata(segments)

    assert metadata["display"] == "Γ — X | Y — Γ | Γ — X'"
    assert metadata["segments"] == [["Γ", "X"], ["Y", "Γ"], ["Γ", "X'"]]
    assert metadata["symprec"] == 1e-5
    assert metadata["with_time_reversal"] is False
    assert format_path_segments(segments) == metadata["display"]


def test_high_symmetry_path_metadata_records_explicit_seekpath_settings() -> None:
    metadata = high_symmetry_path_metadata(
        [("GAMMA", "X")],
        symprec=1e-6,
        with_time_reversal=True,
    )

    assert metadata["display"] == "Γ — X"
    assert metadata["symprec"] == 1e-6
    assert metadata["with_time_reversal"] is True
