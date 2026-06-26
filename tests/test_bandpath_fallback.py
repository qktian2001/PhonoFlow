import numpy as np
import sys
import types

from phonoflow.analysis.bandpath import (
    DEFAULT_SEEKPATH_SYMPREC,
    DEFAULT_SEEKPATH_WITH_TIME_REVERSAL,
    get_band_path,
)


class BrokenAtoms:
    @property
    def cell(self):
        raise RuntimeError("force fallback")


def test_bandpath_fallback_returns_basic_path():
    band_path = get_band_path(BrokenAtoms(), mode="auto", npoints=5)

    assert band_path.source == "fallback"
    assert band_path.labels == ["Γ", "X"]
    assert len(band_path.qpoints) == 1
    assert band_path.qpoints[0].shape == (5, 3)
    assert np.allclose(band_path.qpoints[0][0], [0.0, 0.0, 0.0])
    assert np.allclose(band_path.qpoints[0][-1], [0.5, 0.0, 0.0])


class MinimalCell:
    array = np.eye(3)


class MinimalAtoms:
    cell = MinimalCell()

    def get_scaled_positions(self):
        return np.array([[0.0, 0.0, 0.0]])

    def get_atomic_numbers(self):
        return [14]


def test_seekpath_explicit_path_segments_and_ticks(monkeypatch):
    calls = []

    def fake_get_explicit_k_path(structure, **kwargs):
        calls.append(kwargs)
        return {
            "explicit_kpoints_rel": [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.625, 0.25, 0.625],
                [0.375, 0.375, 0.75],
                [0.0, 0.0, 0.0],
            ],
            "explicit_kpoints_linearcoord": [0.0, 1.0, 1.5, 1.5, 2.5],
            "explicit_kpoints_labels": ["GAMMA", "X", "U", "K", "GAMMA"],
        }

    fake_seekpath = types.SimpleNamespace(
        get_explicit_k_path=fake_get_explicit_k_path
    )
    monkeypatch.setitem(sys.modules, "seekpath", fake_seekpath)

    band_path = get_band_path(MinimalAtoms(), mode="auto")

    assert band_path.source == "seekpath"
    assert calls == [
        {
            "symprec": DEFAULT_SEEKPATH_SYMPREC,
            "with_time_reversal": DEFAULT_SEEKPATH_WITH_TIME_REVERSAL,
        }
    ]
    assert band_path.labels == ["Γ", "X", "X", "U", "K", "Γ"]
    assert band_path.segments == [("Γ", "X"), ("X", "U"), ("K", "Γ")]
    assert [segment.shape[0] for segment in band_path.qpoints] == [2, 2, 2]
