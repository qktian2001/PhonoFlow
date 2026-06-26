"""Structured phonon band data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BandSegment:
    """One continuous high-symmetry q-point path segment."""

    index: int
    start_label: str
    end_label: str
    distances: np.ndarray
    qpoints: np.ndarray
    frequencies: np.ndarray

    @property
    def nqpoint(self) -> int:
        return int(self.distances.shape[0])

    @property
    def n_branches(self) -> int:
        if self.frequencies.ndim != 2:
            return 0
        return int(self.frequencies.shape[1])


@dataclass(frozen=True)
class BandData:
    """Structured Phonopy band data ready for export and plotting."""

    segments: list[BandSegment]
    tick_positions: list[float]
    tick_labels: list[str]
    frequency_unit: str = "THz"
    imag_threshold: float = -0.1

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def n_qpoints(self) -> int:
        return sum(segment.nqpoint for segment in self.segments)

    @property
    def n_branches(self) -> int:
        for segment in self.segments:
            if segment.n_branches:
                return segment.n_branches
        return 0

    @property
    def all_frequencies(self) -> np.ndarray:
        arrays = [segment.frequencies.reshape(-1) for segment in self.segments if segment.frequencies.size]
        return np.concatenate(arrays) if arrays else np.array([], dtype=float)

    @property
    def minimum_frequency(self) -> float | None:
        frequencies = self.all_frequencies
        return float(np.min(frequencies)) if frequencies.size else None

    @property
    def maximum_frequency(self) -> float | None:
        frequencies = self.all_frequencies
        return float(np.max(frequencies)) if frequencies.size else None

    @property
    def has_imaginary_frequency(self) -> bool:
        minimum = self.minimum_frequency
        return bool(minimum is not None and minimum < self.imag_threshold)


def band_data_to_metadata(band_data: BandData) -> dict[str, Any]:
    """Return public metadata for result.json and exported metadata JSON."""

    return {
        "frequency_unit": band_data.frequency_unit,
        "n_branches": band_data.n_branches,
        "n_qpoints": band_data.n_qpoints,
        "n_segments": band_data.n_segments,
        "tick_positions": band_data.tick_positions,
        "tick_labels": band_data.tick_labels,
        "minimum_frequency_THz": band_data.minimum_frequency,
        "maximum_frequency_THz": band_data.maximum_frequency,
        "has_imaginary_frequency": band_data.has_imaginary_frequency,
        "imag_threshold_THz": band_data.imag_threshold,
    }
