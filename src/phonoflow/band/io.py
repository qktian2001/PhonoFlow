"""Build structured band data from Phonopy outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phonoflow.band.data import BandData, BandSegment
from phonoflow.band.labels import collapse_tick_labels


def band_data_from_phonopy_dict(
    band_structure: dict[str, Any],
    band_path: Any,
    imag_threshold: float = -0.1,
) -> BandData:
    """Build band data from Phonopy frequencies and the explicit seekpath path."""

    frequencies = band_structure.get("frequencies")
    if frequencies is None:
        raise RuntimeError("Phonopy band structure did not contain frequencies.")
    segments: list[BandSegment] = []
    endpoint_ticks: list[float] = []
    endpoint_labels: list[str] = []
    for index, segment_frequencies in enumerate(frequencies):
        qpoints = np.asarray(band_path.qpoints[index], dtype=float)
        distances = np.asarray(band_path.segment_linearcoords[index], dtype=float)
        frequencies_array = np.asarray(segment_frequencies, dtype=float)
        if frequencies_array.ndim == 1:
            frequencies_array = frequencies_array[:, None]
        start_label, end_label = band_path.segments[index]
        segments.append(
            BandSegment(
                index=index,
                start_label=start_label,
                end_label=end_label,
                distances=distances,
                qpoints=qpoints,
                frequencies=frequencies_array,
            )
        )
        endpoint_ticks.extend([float(distances[0]), float(distances[-1])])
        endpoint_labels.extend([start_label, end_label])

    tick_positions, tick_labels = collapse_tick_labels(endpoint_ticks, endpoint_labels)
    return BandData(
        segments=segments,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        imag_threshold=imag_threshold,
    )


def load_band_yaml_segments(
    band_yaml_path: Path,
    labels: list[str] | None = None,
    imag_threshold: float = -0.1,
) -> BandData:
    """Load a Phonopy ``band.yaml`` file as explicit path segments."""

    with Path(band_yaml_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"band.yaml did not contain a mapping: {band_yaml_path}")

    phonons = data.get("phonon") or []
    if not phonons:
        raise RuntimeError(f"band.yaml contains no phonon entries: {band_yaml_path}")

    segment_nqpoint = [int(item) for item in data.get("segment_nqpoint") or [len(phonons)]]
    raw_labels = _choose_labels(data.get("labels"), labels, len(segment_nqpoint))
    segments: list[BandSegment] = []
    endpoint_ticks: list[float] = []
    endpoint_labels: list[str] = []
    start = 0

    for index, nqpoint in enumerate(segment_nqpoint):
        segment_points = phonons[start : start + nqpoint]
        if not segment_points:
            continue
        distances = np.asarray(
            [float(point.get("distance", 0.0)) for point in segment_points],
            dtype=float,
        )
        qpoints = np.asarray(
            [point.get("q-position", [np.nan, np.nan, np.nan]) for point in segment_points],
            dtype=float,
        )
        frequencies = np.asarray(
            [
                [float(branch.get("frequency", 0.0)) for branch in point.get("band", [])]
                for point in segment_points
            ],
            dtype=float,
        )
        if frequencies.ndim == 1:
            frequencies = frequencies[:, None]

        start_label = raw_labels[2 * index] if 2 * index < len(raw_labels) else f"q{index}_start"
        end_label = (
            raw_labels[2 * index + 1] if 2 * index + 1 < len(raw_labels) else f"q{index}_end"
        )
        segments.append(
            BandSegment(
                index=index,
                start_label=start_label,
                end_label=end_label,
                distances=distances,
                qpoints=qpoints,
                frequencies=frequencies,
            )
        )
        endpoint_ticks.extend([float(distances[0]), float(distances[-1])])
        endpoint_labels.extend([start_label, end_label])
        start += nqpoint

    tick_positions, tick_labels = collapse_tick_labels(endpoint_ticks, endpoint_labels)
    return BandData(
        segments=segments,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        imag_threshold=imag_threshold,
    )


def _choose_labels(yaml_labels: Any, fallback_labels: list[str] | None, n_segments: int) -> list[str]:
    flattened = _flatten_yaml_labels(yaml_labels)
    if len(flattened) >= 2 * n_segments:
        return flattened
    if fallback_labels and len(fallback_labels) >= 2 * n_segments:
        return [str(item) for item in fallback_labels]
    labels: list[str] = []
    for index in range(n_segments):
        labels.extend([f"q{index}_start", f"q{index}_end"])
    return labels


def _flatten_yaml_labels(labels: Any) -> list[str]:
    if not labels:
        return []
    flattened: list[str] = []
    for item in labels:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            flattened.extend([str(item[0]), str(item[1])])
        else:
            flattened.append(str(item))
    return flattened
