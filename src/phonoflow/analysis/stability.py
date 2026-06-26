"""Dynamic stability analysis helpers."""

from __future__ import annotations

from typing import Iterable, Any

import numpy as np


def analyze_stability(frequencies: Iterable[float], imag_threshold: float = -0.1) -> dict[str, Any]:
    """Analyze phonon frequencies in THz for obvious imaginary modes."""

    values = np.asarray(list(frequencies), dtype=float)
    if values.size == 0:
        raise ValueError("frequencies must contain at least one value")

    minimum = float(np.min(values))
    imaginary_mask = values < imag_threshold
    imaginary_count = int(np.count_nonzero(imaginary_mask))
    has_imaginary = bool(minimum < imag_threshold)
    return {
        "dynamically_stable": not has_imaginary,
        "has_imaginary_frequency": has_imaginary,
        "minimum_frequency_THz": minimum,
        "imaginary_mode_count": imaginary_count,
        "imaginary_mode_ratio": float(imaginary_count / values.size),
        "imag_threshold_THz": float(imag_threshold),
    }
