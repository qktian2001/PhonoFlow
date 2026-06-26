"""Plot structured phonon band data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from phonoflow.band.data import BandData
from phonoflow.band.io import load_band_yaml_segments


def plot_phonon_band(
    band_data: BandData,
    output_png: Path,
    title: str = "Phonon dispersion",
    dpi: int = 300,
) -> None:
    """Plot phonon branches from segmented Phonopy distance data."""

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for segment in band_data.segments:
        x = segment.distances
        y = segment.frequencies
        for branch in range(segment.n_branches):
            ax.plot(x, y[:, branch], color="#176b87", linewidth=1.15)

    for tick in band_data.tick_positions:
        ax.axvline(tick, color="#d8e0eb", linewidth=0.7, zorder=0)
    ax.axhline(0.0, color="#667085", linestyle="--", linewidth=0.8)
    ax.set_xticks(band_data.tick_positions)
    ax.set_xticklabels(band_data.tick_labels)
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_xlabel("Wave vector")
    ax.set_ylabel(f"Frequency ({band_data.frequency_unit})")
    ax.set_title(title)
    ax.grid(True, axis="y", color="#e6edf5", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    _set_frequency_limits(ax, band_data.all_frequencies)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_phonon_band_from_band_yaml(
    band_yaml_path: Path,
    output_png: Path,
    title: str = "Phonon dispersion",
    dpi: int = 300,
    labels: list[str] | None = None,
    imag_threshold: float = -0.1,
) -> BandData:
    """Plot ``phonon_band.png`` directly from a Phonopy ``band.yaml`` file."""

    band_data = load_band_yaml_segments(band_yaml_path, labels=labels, imag_threshold=imag_threshold)
    plot_phonon_band(band_data, output_png, title=title, dpi=dpi)
    return band_data


def _set_frequency_limits(ax: Any, frequencies: np.ndarray) -> None:
    finite = np.asarray(frequencies[np.isfinite(frequencies)], dtype=float)
    if finite.size == 0:
        return

    min_freq = float(np.min(finite))
    max_freq = float(np.max(finite))
    if abs(min_freq) < 1e-6:
        min_freq = 0.0
    span = max(max_freq - min_freq, 1.0)
    pad = 0.08 * span
    lower = min_freq - pad
    upper = max_freq + pad
    if min_freq >= -1e-6:
        lower = min(lower, -0.05 * max(max_freq, 1.0))
    ax.set_ylim(lower, upper)
