"""Phonon DOS plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_phonon_dos(
    frequencies: np.ndarray,
    total_dos: np.ndarray,
    path: Path,
    dpi: int = 300,
) -> None:
    """Plot total phonon density of states."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.plot(frequencies, total_dos, color="#0f766e", linewidth=1.35)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("DOS")
    ax.axvline(0.0, color="#667085", linestyle="--", linewidth=0.8)
    ax.grid(True, color="#e6edf5", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
