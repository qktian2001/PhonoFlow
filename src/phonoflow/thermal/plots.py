"""Plot helpers for thermal outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def plot_thermal_conductivity(rows: list[dict[str, float]], output_png: Path, dpi: int = 300) -> Path:
    """Plot kxx, kyy, kzz, and trace/3 against temperature."""

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    temps = np.asarray([row["temperature_K"] for row in rows], dtype=float)
    series = [
        ("kxx", "#2563eb"),
        ("kyy", "#0f766e"),
        ("kzz", "#b45309"),
        ("kappa_trace_over_3", "#7c3aed"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=dpi)
    if len(temps) == 1:
        labels = ["kxx", "kyy", "kzz", "trace/3"]
        keys = ["kxx", "kyy", "kzz", "kappa_trace_over_3"]
        values = [float(rows[0][key]) for key in keys]
        colors = ["#2563eb", "#0f766e", "#b45309", "#7c3aed"]
        bars = ax.bar(labels, values, color=colors, width=0.62, edgecolor="#223044", linewidth=0.6)
        ax.bar_label(bars, fmt="%.3g", padding=4, fontsize=8)
        ax.set_xlabel(f"Tensor component at {temps[0]:g} K")
    else:
        x = np.arange(len(temps), dtype=float)
        width = 0.22
        component_series = series[:3]
        offsets = [-width, 0.0, width]
        for (key, color), offset in zip(component_series, offsets, strict=True):
            values = np.asarray([row[key] for row in rows], dtype=float)
            ax.bar(x + offset, values, width=width, label=key, color=color, alpha=0.82)
        trace_values = np.asarray([row["kappa_trace_over_3"] for row in rows], dtype=float)
        ax.plot(x, trace_values, marker="o", linewidth=2.2, label="trace/3", color="#7c3aed")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{temp:g}" for temp in temps])
        ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Lattice thermal conductivity (W m$^{-1}$ K$^{-1}$)")
    ax.set_title("Lattice thermal conductivity")
    ax.grid(True, color="#d7dee8", linewidth=0.6, alpha=0.7)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    if len(temps) > 1:
        ax.legend(frameon=False)
    ax.margins(y=0.16)
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return output_png


def plot_lifetime(rows: list[dict[str, Any]], output_png: Path, dpi: int = 300) -> Path:
    """Plot lifetime against frequency for rows containing lifetime_ps."""

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    frequencies = np.asarray([row["frequency_THz"] for row in rows], dtype=float)
    lifetimes = np.asarray([row["lifetime_ps"] for row in rows], dtype=float)
    mask = np.isfinite(frequencies) & np.isfinite(lifetimes) & (lifetimes > 0)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=dpi)
    ax.scatter(frequencies[mask], lifetimes[mask], s=10, alpha=0.55, color="#db2777")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Phonon lifetime (ps, log scale)")
    ax.set_title("Phonon lifetime")
    if np.any(mask):
        ax.set_yscale("log")
    ax.grid(True, color="#d7dee8", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    return output_png
