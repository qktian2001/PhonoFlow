"""Phonon group velocity postprocessing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from phonoflow.analysis.postprocessing_diagnostics import write_group_velocity_diagnostics


GROUP_VELOCITY_CSV = "phonon_group_velocity.csv"
GROUP_VELOCITY_PNG = "phonon_group_velocity.png"
GROUP_VELOCITY_DIAGNOSTICS = "phonon_group_velocity_diagnostics.json"
THZ_ANGSTROM_TO_KM_PER_S = 0.1


def compute_phonon_group_velocity(
    phonon: Any | None = None,
    output_dir: Path | str | None = None,
    mesh: list[int] | tuple[int, int, int] | None = None,
    unit: str = "km/s",
    plot: bool = True,
    overwrite: bool = True,
    logger: Callable[[str], None] | None = None,
    dpi: int = 300,
    **_: Any,
) -> dict[str, Any]:
    """Compute mesh phonon group velocities and write CSV/PNG outputs.

    Phonopy reports group velocities in THz*Angstrom for the default frequency
    unit convention used by this project. Since 1 THz * 1 Angstrom = 100 m/s,
    the conversion to km/s is 1 THz*Angstrom = 0.1 km/s.
    """

    if unit != "km/s":
        return _unavailable(f"Unsupported group velocity unit: {unit!r}.")
    if phonon is None:
        return _unavailable("A Phonopy object is required for group velocity calculation.")
    if output_dir is None:
        return _unavailable("An output directory is required for group velocity calculation.")
    if mesh is None:
        return _unavailable("A phonon mesh is required for group velocity calculation.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_path = output_path / GROUP_VELOCITY_CSV
    plot_path = output_path / GROUP_VELOCITY_PNG
    if not overwrite and data_path.exists() and (not plot or plot_path.exists()):
        return _existing_result(data_path, plot_path if plot_path.exists() else None)

    try:
        _log(logger, f"Computing phonon group velocities on gamma-centered mesh={list(mesh)}")
        phonon.run_mesh(mesh, with_group_velocities=True, is_gamma_center=True)
        mesh_dict = phonon.get_mesh_dict()
        frequencies = np.asarray(mesh_dict.get("frequencies"), dtype=float)
        group_velocities = np.asarray(mesh_dict.get("group_velocities"), dtype=float)
        if frequencies.size == 0 or group_velocities.size == 0:
            return _unavailable("Phonopy did not return group velocity data.")
        result = write_group_velocity_outputs(
            frequencies=frequencies,
            group_velocities_thz_angstrom=group_velocities,
            output_dir=output_path,
            plot=plot,
            dpi=dpi,
        )
        _log(logger, "Wrote phonon group velocity CSV and plot")
        return result
    except Exception as exc:
        return _unavailable(f"Group velocity calculation failed: {exc}")


def write_group_velocity_outputs(
    frequencies: np.ndarray,
    group_velocities_thz_angstrom: np.ndarray,
    output_dir: Path | str,
    plot: bool = True,
    dpi: int = 300,
) -> dict[str, Any]:
    """Write group velocity CSV and optional scatter plot from array data."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frequencies = np.asarray(frequencies, dtype=float)
    group_velocities_raw = np.asarray(group_velocities_thz_angstrom, dtype=float)
    _validate_shapes(frequencies, group_velocities_raw)

    group_velocities = convert_thz_angstrom_to_km_s(group_velocities_raw)
    magnitudes = np.linalg.norm(group_velocities, axis=2)
    data_path = output_path / GROUP_VELOCITY_CSV
    _write_group_velocity_csv(data_path, frequencies, group_velocities, magnitudes)

    plot_file: str | None = None
    negative_frequency_points = int(np.count_nonzero(frequencies < 0.0))
    if plot:
        plot_path = output_path / GROUP_VELOCITY_PNG
        plot_group_velocity_scatter(
            frequencies=frequencies,
            magnitudes=magnitudes,
            path=plot_path,
            dpi=dpi,
        )
        plot_file = plot_path.name

    diagnostics_path = output_path / GROUP_VELOCITY_DIAGNOSTICS
    diagnostics = write_group_velocity_diagnostics(
        frequencies=frequencies,
        magnitudes=magnitudes,
        source_file=data_path,
        plot_file=output_path / plot_file if plot_file else None,
        output_path=diagnostics_path,
    )

    return {
        "available": True,
        "reason": None,
        "data_file": data_path.name,
        "plot_file": plot_file,
        "diagnostics_file": diagnostics_path.name,
        "unit": "km/s",
        "x_axis": "frequency_THz",
        "y_axis": "group_velocity_km_s",
        "n_points": int(frequencies.size),
        "max_abs_velocity": float(np.nanmax(magnitudes)) if magnitudes.size else None,
        "mean_abs_velocity": float(np.nanmean(magnitudes)) if magnitudes.size else None,
        "method": "phonopy mesh group velocities",
        "negative_frequency_points": negative_frequency_points,
        "plot_note": (
            "Negative frequencies were excluded from the group velocity scatter plot."
            if negative_frequency_points
            else None
        ),
        "diagnostics": diagnostics,
    }


def convert_thz_angstrom_to_km_s(values: np.ndarray) -> np.ndarray:
    """Convert Phonopy THz*Angstrom group velocities to km/s."""

    return np.asarray(values, dtype=float) * THZ_ANGSTROM_TO_KM_PER_S


def plot_group_velocity_scatter(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    path: Path | str,
    dpi: int = 300,
) -> None:
    """Plot |v_g| against phonon frequency."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frequency_values = np.asarray(frequencies, dtype=float).reshape(-1)
    velocity_values = np.asarray(magnitudes, dtype=float).reshape(-1)
    mask = np.isfinite(frequency_values) & np.isfinite(velocity_values) & (frequency_values >= 0.0)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    if np.any(mask):
        ax.scatter(
            frequency_values[mask],
            velocity_values[mask],
            s=9,
            alpha=0.55,
            color="#7c3aed",
            edgecolors="none",
        )
    ax.set_title("Phonon group velocity")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Group velocity (km/s)")
    ax.grid(True, color="#d8e0eb", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_group_velocity_csv(
    path: Path,
    frequencies: np.ndarray,
    group_velocities: np.ndarray,
    magnitudes: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "q_index",
                "branch_index",
                "frequency_THz",
                "vg_x_km_s",
                "vg_y_km_s",
                "vg_z_km_s",
                "vg_abs_km_s",
            ]
        )
        for q_index in range(frequencies.shape[0]):
            for branch_index in range(frequencies.shape[1]):
                writer.writerow(
                    [
                        q_index,
                        branch_index,
                        f"{float(frequencies[q_index, branch_index]):.10f}",
                        f"{float(group_velocities[q_index, branch_index, 0]):.10f}",
                        f"{float(group_velocities[q_index, branch_index, 1]):.10f}",
                        f"{float(group_velocities[q_index, branch_index, 2]):.10f}",
                        f"{float(magnitudes[q_index, branch_index]):.10f}",
                    ]
                )


def _validate_shapes(frequencies: np.ndarray, group_velocities: np.ndarray) -> None:
    if frequencies.ndim != 2:
        raise ValueError(f"frequencies must have shape (n_qpoints, n_branches), got {frequencies.shape}.")
    if group_velocities.ndim != 3 or group_velocities.shape[2] != 3:
        raise ValueError(
            "group velocities must have shape (n_qpoints, n_branches, 3), "
            f"got {group_velocities.shape}."
        )
    if group_velocities.shape[:2] != frequencies.shape:
        raise ValueError(
            "frequency and group-velocity dimensions do not match: "
            f"{frequencies.shape} vs {group_velocities.shape}."
        )


def _existing_result(data_path: Path, plot_path: Path | None) -> dict[str, Any]:
    return {
        "available": True,
        "reason": None,
        "data_file": data_path.name,
        "plot_file": plot_path.name if plot_path else None,
        "diagnostics_file": GROUP_VELOCITY_DIAGNOSTICS if (data_path.parent / GROUP_VELOCITY_DIAGNOSTICS).exists() else None,
        "unit": "km/s",
        "x_axis": "frequency_THz",
        "y_axis": "group_velocity_km_s",
        "method": "phonopy mesh group velocities",
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "data_file": None,
        "plot_file": None,
    }


def _log(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)
