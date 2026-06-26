"""Diagnostics for phonon postprocessing outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from phonoflow.band.data import BandData


def write_band_diagnostics(
    *,
    band_data: BandData,
    source_file: Path,
    plot_file: Path,
    output_path: Path,
    high_symmetry_path: dict[str, Any] | None = None,
    generated_by: str = "phonopy band structure",
) -> dict[str, Any]:
    """Write diagnostics for a phonon dispersion output."""

    frequencies = band_data.all_frequencies
    payload = {
        "source_file": Path(source_file).name,
        "plot_file": Path(plot_file).name,
        "generated_by": generated_by,
        "n_qpoints": band_data.n_qpoints,
        "n_branches": band_data.n_branches,
        "n_segments": band_data.n_segments,
        "min_frequency_THz": _finite_min(frequencies),
        "max_frequency_THz": _finite_max(frequencies),
        "has_imaginary_frequency": band_data.has_imaginary_frequency,
        "imag_threshold_THz": band_data.imag_threshold,
        "unit": band_data.frequency_unit,
        "high_symmetry_path": high_symmetry_path or {},
        "tick_positions": band_data.tick_positions,
        "tick_labels": band_data.tick_labels,
        "file_exists": Path(source_file).exists(),
        "file_size": _file_size(source_file),
        "plot_exists": Path(plot_file).exists(),
        "plot_size": _file_size(plot_file),
        "warnings": [],
    }
    return _write_json(payload, output_path)


def write_dos_diagnostics(
    *,
    frequencies: np.ndarray,
    total_dos: np.ndarray,
    source_file: Path,
    plot_file: Path,
    output_path: Path,
    generated_by: str = "phonopy total DOS",
) -> dict[str, Any]:
    """Write diagnostics for total phonon DOS data."""

    frequencies = np.asarray(frequencies, dtype=float)
    total_dos = np.asarray(total_dos, dtype=float)
    payload = {
        "source_file": Path(source_file).name,
        "plot_file": Path(plot_file).name,
        "generated_by": generated_by,
        "n_qpoints": None,
        "n_branches": None,
        "n_dos_points": int(frequencies.size),
        "min_frequency_THz": _finite_min(frequencies),
        "max_frequency_THz": _finite_max(frequencies),
        "frequency_range_THz": [_finite_min(frequencies), _finite_max(frequencies)],
        "dos_min": _finite_min(total_dos),
        "dos_max": _finite_max(total_dos),
        "has_imaginary_frequency": bool(np.any(frequencies < 0.0)) if frequencies.size else False,
        "unit": "THz",
        "dos_unit": "arbitrary units from phonopy total DOS",
        "file_exists": Path(source_file).exists(),
        "file_size": _file_size(source_file),
        "plot_exists": Path(plot_file).exists(),
        "plot_size": _file_size(plot_file),
        "warnings": [],
    }
    return _write_json(payload, output_path)


def write_group_velocity_diagnostics(
    *,
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    source_file: Path,
    plot_file: Path | None,
    output_path: Path,
    generated_by: str = "phonopy mesh group velocities",
) -> dict[str, Any]:
    """Write diagnostics for frequency-resolved group velocities."""

    frequencies = np.asarray(frequencies, dtype=float)
    magnitudes = np.asarray(magnitudes, dtype=float)
    payload = {
        "source_file": Path(source_file).name,
        "plot_file": Path(plot_file).name if plot_file else None,
        "generated_by": generated_by,
        "n_qpoints": int(frequencies.shape[0]) if frequencies.ndim >= 1 else 0,
        "n_branches": int(frequencies.shape[1]) if frequencies.ndim >= 2 else 0,
        "n_points": int(frequencies.size),
        "min_frequency_THz": _finite_min(frequencies),
        "max_frequency_THz": _finite_max(frequencies),
        "has_imaginary_frequency": bool(np.any(frequencies < 0.0)) if frequencies.size else False,
        "unit": "km/s",
        "min_abs_velocity": _finite_min(magnitudes),
        "max_abs_velocity": _finite_max(magnitudes),
        "mean_abs_velocity": _finite_mean(magnitudes),
        "file_exists": Path(source_file).exists(),
        "file_size": _file_size(source_file),
        "plot_exists": bool(plot_file and Path(plot_file).exists()),
        "plot_size": _file_size(plot_file) if plot_file else None,
        "warnings": [],
    }
    return _write_json(payload, output_path)


def write_lifetime_diagnostics(
    *,
    rows: list[dict[str, Any]],
    source_file: Path,
    data_file: Path,
    plot_file: Path,
    output_path: Path,
    source: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Write diagnostics for phonon lifetime extraction."""

    frequencies = np.asarray([row.get("frequency_THz", np.nan) for row in rows], dtype=float)
    lifetimes = np.asarray([row.get("lifetime_ps", np.nan) for row in rows], dtype=float)
    finite_positive = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]
    payload = {
        "source_file": Path(source_file).name,
        "data_file": Path(data_file).name,
        "plot_file": Path(plot_file).name,
        "generated_by": f"phono3py kappa HDF5 {source} dataset",
        "n_qpoints": _count_unique(rows, "q_index"),
        "n_branches": _count_unique(rows, "branch_index"),
        "n_points": len(rows),
        "min_frequency_THz": _finite_min(frequencies),
        "max_frequency_THz": _finite_max(frequencies),
        "has_imaginary_frequency": bool(np.any(frequencies < 0.0)) if frequencies.size else False,
        "unit": "ps",
        "min_lifetime_ps": _finite_min(finite_positive),
        "max_lifetime_ps": _finite_max(finite_positive),
        "mean_lifetime_ps": _finite_mean(finite_positive),
        "n_nan": int(np.count_nonzero(np.isnan(lifetimes))),
        "n_inf": int(np.count_nonzero(np.isinf(lifetimes))),
        "n_zero": int(np.count_nonzero(lifetimes == 0.0)),
        "n_non_positive": int(np.count_nonzero(np.isfinite(lifetimes) & (lifetimes <= 0.0))),
        "file_exists": Path(data_file).exists(),
        "file_size": _file_size(data_file),
        "plot_exists": Path(plot_file).exists(),
        "plot_size": _file_size(plot_file),
        "warnings": list(warnings or []),
    }
    return _write_json(payload, output_path)


def _write_json(payload: dict[str, Any], output_path: Path) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return payload | {"diagnostics_file": output_path.name}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _file_size(path: Path | str | None) -> int | None:
    if path is None:
        return None
    path = Path(path)
    return int(path.stat().st_size) if path.exists() and path.is_file() else None


def _finite_min(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.min(finite)) if finite.size else None


def _finite_max(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else None


def _finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else None


def _count_unique(rows: list[dict[str, Any]], key: str) -> int:
    values = {row.get(key) for row in rows if row.get(key) is not None}
    return len(values)
