"""Read, write, and plot phono3py thermal-conductivity outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from phonoflow.analysis.postprocessing_diagnostics import write_lifetime_diagnostics
from phonoflow.thermal.plots import plot_lifetime, plot_thermal_conductivity


KAPPA_COLUMNS = [
    "temperature_K",
    "kxx",
    "kyy",
    "kzz",
    "kxy",
    "kyz",
    "kxz",
    "kappa_trace_over_3",
]


def kappa_mesh_tag(mesh: list[int] | tuple[int, int, int]) -> str:
    """Return the phono3py filename suffix for one q-mesh."""

    return "".join(str(int(value)) for value in mesh)


def select_kappa_hdf5_path(outdir: Path, mesh: list[int] | tuple[int, int, int]) -> Path:
    """Prefer the kappa HDF5 file matching the active q-mesh."""

    outdir = Path(outdir)
    preferred = outdir / f"kappa-m{kappa_mesh_tag(mesh)}.hdf5"
    if preferred.exists() and preferred.is_file():
        return preferred
    matches = sorted(path for path in outdir.glob("kappa-m*.hdf5") if path.is_file())
    if not matches:
        raise FileNotFoundError("No kappa-m*.hdf5 file was written.")
    return matches[-1]


def parse_kappa_hdf5(path: Path) -> dict[str, Any]:
    """Parse a phono3py kappa HDF5 file into stable row dictionaries.

    phono3py stores tensor components in the order xx, yy, zz, yz, xz, xy.
    The public CSV keeps the more familiar kxy, kyz, kxz column order.
    """

    path = Path(path)
    with h5py.File(path, "r") as handle:
        temperatures = np.asarray(handle["temperature"], dtype=float)
        kappa = np.asarray(handle["kappa"], dtype=float)

    if kappa.ndim == 3:
        # Shape may include a sigma axis. The first sigma is the default no-smearing result.
        kappa = kappa[0]
    if kappa.ndim != 2 or kappa.shape[1] < 6:
        raise ValueError(f"Unexpected kappa dataset shape in {path}: {kappa.shape}")

    rows = []
    for temperature, tensor in zip(temperatures, kappa):
        kxx, kyy, kzz, kyz, kxz, kxy = [float(value) for value in tensor[:6]]
        rows.append(
            {
                "temperature_K": float(temperature),
                "kxx": kxx,
                "kyy": kyy,
                "kzz": kzz,
                "kxy": kxy,
                "kyz": kyz,
                "kxz": kxz,
                "kappa_trace_over_3": float((kxx + kyy + kzz) / 3.0),
            }
        )
    return {"rows": rows, "source_file": path.name}


def inspect_kappa_hdf5(path: Path) -> dict[str, Any]:
    """Return JSON-safe dataset metadata for a phono3py kappa HDF5 file."""

    path = Path(path)
    fields: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key in sorted(handle.keys()):
            dataset = handle[key]
            fields[key] = {
                "shape": [int(value) for value in getattr(dataset, "shape", ())],
                "dtype": str(getattr(dataset, "dtype", "")),
            }
    return {
        "source_file": path.name,
        "fields": fields,
        "fields_found": sorted(fields),
    }


def write_thermal_conductivity_csv(rows: list[dict[str, float]], path: Path) -> Path:
    """Write thermal conductivity rows."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=KAPPA_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: f"{float(row[column]):.10f}" if column in row else ""
                    for column in KAPPA_COLUMNS
                }
            )
    return path


def summarize_kappa(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Return a JSON-friendly temperature-keyed conductivity summary."""

    return {
        f"{row['temperature_K']:g}": {
            key: float(row[key])
            for key in ["kxx", "kyy", "kzz", "kxy", "kyz", "kxz", "kappa_trace_over_3"]
        }
        for row in rows
    }


def extract_lifetime_from_hdf5(path: Path, outdir: Path, dpi: int = 300) -> dict[str, Any]:
    """Extract phonon lifetime from a phono3py kappa HDF5 file.

    phono3py may either write a direct ``lifetime`` dataset or only ``gamma``.
    phono3py's own conductivity utilities document ``tau = 1 / (2 * 2 * pi *
    gamma)``. Since gamma in the kappa HDF5 is in THz, ``1 / THz`` is ps, so the
    converted lifetime is reported in ps. No conversion is attempted when both
    datasets are absent.
    """

    path = Path(path)
    outdir = Path(outdir)
    with h5py.File(path, "r") as handle:
        if "lifetime" in handle:
            raw_lifetime = np.asarray(handle["lifetime"], dtype=float)
            source = "lifetime"
            raw_gamma = None
        elif "gamma" in handle:
            gamma = np.asarray(handle["gamma"], dtype=float)
            raw_lifetime = _gamma_to_lifetime_ps(gamma)
            source = "gamma"
            raw_gamma = gamma
        else:
            return {
                "available": False,
                "reason": "No lifetime or gamma dataset was found in the phono3py kappa HDF5 file.",
            }
        frequencies = np.asarray(handle.get("frequency", []), dtype=float)
        temperatures = np.asarray(handle.get("temperature", []), dtype=float)

    lifetime_array = _normalize_temperature_q_branch(raw_lifetime)
    if lifetime_array is None:
        return {"available": False, "reason": f"Unexpected {source} dataset shape: {raw_lifetime.shape}"}
    gamma_array = _normalize_temperature_q_branch(raw_gamma) if raw_gamma is not None else None

    if temperatures.size == 0:
        temperatures = np.arange(lifetime_array.shape[0], dtype=float)

    rows: list[dict[str, Any]] = []
    for t_index, temperature in enumerate(temperatures):
        if t_index >= lifetime_array.shape[0]:
            break
        for q_index in range(lifetime_array.shape[1]):
            for branch_index in range(lifetime_array.shape[2]):
                frequency = (
                    float(frequencies[q_index, branch_index])
                    if frequencies.ndim == 2 and q_index < frequencies.shape[0]
                    else float("nan")
                )
                rows.append(
                    {
                        "temperature_K": float(temperature),
                        "q_index": int(q_index),
                        "branch_index": int(branch_index),
                        "frequency_THz": frequency,
                        "lifetime_ps": float(lifetime_array[t_index, q_index, branch_index]),
                        "gamma_or_linewidth_raw": (
                            float(gamma_array[t_index, q_index, branch_index])
                            if gamma_array is not None
                            else ""
                        ),
                    }
                )

    csv_path = outdir / "phonon_lifetime.csv"
    png_path = outdir / "phonon_lifetime.png"
    diagnostics_path = outdir / "phonon_lifetime_diagnostics.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "temperature_K",
            "q_index",
            "branch_index",
            "frequency_THz",
            "lifetime_ps",
            "gamma_or_linewidth_raw",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    plot_lifetime(rows, png_path, dpi=dpi)
    finite_positive = np.asarray(
        [
            row["lifetime_ps"]
            for row in rows
            if np.isfinite(float(row["lifetime_ps"])) and float(row["lifetime_ps"]) > 0
        ],
        dtype=float,
    )
    warnings: list[str] = []
    if source == "gamma":
        warnings.append(
            "Lifetime was converted from phono3py gamma using tau = 1 / (4*pi*gamma); "
            "gamma is in THz and the resulting lifetime is in ps."
        )
    diagnostics = write_lifetime_diagnostics(
        rows=rows,
        source_file=path,
        data_file=csv_path,
        plot_file=png_path,
        output_path=diagnostics_path,
        source=source,
        warnings=warnings,
    )
    return {
        "available": True,
        "data_file": csv_path.name,
        "plot_file": png_path.name,
        "diagnostics_file": diagnostics_path.name,
        "unit": "ps",
        "n_points": len(rows),
        "source": source,
        "mean_lifetime_ps": float(np.mean(finite_positive)) if finite_positive.size else None,
        "max_lifetime_ps": float(np.max(finite_positive)) if finite_positive.size else None,
        "reason": None,
        "warnings": warnings,
        "diagnostics": diagnostics,
    }


def _gamma_to_lifetime_ps(gamma: np.ndarray) -> np.ndarray:
    """Convert phono3py gamma in THz to lifetime in ps.

    phono3py defines lifetime as ``tau = 1 / (2 * 2 * pi * gamma)``. Because
    gamma is in THz cycles per second and ``1 THz^-1 = 1 ps``, the result is ps.
    Non-positive gamma values are kept as NaN to avoid fabricating lifetimes.
    """

    gamma = np.asarray(gamma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lifetime = np.where(gamma > 0, 1.0 / (4.0 * np.pi * gamma), np.nan)
    return lifetime


def _normalize_temperature_q_branch(array: np.ndarray | None) -> np.ndarray | None:
    """Return an array with shape (temperature, q-point, branch) when possible."""

    if array is None:
        return None
    array = np.asarray(array, dtype=float)
    if array.ndim == 3:
        return array
    if array.ndim == 4:
        # Some phono3py outputs include a leading sigma axis. Use the first
        # default-no-smearing slice, matching parse_kappa_hdf5.
        return array[0]
    if array.ndim == 2:
        return array[None, :, :]
    return None
