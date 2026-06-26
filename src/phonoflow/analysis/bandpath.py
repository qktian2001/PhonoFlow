"""Band path helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phonoflow.band.labels import collapse_tick_labels, normalize_band_label
from phonoflow.kpath.dimensionality import infer_dimensionality_by_vacuum
from phonoflow.kpath.kpath_ase_2d import display_path_string, generate_ase_2d_kpath
from phonoflow.kpath.schema import KPathResult, serialize_kpath_result

DEFAULT_ASE_2D_EPS = 2e-4
DEFAULT_SEEKPATH_SYMPREC = 1e-5
DEFAULT_SEEKPATH_WITH_TIME_REVERSAL = False
PATH_SEGMENT_SEPARATOR = " \N{EM DASH} "
PATH_BREAK_SEPARATOR = " | "


@dataclass(frozen=True)
class BandPath:
    """Q-point path and labels for Phonopy band calculation."""

    qpoints: list[np.ndarray]
    labels: list[str]
    source: str
    segments: list[tuple[str, str]]
    explicit_kpoints_rel: np.ndarray
    explicit_kpoints_linearcoord: np.ndarray
    explicit_kpoints_labels: list[str]
    segment_linearcoords: list[np.ndarray]


def get_band_path(
    atoms: Any,
    mode: str = "auto",
    npoints: int = 101,
    symprec: float = DEFAULT_SEEKPATH_SYMPREC,
    with_time_reversal: bool = DEFAULT_SEEKPATH_WITH_TIME_REVERSAL,
    **kwargs: Any,
) -> BandPath:
    """Return a Phonopy-compatible q-point path."""

    if mode not in {"auto", "3d_seekpath", "2d_ase", "custom"}:
        raise NotImplementedError(f"Unsupported k-path mode: {mode}")

    try:
        kpath = generate_kpath(
            atoms,
            kpath_mode=mode,
            npoints=int(npoints),
            symprec=float(symprec),
            with_time_reversal=bool(with_time_reversal),
            **kwargs,
        )
        return band_path_from_kpath_result(kpath, npoints=int(npoints))
    except Exception:
        return _fallback_band_path(npoints=npoints)


def build_band_path(*args: Any, **kwargs: Any) -> BandPath:
    """Backward-compatible alias for band path construction."""

    return get_band_path(*args, **kwargs)


def generate_kpath(
    atoms: Any,
    kpath_mode: str = "auto",
    npoints: int = 101,
    symprec: float = DEFAULT_SEEKPATH_SYMPREC,
    with_time_reversal: bool = DEFAULT_SEEKPATH_WITH_TIME_REVERSAL,
    path_labels: list[str] | None = None,
    special_points: dict[str, list[float]] | None = None,
    path_segments: list[tuple[str, str]] | None = None,
) -> KPathResult:
    """Return a unified k-path description for 2D and 3D structures."""

    mode = str(kpath_mode).lower()
    if mode == "custom":
        return _generate_custom_kpath(
            atoms,
            path_labels=path_labels,
            special_points=special_points,
            path_segments=path_segments,
        )
    if mode == "3d_seekpath":
        return _generate_3d_seekpath_kpath(
            atoms,
            npoints=npoints,
            symprec=symprec,
            with_time_reversal=with_time_reversal,
        )
    if mode == "2d_ase":
        return generate_ase_2d_kpath(atoms, npoints=npoints, eps=max(float(symprec), DEFAULT_ASE_2D_EPS))
    if mode == "auto":
        try:
            dimensionality = infer_dimensionality_by_vacuum(atoms)
        except Exception:
            dimensionality = {"dimension": "3D"}
        if dimensionality.get("dimension") == "2D":
            return generate_ase_2d_kpath(atoms, npoints=npoints, eps=max(float(symprec), DEFAULT_ASE_2D_EPS))
        return _generate_3d_seekpath_kpath(
            atoms,
            npoints=npoints,
            symprec=symprec,
            with_time_reversal=with_time_reversal,
        )
    raise ValueError(f"Unsupported kpath_mode: {kpath_mode}")


def band_path_from_kpath_result(kpath: KPathResult, npoints: int = 101) -> BandPath:
    existing = kpath.metadata.get("band_path")
    if isinstance(existing, BandPath):
        return existing

    if len(kpath.phonopy_band) != len(kpath.path_segments):
        raise RuntimeError("KPathResult phonopy_band and path_segments are inconsistent.")

    qpoint_segments: list[np.ndarray] = []
    linear_segments: list[np.ndarray] = []
    endpoint_labels: list[str] = []
    segments: list[tuple[str, str]] = []
    explicit_qpoints: list[np.ndarray] = []
    explicit_linearcoords: list[np.ndarray] = []
    explicit_labels: list[str] = []
    distance_offset = 0.0

    for segment_points, (start_label, end_label) in zip(kpath.phonopy_band, kpath.path_segments, strict=False):
        start = np.asarray(segment_points[0], dtype=float)
        end = np.asarray(segment_points[1], dtype=float)
        qpoint_segment = np.linspace(start, end, int(npoints))
        deltas = np.diff(qpoint_segment, axis=0)
        lengths = np.linalg.norm(deltas, axis=1)
        linear_segment = distance_offset + np.concatenate(([0.0], np.cumsum(lengths)))
        distance_offset = float(linear_segment[-1])

        start_display = normalize_band_label(start_label)
        end_display = normalize_band_label(end_label)
        labels = [""] * len(qpoint_segment)
        labels[0] = start_display
        labels[-1] = end_display

        qpoint_segments.append(qpoint_segment)
        linear_segments.append(linear_segment)
        endpoint_labels.extend([start_display, end_display])
        segments.append((start_display, end_display))
        explicit_qpoints.append(qpoint_segment)
        explicit_linearcoords.append(linear_segment)
        explicit_labels.extend(labels)

    source = "ase-2d" if kpath.mode == "2d_ase" else kpath.source
    return BandPath(
        qpoints=qpoint_segments,
        labels=endpoint_labels,
        source=source,
        segments=segments,
        explicit_kpoints_rel=np.vstack(explicit_qpoints),
        explicit_kpoints_linearcoord=np.concatenate(explicit_linearcoords),
        explicit_kpoints_labels=explicit_labels,
        segment_linearcoords=linear_segments,
    )


def _generate_3d_seekpath_kpath(
    atoms: Any,
    *,
    npoints: int = 101,
    symprec: float = DEFAULT_SEEKPATH_SYMPREC,
    with_time_reversal: bool = DEFAULT_SEEKPATH_WITH_TIME_REVERSAL,
) -> KPathResult:
    import seekpath

    band_path = _seekpath_explicit_band_path(
        atoms,
        symprec=float(symprec),
        with_time_reversal=bool(with_time_reversal),
    )
    try:
        path_data = seekpath.get_path(
            seekpath_cell_from_atoms(atoms),
            symprec=float(symprec),
            with_time_reversal=bool(with_time_reversal),
        )
        path_segments = [(str(start), str(end)) for start, end in path_data.get("path", [])]
        point_coords = {
            str(label): [float(value) for value in coords]
            for label, coords in path_data.get("point_coords", {}).items()
        }
        primitive_lattice = np.asarray(path_data.get("primitive_lattice"), dtype=float).tolist()
        reciprocal_lattice = np.asarray(path_data.get("reciprocal_primitive_lattice"), dtype=float).tolist()
    except Exception:
        path_segments = [(str(start), str(end)) for start, end in band_path.segments]
        point_coords = {}
        for (start, end), segment in zip(path_segments, band_path.qpoints, strict=False):
            point_coords.setdefault(str(start), [float(value) for value in np.asarray(segment[0], dtype=float)])
            point_coords[str(end)] = [float(value) for value in np.asarray(segment[-1], dtype=float)]
        primitive_lattice = np.asarray(atoms.cell.array, dtype=float).tolist()
        try:
            reciprocal_lattice = np.asarray(atoms.cell.reciprocal().array, dtype=float).tolist()
        except Exception:
            reciprocal_lattice = np.linalg.pinv(np.asarray(atoms.cell.array, dtype=float)).T.tolist()
    path_labels = _flatten_path_labels(path_segments)
    phonopy_band = [
        [point_coords[start], point_coords[end]]
        for start, end in path_segments
        if start in point_coords and end in point_coords
    ]
    return KPathResult(
        mode="3d_seekpath",
        dimensionality="3D",
        source="seekpath",
        bravais=None,
        path_string="-".join(path_labels),
        display_path=display_path_string(path_labels),
        path_labels=path_labels,
        special_points=point_coords,
        path_segments=path_segments,
        phonopy_band=phonopy_band,
        metadata={
            "band_path": band_path,
            "symprec": float(symprec),
            "with_time_reversal": bool(with_time_reversal),
            "primitive_lattice": primitive_lattice,
            "reciprocal_lattice": reciprocal_lattice,
            "seekpath_path": [[start, end] for start, end in path_segments],
        },
    )


def _generate_custom_kpath(
    atoms: Any,
    *,
    path_labels: list[str] | None = None,
    special_points: dict[str, list[float]] | None = None,
    path_segments: list[tuple[str, str]] | None = None,
) -> KPathResult:
    if not special_points:
        raise ValueError("custom kpath_mode requires special_points.")
    if path_segments is None:
        if not path_labels or len(path_labels) < 2:
            raise ValueError("custom kpath_mode requires path_labels or path_segments.")
        path_segments = list(zip(path_labels[:-1], path_labels[1:], strict=False))
    labels = path_labels or _flatten_path_labels(path_segments)
    dimensionality = infer_dimensionality_by_vacuum(atoms)
    phonopy_band = [
        [list(map(float, special_points[start])), list(map(float, special_points[end]))]
        for start, end in path_segments
    ]
    return KPathResult(
        mode="custom",
        dimensionality=str(dimensionality.get("dimension", "3D")),
        source="custom",
        bravais=None,
        path_string="-".join(labels),
        display_path=display_path_string(labels),
        path_labels=[str(label) for label in labels],
        special_points={str(label): [float(value) for value in coords] for label, coords in special_points.items()},
        path_segments=[(str(start), str(end)) for start, end in path_segments],
        phonopy_band=phonopy_band,
        vacuum_axis=dimensionality.get("vacuum_axis"),
        vacuum_axis_name=dimensionality.get("vacuum_axis_name"),
        metadata={"reason": "User-supplied custom k-path."},
    )


def _seekpath_explicit_band_path(
    atoms: Any,
    *,
    symprec: float = DEFAULT_SEEKPATH_SYMPREC,
    with_time_reversal: bool = DEFAULT_SEEKPATH_WITH_TIME_REVERSAL,
) -> BandPath:
    import seekpath

    structure = seekpath_cell_from_atoms(atoms)
    path_data = seekpath.get_explicit_k_path(
        structure,
        symprec=float(symprec),
        with_time_reversal=bool(with_time_reversal),
    )
    explicit_qpoints = np.asarray(path_data["explicit_kpoints_rel"], dtype=float)
    explicit_linear = np.asarray(path_data["explicit_kpoints_linearcoord"], dtype=float)
    explicit_labels = [
        normalize_band_label(label) if label else "" for label in path_data["explicit_kpoints_labels"]
    ]
    if explicit_qpoints.size == 0:
        raise RuntimeError("seekpath returned no explicit k-points.")

    segment_slices = _split_explicit_segments(explicit_linear, explicit_labels)
    qpoint_segments: list[np.ndarray] = []
    linear_segments: list[np.ndarray] = []
    endpoint_labels: list[str] = []
    segments: list[tuple[str, str]] = []
    for start, stop in segment_slices:
        qpoint_segment = explicit_qpoints[start:stop]
        linear_segment = explicit_linear[start:stop]
        start_label = _first_label(explicit_labels[start:stop], default=f"q{len(segments)}_start")
        end_label = _last_label(explicit_labels[start:stop], default=f"q{len(segments)}_end")
        qpoint_segments.append(qpoint_segment)
        linear_segments.append(linear_segment)
        endpoint_labels.extend([start_label, end_label])
        segments.append((start_label, end_label))

    return BandPath(
        qpoints=qpoint_segments,
        labels=endpoint_labels,
        source="seekpath",
        segments=segments,
        explicit_kpoints_rel=explicit_qpoints,
        explicit_kpoints_linearcoord=explicit_linear,
        explicit_kpoints_labels=explicit_labels,
        segment_linearcoords=linear_segments,
    )


def seekpath_cell_from_atoms(atoms: Any) -> tuple[Any, Any, Any]:
    """Return the cell tuple consumed by SeekPath from an ASE-like Atoms object."""

    return (
        atoms.cell.array,
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers(),
    )


def format_path_segments(segments: list[tuple[Any, Any]]) -> str:
    """Format high-symmetry path segments using segment ordering."""

    formatted = [
        f"{normalize_band_label(str(start))}{PATH_SEGMENT_SEPARATOR}{normalize_band_label(str(end))}"
        for start, end in segments
        if str(start).strip() and str(end).strip()
    ]
    return PATH_BREAK_SEPARATOR.join(formatted) if formatted else "not available"


def high_symmetry_path_metadata(
    segments: list[tuple[Any, Any]],
    *,
    source: str = "seekpath",
    symprec: float = DEFAULT_SEEKPATH_SYMPREC,
    with_time_reversal: bool = DEFAULT_SEEKPATH_WITH_TIME_REVERSAL,
    kpath: KPathResult | None = None,
) -> dict[str, Any]:
    """Return result.json/summary metadata from explicit path segments."""

    clean_segments = [
        [normalize_band_label(str(start)), normalize_band_label(str(end))]
        for start, end in segments
        if str(start).strip() and str(end).strip()
    ]
    labels: list[str] = []
    for start, end in clean_segments:
        labels.extend([start, end])
    metadata = {
        "available": bool(clean_segments),
        "source": source,
        "symprec": float(symprec),
        "with_time_reversal": bool(with_time_reversal) if source == "seekpath" else None,
        "labels": labels,
        "segments": clean_segments,
        "display": format_path_segments([(start, end) for start, end in clean_segments]),
    }
    if kpath is not None:
        metadata.update(
            {
                "mode": kpath.mode,
                "dimensionality": kpath.dimensionality,
                "bravais": kpath.bravais,
                "path_string": kpath.path_string,
                "display_path": kpath.display_path,
                "path_labels": list(kpath.path_labels),
                "special_points": {
                    str(label): [float(value) for value in coords]
                    for label, coords in kpath.special_points.items()
                },
                "path_segments": [[str(start), str(end)] for start, end in kpath.path_segments],
                "vacuum_axis": kpath.vacuum_axis,
                "vacuum_axis_name": kpath.vacuum_axis_name,
                "path_generator": "ASE Cell.bandpath" if kpath.source == "ase_cell_bandpath" else "SeekPath",
                "warning": kpath.warning,
            }
        )
    return metadata


def _fallback_band_path(npoints: int) -> BandPath:
    gamma = np.array([0.0, 0.0, 0.0])
    x_point = np.array([0.5, 0.0, 0.0])
    qpoints = np.linspace(gamma, x_point, npoints)
    linear = np.linspace(0.0, 1.0, npoints)
    gamma_label = normalize_band_label("G")
    labels = [""] * npoints
    labels[0] = gamma_label
    labels[-1] = "X"
    return BandPath(
        qpoints=[qpoints],
        labels=[gamma_label, "X"],
        source="fallback",
        segments=[(gamma_label, "X")],
        explicit_kpoints_rel=qpoints,
        explicit_kpoints_linearcoord=linear,
        explicit_kpoints_labels=labels,
        segment_linearcoords=[linear],
    )


def _split_explicit_segments(linearcoord: np.ndarray, labels: list[str]) -> list[tuple[int, int]]:
    """Split explicit SeekPath output into labeled path segments."""

    if len(linearcoord) == 0:
        return []
    label_indices = [index for index, label in enumerate(labels) if label]
    if len(label_indices) < 2:
        return [(0, len(linearcoord))]

    segments: list[tuple[int, int]] = []
    for start, end in zip(label_indices, label_indices[1:], strict=False):
        same_position = abs(float(linearcoord[end]) - float(linearcoord[start])) < 1e-8
        if same_position and labels[start] != labels[end]:
            continue
        if end > start:
            segments.append((start, end + 1))
    return segments or [(0, len(linearcoord))]


def _first_label(labels: list[str], default: str) -> str:
    for label in labels:
        if label:
            return label
    return default


def _last_label(labels: list[str], default: str) -> str:
    for label in reversed(labels):
        if label:
            return label
    return default


def _flatten_path_labels(segments: list[tuple[str, str]]) -> list[str]:
    labels: list[str] = []
    for start, end in segments:
        if not labels:
            labels.append(str(start))
        elif labels[-1] != str(start):
            labels.append(str(start))
        labels.append(str(end))
    return labels


def collapse_path_ticks(ticks: list[float], labels: list[str]) -> tuple[list[float], list[str]]:
    """Collapse same-position explicit labels into plot ticks."""

    return collapse_tick_labels(ticks, labels)


def write_band_path_json(
    band_path: BandPath,
    path: Path,
    labels_for_plot: list[str] | None = None,
    tick_positions: list[float] | None = None,
    high_symmetry_path: dict[str, Any] | None = None,
    kpath: KPathResult | None = None,
    requested_mode: str | None = None,
) -> None:
    """Write the explicit high-symmetry path used for the band calculation."""

    if labels_for_plot is None or tick_positions is None:
        tick_positions, labels_for_plot = collapse_path_ticks(
            band_path.explicit_kpoints_linearcoord.tolist(),
            band_path.explicit_kpoints_labels,
        )
    metadata = high_symmetry_path or high_symmetry_path_metadata(
        band_path.segments,
        source=band_path.source,
        kpath=kpath,
    )
    payload = {
        "path_source": band_path.source,
        "bandpath_symprec": metadata.get("symprec"),
        "bandpath_with_time_reversal": metadata.get("with_time_reversal"),
        "seekpath_symprec": metadata.get("symprec") if band_path.source == "seekpath" else None,
        "seekpath_with_time_reversal": metadata.get("with_time_reversal") if band_path.source == "seekpath" else None,
        "explicit_kpoints_rel": band_path.explicit_kpoints_rel.tolist(),
        "explicit_kpoints_linearcoord": band_path.explicit_kpoints_linearcoord.tolist(),
        "explicit_kpoints_labels": band_path.explicit_kpoints_labels,
        "tick_positions": tick_positions,
        "tick_labels": labels_for_plot,
        "segments": [[start, end] for start, end in band_path.segments],
        "labels": band_path.labels,
        "labels_for_plot": labels_for_plot,
        "npoints_per_segment": [int(len(segment)) for segment in band_path.qpoints],
        "kpath": serialize_kpath_result(kpath, requested_mode=requested_mode) if kpath is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_band_yaml_path_metadata(
    band_yaml_path: Path,
    band_path: BandPath,
    high_symmetry_path: dict[str, Any] | None = None,
    kpath: KPathResult | None = None,
    requested_mode: str | None = None,
) -> None:
    """Embed PhonoFlow path labels into Phonopy ``band.yaml``."""

    path = Path(band_yaml_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"band.yaml did not contain a mapping: {path}")

    metadata = high_symmetry_path or high_symmetry_path_metadata(
        band_path.segments,
        source=band_path.source,
        kpath=kpath,
    )
    data["labels"] = [[start, end] for start, end in band_path.segments]
    data["phonoflow_high_symmetry_path"] = metadata
    data["phonoflow_bandpath"] = {
        "symprec": metadata.get("symprec"),
        "with_time_reversal": metadata.get("with_time_reversal"),
        "source": band_path.source,
        "kpath": serialize_kpath_result(kpath, requested_mode=requested_mode) if kpath is not None else None,
    }
    data["phonoflow_seekpath"] = {
        "symprec": metadata.get("symprec") if band_path.source == "seekpath" else None,
        "with_time_reversal": metadata.get("with_time_reversal") if band_path.source == "seekpath" else None,
        "source": band_path.source,
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
