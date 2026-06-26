from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class KPathResult:
    mode: str
    dimensionality: Literal["2D", "3D"]
    source: str
    bravais: Optional[str]
    path_string: str
    display_path: str
    path_labels: list[str]
    special_points: dict[str, list[float]]
    path_segments: list[tuple[str, str]]
    phonopy_band: list[list[list[float]]]
    vacuum_axis: Optional[int] = None
    vacuum_axis_name: Optional[str] = None
    warning: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def serialize_kpath_result(result: KPathResult, requested_mode: str | None = None) -> dict[str, Any]:
    return {
        "mode": requested_mode or result.mode,
        "resolved_mode": result.mode,
        "dimensionality": result.dimensionality,
        "source": result.source,
        "path_generator": "ASE Cell.bandpath" if result.source == "ase_cell_bandpath" else "SeekPath",
        "bravais": result.bravais,
        "path_string": result.path_string,
        "display_path": result.display_path,
        "path_labels": list(result.path_labels),
        "special_points": {
            str(label): [float(value) for value in coords]
            for label, coords in result.special_points.items()
        },
        "path_segments": [[str(start), str(end)] for start, end in result.path_segments],
        "phonopy_band": [
            [[float(value) for value in point] for point in segment]
            for segment in result.phonopy_band
        ],
        "vacuum_axis": result.vacuum_axis_name if result.vacuum_axis_name is not None else result.vacuum_axis,
        "vacuum_axis_index": result.vacuum_axis,
        "vacuum_axis_name": result.vacuum_axis_name,
        "warning": result.warning,
        "metadata": _jsonable_metadata(result.metadata),
    }


def _jsonable_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        payload: dict[str, Any] = {}
        for key, item in value.items():
            if key == "band_path":
                continue
            payload[str(key)] = _jsonable_metadata(item)
        return payload
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable_metadata(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_metadata(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
