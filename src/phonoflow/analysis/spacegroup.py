"""Space-group diagnostics using spglib."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def analyze_spacegroup(
    atoms: Any,
    symprec: float = 1e-5,
    angle_tolerance: float = -1.0,
) -> dict[str, Any]:
    """Analyze the space group of an ASE Atoms object.

    Failures are returned as structured diagnostics instead of aborting the
    workflow, because symmetry detection is a reporting aid.
    """

    try:
        import spglib

        cell = (
            atoms.cell.array,
            atoms.get_scaled_positions(wrap=True),
            atoms.get_atomic_numbers(),
        )
        dataset = spglib.get_symmetry_dataset(
            cell,
            symprec=float(symprec),
            angle_tolerance=float(angle_tolerance),
        )
        if dataset is None:
            raise RuntimeError("spglib returned no symmetry dataset")

        number = _dataset_get(dataset, "number")
        international = _dataset_get(dataset, "international")
        hall = _dataset_get(dataset, "hall")
        pointgroup = _dataset_get(dataset, "pointgroup")
        return {
            "spacegroup_number": int(number) if number is not None else None,
            "international_symbol": str(international) if international is not None else None,
            "hall_symbol": str(hall) if hall is not None else None,
            "pointgroup": str(pointgroup) if pointgroup is not None else None,
            "crystal_system": crystal_system_from_number(number),
            "symprec": float(symprec),
            "angle_tolerance": float(angle_tolerance),
            "dataset_available": True,
            "error": None,
        }
    except Exception as exc:
        return {
            "spacegroup_number": None,
            "international_symbol": None,
            "hall_symbol": None,
            "pointgroup": None,
            "crystal_system": None,
            "symprec": float(symprec),
            "angle_tolerance": float(angle_tolerance),
            "dataset_available": False,
            "error": str(exc),
        }


def build_spacegroup_report(
    initial: dict[str, Any],
    final: dict[str, Any] | None,
    symprec: float,
    angle_tolerance: float,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build a before/after space-group comparison report."""

    warnings: list[str] = []
    if final is None:
        final = dict(initial)
        warnings.append("Dry run did not execute relaxation; final space group is reported as same as initial.")

    changed = _spacegroup_key(initial) != _spacegroup_key(final)
    if not initial.get("dataset_available") or not final.get("dataset_available"):
        warnings.append("One or more space-group datasets were unavailable; compare symmetry with care.")

    if changed:
        change_summary = (
            f"Space group changed from {_format_spacegroup(initial)} to "
            f"{_format_spacegroup(final)}. Check relaxation settings and symprec."
        )
    else:
        change_summary = f"Space group preserved: {_format_spacegroup(final)}."
    if dry_run:
        change_summary = f"Dry run: {change_summary}"

    return {
        "symprec": float(symprec),
        "angle_tolerance": float(angle_tolerance),
        "initial": initial,
        "final": final,
        "changed": bool(changed),
        "change_summary": change_summary,
        "warnings": warnings,
    }


def write_spacegroup_report(report: dict[str, Any], json_path: Path, text_path: Path) -> None:
    """Write JSON and text space-group reports."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    text_path.write_text(format_spacegroup_report(report), encoding="utf-8")


def format_spacegroup_report(report: dict[str, Any]) -> str:
    """Format a readable space-group report."""

    initial = report.get("initial") or {}
    final = report.get("final") or {}
    warnings = report.get("warnings") or []
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "None"
    return "\n".join(
        [
            "# PhonoFlow Space Group Report",
            "",
            "## Symmetry tolerance",
            f"symprec: {report.get('symprec')}",
            f"angle_tolerance: {report.get('angle_tolerance')}",
            "",
            "## Initial structure",
            f"Space group: {_format_spacegroup(initial)}",
            f"Point group: {initial.get('pointgroup')}",
            f"Crystal system: {initial.get('crystal_system')}",
            f"Dataset available: {initial.get('dataset_available')}",
            f"Error: {initial.get('error')}",
            "",
            "## Relaxed structure",
            f"Space group: {_format_spacegroup(final)}",
            f"Point group: {final.get('pointgroup')}",
            f"Crystal system: {final.get('crystal_system')}",
            f"Dataset available: {final.get('dataset_available')}",
            f"Error: {final.get('error')}",
            "",
            "## Comparison",
            f"Changed: {'Yes' if report.get('changed') else 'No'}",
            f"Summary: {report.get('change_summary')}",
            "",
            "## Warnings",
            warning_text,
            "",
        ]
    )


def crystal_system_from_number(number: Any) -> str | None:
    """Return the crystal system for an international space-group number."""

    if number is None:
        return None
    number = int(number)
    if 1 <= number <= 2:
        return "triclinic"
    if number <= 15:
        return "monoclinic"
    if number <= 74:
        return "orthorhombic"
    if number <= 142:
        return "tetragonal"
    if number <= 167:
        return "trigonal"
    if number <= 194:
        return "hexagonal"
    if number <= 230:
        return "cubic"
    return None


def _dataset_get(dataset: Any, key: str) -> Any:
    if isinstance(dataset, dict):
        return dataset.get(key)
    return getattr(dataset, key, None)


def _spacegroup_key(data: dict[str, Any]) -> tuple[Any, Any]:
    return data.get("spacegroup_number"), data.get("international_symbol")


def _format_spacegroup(data: dict[str, Any]) -> str:
    symbol = data.get("international_symbol")
    number = data.get("spacegroup_number")
    if symbol and number:
        return f"{symbol} (No. {number})"
    if symbol:
        return str(symbol)
    if number:
        return f"No. {number}"
    return "unavailable"
