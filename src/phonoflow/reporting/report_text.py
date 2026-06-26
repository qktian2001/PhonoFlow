"""Human-readable text report writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def format_stability_text(report: dict[str, Any]) -> str:
    """Format a stability report as plain text."""

    stable = "Yes" if report.get("dynamically_stable") else "No"
    return "\n".join(
        [
            f"Project: {report.get('project')}",
            f"Backend: {report.get('backend')}",
            f"Dynamically stable: {stable}",
            f"Minimum frequency: {report.get('minimum_frequency_THz', 0.0):.3f} THz",
            f"Imaginary mode count: {report.get('imaginary_mode_count')}",
            f"Imaginary mode ratio: {report.get('imaginary_mode_ratio', 0.0):.3f}",
            f"Note: {report.get('notes')}",
            "",
        ]
    )


def write_stability_text(report: dict[str, Any], path: Path) -> None:
    """Write a human-readable stability report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_stability_text(report), encoding="utf-8")
